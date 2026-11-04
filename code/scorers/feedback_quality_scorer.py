import os

# Set CUBLAS environment variable BEFORE importing torch for deterministic behavior
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')

import sys
import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import weave
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# Make training utilities importable
CODE_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = CODE_ROOT / 'training'
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from feedback_quality_training_utils import (
    SEED,
    set_seed,
    RegressionModelWithClipping,
    build_input_text,
)

# Tasks
MODERNBERT_TASKS = ['correctness', 'error-detection', 'specificity']

# Default base directories for trained models. They can be overridden without
# editing the code, which is useful when running from a submission checkout.
MODELS_BASE_DIR = Path(os.environ.get('STORYS_MODELS_DIR', '/storys/models_final'))
MODELS_FBONLY_BASE_DIR = Path(
    os.environ.get('STORYS_FBONLY_MODELS_DIR', '/storys/models_fbonly_final')
)
MODELS_QWENRM_BASE_DIR = Path(
    os.environ.get('STORYS_QWENRM_MODELS_DIR', '/storys/models_qwen')
)


# Set seed for reproducibility
set_seed(SEED)


class FeedbackQualityScorer:
    """Scorer for feedback quality using trained task-specific regression models."""
    
    def __init__(self, device, model_path: str = None, max_length: Optional[int] = None, fbonly: bool = False, qwenrm: bool = False):
        """
        Initialize the feedback quality scorer.
        
        Args:
            device: torch device or device string
            model_path: Path to the trained model checkpoint directory
            max_length: Maximum sequence length for tokenization. If None, read from model hyperparameters.
            fbonly: Whether to use only feedback (no story/essay) as input
            qwenrm: Whether to use Qwen reward models
        """
        self.device = self._process_device(device)
        self.max_length = max_length
        self._max_length_overridden = max_length is not None
        self.task_names = MODERNBERT_TASKS
        self.fbonly = fbonly
        self.qwenrm = qwenrm
        
        if self.qwenrm and self.fbonly:
            raise ValueError("Qwen reward models are not compatible with feedback-only input. Set fbonly=False if using qwenrm=True.")

        # Tokenizer is loaded from checkpoint settings when models are loaded.
        self.tokenizer = None
        
        # Load models        
        self.models = self._load_task_models(model_path)

        # Fallback defaults when metadata is missing.
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                elif self.tokenizer.eos_token_id is not None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                else:
                    raise ValueError("Tokenizer has neither pad_token nor eos_token.")
        if self.max_length is None:
            self.max_length = 512

        for model in self.models.values():
            model.eval()
        self.model = None
        
        # Cache for predictions to avoid redundant forward passes
        self._cache = {}
    
    def _process_device(self, device):
        """Convert device to torch.device."""
        if isinstance(device, torch.device):
            return device
        elif isinstance(device, str):
            return torch.device(device)
        elif isinstance(device, int):
            return torch.device(f'cuda:{device}' if device >= 0 else 'cpu')
        return torch.device('cpu')

    def _load_model_settings(self, model_dir: Path) -> dict:
        """Load per-task model settings from hyperparameters if available."""
        settings = {
            "model_name": "answerdotai/ModernBERT-large",
            "max_length": 512,
            "fbonly": False,
        }
        hyperparams_path = model_dir / "hyperparameters.json"
        if hyperparams_path.exists():
            with open(hyperparams_path, "r") as f:
                hyperparams = json.load(f)
            settings["model_name"] = hyperparams.get("model_name", settings["model_name"])
            settings["max_length"] = int(hyperparams.get("max_length", settings["max_length"]))
            settings["fbonly"] = bool(hyperparams.get("fbonly", settings["fbonly"]))
        return settings

    def _load_tokenizer(self, tokenizer_source: Union[Path, str]):
        """Load tokenizer and ensure a valid pad token is set."""
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                raise ValueError(f"Tokenizer has no pad/eos token for {tokenizer_source}")
        return tokenizer
    
    def _load_regression_model(self, checkpoint_path, num_labels=1, model_name="answerdotai/ModernBERT-large"):
        """Load a regression model from checkpoint and wrap with clipping."""
        checkpoint_path = Path(checkpoint_path)
        
        # If checkpoint_path is a directory (Trainer saved model), check if it has config.json
        if checkpoint_path.is_dir():
            config_file = checkpoint_path / 'config.json'
            if config_file.exists():
                # Proper Trainer-saved model with config
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint_path,
                    num_labels=num_labels,
                    problem_type="regression",
                    ignore_mismatched_sizes=True
                )
                # Wrap with clipping
                model = RegressionModelWithClipping(base_model)
                return model
            else:
                # Directory exists but no config - look for pytorch_model.bin
                model_file = checkpoint_path / 'pytorch_model.bin'
                if model_file.exists():
                    checkpoint_path = model_file
                else:
                    # Try model.safetensors
                    model_file = checkpoint_path / 'model.safetensors'
                    if model_file.exists():
                        checkpoint_path = model_file
        
        # If we get here, we have a checkpoint file (or directory without config)
        # Load into fresh model with custom config
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        config.problem_type = "regression"
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Load checkpoint weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle both direct state_dict and wrapped checkpoint formats
        if 'model_state_dict' in checkpoint:
            base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            base_model.load_state_dict(checkpoint)
        
        # Wrap with clipping
        model = RegressionModelWithClipping(base_model)
        return model 
    
    
    def _load_task_models(self, base_path: Union[Path, str] = None):
        """Load separate models for each task."""
        if base_path is None:
            if not self.fbonly and not self.qwenrm:
                base_path = MODELS_BASE_DIR
            elif self.fbonly and not self.qwenrm:
                base_path = MODELS_FBONLY_BASE_DIR
            elif self.qwenrm:
                base_path = MODELS_QWENRM_BASE_DIR

        base_path = Path(base_path)
        models = {}
        for task in self.task_names:
            task_dir = task.replace('-', '_')
            model_dir = base_path / f"{task_dir}"
            
            if not model_dir.exists():
                raise FileNotFoundError(f"Could not find model directory for task '{task}' in {model_dir}")

            settings = self._load_model_settings(model_dir)

            if self.tokenizer is None:
                tokenizer_source = model_dir if (model_dir / "tokenizer_config.json").exists() else settings["model_name"]
                self.tokenizer = self._load_tokenizer(tokenizer_source)
                if not self._max_length_overridden:
                    self.max_length = settings["max_length"]
                print(
                    f"Loaded scorer tokenizer from {tokenizer_source}; "
                    f"max_length={self.max_length}"
                )

            if self.qwenrm and settings.get("fbonly", False):
                raise ValueError(
                    f"Qwen reward model for task '{task}' was trained with fbonly=True, "
                    "which is incompatible with qwenrm scoring."
                )
            
            print(f"Loading model for '{task}' from {model_dir}")
            
            model = self._load_regression_model(
                model_dir,
                num_labels=1,
                model_name=settings["model_name"]
            )
            model = model.to(self.device)
            
            models[task] = model
        
        return models
    
    def _predict(self, story: str, feedback: str) -> dict:
        """
        Internal method to get predictions for all tasks.
        Uses caching to avoid redundant forward passes for the same input.
        
        Args:
            story: The story text
            feedback: The feedback text
        
        Returns:
            Dictionary mapping task names to prediction scores
        """
        # Create cache key from story and feedback
        cache_key = (story, feedback)
        
        # Return cached result if available
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Format text exactly as during training
        text = build_input_text(story, feedback, self.fbonly)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            # one forward pass per task
            result = {}
            for task in self.task_names:
                outputs = self.models[task](input_ids, attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                prediction = logits.cpu().numpy().squeeze()
                if prediction.ndim == 0:
                    prediction = float(prediction)
                result[task] = float(prediction)
        
        # Cache the result
        self._cache[cache_key] = result
        
        return result
    
    @weave.op()
    def get_correctness_score(self, story: str, feedback: str) -> float:
        """Get correctness score for feedback."""
        return self._predict(story, feedback)['correctness']
    
    @weave.op()
    def get_error_detection_score(self, story: str, feedback: str) -> float:
        """Get error-detection score for feedback."""
        return self._predict(story, feedback)['error-detection']
    
    @weave.op()
    def get_specificity_score(self, story: str, feedback: str) -> float:
        """Get specificity score for feedback."""
        return self._predict(story, feedback)['specificity']
    
    @weave.op()
    def get_all_scores(self, story: str, feedback: str) -> dict:
        """Get all task scores for feedback in one forward pass."""
        return self._predict(story, feedback)