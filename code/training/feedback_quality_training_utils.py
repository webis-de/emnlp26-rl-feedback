#!/usr/bin/env python3
"""
Shared utilities for single-task ModernBERT training.
Contains dataset, model, and metrics functions used across all task-specific scripts.
"""

import os
from typing import Optional

# Set CUBLAS environment variable BEFORE importing torch for deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    default_data_collator
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import wandb
import torch.nn as nn
import torch.nn.functional as F

# Seeds for reproducibility
SEED = 8549


def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms (may impact performance)
        torch.use_deterministic_algorithms(True)
        # Disable benchmark for deterministic behavior
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# Set seed at module import
set_seed(SEED)


def _is_missing_value(value) -> bool:
    """Return True for None/NaN values that should be treated as missing."""
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))


def build_input_text(story: str, feedback: str, fbonly: bool = False) -> str:
    """Build model input text using the shared training/scoring format."""
    if _is_missing_value(feedback):
        raise ValueError("Missing required field 'feedback'")
    if not fbonly and _is_missing_value(story):
        raise ValueError("Missing required field 'story'")

    if fbonly:
        return f"{feedback}"
    return f"Text: {story}\n\nFeedback: {feedback}"


class HuberLossTrainer(Trainer):
    """Custom Trainer using Huber loss instead of MSE.
    
    Huber loss is less sensitive to outliers than MSE, which helps models
    learn to produce diverse predictions instead of regressing to the mean.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Huber loss: smooth transition between L1 (MAE) and L2 (MSE)
        # delta controls the transition point
        # delta=0.5 is good for normalized scores in ~[0, 1] range
        loss_fn = nn.HuberLoss(delta=0.5, reduction='mean')
        loss = loss_fn(logits.squeeze(-1), labels)
        
        return (loss, outputs) if return_outputs else loss


class MAETrainer(Trainer):
    """Custom Trainer using Mean Absolute Error (L1 loss).
    
    MAE encourages diverse predictions by treating all errors equally,
    regardless of magnitude (unlike MSE which heavily penalizes large errors).
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        loss_fn = nn.L1Loss(reduction='mean')
        loss = loss_fn(logits.squeeze(-1), labels)
        
        return (loss, outputs) if return_outputs else loss


class SingleTaskHumanRatingsDataset(Dataset):
    """Dataset for human ratings with text inputs - single task version."""
    
    def __init__(
        self,
        filepath: str,
        tokenizer,
        task_field: str,
        fbonly=False,
        max_length: int = 1024,
        skip_invalid_rows: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_field = task_field
        self.fbonly = fbonly
        self.data = []
        self.num_skipped = 0
        self.skipped_missing_label = 0
        self.skipped_invalid_label = 0
        self.skipped_invalid_text = 0
        
        total_rows = 0
        
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                total_rows += 1
                item = json.loads(line)
                
                # Get the score for this task
                if task_field not in item:
                    if skip_invalid_rows:
                        self.num_skipped += 1
                        self.skipped_missing_label += 1
                        continue
                    raise KeyError(
                        f"{filepath}:{line_num}: missing required label field '{task_field}'"
                    )
                val = item[task_field]
                if _is_missing_value(val):
                    if skip_invalid_rows:
                        self.num_skipped += 1
                        self.skipped_missing_label += 1
                        continue
                    raise ValueError(
                        f"{filepath}:{line_num}: label field '{task_field}' is missing/null"
                    )
                try:
                    score = float(val)
                except (TypeError, ValueError):
                    if skip_invalid_rows:
                        self.num_skipped += 1
                        self.skipped_invalid_label += 1
                        continue
                    raise ValueError(
                        f"{filepath}:{line_num}: label field '{task_field}' could not be converted to float: {val!r}"
                    )
                
                # Create input text from story and feedback
                story = item.get('story')
                feedback = item.get('feedback')
                try:
                    text = build_input_text(story, feedback, self.fbonly)
                except ValueError as exc:
                    if skip_invalid_rows:
                        self.num_skipped += 1
                        self.skipped_invalid_text += 1
                        continue
                    raise ValueError(f"{filepath}:{line_num}: {exc}") from exc
                
                self.data.append({
                    'text': text,
                    'score': score,
                    'example_id': item.get('example_id', '')
                })

        if self.num_skipped > 0:
            print(
                f"Skipped {self.num_skipped}/{total_rows} rows from {filepath} for task '{task_field}' "
                f"(missing/null label: {self.skipped_missing_label}, "
                f"non-numeric label: {self.skipped_invalid_label}, "
                f"missing story/feedback: {self.skipped_invalid_text})"
            )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['score'], dtype=torch.float32),
        }


class RegressionModelWithClipping(nn.Module):
    """Wrapper that clips regression output to [0, 1] range."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Copy important attributes from base model so Trainer recognizes it
        self.config = base_model.config
        self.device = base_model.device
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Forward through base model with explicit parameters
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        # Clip logits to [0, 1] range
        if hasattr(outputs, 'logits'):
            outputs.logits = torch.clamp(outputs.logits, min=0.0, max=1.0)
        return outputs
    
    def __getattr__(self, name):
        """Delegate attribute access to base_model for Trainer compatibility."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def create_regression_model(model_name: str, dropout: float = 0.1):
    """Create a ModernBERT model configured for regression with output clipped to [0, 1]."""
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 1
    config.problem_type = "regression"
    config.hidden_dropout_prob = dropout
    config.classifier_dropout = dropout
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )
    
    # Wrap with clipping to constrain output to [0, 1]
    model = RegressionModelWithClipping(base_model)
    return model


def compute_metrics_for_trainer(eval_pred):
    """Compute regression and ranking metrics for Trainer.
    
    Primary metric for ranking: Pearson correlation coefficient.
    Measures how well the predictions correlate with labels.
    """
    predictions, labels = eval_pred
    if len(predictions.shape) > 1:
        predictions = predictions.squeeze()
    
    mse = mean_squared_error(labels, predictions)
    
    # Ranking metric - most important for feedback quality scoring
    pearson_corr, _ = pearsonr(labels, predictions)
    
    # Track prediction diversity to detect median prediction issue
    pred_std = np.std(predictions)
    pred_mean = np.mean(predictions)
    label_std = np.std(labels)
    label_mean = np.mean(labels)
    
    return {
        'pearson': pearson_corr,    # PRIMARY: correlation (higher is better)
        'mse': mse,                 # Secondary: absolute error
        'pred_std': pred_std,
        'pred_mean': pred_mean,
        'label_std': label_std,
        'label_mean': label_mean,
    }


def get_default_config(model_name: Optional[str] = None, max_length: Optional[int] = None):
    """Get default training configuration."""
    resolved_model_name = model_name or 'answerdotai/ModernBERT-large'
    resolved_max_length = max_length if max_length is not None else 1024

    return {
        'train_file': '../../data/stories/human_ratings_train.jsonl',
        'dev_file': '../../data/stories/human_ratings_dev.jsonl',
        'test_file': '../../data/stories/human_ratings_test.jsonl',
        'model_name': resolved_model_name,
        'max_length': resolved_max_length,
        'num_workers': 4,
        'logging_steps': 100,  # Optimization options
        'label_smoothing': 0.0,  # Can try 0.05-0.1 to encourage diversity
        'gradient_clip_norm': 1.0,  # Gradient clipping    
    }


def train_task_model(
    task_field: str,
    best_hyperparameters: dict,
    output_dir: Path,
    loss_type: str = 'mse',
    fbonly: bool = False,
    model_name: Optional[str] = None,
    max_length: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
):
    """
    Train a single-task model with the given task field and hyperparameters.
    
    Args:
        task_field: The task field name (e.g., 'correctness')
        best_hyperparameters: Dict with keys: learning_rate, batch_size, weight_decay, 
                             warmup_ratio, dropout, num_epochs
        loss_type: 'mse', 'huber', or 'mae'. Use 'huber' or 'mae' to encourage diverse predictions.
        fbonly: Whether to use only feedback as input.
        model_name: Base model checkpoint to fine-tune.
        max_length: Maximum token length for tokenization.
        gradient_accumulation_steps: Number of gradient accumulation steps.
    """
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    # Set seed at start of training for maximum reproducibility
    set_seed(SEED)
    
    # Enable TensorFloat32 for better performance on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training model for task: {task_field}")
    print(f"Random seed: {SEED}")
    
    # Data and model configuration
    config = get_default_config(model_name=model_name, max_length=max_length)
    run_prefix = config['model_name'].split('/')[-1].replace('-', '_').replace('.', '_').lower()
    
    print(f"Using best hyperparameters: {best_hyperparameters}")
    print(f"Model checkpoint: {config['model_name']}")
    print(f"Max input length: {config['max_length']}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Load tokenizer and create datasets
    print(f"Loading tokenizer from {config['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # ModernBERT tokenizer doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading datasets...")
    train_dataset = SingleTaskHumanRatingsDataset(config['train_file'], tokenizer, task_field, fbonly, config['max_length'])
    dev_dataset = SingleTaskHumanRatingsDataset(config['dev_file'], tokenizer, task_field, fbonly, config['max_length'])
    test_dataset = SingleTaskHumanRatingsDataset(config['test_file'], tokenizer, task_field, fbonly, config['max_length'])
    
    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")
    
    if len(train_dataset) == 0:
        print(f"ERROR: No training samples found for task '{task_field}'!")
        return
    
    # Setup training arguments with best hyperparameters
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=best_hyperparameters['learning_rate'],
        per_device_train_batch_size=best_hyperparameters['batch_size'],
        per_device_eval_batch_size=best_hyperparameters['batch_size'],
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=best_hyperparameters['num_epochs'],
        weight_decay=best_hyperparameters['weight_decay'],
        warmup_ratio=best_hyperparameters['warmup_ratio'],
        logging_steps=config['logging_steps'],
        load_best_model_at_end=True,
        metric_for_best_model="pearson",  # Use Pearson for ranking performance
        greater_is_better=True,  # Higher Pearson is better
        save_total_limit=10,
        report_to="wandb",
        run_name=f"{run_prefix}_{task_field}",
        eval_on_start=True,
        seed=SEED,
        dataloader_num_workers=0,  # Set to 0 for deterministic behavior
        dataloader_pin_memory=True,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        max_grad_norm=config['gradient_clip_norm'],  # Add gradient clipping
        label_smoothing_factor=config['label_smoothing'],  # Optional smoothing
    )
    
    # Train final model with best hyperparameters
    print("\n" + "="*60)
    print(f"Training {task_field} model with best hyperparameters...")
    print("="*60)
    
    wandb.init(
        project='human-ratings',
        name=f"{run_prefix}_{task_field}",
        config={**best_hyperparameters, **config}
    )
    
    # Initialize model with best hyperparameters
    model = create_regression_model(config['model_name'], dropout=best_hyperparameters['dropout'])
    
    # Create trainer with appropriate loss function
    print(f"\nUsing loss function: {loss_type.upper()}")
    if loss_type.lower() == 'huber':
        trainer = HuberLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics_for_trainer,
            data_collator=default_data_collator,  # Use default collator for pre-tokenized data
            tokenizer=tokenizer,
        )
    elif loss_type.lower() == 'mae':
        trainer = MAETrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics_for_trainer,
            data_collator=default_data_collator,  # Use default collator for pre-tokenized data
            tokenizer=tokenizer,
        )
    else:  # default to standard Trainer (MSE loss)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics_for_trainer,
            data_collator=default_data_collator,  # Use default collator for pre-tokenized data
            tokenizer=tokenizer,
        )
    
    # Train
    trainer.train()
    
    # Save best model - save only the base model, not the wrapper
    # This ensures compatibility when loading with AutoModelForSequenceClassification
    best_model_final = output_dir / "best_model"
    trainer.model.base_model.save_pretrained(str(best_model_final))
    tokenizer.save_pretrained(str(best_model_final))
    
    # Save hyperparameters
    all_config = {**best_hyperparameters, **config, 'task_field': task_field}
    with open(best_model_final / "hyperparameters.json", 'w') as f:
        json.dump(all_config, f, indent=2)
    
    print(f"Final best model saved to: {best_model_final}")

    
    # Evaluate on dev set
    print("\nEvaluating on dev set...")
    dev_metrics = trainer.evaluate(dev_dataset, metric_key_prefix='eval')
    
    print("\nDev metrics:")
    for key, value in dev_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix='test')
    
    print("\nTest metrics:")
    for key, value in test_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    # Log test results
    wandb.run.summary['test_pearson'] = test_metrics.get('test_pearson', 0)
    wandb.run.summary['test_mse'] = test_metrics.get('test_mse', 0)
    wandb.run.summary['task_field'] = task_field
    
    print(f"\nFinal results for {task_field}:")
    print(f"  Test Pearson (ranking): {test_metrics.get('test_pearson', 0):.4f}")
    print(f"  Test MSE (absolute):    {test_metrics.get('test_mse', 0):.4f}")
    
    wandb.finish()
