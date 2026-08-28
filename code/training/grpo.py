from curses import meta
import torch
import os
import sys
from pathlib import Path

# Set CUBLAS environment variable BEFORE importing torch for deterministic behavior
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')

import argparse
import re
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
import logging
import wandb
import weave

# Ensure parent of project root is on sys.path for 'storys' package imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARENT_ROOT = PROJECT_ROOT.parent
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))

# Local models root (sibling of project root)
LOCAL_MODELS_ROOT = PARENT_ROOT / "transformer-models"

from storys.code.utils.prompts import get_prompt_template
from storys.code.scorers.feedback_quality_scorer import FeedbackQualityScorer

# Seed for reproducibility
SEED = 8549

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


DATASET_SPLITS = {
    "stories": ("stories/stories_train.jsonl", "stories/stories_dev.jsonl"),
    "stories_us": ("stories/stories_train_us.jsonl", "stories/stories_dev.jsonl"),
    "storal_matched": (
        "storal_matched/storal_en_train_stories.jsonl",
        "storal_matched/storal_en_valid_stories.jsonl",
    ),
    "writingprompts_matched": (
        "writingprompts_matched/writingprompts_train_stories.jsonl",
        "writingprompts_matched/writingprompts_validation_stories.jsonl",
    ),
    "comb": ("comb/comb_train.jsonl", "comb/comb_dev.jsonl"),
    "leaf_ds": ("leaf_ds/leaf_ds_train.jsonl", "leaf_ds/leaf_ds_dev.jsonl"),
    "leaf_corr": ("leaf/leaf_train_corr.jsonl", "leaf_ds/leaf_ds_dev.jsonl"),
    "leaf_us": ("leaf/leaf_train_us.jsonl", "leaf/leaf_dev_fast.jsonl"),
    "reviews": ("reviews/reviews_train.jsonl", "reviews/reviews_dev.jsonl"),
    "reviews_corr": ("reviews/reviews_train_corr.jsonl", "reviews/reviews_dev_corr.jsonl"),
    "reviews_us": ("reviews/reviews_train_us.jsonl", "reviews/reviews_dev.jsonl"),
    "helpsteer2": ("helpsteer2/helpsteer2_train.jsonl", "helpsteer2/helpsteer2_dev.jsonl"),
    "helpsteer2_corr": (
        "helpsteer2/helpsteer2_train_corr.jsonl",
        "helpsteer2/helpsteer2_dev_corr.jsonl",
    ),
    "helpsteer2_us": ("helpsteer2/helpsteer2_train_us.jsonl", "helpsteer2/helpsteer2_dev.jsonl"),
}


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set all random seeds to {seed} for reproducibility")


def patch_lm_head_dtype_guard(model: torch.nn.Module) -> int:
    """Patch lm_head.forward to cast hidden states to lm_head weight dtype.

    Returns:
        Number of lm_head modules patched.
    """
    patched = 0
    seen = set()

    for module in model.modules():
        lm_head = getattr(module, "lm_head", None)
        if lm_head is None or not isinstance(lm_head, torch.nn.Module):
            continue

        if id(lm_head) in seen:
            continue
        seen.add(id(lm_head))

        if getattr(lm_head, "_dtype_guard_patched", False):
            continue

        original_forward = lm_head.forward

        def guarded_forward(hidden_states, *args, _original_forward=original_forward, _lm_head=lm_head, **kwargs):
            weight = getattr(_lm_head, "weight", None)
            target_dtype = weight.dtype if weight is not None else hidden_states.dtype
            if hidden_states.dtype != target_dtype:
                hidden_states = hidden_states.to(target_dtype)
            return _original_forward(hidden_states, *args, **kwargs)

        lm_head.forward = guarded_forward
        lm_head._dtype_guard_patched = True
        patched += 1

    return patched


def ensure_grpo_trainer_grad_accum_compat(trainer: GRPOTrainer) -> int:
    """Ensure GRPOTrainer exposes grad-accum state needed by eval-on-start.

    Some TRL builds access `self.current_gradient_accumulation_steps` during
    evaluation before the training loop has initialized it.
    """
    grad_steps = getattr(getattr(trainer, "args", None), "gradient_accumulation_steps", 1)
    try:
        grad_steps = int(grad_steps)
    except (TypeError, ValueError):
        grad_steps = 1
    grad_steps = max(1, grad_steps)

    current_steps = getattr(trainer, "current_gradient_accumulation_steps", None)
    try:
        current_steps = int(current_steps) if current_steps is not None else None
    except (TypeError, ValueError):
        current_steps = None

    trainer.current_gradient_accumulation_steps = max(1, current_steps) if current_steps is not None else grad_steps
    return trainer.current_gradient_accumulation_steps


def extract_story_from_prompt(prompt: str, prompt_template: str) -> str:
    """
    Extract the original story text from a formatted prompt.
    
    Args:
        prompt: The full formatted prompt with chat template and special tokens
        prompt_template: The prompt template used (e.g., from prompts.py)
    
    Returns:
        The extracted story text
    """
    # Remove special tokens and chat template markers
    text = prompt
    text = re.sub(r'<bos>|<eos>|<start_of_turn>|<end_of_turn>|user|model|assistant', '', text)
    
    # First, try to extract text within triple quotes (for feedback_only template)
    triple_quote_match = re.search(r'"""(.*?)"""', text, re.DOTALL)
    if triple_quote_match:
        return triple_quote_match.group(1).strip()
    
    # Extract the instruction pattern from the template
    # The template has {story} as a placeholder, so we split on that
    template_parts = prompt_template.split('{story}')
    
    if len(template_parts) == 2:
        before_story = template_parts[0].strip()
        after_story = template_parts[1].strip()
        
        # Build patterns to find the story
        # Use the last few words of before_story and first few words of after_story
        before_words = before_story.split()[-3:] if len(before_story.split()) >= 3 else before_story.split()
        after_words = after_story.split()[:3] if len(after_story.split()) >= 3 else after_story.split()
        
        if before_words and after_words:
            # Build pattern: last words of instruction + story + first words of next instruction
            before_pattern = r'\s+'.join(re.escape(word) for word in before_words)
            after_pattern = r'\s+'.join(re.escape(word) for word in after_words)
            pattern = before_pattern + r'\s*[\n\r]+(.*?)\s*[\n\r]+' + after_pattern
            
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                return extracted
        
        # Fallback: use simpler pattern with just "following text:" anchor
        match = re.search(r'following text:?\s*[\n\r]+(.*?)\s*[\n\r]+(?:The feedback should|If the story has no problems)', 
                          text, re.DOTALL | re.IGNORECASE)
        if match:
            story = match.group(1).strip()
            return story
    
    # Another fallback: try to extract text between common instruction patterns
    match = re.search(r'following text:?\s*[\n\r]+(.*?)(?:\s*[\n\r]+The feedback should|\s*[\n\r]+If the story has no problems|\s*[\n\r]+Feedback:)', 
                      text, re.DOTALL | re.IGNORECASE)
    if match:
        story = match.group(1).strip()
        # Remove any remaining "Title:" prefix that might have been added
        story = re.sub(r'^Title:\s*.*?\n', '', story, flags=re.MULTILINE)
        return story
    
    # Last resort: return the text as-is, removing obvious instruction markers
    text = re.sub(r'Write.*?feedback.*?following text:?\s*[\n\r]+', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\s*[\n\r]+The feedback should.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\s*[\n\r]+If the story has no problems.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\s*[\n\r]+Feedback:?\s*$', '', text, flags=re.IGNORECASE)
    return text.strip()


def _normalize_story_example(example: dict) -> dict:
    """Build a unified item for story JSONL examples."""
    return {
        "story": example.get("story", ""),
        "noise": example.get("noise", "unknown"),
        "gold_story": example.get("gold_story", ""),
    }


def _normalize_leaf_example(example: dict) -> dict:
    """Build a unified item for LEAF essay examples."""
    title = example.get("essay_title", "").strip()
    if "essay_text" in example:
        essay = example.get("essay_text", "").strip()
    else:
        essay = example.get("essay", "").strip()
    
    # Combine title and essay for prompting
    story = f"Title: {title}\n{essay}" if title else essay
    
    return {
        "story": story,
        "title": title,
        "essay_text": essay,
    }


def get_reward_component_keys(mode: str) -> list[str]:
    """Return scorer keys to aggregate for a configured reward mode."""
    mode_to_keys = {
        "all": ["correctness", "error-detection", "specificity"],
        "corr": ["correctness"],
        "err": ["error-detection"],
        "spec": ["specificity"],
        "no_corr": ["error-detection", "specificity"],
        "no_err": ["correctness", "specificity"],
        "no_spec": ["correctness", "error-detection"],
    }
    return mode_to_keys[mode]


def main():
    # Set seed for reproducibility
    set_seed(SEED)
    
    parser = argparse.ArgumentParser(description="Train a GRPO model with configurable hyperparameters.")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-2-9b-it", help="The name of the model to train.")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory for the trained model.")
    parser.add_argument("--wandb_project", type=str, default="gap-feedback", help="W&B project name for weave tracing.")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID to resume (optional). If not provided, creates a new run.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from (optional).")

    # Training configuration flags
    parser.add_argument("--disable_eval_on_start", action="store_true", help="Disable evaluation at the start of training.")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit", help="Optimizer to use.")

    # Approach flags
    parser.add_argument("--prompt", type=str, default="bl_full", help="The prompt to use for training.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="stories",
        choices=list(DATASET_SPLITS.keys()),
        help="Dataset to train on (story-format, matched, leaf, reviews, or HelpSteer2 variants).",
    )

    # === Priority 1 Hyperparameters ===
    parser.add_argument("--num_train_epochs", type=float, default=5, 
                        help="Number of training epochs. Recommended: [1-5]")
    
    # Learning rate: Try [1e-6, 5e-6, 1e-5]
    parser.add_argument("--learning_rate", type=float, default=5e-6, 
                        help="Learning rate for training. Recommended: [1e-6, 5e-6, 1e-5]")
    
    # LoRA rank: Try [16, 32, 64]
    parser.add_argument("--lora_r", type=int, default=32, 
                        help="LoRA rank. Higher = more expressiveness but more memory. Recommended: [16, 32, 64]")
    
    # LoRA alpha: Typically 2x the rank
    parser.add_argument("--lora_alpha", type=int, default=None, 
                        help="LoRA alpha. If not set, defaults to 2x lora_r. Recommended: 2x lora_r")
    
    # Beta (KL penalty): Try [0.001, 0.005, 0.01]
    parser.add_argument("--beta", type=float, default=0.005, 
                        help="KL divergence penalty coefficient. Higher = stays closer to base model. Recommended: [0.001, 0.005, 0.01]")

    parser.add_argument("--eval_steps", type=float, default=50, 
                        help="Number of steps between evaluations. Recommended: [50-250]")

    parser.add_argument("--fbonly", action="store_true", help="Use feedback-only input for reward models (no story/essay etc. context).")

    parser.add_argument("--qwenrm", action="store_true", help="Use qwen reward models.")
    parser.add_argument(
        "--reward_components",
        type=str,
        default="all",
        choices=["all", "corr", "err", "spec", "no_corr", "no_err", "no_spec"],
        help=(
            "Reward component selection: all (default), corr, err, spe, "
            "no_corr, no_err, no_spec."
        ),
    )

    args = parser.parse_args()
    reward_component_keys = get_reward_component_keys(args.reward_components)

    # Set lora_alpha to 2x lora_r if not provided
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_r * 2
        logger.info(f"Setting lora_alpha to 2x lora_r: {args.lora_alpha}")

    # Auto-detect the latest checkpoint if resume path points to output directory
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint and os.path.isdir(resume_checkpoint):
        # Check if this is a checkpoint directory (has trainer_state.json)
        if not os.path.exists(os.path.join(resume_checkpoint, "trainer_state.json")):
            # This is likely the output directory, find the latest checkpoint
            import glob
            checkpoints = glob.glob(os.path.join(resume_checkpoint, "checkpoint-*"))
            if checkpoints:
                # Sort by checkpoint number
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
                resume_checkpoint = checkpoints[-1]
                logger.info(f"Auto-detected latest checkpoint: {resume_checkpoint}")
            else:
                logger.warning(f"No checkpoints found in {resume_checkpoint}, starting fresh")
                resume_checkpoint = None
        else:
            logger.info(f"Using checkpoint directory: {resume_checkpoint}")

    args.resume_from_checkpoint = resume_checkpoint

    # Log hyperparameters
    logger.info("=" * 80)
    logger.info("Hyperparameter Configuration:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Model Name: {args.model_name}")
    logger.info(f"  Feedback-only Input: {args.fbonly}")
    logger.info(f"  Qwen Reward Models: {args.qwenrm}")
    logger.info(f"  Training Epochs: {args.num_train_epochs}")
    logger.info(f"  Prompt Template: {args.prompt}")
    logger.info(f"  Reward Components Mode: {args.reward_components}")
    logger.info(f"  Active Reward Components: {', '.join(reward_component_keys)}")
    logger.info(f"  Learning Rate: {args.learning_rate}")
    logger.info(f"  LoRA Rank (r): {args.lora_r}")
    logger.info(f"  LoRA Alpha: {args.lora_alpha}")
    logger.info(f"  Beta (KL penalty): {args.beta}")
    logger.info("=" * 80)

    # Log environment configuration (set by launch scripts)
    logger.info("=" * 80)
    logger.info("Job Isolation Configuration (from environment):")
    logger.info(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
    logger.info(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")
    logger.info(f"  TORCHELASTIC_RUN_ID: {os.environ.get('TORCHELASTIC_RUN_ID', 'not set')}")
    logger.info(f"  VLLM_INSTANCE_ID: {os.environ.get('VLLM_INSTANCE_ID', 'not set')}")
    logger.info(f"  TMPDIR: {os.environ.get('TMPDIR', 'not set')}")
    logger.info(f"  XDG_CACHE_HOME: {os.environ.get('XDG_CACHE_HOME', 'not set')}")
    logger.info(f"  TRITON_CACHE_DIR: {os.environ.get('TRITON_CACHE_DIR', 'not set')}")
    logger.info(f"  TORCH_COMPILE_CACHE_DIR: {os.environ.get('TORCH_COMPILE_CACHE_DIR', 'not set')}")
    logger.info(f"  WANDB_DIR: {os.environ.get('WANDB_DIR', 'not set')}")
    logger.info(f"  Model cache: Using shared HuggingFace cache (read-only, no conflicts)")
    logger.info("=" * 80)

    # Check if this is the main process (for distributed training)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0

    # Initialize wandb only on main process
    if is_main_process:
        run_name = args.output_dir.split("/")[-1]

        # Configure WandB resumption
        if args.wandb_run_id:
            # Explicitly resume a specific WandB run
            wandb.init(
                project=args.wandb_project,
                id=args.wandb_run_id,
                resume="must",
                config={
                    "dataset": args.dataset,
                    "prompt": args.prompt,
                    "learning_rate": args.learning_rate,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "beta": args.beta,
                }
            )
            logger.info(f"Resuming WandB run: {args.wandb_run_id}")
        else:
            # Create a new run
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "dataset": args.dataset,
                    "prompt": args.prompt,
                    "learning_rate": args.learning_rate,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "beta": args.beta,
                }
            )
            logger.info(f"Created new WandB run: {run_name} (id: {wandb.run.id})")

        # Initialize Weave for tracing (will use the existing wandb run)
        weave.init(args.wandb_project)
        logger.info(f"Initialized Weave tracing (sharing WandB run)")
    else:
        logger.info(f"Skipping WandB/Weave initialization on worker process (local_rank={local_rank})")

    # --- Load Reward Models ---
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Load scorers

    # Feedback Quality Scorer
    feedback_quality_scorer = FeedbackQualityScorer(device, fbonly=args.fbonly, qwenrm=args.qwenrm)
    logger.info(f"Memory after feedback quality scorer: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

    # --- GRPOTrainer ---
    # Load tokenizer and model on main process first to avoid checkpoint loading race conditions
    logger.info("Loading tokenizer...")
    model_path = (LOCAL_MODELS_ROOT / args.model_name).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add barrier to ensure all processes wait until main process has downloaded model
    if torch.cuda.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        logger.info(f"Process {local_rank} passed barrier after tokenizer load")

    
    prompt_template = get_prompt_template(args.prompt)

    def prepare_dataset(examples):
        """Prepare prompts for a batch of examples.
        
        Auto-detects dataset format (story vs LEAF) and normalizes accordingly.
        Note: Always expects batched input since called with batched=True.
        Returns dicts with list values for all keys.
        """
        prompts = []
        
        # Get the length of the batch from any list field
        batch_keys = [k for k, v in examples.items() if isinstance(v, list)]
        if not batch_keys:
            raise ValueError(
                f"No list fields found in batch. Available fields: {list(examples.keys())}"
            )
        batch_size = len(examples[batch_keys[0]])
        
        for idx in range(batch_size):
            # Reconstruct single example from batched structure
            single_example = {}
            for key, values in examples.items():
                if isinstance(values, list) and idx < len(values):
                    single_example[key] = values[idx]

            # Detect dataset type per example (required for mixed/shuffled comb batches)
            essay_title = single_example.get("essay_title")
            essay_text = single_example.get("essay_text")
            essay = single_example.get("essay")
            has_leaf_fields = any(
                value is not None and str(value).strip() != ""
                for value in (essay_title, essay_text, essay)
            )
            
            # Normalize using appropriate function - this ensures story field exists
            if has_leaf_fields:
                normalized = _normalize_leaf_example(single_example)
            else:
                normalized = _normalize_story_example(single_example)
            
            story_text = normalized.get("story", "")
            if not story_text:
                prompts.append("")
                continue
            
            formatted = prompt_template.format(story=story_text)
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": formatted}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(chat_prompt)
        
        return {"prompt": prompts}

    # Load datasets based on selection
    logger.info(f"Loading dataset: {args.dataset}")
    
    data_root = PROJECT_ROOT / "data"
    train_rel, eval_rel = DATASET_SPLITS[args.dataset]
    train_files = str(data_root / train_rel)
    eval_files = str(data_root / eval_rel)
    
    dataset = load_dataset("json", data_files=train_files, split="train")
    dataset = dataset.map(prepare_dataset, batched=True, load_from_cache_file=False)
    logger.info(f"Training dataset size: {len(dataset)}")
    
    eval_dataset = load_dataset("json", data_files=eval_files, split="train")
    eval_dataset = eval_dataset.map(prepare_dataset, batched=True, load_from_cache_file=False)
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        eval_on_start=not args.disable_eval_on_start,
        log_completions=True,
        max_completion_length=512,  # Reduced from 1024 to save memory
        max_prompt_length=1024,  # Reduced from 2048 to save memory
        scale_rewards=False,
        gradient_accumulation_steps=16,  # Increased from 8 to maintain effective batch size
        optim=args.optimizer,
        bf16=True,
        label_names=[],
        use_vllm=False,  # Disabled because vLLM doesn't support quantized models
        loss_type="dr_grpo",
        mask_truncated_completions=True,
        learning_rate=args.learning_rate,  # Now configurable
        lr_scheduler_type="cosine",
        beta=args.beta,  # Now configurable
        disable_dropout=True,
        report_to="wandb" if is_main_process else "none",
        num_train_epochs=args.num_train_epochs, # prev: 5
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_checkpointing=False,  # Disable gradient checkpointing to avoid DDP issues with LoRA
        seed=SEED,  # For reproducibility
        dataloader_num_workers=0,  # Deterministic data loading
        dataloader_pin_memory=True,
    )

    peft_config = LoraConfig(
        peft_type="LORA",
        r=args.lora_r,  # Now configurable
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=args.lora_alpha,  # Now configurable
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Quantization config to reduce memory usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info("Using 4-bit quantization to reduce memory usage")

    # Load quantized model
    logger.info(f"Process {local_rank} loading quantized model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=bnb_config,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
    )
    
    # Explicitly disable gradient checkpointing to avoid DDP issues
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
        logger.info("Explicitly disabled gradient checkpointing on model")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    patched_count = patch_lm_head_dtype_guard(model)
    logger.info(f"Applied lm_head dtype guard patches: {patched_count}")
    logger.info(f"Memory after loading quantized model: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

    logger.info(f"Process {local_rank} initializing GRPOTrainer")
    
    # Define reward function that extracts story from formatted prompts
    def reward_func(prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            # Extract the original story from the formatted prompt
            story = extract_story_from_prompt(prompt, prompt_template)
            # Get all scores and average them
            scores = feedback_quality_scorer.get_all_scores(story, completion)
            core_scores = [scores.get(component_key) for component_key in reward_component_keys]
            core_scores = [s for s in core_scores if s is not None]
            avg_score = sum(core_scores) / len(core_scores) if core_scores else 0.0
            rewards.append(avg_score)
        return rewards
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],  # Use our custom reward function
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    patched_trainer_count = patch_lm_head_dtype_guard(trainer.model)
    logger.info(f"Re-applied lm_head dtype guard patches after trainer init: {patched_trainer_count}")

    compat_grad_steps = ensure_grpo_trainer_grad_accum_compat(trainer)
    logger.info(
        f"Initialized GRPOTrainer eval compatibility: current_gradient_accumulation_steps={compat_grad_steps}"
    )
    
    # Try to set static graph for DDP compatibility with LoRA
    # This must be done after the model is wrapped by DDP (which happens inside trainer init)
    # We'll do this in a hook after first forward pass
    logger.info(f"Memory after GRPOTrainer init: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
    logger.info(f"Memory reserved: {torch.cuda.memory_reserved(device)/1024**3:.2f} GB")
    
    # Set static graph on the DDP model if available
    if hasattr(trainer.model, 'module'):
        # Model is wrapped in DDP
        if hasattr(trainer.model, '_set_static_graph'):
            trainer.model._set_static_graph()
            logger.info("Set static graph on DDP wrapper for LoRA compatibility")
    elif hasattr(trainer.model, '_set_static_graph'):
        # Direct model (not wrapped)
        trainer.model._set_static_graph()
        logger.info("Set static graph on model for LoRA compatibility")


    # Log checkpoint resumption info
    if args.resume_from_checkpoint:
        logger.info(f"=" * 80)
        logger.info(f"RESUMING TRAINING FROM CHECKPOINT: {args.resume_from_checkpoint}")
        logger.info(f"=" * 80)
        # Try to read trainer state to show what step we're resuming from
        import json
        trainer_state_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
                logger.info(f"Resuming from global step: {trainer_state.get('global_step', 'unknown')}")
                logger.info(f"Resuming from epoch: {trainer_state.get('epoch', 'unknown')}")
                logger.info(f"Best metric so far: {trainer_state.get('best_metric', 'unknown')}")
        else:
            logger.warning(f"trainer_state.json not found in {args.resume_from_checkpoint}")
    else:
        logger.info("Starting training from scratch (no checkpoint)")

    compat_grad_steps = ensure_grpo_trainer_grad_accum_compat(trainer)
    logger.info(f"Pre-train GRPO gradient accumulation state: {compat_grad_steps}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

if __name__ == "__main__":
    main()
