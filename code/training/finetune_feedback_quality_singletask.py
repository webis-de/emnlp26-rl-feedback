#!/usr/bin/env python3
"""
Fine-tune feedback quality scorer (ModernBERT-large) on human ratings data for regression.
Train separate models for each score field.

This script provides a command-line interface to the shared training utilities.
"""

import os

# Set CUBLAS environment variable BEFORE importing torch for deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

import argparse
import json
from pathlib import Path
import torch
import wandb
import optuna
from transformers import AutoTokenizer, Trainer, TrainingArguments, default_data_collator

from feedback_quality_training_utils import (
    train_task_model, 
    SEED, 
    set_seed,
    SingleTaskHumanRatingsDataset,
    create_regression_model,
    compute_metrics_for_trainer,
    get_default_config
)

# Set seed for reproducibility
set_seed(SEED)

DEFAULT_MODEL_NAME = "answerdotai/ModernBERT-large"


def _model_tag(model_name: str) -> str:
    """Create stable model tag for run/output naming."""
    if model_name == DEFAULT_MODEL_NAME:
        return "modernbert"
    return model_name.split("/")[-1].replace("-", "_").replace(".", "_").lower()


def run_hyperparameter_tuning(
    task_field: str,
    n_trials: int = 20,
    tuning_epochs: int = 3,
    fbonly: bool = False,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = 1024,
    gradient_accumulation_steps: int = 1,
):
    """
    Run Optuna hyperparameter tuning to maximize Pearson correlation.
    
    Args:
        task_field: The task to optimize for
        n_trials: Number of trials to run
        tuning_epochs: Number of epochs per trial
    
    Returns:
        dict: Best hyperparameters found
    """
    print(f"\n{'='*60}")
    print(f"Starting hyperparameter tuning for {task_field}")
    print(f"Optimizing Pearson correlation with {n_trials} trials")
    print(f"Model: {model_name}, max_length: {max_length}, grad_accum: {gradient_accumulation_steps}")
    print(f"{'='*60}\n")
    
    # Get default config
    config = get_default_config(model_name=model_name, max_length=max_length)
    model_tag = _model_tag(config['model_name'])
    
    # Load tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # ModernBERT tokenizer doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = SingleTaskHumanRatingsDataset(config['train_file'], tokenizer, task_field, fbonly, config['max_length'])
    dev_dataset = SingleTaskHumanRatingsDataset(config['dev_file'], tokenizer, task_field, fbonly, config['max_length'])
    
    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found for task '{task_field}'!")
    
    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}")
    
    # Initialize wandb for tuning
    if not fbonly:
        wandb_name = f"{model_tag}_{task_field}_tuning"
    else:
        wandb_name = f"{model_tag}_{task_field}_tuning_fbonly"
    wandb.init(
        project='human-ratings-hpo',
        name=wandb_name,
        config={
            'task_field': task_field,
            'n_trials': n_trials,
            'model_name': config['model_name'],
            'max_length': config['max_length'],
            'gradient_accumulation_steps': gradient_accumulation_steps,
        }
    )
    
    def objective(trial):
        """Optuna objective function to maximize Pearson correlation."""
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
        if "modernbert" in config['model_name'].lower():
            batch_sizes = [8, 16, 32]
        else:
            batch_sizes = [1, 2, 4, 8]
        batch_size = trial.suggest_categorical('batch_size', batch_sizes)
        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.3)
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.3)
        dropout = trial.suggest_float('dropout', 0.0, 0.3)
        
        # Create model
        model = create_regression_model(config['model_name'], dropout=dropout)
        
        # Training arguments for this trial
        if not fbonly:
            output_dir = Path(f"../../models/{model_tag}_{task_field.replace('-', '_')}_{trial.number}")
        else:
            output_dir = Path(f"../../models/{model_tag}_{task_field.replace('-', '_')}_fbonly_{trial.number}")
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            eval_strategy="epoch",
            save_strategy="no",  # Don't save during tuning
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=tuning_epochs,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            logging_steps=100,
            seed=SEED,
            dataloader_num_workers=0,  # Deterministic
            dataloader_pin_memory=True,
            bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
            fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
            report_to="wandb", # Log to wandb
        )
        
        # Create trainer
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
        
        # Evaluate and get Pearson
        metrics = trainer.evaluate(dev_dataset)
        pearson = metrics.get('eval_pearson', 0.0)
        
        # Log to wandb
        wandb.log({
            'trial': trial.number,
            'pearson': pearson,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'warmup_ratio': warmup_ratio,
            'dropout': dropout,
        })
        
        # Clean up trial directory
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        
        return pearson
    
    # Create and run study
    study = optuna.create_study(
        direction='maximize',  # Maximize Pearson correlation
        study_name=f'{model_tag}_{task_field}_pearson'
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n{'='*60}")
    print(f"Best hyperparameters found for {task_field}:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"Best Pearson: {best_value:.4f}")
    print(f"{'='*60}\n")
    
    # Log best results
    wandb.run.summary['best_pearson'] = best_value
    wandb.run.summary['best_params'] = best_params
    wandb.finish()
    
    # Convert to training format
    hyperparameters = {
        'learning_rate': best_params['learning_rate'],
        'batch_size': best_params['batch_size'],
        'weight_decay': best_params['weight_decay'],
        'warmup_ratio': best_params['warmup_ratio'],
        'dropout': best_params['dropout'],
        'num_epochs': 3,
    }
    
    return hyperparameters


def main():
    parser = argparse.ArgumentParser(description="Fine-tune feedback quality scorer on single task human ratings")
    
    # Task argument
    parser.add_argument('--task_field', type=str, required=True,
                       choices=['correctness', 'error-detection', 'specificity'],
                       help='Which score field to train on')
    
    # Hyperparameter tuning
    parser.add_argument('--tune_hyperparameters', action='store_true',
                       help='Enable hyperparameter tuning with Optuna (maximizes Pearson correlation)')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of Optuna trials for hyperparameter tuning')
    parser.add_argument('--tuning_epochs', type=int, default=3,
                       help='Number of epochs per trial during hyperparameter tuning')

    # Model and sequence settings
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                       help='Base model checkpoint to fine-tune for regression')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum token length for inputs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    
    # Hyperparameter arguments (used if not tuning)
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    
    # Loss function
    parser.add_argument('--loss_type', type=str, default='mse',
                       choices=['mse', 'huber', 'mae'],
                       help='Loss function type')
    
    parser.add_argument('--fbonly', action='store_true',
                       help='Get only the feedback as input, not the story or essay context.')
    
    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1")

    model_tag = _model_tag(args.model_name)
    
    if not args.fbonly:
        output_dir = Path(f"../../models/{model_tag}_{args.task_field.replace('-', '_')}_best")
    else:
        output_dir = Path(f"../../models/{model_tag}_{args.task_field.replace('-', '_')}_fbonly_best")
    
    if args.tune_hyperparameters:
        # Run hyperparameter tuning to maximize Pearson correlation
        best_hyperparameters = run_hyperparameter_tuning(
            args.task_field, 
            n_trials=args.n_trials,
            tuning_epochs=args.tuning_epochs, 
            fbonly=args.fbonly,
            model_name=args.model_name,
            max_length=args.max_length,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        
        # Save best hyperparameters
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "best_hyperparameters.json", 'w') as f:
            json.dump(best_hyperparameters, f, indent=2)
        
        print(f"\nBest hyperparameters saved to {output_dir / 'best_hyperparameters.json'}")
    else:
        # Use provided hyperparameters
        best_hyperparameters = {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'warmup_ratio': args.warmup_ratio,
            'dropout': args.dropout,
            'num_epochs': args.num_epochs,
        }
    
    print(f"\nTraining final model for task: {args.task_field}")
    print(f"Model: {args.model_name}")
    print(f"Max length: {args.max_length}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Using seed: {SEED}")
    print(f"Hyperparameters: {best_hyperparameters}")
    print(f"Loss type: {args.loss_type}\n")
    
    # Train final model using shared utilities
    train_task_model(
        args.task_field,
        best_hyperparameters,
        output_dir,
        loss_type=args.loss_type,
        fbonly=args.fbonly,
        model_name=args.model_name,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


if __name__ == "__main__":
    main()
