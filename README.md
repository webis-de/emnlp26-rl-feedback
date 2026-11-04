# Reproduction Guide

This directory contains the code and data needed to reproduce the feedback-quality regression models, GRPO training, feedback generation, and automatic evaluation.

## 1. Expected checkout layout

The Python modules import `storys.code`, so the submission should be placed inside a directory named `storys`, or the parent directory containing the `storys` package must be added to `PYTHONPATH`.

The code expects the following layout:

```text
storys/
  submission/
    code/
    data/
    models/                 # intermediate regression-model outputs
    models_final/            # scorers used by GRPO/evaluation
  transformer-models/        # local causal-language-model directories
```

Set the paths once for the rest of the commands:

```bash
export STORYS_ROOT=/path/to/storys
export SUBMISSION_ROOT="$STORYS_ROOT/submission"
export PYTHONPATH="$STORYS_ROOT:$PYTHONPATH"
```

The local causal models are searched for under:

```text
$STORYS_ROOT/transformer-models/
```

The reward-model directory can be overridden with:

```bash
export STORYS_MODELS_DIR="$SUBMISSION_ROOT/models_final"
export STORYS_FBONLY_MODELS_DIR="$SUBMISSION_ROOT/models_fbonly_final"
export STORYS_QWENRM_MODELS_DIR="$SUBMISSION_ROOT/models_qwen"
```

## 2. Install dependencies

Use a Python environment with a PyTorch build compatible with the available CUDA version.

```bash
python -m venv "$STORYS_ROOT/.venv"
source "$STORYS_ROOT/.venv/bin/activate"

pip install torch
pip install transformers datasets accelerate peft trl bitsandbytes
pip install numpy pandas scikit-learn optuna
pip install wandb weave
```

Optional dependencies:

```bash
pip install openai google-genai choix
```

`bitsandbytes` and a CUDA-enabled PyTorch installation are required for the standard 4-bit GRPO setup. CPU execution is suitable for small smoke tests and scorer inspection, but not for the intended large-model GRPO experiments.

## 3. Train the regression reward models

The regression models are trained independently for three tasks:

- `correctness`
- `error-detection`
- `specificity`

The training utility reads these files:

```text
$SUBMISSION_ROOT/data/storyfeedback/human_ratings_train.jsonl
$SUBMISSION_ROOT/data/storyfeedback/human_ratings_dev.jsonl
```

Run the commands from the training directory because the script uses relative data and output paths:

```bash
cd "$SUBMISSION_ROOT/code/training"

python finetune_feedback_quality_singletask.py \
  --task_field correctness \
  --model_name answerdotai/ModernBERT-large \
  --max_length 1024 \
  --learning_rate 3e-5 \
  --batch_size 16 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --dropout 0.1 \
  --num_epochs 3

python finetune_feedback_quality_singletask.py \
  --task_field error-detection \
  --model_name answerdotai/ModernBERT-large \
  --max_length 1024 \
  --learning_rate 3e-5 \
  --batch_size 16 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --dropout 0.1 \
  --num_epochs 3

python finetune_feedback_quality_singletask.py \
  --task_field specificity \
  --model_name answerdotai/ModernBERT-large \
  --max_length 1024 \
  --learning_rate 3e-5 \
  --batch_size 16 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --dropout 0.1 \
  --num_epochs 3
```

The final checkpoints are written to:

```text
$SUBMISSION_ROOT/models/modernbert_correctness_best/
$SUBMISSION_ROOT/models/modernbert_error_detection_best/
$SUBMISSION_ROOT/models/modernbert_specificity_best/
```

GRPO and automatic evaluation expect one directory per task under `models_final`. Prepare that layout after training:

```bash
mkdir -p "$SUBMISSION_ROOT/models_final/correctness"
mkdir -p "$SUBMISSION_ROOT/models_final/error_detection"
mkdir -p "$SUBMISSION_ROOT/models_final/specificity"

cp -a "$SUBMISSION_ROOT/models/modernbert_correctness_best/." \
  "$SUBMISSION_ROOT/models_final/correctness/"
cp -a "$SUBMISSION_ROOT/models/modernbert_error_detection_best/." \
  "$SUBMISSION_ROOT/models_final/error_detection/"
cp -a "$SUBMISSION_ROOT/models/modernbert_specificity_best/." \
  "$SUBMISSION_ROOT/models_final/specificity/"

export STORYS_MODELS_DIR="$SUBMISSION_ROOT/models_final"
```

### Optional Optuna tuning

Optuna maximizes Pearson correlation on the dev split. For example:

```bash
cd "$SUBMISSION_ROOT/code/training"
python finetune_feedback_quality_singletask.py \
  --task_field correctness \
  --tune_hyperparameters \
  --n_trials 20 \
  --tuning_epochs 3
```

The search space is:

```text
learning_rate: log-uniform [1e-6, 1e-4]
batch_size:     [8, 16, 32] for ModernBERT
weight_decay:   uniform [0.0, 0.3]
warmup_ratio:   uniform [0.0, 0.3]
dropout:        uniform [0.0, 0.3]
```

Repeat tuning for each reward task before preparing `models_final`.

## 4. Evaluate the regression scorers

The scorer evaluation compares random, mean, and trained single-task baselines:

```bash
cd "$SUBMISSION_ROOT/code/scorers"
python evaluate_quality_scorer.py
```

The configured outputs are:

```text
$SUBMISSION_ROOT/data/scorer_eval/model_metrics_comparison.csv
$SUBMISSION_ROOT/data/scorer_eval/model_metrics_table.tex
```

The scorer uses the test file:

```text
$SUBMISSION_ROOT/data/storyfeedback/human_ratings_test.jsonl
```

## 5. Train GRPO

The standard GRPO setup uses:

```text
base model:       local model under transformer-models/
quantization:     4-bit NF4 with bfloat16 compute
adaptation:       LoRA
LoRA rank:        16 in the experiment launchers, 32 in the Python default
LoRA alpha:       32 in the experiment launchers
learning rate:    5e-6
KL coefficient:   0.002 in the experiment launchers
epochs:           5
gradient steps:   16
reward:           mean of correctness, error-detection, specificity
```

Make sure the model directory exists, for example:

```text
$STORYS_ROOT/transformer-models/gemma-2-9b-it/
```

Launch a four-GPU run with the full story prompt:

```bash
cd "$STORYS_ROOT"
torchrun --nproc_per_node=4 \
  "$SUBMISSION_ROOT/code/training/grpo.py" \
  --model_name gemma-2-9b-it \
  --output_dir "$SUBMISSION_ROOT/grpo/gemma_bl" \
  --dataset stories \
  --prompt bl_full \
  --learning_rate 5e-6 \
  --lora_r 16 \
  --lora_alpha 32 \
  --beta 0.002 \
  --num_train_epochs 5 \
  --eval_steps 50
```

Launch the feedback-only prompt:

```bash
torchrun --nproc_per_node=4 \
  "$SUBMISSION_ROOT/code/training/grpo.py" \
  --model_name gemma-2-9b-it \
  --output_dir "$SUBMISSION_ROOT/grpo/gemma_bo" \
  --dataset stories \
  --prompt bl_only \
  --learning_rate 5e-6 \
  --lora_r 16 \
  --lora_alpha 32 \
  --beta 0.002 \
  --num_train_epochs 5 \
  --eval_steps 50
```

Replace `gemma-2-9b-it` with the local Llama or Qwen model directory to reproduce the other model families. The supported dataset names are defined in `code/training/grpo.py`, including `stories`, `storal_matched`, and `writingprompts_matched`. In this submission, those two matched names use the bundled files under `data/storal/` and `data/writingprompts/`.

Reward ablations can be run with:

```bash
--reward_components corr
--reward_components err
--reward_components spec
--reward_components no_corr
--reward_components no_err
--reward_components no_spec
```

For checkpoint continuation:

```bash
--resume_from_checkpoint "$SUBMISSION_ROOT/grpo/gemma_bl"
```

The script automatically selects the latest `checkpoint-*` directory when the supplied path is an output directory.

## 6. Generate feedback

Generate feedback from a local GRPO checkpoint:

```bash
cd "$STORYS_ROOT"
python "$SUBMISSION_ROOT/code/predictions/predict.py" \
  --backend hf \
  --model gemma-2-9b-it \
  --checkpoint_root "$SUBMISSION_ROOT/grpo/gemma_bl" \
  --prompt_type bl_full \
  --data_file "$SUBMISSION_ROOT/data/storyfeedback/stories_test.jsonl" \
  --output_jsonl "$SUBMISSION_ROOT/predictions/gemma_bl_test.jsonl" \
  --override
```

For a smoke test, restrict the number of examples:

```bash
python "$SUBMISSION_ROOT/code/predictions/predict.py" \
  --backend hf \
  --model gemma-2-9b-it \
  --checkpoint_root "$SUBMISSION_ROOT/grpo/gemma_bl" \
  --data_file "$SUBMISSION_ROOT/data/storyfeedback/stories_test.jsonl" \
  --output_jsonl "$SUBMISSION_ROOT/predictions/smoke_test.jsonl" \
  --max_examples 10 \
  --override
```

To generate with the base model without an adapter:

```bash
python "$SUBMISSION_ROOT/code/predictions/predict.py" \
  --backend hf \
  --model gemma-2-9b-it \
  --use_base_model_only \
  --data_file "$SUBMISSION_ROOT/data/storyfeedback/stories_test.jsonl" \
  --output_jsonl "$SUBMISSION_ROOT/predictions/gemma_base_test.jsonl" \
  --override
```

The generator resumes an existing output file by default. Use `--override` to replace it.

## 7. Automatically evaluate generated feedback

```bash
cd "$STORYS_ROOT"
python "$SUBMISSION_ROOT/code/eval/evaluate.py" \
  --input_file "$SUBMISSION_ROOT/predictions/gemma_bl_test.jsonl" \
  --output_file "$SUBMISSION_ROOT/eval_results/gemma_bl_test.jsonl"
```

Each output record is enriched with:

- feedback character and token length
- direct citation n-gram statistics
- `is_perfect_as_is`
- correctness, error-detection, and specificity scores
- average feedback quality score
- trigram repetition metrics
- within-story diversity information

For LEAF essays, add `--leaf_mode`.

The scorer defaults to `$STORYS_MODELS_DIR`. To use another scorer directory, set `STORYS_MODELS_DIR` before running the command.

## 8. Compare evaluated files

Open the HTML comparator in a browser:

```text
$SUBMISSION_ROOT/code/eval/results/compare.html
```

Upload one or more enriched JSONL files. It provides:

- overall corpus metrics
- noise-specific metrics
- aligned story-version comparisons
- per-version reward-model scores
- CSV export

The HTML file runs locally in the browser and does not compute reward-model predictions itself; upload files produced by `evaluate.py`.

## 9. Optional LLM-as-a-judge evaluation

The submitted API client code should be configured with the desired provider credentials before use. Then run the judge with a pair of evaluated approaches, for example:

```bash
cd "$SUBMISSION_ROOT/code/eval"
python judge_llm.py \
  --approaches gemma_bl,gemma_bo \
  --max_items 100 \
  --runs 3
```

Results are written under `code/eval/judge_results/`.

## 10. Reproducibility checklist

Use the same:

- dataset files and split definitions
- base-model checkpoints
- reward-model checkpoints
- prompt template (`bl_full` or `bl_only`)
- PyTorch, Transformers, TRL, PEFT, and bitsandbytes versions
- number of GPUs
- CUDA version
- seed `8549`

The repository currently has no pinned `requirements.txt`; record the installed package versions when producing final experiment results.
