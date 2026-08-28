#!/usr/bin/env python3
"""
Shared core logic for quality-scorer evaluation scripts.

The three variant scripts (evaluate_quality_scorer.py, evaluate_quality_scorer_fbonly.py,
evaluate_quality_scorer_qwenrm.py) all use this module.  Each variant calls
``run_evaluation(config)`` with a small config dict that captures its unique settings.
"""
import os

# Set CUBLAS environment variable BEFORE importing torch for deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))

import pandas as pd
import numpy as np
import json
import torch
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import matplotlib.pyplot as plt

from feedback_quality_training_utils import (
    create_regression_model,
    RegressionModelWithClipping,
    SEED,
    set_seed,
    build_input_text,
)


# Set seed at module import for reproducibility
set_seed(SEED)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(model_dir):
    """Find the best checkpoint from hyperparameter tuning or latest checkpoint."""
    model_path = Path(model_dir)
    if not model_path.exists():
        return None
    return model_path


def load_regression_model(checkpoint_path, num_labels=1, model_name="answerdotai/ModernBERT-large"):
    """Load a regression model from checkpoint and wrap with clipping."""
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        config_file = checkpoint_path / 'config.json'
        if config_file.exists():
            base_model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_path,
                num_labels=num_labels,
                problem_type="regression",
                ignore_mismatched_sizes=True
            )
            return RegressionModelWithClipping(base_model)
        else:
            model_file = checkpoint_path / 'pytorch_model.bin'
            if not model_file.exists():
                model_file = checkpoint_path / 'model.safetensors'
            if model_file.exists():
                checkpoint_path = model_file

    model = create_regression_model(model_name, dropout=0.1)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.base_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.base_model.load_state_dict(checkpoint)
    return model


def load_model_settings(model_path: Path):
    """Load per-model settings from saved hyperparameters.json."""
    settings = {
        'model_name': "answerdotai/ModernBERT-large",
        'max_length': 512,
        'fbonly': False,
    }
    hp_path = model_path / 'hyperparameters.json'
    if hp_path.exists():
        with open(hp_path, 'r') as f:
            hp = json.load(f)
        settings['model_name'] = hp.get('model_name', settings['model_name'])
        settings['max_length'] = int(hp.get('max_length', settings['max_length']))
        settings['fbonly'] = bool(hp.get('fbonly', settings['fbonly']))
    return settings


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(filepath):
    """Load JSONL data."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def assert_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    df_name: str,
    check_nulls: bool = True,
):
    """Fail fast when required columns are missing and optionally null."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{df_name}: missing required column(s): {', '.join(missing_columns)}"
        )

    if not check_nulls:
        return

    for col in required_columns:
        missing_mask = df[col].isna()
        if missing_mask.any():
            missing_count = int(missing_mask.sum())
            sample_rows = df.index[missing_mask][:5].tolist()
            raise ValueError(
                f"{df_name}: column '{col}' has {missing_count} missing/null value(s); "
                f"sample row indices: {sample_rows}"
            )


def _get_valid_numeric_values(df: pd.DataFrame, column: str, df_name: str) -> np.ndarray:
    """Return valid numeric values for a column, skipping missing/non-numeric rows."""
    numeric_values = pd.to_numeric(df[column], errors='coerce')
    invalid_mask = numeric_values.isna()
    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        sample_rows = df.index[invalid_mask][:5].tolist()
        print(
            f"{df_name}: skipped {invalid_count} row(s) where '{column}' is missing/non-numeric; "
            f"sample row indices: {sample_rows}"
        )
    return numeric_values[~invalid_mask].to_numpy(dtype=float)


def _filter_rows_for_task(
    df: pd.DataFrame,
    task: str,
    required_text_columns: list[str],
    df_name: str,
) -> pd.DataFrame:
    """Filter rows to those with non-null task label and required text fields."""
    required_columns = [task] + required_text_columns
    missing_mask = df[required_columns].isna().any(axis=1)
    filtered_df = df.loc[~missing_mask].copy()

    skipped_count = int(missing_mask.sum())
    if skipped_count > 0:
        missing_counts = df.loc[missing_mask, required_columns].isna().sum()
        missing_parts = [
            f"{col}: {int(count)}"
            for col, count in missing_counts.items()
            if int(count) > 0
        ]
        print(
            f"{df_name}: skipped {skipped_count} row(s) for task '{task}' due to missing required fields "
            f"({', '.join(missing_parts)})"
        )

    # Coerce labels to numeric and drop any remaining invalid rows.
    filtered_df[task] = pd.to_numeric(filtered_df[task], errors='coerce')
    invalid_label_mask = filtered_df[task].isna()
    if invalid_label_mask.any():
        invalid_count = int(invalid_label_mask.sum())
        sample_rows = filtered_df.index[invalid_label_mask][:5].tolist()
        print(
            f"{df_name}: skipped {invalid_count} row(s) for task '{task}' due to non-numeric labels; "
            f"sample row indices: {sample_rows}"
        )
        filtered_df = filtered_df.loc[~invalid_label_mask].copy()

    return filtered_df


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_pearson_standard_error(pred, true, n_bootstrap=1000, seed=SEED):
    """Estimate Pearson correlation standard error via bootstrap resampling."""
    n = len(true)
    if n < 3:
        return np.nan

    rng = np.random.RandomState(seed)
    boot_corrs = []
    for _ in range(n_bootstrap):
        sample_idx = rng.randint(0, n, size=n)
        sample_true = true[sample_idx]
        sample_pred = pred[sample_idx]
        if np.std(sample_true) == 0 or np.std(sample_pred) == 0:
            continue
        corr, _ = pearsonr(sample_true, sample_pred)
        if not np.isnan(corr):
            boot_corrs.append(corr)

    if len(boot_corrs) < 2:
        return np.nan
    return float(np.std(boot_corrs, ddof=1))


def compute_metrics_for_task(predictions, labels):
    """Compute Pearson, Spearman, and MSE metrics for a single task."""
    mask = ~np.isnan(labels)
    pred = predictions[mask]
    true = labels[mask]

    if len(true) == 0:
        return {}

    if np.std(pred) == 0 or np.std(true) == 0:
        pearson_corr = np.nan
        pearson_se = np.nan
        spearman_corr = np.nan
    else:
        pearson_corr, _ = pearsonr(true, pred)
        spearman_corr, _ = spearmanr(true, pred)
        pearson_se = compute_pearson_standard_error(pred, true)

    squared_errors = (pred - true) ** 2
    mse = np.mean(squared_errors)
    mse_std = np.std(squared_errors, ddof=1) if len(squared_errors) > 1 else 0.0

    return {
        'pearson': pearson_corr,
        'pearson_se': pearson_se,
        'spearman': spearman_corr,
        'mse': mse,
        'mse_std': mse_std,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _latex_num(x: float, decimals: int = 3) -> str:
    """Format a float without leading zero before the decimal point."""
    s = f"{x:.{decimals}f}"
    if s.startswith("-0."):
        return "-." + s[3:]
    if s.startswith("0."):
        return "." + s[2:]
    return s


def plot_predictions_vs_actual(
    predictions, labels, task, model_type,
    output_dir='../../data/scorer_eval',
    prefix: str = '',
):
    """Create and save scatter + residual plots of predicted vs actual scores."""
    mask = ~np.isnan(labels)
    pred = predictions[mask]
    true = labels[mask]

    if len(true) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(true, pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    min_val = min(true.min(), pred.min())
    max_val = max(true.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    z = np.polyfit(true, pred, 1)
    p = np.poly1d(z)
    ax.plot(true, p(true), 'b-', lw=2, alpha=0.7, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    ax.set_xlabel('Actual Score', fontsize=12)
    ax.set_ylabel('Predicted Score', fontsize=12)
    ax.set_title(f'{model_type.title()} - {task.title()}\nPredicted vs Actual', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    residuals = pred - true
    ax.scatter(true, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Actual Score', fontsize=12)
    ax.set_ylabel('Residuals (Pred - Actual)', fontsize=12)
    ax.set_title(f'{model_type.title()} - {task.title()}\nResidual Plot', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_task = task.replace('/', '_').replace(' ', '_')
    safe_model = model_type.replace('/', '_').replace(' ', '_')
    filename = output_path / f'{prefix}predictions_{safe_model}_{safe_task}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {filename}")


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_random_baseline(train_df, test_df, tasks, seed=SEED):
    """Evaluate random baseline (predictions uniform in training range)."""
    results = []
    predictions_data = {}
    rng = np.random.RandomState(seed)

    for task in tasks:
        train_values = _get_valid_numeric_values(train_df, task, f"train_df ({task})")
        if len(train_values) == 0:
            raise ValueError(f"train_df: task '{task}' has no valid labels")
        train_min, train_max = train_values.min(), train_values.max()

        test_true = _get_valid_numeric_values(test_df, task, f"test_df ({task})")
        if len(test_true) == 0:
            raise ValueError(f"test_df: task '{task}' has no valid labels")

        test_pred = rng.uniform(train_min, train_max, len(test_true))
        metrics = compute_metrics_for_task(test_pred, test_true)
        for metric_name, value in metrics.items():
            results.append({'model_type': 'random-baseline', 'task': task,
                             'metric': metric_name, 'value': value})
        predictions_data[task] = (test_pred, test_true)

    return results, predictions_data


def evaluate_mean_baseline(train_df, test_df, tasks):
    """Evaluate mean baseline (predict training mean for every sample)."""
    results = []
    predictions_data = {}

    for task in tasks:
        train_values = _get_valid_numeric_values(train_df, task, f"train_df ({task})")
        if len(train_values) == 0:
            raise ValueError(f"train_df: task '{task}' has no valid labels")

        train_mean = float(np.mean(train_values))

        test_true = _get_valid_numeric_values(test_df, task, f"test_df ({task})")
        if len(test_true) == 0:
            raise ValueError(f"test_df: task '{task}' has no valid labels")

        test_pred = np.full(len(test_true), train_mean)
        if len(test_pred) > 0:
            test_pred[0] += 0.0001  # avoid constant -> undefined Pearson
        metrics = compute_metrics_for_task(test_pred, test_true)
        for metric_name, value in metrics.items():
            results.append({'model_type': 'mean-baseline', 'task': task,
                             'metric': metric_name, 'value': value})
        predictions_data[task] = (test_pred, test_true)

    return results, predictions_data


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate_singletask_model(
    model_path,
    test_df,
    task,
    device,
    fbonly_override: Optional[bool] = None,
):
    """Evaluate a single-task regression model.

    Args:
        model_path: Path to the saved model directory.
        test_df: Test DataFrame.
        task: Task name (column in test_df).
        device: torch device.
        fbonly_override: If not None, overrides the ``fbonly`` flag read from
            ``hyperparameters.json``.  Pass ``False`` for the base scorer,
            ``True`` for the fbonly scorer, and ``None`` (default) for the
            qwenrm scorer which reads the flag from the checkpoint.
    """
    print(f"\nEvaluating single-task model for {task} from {model_path}")

    model_path = Path(model_path)
    settings = load_model_settings(model_path)
    if fbonly_override is not None:
        settings['fbonly'] = fbonly_override

    # Load tokenizer from checkpoint when possible to avoid tokenizer/model mismatch
    tokenizer_source = (
        model_path if (model_path / 'tokenizer_config.json').exists()
        else settings['model_name']
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError(f"Tokenizer has no pad/eos token for {tokenizer_source}")

    print(f"Using tokenizer from: {tokenizer_source}")
    print(f"Using model_name: {settings['model_name']}")
    print(f"Using max_length={settings['max_length']}, fbonly={settings['fbonly']}")

    model = load_regression_model(model_path, num_labels=1, model_name=settings['model_name'])
    model = model.to(device)
    model.eval()

    required_text_columns = ['feedback'] if settings['fbonly'] else ['story', 'feedback']
    assert_required_columns(test_df, [task] + required_text_columns, 'test_df', check_nulls=False)

    test_subset = _filter_rows_for_task(test_df, task, required_text_columns, 'test_df')
    if len(test_subset) == 0:
        raise ValueError(
            f"test_df: no valid rows remain for task '{task}' after filtering missing labels/text"
        )

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for _, row in tqdm(test_subset.iterrows(), total=len(test_subset), desc=f"Predicting {task}"):
            text = build_input_text(row['story'], row['feedback'], settings['fbonly'])
            encoding = tokenizer(
                text,
                max_length=settings['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            output = model(input_ids, attention_mask)
            logits = output.logits if hasattr(output, 'logits') else output
            prediction = logits.cpu().numpy().squeeze()
            if prediction.ndim == 0:
                prediction = float(prediction)
            all_predictions.append(prediction)
            all_labels.append(row[task])

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    metrics = compute_metrics_for_task(all_predictions, all_labels)
    results = [
        {'model_type': 'single-task', 'task': task, 'metric': name, 'value': value}
        for name, value in metrics.items()
    ]
    return results, (all_predictions, all_labels)


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def _build_latex_table(df, tasks, task_order, task_abbrev, model_display):
    """Build a LaTeX tabular string from evaluation results."""
    latex_data = {}
    for task in tasks:
        latex_data[task] = {}
        for model_type in model_display:
            pearson_df = df[(df['model_type'] == model_type) & (df['task'] == task) & (df['metric'] == 'pearson')]
            pearson_se_df = df[(df['model_type'] == model_type) & (df['task'] == task) & (df['metric'] == 'pearson_se')]
            latex_data[task][model_type] = {
                'pearson': pearson_df['value'].values[0] if not pearson_df.empty else None,
                'pearson_se': pearson_se_df['value'].values[0] if not pearson_se_df.empty else None,
            }

    rows = ["\\begin{tabular}{lrrrr}", "\\toprule"]
    header = "Model       & " + " & ".join([task_abbrev[t] for t in task_order]) + " \\\\"
    rows += [header, "\\midrule"]

    for model_type, label in model_display.items():
        row_parts = [label]
        for task in task_order:
            pearson = latex_data[task][model_type]['pearson']
            pearson_se = latex_data[task][model_type]['pearson_se']
            if pearson is not None and not np.isnan(pearson):
                pearson_str = _latex_num(pearson)
                if pearson_se is not None and not np.isnan(pearson_se):
                    cell = f"{pearson_str} \\scriptsize $\\pm {_latex_num(pearson_se)}$"
                else:
                    cell = pearson_str
            else:
                cell = "---"
            row_parts.append(cell)
        rows.append(" & ".join(row_parts) + " \\\\")

    rows += ["\\bottomrule", "\\end{tabular}"]
    return '\n'.join(rows)


# ---------------------------------------------------------------------------
# Unified evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(config: dict):
    """Run the full scorer evaluation pipeline.

    Config keys:
        train_data (str): Path to training JSONL.
        test_data (str): Path to test JSONL.
        tasks (list[str]): Task names.
        get_model_path (callable): ``task_dir -> str`` mapping task dir to model path.
        check_path_exists (bool): If True, skip model if path does not exist.
        fbonly_override (bool | None): Forwarded to ``evaluate_singletask_model``.
        csv_filename (str): Output CSV path.
        tex_filename (str): Output LaTeX table path.
        plot_prefix (str): Filename prefix for scatter plots (e.g. 'fbonly_').
        single_task_label (str): Display name for the single-task model in the LaTeX table.
    """
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using SEED: {SEED}")

    tasks = config['tasks']
    fbonly_override = config.get('fbonly_override', None)
    plot_prefix = config.get('plot_prefix', '')

    print("Loading data...")
    train_df = load_data(config['train_data'])
    test_df = load_data(config['test_data'])

    # Verify required task columns exist. Per-task evaluators skip invalid rows and report counts.
    required_columns = list(tasks)
    assert_required_columns(train_df, required_columns, 'train_df', check_nulls=False)
    assert_required_columns(test_df, required_columns, 'test_df', check_nulls=False)

    all_results = []

    print("\n" + "=" * 50)
    print("Evaluating Random Baseline")
    print("=" * 50)
    random_results, _ = evaluate_random_baseline(train_df, test_df, tasks)
    all_results.extend(random_results)

    print("\n" + "=" * 50)
    print("Evaluating Mean Baseline")
    print("=" * 50)
    baseline_results, _ = evaluate_mean_baseline(train_df, test_df, tasks)
    all_results.extend(baseline_results)

    for task in tasks:
        task_dir = task.replace('-', '_')
        model_path_str = config['get_model_path'](task_dir)
        model_path = Path(model_path_str)

        if config.get('check_path_exists', False) and not model_path.exists():
            print(f"\nWarning: model not found at {model_path}")
            continue

        print("\n" + "=" * 50)
        print(f"Evaluating Single-Task Model: {task}")
        print("=" * 50)
        singletask_results, (pred, actual) = evaluate_singletask_model(
            model_path, test_df, task, device, fbonly_override=fbonly_override
        )
        all_results.extend(singletask_results)
        plot_predictions_vs_actual(pred, actual, task, 'single-task', prefix=plot_prefix)

    df = pd.DataFrame(all_results)

    model_types = ['random-baseline', 'mean-baseline', 'single-task']
    for model_type in model_types:
        model_df = df[df['model_type'] == model_type]
        for metric in ['mse']:
            metric_values = model_df[model_df['metric'] == metric]['value']
            if len(metric_values) > 0:
                all_results.append({
                    'model_type': model_type, 'task': 'overall',
                    'metric': f'hmean_{metric}', 'value': stats.hmean(metric_values)
                })
            if len(metric_values) > 1:
                all_results.append({
                    'model_type': model_type, 'task': 'overall',
                    'metric': f'std_{metric}', 'value': metric_values.std(ddof=1)
                })

    df = pd.DataFrame(all_results)
    df = df.sort_values(by=['task', 'model_type', 'metric']).reset_index(drop=True)

    csv_filename = config['csv_filename']
    df.to_csv(csv_filename, index=False)
    print(f"\n{'=' * 50}")
    print(f"Successfully created '{csv_filename}'")
    print(f"{'=' * 50}")

    print("\n" + "=" * 50)
    print("Generating LaTeX table")
    print("=" * 50)

    task_order = ['specificity', 'correctness', 'error-detection']
    task_abbrev = {
        'specificity': 'Specific.', 'correctness': 'Correct.', 'error-detection': 'Error Det.'
    }
    model_display = {
        'random-baseline': 'Random',
        'mean-baseline': 'Mean',
        'single-task': config.get('single_task_label', 'ModernBERT'),
    }

    latex_str = _build_latex_table(df, tasks, task_order, task_abbrev, model_display)
    tex_filename = config['tex_filename']
    with open(tex_filename, 'w') as f:
        f.write(latex_str)
    print(f"\nLaTeX table saved to '{tex_filename}'")
