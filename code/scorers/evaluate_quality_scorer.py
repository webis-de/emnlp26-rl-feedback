#!/usr/bin/env python3
"""
Evaluate trained ModernBERT (base) quality scorer on test data.

This script is a thin wrapper around scorer_core.run_evaluation().
Variant-specific settings: full story+feedback input, models_final/ model directory.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))
from scorer_core import run_evaluation



def main():
    run_evaluation({
        'train_data': '../../data/stories/human_ratings_train.jsonl',
        'test_data': '../../data/stories/human_ratings_test.jsonl',
        'tasks': ['correctness', 'error-detection', 'specificity'],
        'get_model_path': lambda task_dir: f'../../models_final/{task_dir}',
        'check_path_exists': False,
        'fbonly_override': False,
        'csv_filename': '../../data/scorer_eval/model_metrics_comparison.csv',
        'tex_filename': '../../data/scorer_eval/model_metrics_table.tex',
        'plot_prefix': '',
        'single_task_label': 'ModernBERT',
    })


if __name__ == '__main__':
    main()
