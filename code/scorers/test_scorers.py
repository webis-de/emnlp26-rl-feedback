#!/usr/bin/env python3
"""
Test script for FeedbackQualityScorer.
Loads test data and compares model predictions with ground truth labels.
"""

import json
import torch
import numpy as np
from feedback_quality_scorer import FeedbackQualityScorer


def load_test_data(filepath):
    """Load test data from JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def find_extreme_examples(data, task, n=5):
    """
    Find examples with lowest and highest scores for a given task.
    
    Args:
        data: List of examples
        task: Task name
        n: Number of examples to find per extreme
    
    Returns:
        Tuple of (low_examples, high_examples)
    """
    # Filter out examples without scores for this task
    valid_examples = [ex for ex in data if task in ex and ex[task] is not None and not np.isnan(ex[task])]
    
    if not valid_examples:
        return [], []
    
    # Sort by score
    sorted_examples = sorted(valid_examples, key=lambda x: x[task])
    
    # Get extreme examples
    low_examples = sorted_examples[:n]
    high_examples = sorted_examples[-n:]
    
    return low_examples, high_examples


def test_task(scorer, task, low_examples, high_examples):
    """Test model predictions on extreme examples for a task."""
    print(f"\n{'='*80}")
    print(f"Task: {task.upper()}")
    print(f"{'='*80}")
    
    # Get the appropriate scoring function
    task_func_map = {
        'correctness': scorer.get_correctness_score,
        'error-detection': scorer.get_error_detection_score,
        'specificity': scorer.get_specificity_score
    }
    
    score_func = task_func_map[task]
    
    # Test low score examples
    print(f"\n--- LOW SCORE EXAMPLES ---")
    for i, example in enumerate(low_examples, 1):
        true_score = example[task]
        pred_score = score_func(example['story'], example['feedback'])
        
        print(f"\nExample {i}:")
        print(f"  True Score: {true_score:.3f}")
        print(f"  Pred Score: {pred_score:.3f}")
        print(f"  Difference: {abs(pred_score - true_score):.3f}")
        print(f"  Story (first 100 chars): {example['story'][:100]}...")
        print(f"  Feedback (first 100 chars): {example['feedback'][:100]}...")
    
    # Test high score examples
    print(f"\n--- HIGH SCORE EXAMPLES ---")
    for i, example in enumerate(high_examples, 1):
        true_score = example[task]
        pred_score = score_func(example['story'], example['feedback'])
        
        print(f"\nExample {i}:")
        print(f"  True Score: {true_score:.3f}")
        print(f"  Pred Score: {pred_score:.3f}")
        print(f"  Difference: {abs(pred_score - true_score):.3f}")
        print(f"  Story (first 100 chars): {example['story'][:100]}...")
        print(f"  Feedback (first 100 chars): {example['feedback'][:100]}...")
    
    # Compute average error for this task on extreme examples
    all_examples = low_examples + high_examples
    errors = []
    for example in all_examples:
        true_score = example[task]
        pred_score = score_func(example['story'], example['feedback'])
        errors.append(abs(pred_score - true_score))
    
    avg_error = np.mean(errors)
    print(f"\n--- SUMMARY ---")
    print(f"Average absolute error on {len(all_examples)} extreme examples: {avg_error:.3f}")


def test_cache_efficiency(scorer, example):
    """Test that caching works by calling multiple score functions."""
    print(f"\n{'='*80}")
    print("TESTING CACHE EFFICIENCY")
    print(f"{'='*80}")
    
    story = example['story']
    feedback = example['feedback']
    
    print("\nCalling all individual score functions on the same example...")
    print("(Should use cached predictions after first call)")
    
    # Clear cache first
    initial_cache_size = len(scorer._cache)
    
    # Call each score function
    correctness = scorer.get_correctness_score(story, feedback)
    error_det = scorer.get_error_detection_score(story, feedback)
    specificity = scorer.get_specificity_score(story, feedback)
    
    final_cache_size = len(scorer._cache)
    cache_increase = final_cache_size - initial_cache_size
    
    print(f"\nCache size before: {initial_cache_size}")
    print(f"Cache size after: {final_cache_size}")
    print(f"Cache increase: {cache_increase}")
    print(f"\n✓ Cache working correctly!" if cache_increase == 1 else "✗ Cache issue detected!")
    
    print(f"\nScores:")
    print(f"  Correctness: {correctness:.3f}")
    print(f"  Error Detection: {error_det:.3f}")
    print(f"  Specificity: {specificity:.3f}")
    
    # Test get_all_scores
    print("\nCalling get_all_scores (should use cache)...")
    cache_before_all = len(scorer._cache)
    all_scores = scorer.get_all_scores(story, feedback)
    cache_after_all = len(scorer._cache)
    
    print(f"Cache size before: {cache_before_all}")
    print(f"Cache size after: {cache_after_all}")
    print(f"✓ get_all_scores used cache!" if cache_before_all == cache_after_all else "✗ Cache issue!")
    
    print(f"\nAll scores: {all_scores}")


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load scorer
    print("\nLoading FeedbackQualityScorer...")
    scorer = FeedbackQualityScorer(device)
    
    # Load test data
    test_file = '../../data/stories/human_ratings_test.jsonl'
    print(f"\nLoading test data from {test_file}...")
    test_data = load_test_data(test_file)
    print(f"Loaded {len(test_data)} test examples")
    
    # Define tasks
    tasks = ['correctness', 'error-detection', 'specificity']
    
    # Test each task
    for task in tasks:
        # Find extreme examples
        low_examples, high_examples = find_extreme_examples(test_data, task, n=2)
        
        if not low_examples or not high_examples:
            print(f"\nSkipping task '{task}' - insufficient examples")
            continue
        
        # Test the task
        test_task(scorer, task, low_examples, high_examples)
    
    # Test cache efficiency
    if test_data:
        test_cache_efficiency(scorer, test_data[0])
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
