import argparse
import json
import os
import re
import sys
import torch
from pathlib import Path
from collections import defaultdict
import itertools
from typing import Optional

# Add parent directory to path to enable imports from storys package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from storys.code.scorers.feedback_quality_scorer import FeedbackQualityScorer, SEED, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = (SCRIPT_DIR.parent / "predictions" / "outputs").resolve()
RESULTS_DIR = (SCRIPT_DIR / "results").resolve()


def _resolve_input_path(user_input: str) -> Path:
    """Resolve input file path relative to the configured predictions directory."""
    candidate = Path(user_input)

    if candidate.is_absolute() and candidate.exists():
        return candidate

    if candidate.exists():
        return candidate.resolve()

    joined = (PREDICTIONS_DIR / candidate).resolve()
    if joined.exists():
        return joined

    fallback = (SCRIPT_DIR / candidate).resolve()
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Input file not found: {user_input}")


def _resolve_output_path(input_path: Path, user_output: Optional[str]) -> Path:
    """Place outputs under eval/results, mirroring input subdirectories."""
    if user_output:
        out_path = Path(user_output)
        if not out_path.is_absolute():
            out_path = (SCRIPT_DIR / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    try:
        rel = input_path.resolve().relative_to(PREDICTIONS_DIR)
        out_dir = (RESULTS_DIR / rel.parent).resolve()
    except ValueError:
        out_dir = RESULTS_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"eval_{input_path.name}"


def _normalize_text(text: str) -> list:
    """Lowercase, strip punctuation, split into tokens."""
    text = re.sub(r"[\.,!?;:\-\(\)\[\]\{\}\"']", "", text.lower())
    return [t for t in text.split() if t]


def _extract_trigrams(tokens: list) -> set:
    """Extract all trigrams from token list."""
    if len(tokens) < 3:
        return set()
    return {" ".join(tokens[i:i+3]) for i in range(len(tokens) - 2)}


def _find_longest_match_from(feedback_tokens: list, input_tokens: list, input_positions: dict, start_idx: int) -> tuple:
    """Find the longest contiguous match that starts at feedback[start_idx]."""
    token = feedback_tokens[start_idx]
    candidate_positions = input_positions.get(token, [])
    best_len = 0
    best_input_start = -1

    for input_start in candidate_positions:
        cur_len = 0
        i = start_idx
        j = input_start
        while i < len(feedback_tokens) and j < len(input_tokens) and feedback_tokens[i] == input_tokens[j]:
            cur_len += 1
            i += 1
            j += 1

        if cur_len > best_len:
            best_len = cur_len
            best_input_start = input_start

    return best_len, best_input_start


def _compute_direct_citation_metrics(input_text: str, feedback_text: str, min_ngram: int = 3) -> dict:
    """Count maximal direct citation n-grams from input text that appear in feedback.

    Uses contiguous token overlap after normalization, with a minimum n-gram size.
    Greedily takes the longest match at each feedback token position to avoid
    counting smaller n-grams nested inside larger matched spans.
    """
    try:
        input_tokens = _normalize_text(input_text)
        feedback_tokens = _normalize_text(feedback_text)
    except Exception as e:
        print(f"Error normalizing text. Input: {input_text}, Feedback: {feedback_text}")
        raise e

    if len(input_tokens) < min_ngram or len(feedback_tokens) < min_ngram:
        return {
            "direct_citation_count": 0,
            "direct_citation_total_tokens": 0,
            "direct_citation_longest_ngram": 0,
            "direct_citation_avg_ngram": 0.0,
            "direct_citation_ngrams": [],
        }

    input_positions = defaultdict(list)
    for idx, token in enumerate(input_tokens):
        input_positions[token].append(idx)

    citation_ngrams = []
    citation_lengths = []
    i = 0
    while i < len(feedback_tokens):
        longest_len, _ = _find_longest_match_from(feedback_tokens, input_tokens, input_positions, i)
        if longest_len >= min_ngram:
            citation_ngrams.append(" ".join(feedback_tokens[i:i + longest_len]))
            citation_lengths.append(longest_len)
            i += longest_len
        else:
            i += 1

    if not citation_lengths:
        return {
            "direct_citation_count": 0,
            "direct_citation_total_tokens": 0,
            "direct_citation_longest_ngram": 0,
            "direct_citation_avg_ngram": 0.0,
            "direct_citation_ngrams": [],
        }

    return {
        "direct_citation_count": len(citation_lengths),
        "direct_citation_total_tokens": sum(citation_lengths),
        "direct_citation_longest_ngram": max(citation_lengths),
        "direct_citation_avg_ngram": sum(citation_lengths) / len(citation_lengths),
        "direct_citation_ngrams": citation_ngrams,
    }


def _compute_trigram_repetition(feedbacks: list) -> list:
    """
    Compute per-feedback trigram repetition score.
    
    Score = (# unique trigrams appearing in other feedback) / (# unique trigrams)
    Higher score = more repetitive/less diverse feedback.
    
    Returns list of floats aligned with feedbacks order.
    """
    trigram_sets = []
    trigram_counts = defaultdict(int)
    
    for fb in feedbacks:
        tokens = _normalize_text(fb)
        trigrams = _extract_trigrams(tokens)
        trigram_sets.append(trigrams)
        for tg in trigrams:
            trigram_counts[tg] += 1

    scores = []
    for trigrams in trigram_sets:
        if not trigrams:
            scores.append(0.0)
            continue
        repeated = sum(1 for tg in trigrams if trigram_counts[tg] > 1)
        scores.append(repeated / len(trigrams))
    return scores


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets (safe for empty sets)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _compute_within_story_diversity(records, group_key="gold_story"):
    """Compute diversity of feedback across variations of the same story/prompt.

    Args:
        records: list of prediction records
        group_key: field to group by ('gold_story' for stories, 'gold_essay' for leaf)

    Returns tuple: (group_summaries, overall_summary)
    - group_summaries: list of dicts per group with keys:
        'group_key', 'count', 'avg_pairwise_trigram_overlap', 'avg_pairwise_trigram_disparity'
    - overall_summary: dict with aggregate metrics across groups
    """
    # Group records by the specified key
    by_story = defaultdict(list)
    for i, r in enumerate(records):
        key = r.get(group_key) or r.get("story")
        by_story[key].append(i)

    # Precompute trigram sets for feedbacks
    trigram_sets = []
    for r in records:
        tr = _extract_trigrams(_normalize_text(r.get("feedback", "")))
        trigram_sets.append(tr)

    group_summaries = []
    all_pairwise_overlaps = []

    for gold_story, idxs in by_story.items():
        if not idxs:
            continue

        # Pairwise trigram Jaccard overlap
        overlaps = []
        for i, j in itertools.combinations(idxs, 2):
            overlaps.append(_jaccard(trigram_sets[i], trigram_sets[j]))
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        avg_disparity = 1.0 - avg_overlap

        group_summaries.append({
            "group_key": gold_story,
            "count": len(idxs),
            "avg_pairwise_trigram_overlap": avg_overlap,
            "avg_pairwise_trigram_disparity": avg_disparity,
        })
        if overlaps:
            all_pairwise_overlaps.append(avg_overlap)

    overall = {
        "groups": len(group_summaries),
        "avg_pairwise_trigram_overlap": (sum(all_pairwise_overlaps) / len(all_pairwise_overlaps)) if all_pairwise_overlaps else 0.0,
        "avg_pairwise_trigram_disparity": (1.0 - (sum(all_pairwise_overlaps) / len(all_pairwise_overlaps))) if all_pairwise_overlaps else 0.0,
    }

    return group_summaries, overall

def main(input_file, output_file, leaf_mode=False, fbonly=False, qwenrm=False):
    # Set seed for reproducibility (scorer already has deterministic settings)
    set_seed(SEED)

    if qwenrm and fbonly:
        raise ValueError("--qwenrm is not compatible with --fbonly. Use only one mode.")
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # Skip calculation if output file already exists
    if output_path.exists():
        print(f"Output file {output_path} already exists. Skipping calculation.")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Uses models from models_final by default with Spearman-optimized single-task models
    scorer = FeedbackQualityScorer(device, fbonly=fbonly, qwenrm=qwenrm)

    # Load all records
    records = []
    with open(input_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            if line.strip():
                records.append(json.loads(line))
    
    # Choose grouping field based on mode
    group_counts = defaultdict(int)
    pai_by_group = defaultdict(int)

    # First pass: add length, PAI flags, and quality scores
    for r in records:
        # input

        if not leaf_mode:
            # story mode
            prompt = r.get("story", "")
            group_val = r.get("noise")
        else:
            # leaf mode
            title = r.get("essay_title", "").strip()
            if "essay_text" in r:
                essay = r.get("essay_text", "").strip()
            else:
                essay = r.get("essay", "").strip()

            prompt = f"Title: {title}\n{essay}"    
            metadata = r.get("metadata")
            if isinstance(metadata, dict) and "noise" in metadata:
                group_val = metadata["noise"]
            else:
                group_val = r.get("split", "unknown") # for ai / human

        # output
        completion = r.get("feedback", "")        
        
        # Store for later reuse
        r["group"] = group_val
        group_counts[group_val] += 1

        # Length features
        r["feedback_length_chars"] = len(completion)
        r["feedback_length_tokens"] = len(_normalize_text(completion))

        # Direct citation features (maximal shared n-grams with input text)
        citation_metrics = _compute_direct_citation_metrics(prompt, completion, min_ngram=3)
        r.update(citation_metrics)

        # Perfect-as-is detection
        is_pai = "perfect as-is" in completion.lower() or "perfect-as-is" in completion.lower()
        r["is_perfect_as_is"] = is_pai
        if is_pai:
            pai_by_group[group_val] += 1

        # Quality scoring
        scores = scorer.get_all_scores(prompt, completion)
        core_scores = [
            scores.get("correctness"),
            scores.get("error-detection"),
            scores.get("specificity"),
        ] if scores else []
        core_scores = [s for s in core_scores if s is not None]
        avg_score = sum(core_scores) / len(core_scores) if core_scores else None    
        r["feedback_quality_score"] = avg_score
        r["feedback_quality_scores"] = scores

    # Diversity metrics: overall
    # Compares each feedback against all other feedbacks in the entire dataset
    all_feedbacks = [r.get("feedback", "") for r in records]
    trigram_rep_all = _compute_trigram_repetition(all_feedbacks)
    for r, rep in zip(records, trigram_rep_all):
        r["trigram_repetition_all"] = rep

    # Diversity metrics: by group (noise for stories, labels for enthymemes)
    # Compares each feedback against only the feedbacks in its group (noise type for stories, or actual_gap/random_gap/no_gap for enthymemes)
    group_feedback_indices = defaultdict(list)
    for idx, r in enumerate(records):
        gval = r.get("group", "unknown")
        group_feedback_indices[gval].append(idx)

    for group_val, idxs in group_feedback_indices.items():
        group_feedbacks = [all_feedbacks[i] for i in idxs]
        reps = _compute_trigram_repetition(group_feedbacks)
        for local_i, global_i in enumerate(idxs):
            records[global_i]["trigram_repetition_group"] = reps[local_i]

    # Diversity excluding PAI: overall
    # Compares each feedback against all other feedbacks in the entire dataset, EXCLUDING "perfect as-is" responses
    non_pai_indices = [i for i, r in enumerate(records) if not r["is_perfect_as_is"]]
    non_pai_feedbacks = [all_feedbacks[i] for i in non_pai_indices]
    trigram_rep_non_pai = _compute_trigram_repetition(non_pai_feedbacks) if non_pai_feedbacks else []
    
    for rep, idx_global in zip(trigram_rep_non_pai, non_pai_indices):
        records[idx_global]["trigram_repetition_all_no_pai"] = rep
    for i in range(len(records)):
        if "trigram_repetition_all_no_pai" not in records[i]:
            records[i]["trigram_repetition_all_no_pai"] = None

    # Diversity excluding PAI: by group
    # Compares each feedback against only the feedbacks in its group, EXCLUDING "perfect as-is" responses
    for group_val, idxs in group_feedback_indices.items():
        non_pai_group = [i for i in idxs if not records[i]["is_perfect_as_is"]]
        if non_pai_group:
            feedbacks = [all_feedbacks[i] for i in non_pai_group]
            reps = _compute_trigram_repetition(feedbacks)
            for rep, gi in zip(reps, non_pai_group):
                records[gi]["trigram_repetition_group_no_pai"] = rep
    
    for r in records:
        if "trigram_repetition_group_no_pai" not in r:
            r["trigram_repetition_group_no_pai"] = None

    # Within-story diversity metrics across corrupted/original variations of same story (group by gold_story)
    diversity_group_key = "gold_story"
    if leaf_mode:
        diversity_group_key = "gold_essay"
    group_summaries_by_story, within_overall = _compute_within_story_diversity(records, diversity_group_key)

    # Write enriched output, replacing NaN with None for JSON compatibility
    import math
    def clean_nans(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        elif isinstance(obj, dict):
            return {k: clean_nans(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nans(x) for x in obj]
        else:
            return obj

    with open(output_path, "w", encoding="utf-8") as f_out:
        for r in records:
            cleaned = clean_nans(r)
            f_out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score feedback quality for predictions.")
    parser.add_argument("--input_file", "-i", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--output_file", "-o", type=str, help="Path to output JSONL file.")
    parser.add_argument(
        "--predictions_subdir",
        type=str,
        default="outputs",
        help="Subdirectory under predictions/ used to resolve relative input paths (default: outputs).",
    )
    parser.add_argument("--leaf_mode", action="store_true", help="Evaluate LEAF essay predictions (group by split instead of noise).")
    parser.add_argument("--fbonly", action="store_true", help="Use only feedback (no story/essay) as input to the scorer.")
    parser.add_argument("--qwenrm", action="store_true", help="Use Qwen reward models from models_qwen as scorers.")
    args = parser.parse_args()

    PREDICTIONS_DIR = (SCRIPT_DIR.parent / "predictions" / args.predictions_subdir).resolve()

    input_path = _resolve_input_path(args.input_file)
    output_path = _resolve_output_path(input_path, args.output_file)
    main(input_path, output_path, args.leaf_mode, args.fbonly, args.qwenrm)