#!/usr/bin/env python3
import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = (SCRIPT_DIR / "test_results").resolve()
DEFAULT_OUTPUT_ROOT = (SCRIPT_DIR / "judge_results").resolve()

# API call logging state
_api_call_count = 0
_api_logs = []  # List to store API call logs (first 10 calls)


def _load_judge_instruction_config(config_path: str, assume_all_original: bool = False) -> dict:
    """Load dual instruction config from JSON.

    Expected keys:
      - original_instruction
      - corruption_instruction
    """
    p = Path(config_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Judge instruction config not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Judge instruction config must be a JSON object.")

    original_instruction = str(payload.get("original_instruction", "")).strip()
    corruption_instruction = str(payload.get("corruption_instruction", "")).strip()
    force_original = bool(payload.get("assume_all_original", False) or assume_all_original)

    if not original_instruction:
        raise ValueError("Missing or empty 'original_instruction' in judge instruction config.")

    # If only original instruction is provided, optionally run in "all-original" mode.
    if not corruption_instruction and force_original:
        corruption_instruction = original_instruction
    if not corruption_instruction:
        raise ValueError("Missing or empty 'corruption_instruction' in judge instruction config.")

    return {
        "original_instruction": original_instruction,
        "corruption_instruction": corruption_instruction,
        "assume_all_original": force_original,
    }


def _compute_bradley_terry_rankings(
    results: List["PairItemResult"],
    approaches: List[str],
    output_dir: Path,
    leaderboard_by_approach: Dict[str, dict] = None,
    group_label: str = "overall",
    noise_value: str = "all",
    per_instance_filename: str = "",
) -> List[dict]:
    """Compute Bradley-Terry rankings and return grouped average-rank rows."""
    if not results or not approaches:
        return []

    try:
        import choix
    except ImportError:
        print("Bradley-Terry skipped: missing dependency 'choix'. Install via: pip install choix")
        return []

    approach_to_idx = {a: i for i, a in enumerate(approaches)}
    n = len(approaches)
    by_item = defaultdict(list)
    for r in results:
        by_item[r.item_key].append(r)

    per_instance_rows = []
    rank_acc = defaultdict(list)

    for item_key, item_results in sorted(by_item.items()):
        comparisons = []
        for r in item_results:
            li = approach_to_idx[r.approach_left]
            ri = approach_to_idx[r.approach_right]
            if r.final_winner == "left":
                comparisons.append((li, ri))
            elif r.final_winner == "right":
                comparisons.append((ri, li))
            else:
                # Tie: add symmetric pseudo-results so tie contributes equally.
                comparisons.append((li, ri))
                comparisons.append((ri, li))

        if not comparisons:
            continue

        try:
            params = choix.ilsr_pairwise(n_items=n, data=comparisons, alpha=0.01)
        except Exception as exc:
            print(f"Bradley-Terry failed for item {item_key}: {exc}")
            continue

        order = sorted(range(n), key=lambda i: params[i], reverse=True)
        ranks = {}
        for pos, idx in enumerate(order, start=1):
            ranks[approaches[idx]] = pos
            rank_acc[approaches[idx]].append(pos)

        row = {"item_key": item_key}
        for a in approaches:
            row[f"score__{a}"] = float(params[approach_to_idx[a]])
            row[f"rank__{a}"] = int(ranks[a])
        per_instance_rows.append(row)

    if not per_instance_rows:
        return []

    if per_instance_filename:
        with open(output_dir / per_instance_filename, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["item_key"] + [f"score__{a}" for a in approaches] + [f"rank__{a}" for a in approaches]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_instance_rows)

    avg_rows = []
    leaderboard_by_approach = leaderboard_by_approach or {}
    for a in approaches:
        ranks = rank_acc.get(a, [])
        if not ranks:
            continue
        avg_rank = sum(ranks) / len(ranks)
        lb = leaderboard_by_approach.get(a, {})
        row = {
            "group_label": group_label,
            "noise_value": noise_value,
            "approach": a,
            "avg_rank": avg_rank,
            "instances": len(ranks),
            "wins": int(lb.get("wins", 0)),
            "losses": int(lb.get("losses", 0)),
            "ties": int(lb.get("ties", 0)),
            "decisions": int(lb.get("decisions", 0)),
            "win_rate_excl_ties": float(lb.get("win_rate_excl_ties", 0.0)),
            "tie_rate": float(lb.get("tie_rate", 0.0)),
        }
        avg_rows.append(row)

    avg_rows.sort(key=lambda x: x["avg_rank"])
    if per_instance_filename:
        print(f"Bradley-Terry per-instance ranking ({group_label}): {output_dir / per_instance_filename}")
    return avg_rows


def _build_leaderboard_by_approach(results: List["PairItemResult"]) -> Dict[str, dict]:
    board = defaultdict(lambda: {
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "decisions": 0,
    })

    for r in results:
        left = r.approach_left
        right = r.approach_right
        board[left]["decisions"] += 1
        board[right]["decisions"] += 1
        if r.final_winner == "left":
            board[left]["wins"] += 1
            board[right]["losses"] += 1
        elif r.final_winner == "right":
            board[right]["wins"] += 1
            board[left]["losses"] += 1
        else:
            board[left]["ties"] += 1
            board[right]["ties"] += 1

    out = {}
    for approach, stats in sorted(board.items()):
        non_ties = max(stats["wins"] + stats["losses"], 1)
        decisions = max(stats["decisions"], 1)
        out[approach] = {
            "wins": stats["wins"],
            "losses": stats["losses"],
            "ties": stats["ties"],
            "decisions": stats["decisions"],
            "win_rate_excl_ties": stats["wins"] / non_ties,
            "tie_rate": stats["ties"] / decisions,
        }
    return out


_NOISE_SUFFIX_MAP = {
    "backtranslate": "bt",
    "random_sentence_swap": "swa",
    "delete_one_sentence": "del",
}


def _sanitize_suffix(text: str) -> str:
    val = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    return val or "empty"


def _group_suffix_from_row(row: dict) -> str:
    group_label = str(row.get("group_label", ""))
    noise_value = str(row.get("noise_value", ""))

    if group_label == "overall":
        return "overall"
    if group_label == "corrupted_only":
        return "corrupted"
    if noise_value in _NOISE_SUFFIX_MAP:
        return _NOISE_SUFFIX_MAP[noise_value]
    return _sanitize_suffix(noise_value)


def _group_key(row: dict) -> Tuple[str, str]:
    return (str(row.get("group_label", "")), str(row.get("noise_value", "")))


def _to_wide_rank_rows(grouped_rows: List[dict], approaches: List[str]) -> Tuple[List[dict], List[str]]:
    """Pivot grouped Bradley-Terry rows to one row per approach with metric_* columns."""
    if not grouped_rows:
        return [], []

    metric_names = [
        "avg_rank",
        "instances",
        "wins",
        "losses",
        "ties",
        "decisions",
        "win_rate_excl_ties",
        "tie_rate",
    ]

    by_group = defaultdict(list)
    for row in grouped_rows:
        by_group[_group_key(row)].append(row)

    key_to_suffix = {}
    used_suffixes = set()
    for group_key in sorted(by_group.keys()):
        sample_row = by_group[group_key][0]
        base = _group_suffix_from_row(sample_row)
        suffix = base
        i = 2
        while suffix in used_suffixes:
            suffix = f"{base}_{i}"
            i += 1
        key_to_suffix[group_key] = suffix
        used_suffixes.add(suffix)

    preferred_keys = [
        ("overall", "all"),
        ("corrupted_only", "non_original"),
        ("noise:original", "original"),
    ]
    ordered_keys = [k for k in preferred_keys if k in by_group]
    ordered_keys.extend(sorted(k for k in by_group.keys() if k not in ordered_keys))
    suffixes = [key_to_suffix[k] for k in ordered_keys]

    wide_by_approach = {a: {"approach": a} for a in approaches}

    for group_key in ordered_keys:
        suffix = key_to_suffix[group_key]
        rows = sorted(by_group[group_key], key=lambda x: (x["avg_rank"], x["approach"]))
        for row in rows:
            approach = row["approach"]
            if approach not in wide_by_approach:
                wide_by_approach[approach] = {"approach": approach}
            out = wide_by_approach[approach]
            for metric in metric_names:
                out[f"{metric}_{suffix}"] = row.get(metric)

    wide_rows = [wide_by_approach[a] for a in sorted(wide_by_approach.keys())]

    fieldnames = ["approach"]
    for metric in metric_names:
        fieldnames.extend([f"{metric}_{suffix}" for suffix in suffixes])

    return wide_rows, fieldnames


def _write_bradley_terry_outputs(results: List["PairItemResult"], approaches: List[str], output_dir: Path) -> None:
    """Write Bradley-Terry outputs (overall, corrupted-only, and per-noise groups)."""
    combined_rank_rows = []

    overall_lb = _build_leaderboard_by_approach(results)
    overall_rows = _compute_bradley_terry_rankings(
        results=results,
        approaches=approaches,
        output_dir=output_dir,
        leaderboard_by_approach=overall_lb,
        group_label="overall",
        noise_value="all",
        per_instance_filename="bradley_terry_per_instance.csv",
    )
    combined_rank_rows.extend(overall_rows)

    corrupted_results = [r for r in results if str(r.noise or "") != "original"]
    if corrupted_results:
        corrupted_lb = _build_leaderboard_by_approach(corrupted_results)
        corrupted_rows = _compute_bradley_terry_rankings(
            results=corrupted_results,
            approaches=approaches,
            output_dir=output_dir,
            leaderboard_by_approach=corrupted_lb,
            group_label="corrupted_only",
            noise_value="non_original",
            per_instance_filename="",
        )
        combined_rank_rows.extend(corrupted_rows)
    else:
        print("Bradley-Terry skipped for corrupted_only: no items with noise != original")

    noise_values = sorted({str(r.noise or "") for r in results})
    if "original" in noise_values:
        noise_values = ["original"] + [n for n in noise_values if n != "original"]

    for noise_value in noise_values:
        noise_results = [r for r in results if str(r.noise or "") == noise_value]
        if not noise_results:
            continue
        noise_lb = _build_leaderboard_by_approach(noise_results)
        noise_rows = _compute_bradley_terry_rankings(
            results=noise_results,
            approaches=approaches,
            output_dir=output_dir,
            leaderboard_by_approach=noise_lb,
            group_label=f"noise:{noise_value if noise_value else 'empty'}",
            noise_value=noise_value if noise_value else "empty",
            per_instance_filename="",
        )
        if noise_rows:
            combined_rank_rows.extend(noise_rows)
        else:
            label = noise_value if noise_value else "<empty>"
            print(f"Bradley-Terry skipped for noise group '{label}': no valid comparisons")

    if combined_rank_rows:
        wide_rows, fieldnames = _to_wide_rank_rows(combined_rank_rows, approaches)
        with open(output_dir / "bradley_terry_average_rank.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(wide_rows)
        print(f"Bradley-Terry average rank (wide): {output_dir / 'bradley_terry_average_rank.csv'}")
    else:
        print("Bradley-Terry average rank skipped: no valid grouped results")


def _parse_csv_arg(value: str) -> List[str]:
    return [s.strip() for s in (value or "").split(",") if s.strip()]


def _canonical_winner_label(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in {"left", "a", "feedback_a"}:
        return "left"
    if v in {"right", "b", "feedback_b"}:
        return "right"
    return "tie"


def _majority_vote_from_saved(args) -> None:
    src_paths = [Path(p).resolve() for p in _parse_csv_arg(args.majority_vote_jsonls)]
    if len(src_paths) < 2:
        raise ValueError("Need at least two source JSONL files for majority voting.")

    for src in src_paths:
        if not src.exists():
            raise FileNotFoundError(f"majority-vote source JSONL not found: {src}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else src_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    source_approaches = {}
    grouped = defaultdict(list)
    for src in src_paths:
        rows = _read_jsonl(src)
        approaches_in_src = set()
        for row in rows:
            left = str(row.get("approach_left", ""))
            right = str(row.get("approach_right", ""))
            if left:
                approaches_in_src.add(left)
            if right:
                approaches_in_src.add(right)

            key = (
                str(row.get("pair_key", "")),
                left,
                right,
                str(row.get("item_key", "")),
            )
            grouped[key].append((src.name, row))
        source_approaches[str(src)] = approaches_in_src

    if not source_approaches:
        raise ValueError("No source approaches found in provided JSONL files.")

    common_approaches = set.intersection(*source_approaches.values())
    common_approaches = {a for a in common_approaches if a}
    if len(common_approaches) < 2:
        details = {k: sorted(v) for k, v in source_approaches.items()}
        raise ValueError(
            "Need at least two approaches common to all sources. "
            f"Common approaches: {sorted(common_approaches)} | Per-source approaches: {details}"
        )

    merged_rows = []
    for key, entries in sorted(grouped.items()):
        pair_key, left, right, item_key = key
        if left not in common_approaches or right not in common_approaches:
            continue

        votes = []
        vote_conf = []
        for src_name, row in entries:
            label = _canonical_winner_label(row.get("final_winner", "tie"))
            votes.append(label)
            conf = row.get("final_confidence", 0.0)
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 0.0
            vote_conf.append((label, max(0.0, min(1.0, conf)), src_name))

        counts = Counter(votes)
        max_count = max(counts.values()) if counts else 0
        top = [lbl for lbl, c in counts.items() if c == max_count]
        final_winner = top[0] if len(top) == 1 else "tie"

        supporter_conf = [c for lbl, c, _ in vote_conf if lbl == final_winner]
        if supporter_conf:
            final_conf = sum(supporter_conf) / len(supporter_conf)
        elif vote_conf:
            final_conf = sum(c for _, c, _ in vote_conf) / len(vote_conf)
        else:
            final_conf = 0.0

        exemplar = entries[0][1]
        merged_row = {
            "pair_key": pair_key,
            "approach_left": left,
            "approach_right": right,
            "item_key": item_key,
            "story": exemplar.get("story", ""),
            "gold_story": exemplar.get("gold_story", ""),
            "noise": exemplar.get("noise", ""),
            "prompt_type": exemplar.get("prompt_type", ""),
            "final_winner": final_winner,
            "final_confidence": final_conf,
            "vote_counts": {
                "left": counts.get("left", 0),
                "right": counts.get("right", 0),
                "tie": counts.get("tie", 0),
            },
            "order_consistency": 1.0,
            "run_consistency": (max_count / len(votes)) if votes else 0.0,
            "majority_vote_sources": [src_name for src_name, _ in entries],
            "majority_vote_num_sources": len(entries),
        }
        merged_rows.append(merged_row)

    if not merged_rows:
        raise ValueError("No merged rows were produced from the provided JSONL files.")

    merged_jsonl_path = output_dir / args.majority_vote_output_name
    with open(merged_jsonl_path, "w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    approaches = sorted(common_approaches)
    if len(approaches) < 2:
        raise ValueError("Need at least two approaches after merge to compute Bradley-Terry rankings.")

    results = []
    for row in merged_rows:
        results.append(PairItemResult(
            pair_key=str(row.get("pair_key", "")),
            approach_left=str(row.get("approach_left", "")),
            approach_right=str(row.get("approach_right", "")),
            item_key=str(row.get("item_key", "")),
            story=str(row.get("story", "")),
            gold_story=str(row.get("gold_story", "")),
            noise=str(row.get("noise", "")),
            prompt_type=str(row.get("prompt_type", "")),
            final_winner=_canonical_winner_label(row.get("final_winner", "tie")),
            final_confidence=float(row.get("final_confidence", 0.0) or 0.0),
            vote_counts=row.get("vote_counts", {}),
            order_consistency=float(row.get("order_consistency", 1.0) or 1.0),
            run_consistency=float(row.get("run_consistency", 1.0) or 1.0),
            ballots=[],
        ))

    _write_bradley_terry_outputs(results=results, approaches=approaches, output_dir=output_dir)

    with open(output_dir / "majority_vote_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_jsonls": [str(p) for p in src_paths],
                "merged_jsonl": str(merged_jsonl_path),
                "source_approaches": {k: sorted(v) for k, v in source_approaches.items()},
                "common_approaches": sorted(common_approaches),
                "approaches": approaches,
                "merged_rows": len(merged_rows),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Majority-vote merged JSONL: {merged_jsonl_path}")
    print(f"Merged rows: {len(merged_rows)}")
    print(f"Common approaches kept: {len(common_approaches)}")
    print(f"Approaches: {len(approaches)}")
    print(f"Output directory: {output_dir}")


def _recompute_bradley_terry_from_saved(args) -> None:
    src = Path(args.recompute_bt_from_jsonl).resolve()
    if not src.exists():
        raise FileNotFoundError(f"recompute source JSONL not found: {src}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else src.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded = set(_parse_csv_arg(args.exclude_approaches))
    rows = _read_jsonl(src)
    filtered_rows = []
    for row in rows:
        left = str(row.get("approach_left", ""))
        right = str(row.get("approach_right", ""))
        if left in excluded or right in excluded:
            continue
        filtered_rows.append(row)

    if not filtered_rows:
        raise ValueError("No rows left after filtering. Check --exclude_approaches and source JSONL.")

    approaches = sorted({str(r.get("approach_left", "")) for r in filtered_rows} | {str(r.get("approach_right", "")) for r in filtered_rows})
    approaches = [a for a in approaches if a]
    if len(approaches) < 2:
        raise ValueError("Need at least two approaches after filtering to compute Bradley-Terry rankings.")

    results = []
    for row in filtered_rows:
        results.append(PairItemResult(
            pair_key=str(row.get("pair_key", "")),
            approach_left=str(row.get("approach_left", "")),
            approach_right=str(row.get("approach_right", "")),
            item_key=str(row.get("item_key", "")),
            story=str(row.get("story", "")),
            gold_story=str(row.get("gold_story", "")),
            noise=str(row.get("noise", "")),
            prompt_type=str(row.get("prompt_type", "")),
            final_winner=str(row.get("final_winner", "tie")),
            final_confidence=float(row.get("final_confidence", 0.0) or 0.0),
            vote_counts=row.get("vote_counts", {}),
            order_consistency=float(row.get("order_consistency", 1.0) or 1.0),
            run_consistency=float(row.get("run_consistency", 1.0) or 1.0),
            ballots=[],
        ))

    _write_bradley_terry_outputs(results=results, approaches=approaches, output_dir=output_dir)
    print(f"Recomputed Bradley-Terry from: {src}")
    print(f"Excluded approaches: {sorted(excluded) if excluded else 'none'}")
    print(f"Output directory: {output_dir}")


def _read_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
    return records


def _stable_item_key(record: dict) -> str:
    story = (record.get("story") or "").strip()
    gold_story = (record.get("gold_story") or "").strip()
    noise = str(record.get("noise") or "")
    key_payload = {
        "story": story,
        "gold_story": gold_story,
        "noise": noise,
    }
    encoded = json.dumps(key_payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(encoded.encode("utf-8")).hexdigest()
    return digest


def _approach_name_from_path(path: Path) -> str:
    name = path.stem
    if name.startswith("eval_stories_test_"):
        return name[len("eval_stories_test_"):]
    return name


def _extract_candidate_text(response) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    first = choices[0]
    message = getattr(first, "message", None)
    if not message:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                maybe = item.get("text")
                if maybe:
                    parts.append(maybe)
        return "\n".join(parts)
    return str(content)


def _normalize_judge_json(payload: dict) -> dict:
    """Normalize instruction-based judge response."""
    winner = str(payload.get("winner", "Tie")).strip()
    if winner.lower() in {"a", "feedback_a", "left"}:
        winner = "A"
    elif winner.lower() in {"b", "feedback_b", "right"}:
        winner = "B"
    else:
        winner = "Tie"

    confidence = payload.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    rationale = str(payload.get("rationale", "")).strip()

    return {
        "winner": winner,
        "confidence": confidence,
        "rationale": rationale,
    }


def _parse_json_from_response(text: str) -> dict:
    """Parse instruction-based judge response JSON (no dimensions)."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty model response")

    try:
        obj = json.loads(cleaned)
        if not isinstance(obj, dict):
            raise ValueError("Top-level JSON is not an object")
        return _normalize_judge_json(obj)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")

    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Extracted JSON is not an object")
    return _normalize_judge_json(obj)


def _build_instruction_prompt(instruction_cfg: dict, context: dict, feedback_a: str, feedback_b: str) -> str:
    """Build instruction-based judge prompt with context-aware variants.
    
    Case 1: noise == "original" -> simple prompt
    Case 2: noise != "original" -> augment with original story context
    """
    assume_all_original = bool(instruction_cfg.get("assume_all_original", False))
    noise = "original" if assume_all_original else context.get("noise", "original")
    story = context.get("story", "")
    gold_story = context.get("gold_story", "")
    
    schema = {
        "winner": "A|B|Tie",
        "confidence": "number in [0,1]",
        "rationale": "short justification",
    }
    
    if noise == "original":
        instruction = instruction_cfg["original_instruction"]
        # Case 1: Original story - simpler context
        prompt = (
            f"{instruction}\n\n"
            f"Story:\n{story}\n\n"
            f"Feedback A:\n{feedback_a}\n\n"
            f"Feedback B:\n{feedback_b}\n\n"
            "Return ONLY one JSON object and no extra text.\n"
            f"JSON schema: {json.dumps(schema, ensure_ascii=False)}"
        )
    else:
        instruction = instruction_cfg["corruption_instruction"]
        # Case 2: Corrupted story - include original as context
        prompt = (
            f"{instruction}\n\n"
            "Original story (uncorrupted reference):\n"
            f"{gold_story}\n\n"
            "Corrupted story (the story the feedback is generated for):\n"
            f"{story}\n\n"
            f"[This story is a corruption of the original with corruption type \"{noise}\"]\n\n"
            f"Feedback A:\n{feedback_a}\n\n"
            f"Feedback B:\n{feedback_b}\n\n"
            "Return ONLY one JSON object and no extra text.\n"
            f"JSON schema: {json.dumps(schema, ensure_ascii=False)}"
        )
    
    return prompt


async def _judge_once(
    client,
    model: str,
    prompt: str,
    timeout_s: float,
    max_retries: int,
) -> Tuple[dict, str]:
    global _api_call_count, _api_logs
    last_error = None
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a strict evaluation judge. Return only valid JSON.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                ),
                timeout=timeout_s,
            )
            text = _extract_candidate_text(response)
            parsed = _parse_json_from_response(text)
            
            # Log first 10 API calls with inputs and outputs
            _api_call_count += 1
            if _api_call_count <= 10:
                log_entry = {
                    "call_number": _api_call_count,
                    "attempt": attempt,
                    "input_prompt_preview": prompt,#[:500] + ("..." if len(prompt) > 500 else ""),
                    "output_raw_preview": text,#[:500] + ("..." if len(text) > 500 else ""),
                    "output_parsed": parsed,
                }
                _api_logs.append(log_entry)
            
            return parsed, text
        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"Judge call failed after {max_retries} attempts: {last_error}") from last_error


@dataclass
class PairItemResult:
    pair_key: str
    approach_left: str
    approach_right: str
    item_key: str
    story: str
    gold_story: str
    noise: str
    prompt_type: str
    final_winner: str
    final_confidence: float
    vote_counts: dict
    order_consistency: float
    run_consistency: float
    ballots: List[dict]


def _resolve_api_config(args) -> dict:
    api_key = args.api_key or os.getenv("###") or os.getenv("###")
    if not api_key:
        raise EnvironmentError(
            "Missing API key. Set ### or ###, or pass --api_key."
        )

    return {
        "base_url": args.base_url,
        "api_key": api_key,
    }


def _collect_input_files(input_dir: Path, include_glob: str, selected: List[str]) -> List[Path]:
    files = sorted(input_dir.glob(include_glob))
    if selected:
        selected_set = set(selected)
        files = [p for p in files if _approach_name_from_path(p) in selected_set]

    if len(files) < 2:
        raise ValueError("Need at least two approach files for pairwise judging.")
    return files


def _build_aligned_dataset(files: List[Path]) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, dict]]:
    by_approach: Dict[str, Dict[str, dict]] = {}
    context_by_item: Dict[str, dict] = {}

    for path in files:
        approach = _approach_name_from_path(path)
        rows = _read_jsonl(path)
        bucket = {}
        for row in rows:
            key = _stable_item_key(row)
            bucket[key] = row
            if key not in context_by_item:
                context_by_item[key] = {
                    "story": row.get("story", ""),
                    "gold_story": row.get("gold_story", ""),
                    "noise": row.get("noise", ""),
                    "prompt_type": (row.get("metadata") or {}).get("prompt_type", ""),
                }
        by_approach[approach] = bucket

    item_keys = None
    for approach_map in by_approach.values():
        keys = set(approach_map.keys())
        item_keys = keys if item_keys is None else (item_keys & keys)

    if not item_keys:
        raise ValueError("No aligned items across the selected approach files.")

    sorted_keys = sorted(item_keys)
    by_approach = {
        approach: {k: rows[k] for k in sorted_keys}
        for approach, rows in by_approach.items()
    }
    context_by_item = {k: context_by_item[k] for k in sorted_keys}
    return by_approach, context_by_item


async def _evaluate_pair_item(
    client,
    model: str,
    timeout_s: float,
    max_retries: int,
    instruction_cfg: dict,
    context: dict,
    feedback_left: str,
    feedback_right: str,
    runs: int,
    swap_orders: bool,
) -> Tuple[str, float, dict, float, float, List[dict]]:
    ballots = []
    order_consistency_checks = []

    for run_idx in range(runs):
        order_flags = [False, True] if swap_orders else [False]
        run_ballots = []

        for swapped in order_flags:
            fa = feedback_right if swapped else feedback_left
            fb = feedback_left if swapped else feedback_right
            prompt = _build_instruction_prompt(instruction_cfg, context, fa, fb)
            parsed, raw_text = await _judge_once(
                client=client,
                model=model,
                prompt=prompt,
                timeout_s=timeout_s,
                max_retries=max_retries,
            )

            presented_winner = parsed["winner"]
            if presented_winner == "Tie":
                canonical_winner = "tie"
            elif not swapped and presented_winner == "A":
                canonical_winner = "left"
            elif not swapped and presented_winner == "B":
                canonical_winner = "right"
            elif swapped and presented_winner == "A":
                canonical_winner = "right"
            else:
                canonical_winner = "left"

            ballot = {
                "run": run_idx,
                "swapped": swapped,
                "presented_winner": presented_winner,
                "winner": canonical_winner,
                "confidence": parsed["confidence"],
                "rationale": parsed["rationale"],
                "raw_response": raw_text,
            }
            ballots.append(ballot)
            run_ballots.append(ballot)

        if len(run_ballots) == 2:
            order_consistency_checks.append(1.0 if run_ballots[0]["winner"] == run_ballots[1]["winner"] else 0.0)

    winner_counts = Counter(b["winner"] for b in ballots)
    vote_counts = {
        "left": winner_counts.get("left", 0),
        "right": winner_counts.get("right", 0),
        "tie": winner_counts.get("tie", 0),
    }

    top_count = max(vote_counts.values())
    top_labels = [k for k, v in vote_counts.items() if v == top_count]
    final_winner = top_labels[0] if len(top_labels) == 1 else "tie"

    # Confidence is averaged over ballots that support final winner; fall back to global mean.
    supporter_conf = [b["confidence"] for b in ballots if b["winner"] == final_winner]
    if supporter_conf:
        final_confidence = sum(supporter_conf) / len(supporter_conf)
    else:
        final_confidence = (sum(b["confidence"] for b in ballots) / len(ballots)) if ballots else 0.0

    run_consistency = (top_count / len(ballots)) if ballots else 0.0

    order_consistency = (
        sum(order_consistency_checks) / len(order_consistency_checks)
        if order_consistency_checks
        else 1.0
    )

    return (
        final_winner,
        final_confidence,
        vote_counts,
        order_consistency,
        run_consistency,
        ballots,
    )



async def _run(args):
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError("openai is not installed. Install it with: pip install openai") from exc

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    selected = [s.strip() for s in args.approaches.split(",") if s.strip()] if args.approaches else []
    files = _collect_input_files(input_dir, args.include_glob, selected)
    approaches = [_approach_name_from_path(p) for p in files]
    
    instruction_cfg = _load_judge_instruction_config(
        args.judge_instruction_config,
        assume_all_original=args.assume_all_original,
    )

    by_approach, context_by_item = _build_aligned_dataset(files)
    item_keys = list(context_by_item.keys())

    rng = random.Random(args.seed)
    if args.max_items and args.max_items < len(item_keys):
        item_keys = rng.sample(item_keys, args.max_items)
        item_keys.sort()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (DEFAULT_OUTPUT_ROOT / ts)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_cfg = _resolve_api_config(args)
    client = AsyncOpenAI(**api_cfg)

    jobs = []
    for left, right in combinations(approaches, 2):
        pair_key = f"{left}__vs__{right}"
        for item_key in item_keys:
            jobs.append((pair_key, left, right, item_key))

    # Preflight job-count estimate
    num_pairs = len(list(combinations(approaches, 2)))
    num_items = len(item_keys)
    jobs_per_pair_item = args.runs * (2 if not args.no_swap_orders else 1)
    estimated_api_calls = num_pairs * num_items * jobs_per_pair_item
    print(f"\n=== Preflight Estimate ===")
    print(f"Approaches: {len(approaches)} → {num_pairs} pairs")
    print(f"Items per pair: {num_items}")
    print(f"Runs per item: {args.runs} × {'2 (with order swap)' if not args.no_swap_orders else '1 (no swap)'}")
    print(f"Estimated total API calls: {estimated_api_calls}")
    print(f"Concurrency limit: {args.concurrency}")
    print(f"Pair-item jobs to schedule: {len(jobs)}")
    print(f"Starting evaluation...\n")

    semaphore = asyncio.Semaphore(args.concurrency)
    results: List[PairItemResult] = []
    errors = []
    progress_state = {"completed": 0, "total": len(jobs)}
    progress_interval = 25 #max(1, len(jobs) // 20) if len(jobs) > 0 else 1  # Print ~20 updates

    async def run_job(pair_key: str, left: str, right: str, item_key: str):
        async with semaphore:
            context = context_by_item[item_key]
            left_row = by_approach[left][item_key]
            right_row = by_approach[right][item_key]
            try:
                (
                    final_winner,
                    final_confidence,
                    vote_counts,
                    order_consistency,
                    run_consistency,
                    ballots,
                ) = await _evaluate_pair_item(
                    client=client,
                    model=args.model,
                    timeout_s=args.timeout_s,
                    max_retries=args.max_retries,
                    instruction_cfg=instruction_cfg,
                    context=context,
                    feedback_left=left_row.get("feedback", ""),
                    feedback_right=right_row.get("feedback", ""),
                    runs=args.runs,
                    swap_orders=(not args.no_swap_orders),
                )
                progress_state["completed"] += 1
                if progress_state["completed"] % progress_interval == 0 or progress_state["completed"] == progress_state["total"]:
                    print(f"Progress: {progress_state['completed']}/{progress_state['total']} pair-item jobs completed")
                return PairItemResult(
                    pair_key=pair_key,
                    approach_left=left,
                    approach_right=right,
                    item_key=item_key,
                    story=context.get("story", ""),
                    gold_story=context.get("gold_story", ""),
                    noise=context.get("noise", ""),
                    prompt_type=context.get("prompt_type", ""),
                    final_winner=final_winner,
                    final_confidence=final_confidence,
                    vote_counts=vote_counts,
                    order_consistency=order_consistency,
                    run_consistency=run_consistency,
                    ballots=ballots,
                )
            except Exception as exc:
                print(f"Error evaluating pair {pair_key}, item {item_key}: {exc}")
                errors.append({
                    "pair_key": pair_key,
                    "approach_left": left,
                    "approach_right": right,
                    "item_key": item_key,
                    "error": str(exc),
                })
                progress_state["completed"] += 1
                if progress_state["completed"] % progress_interval == 0 or progress_state["completed"] == progress_state["total"]:
                    print(f"Progress: {progress_state['completed']}/{progress_state['total']} pair-item jobs completed (with {len(errors)} errors)")
                return None

    tasks = [run_job(*job) for job in jobs]
    done = await asyncio.gather(*tasks)
    for item in done:
        if item is not None:
            results.append(item)

    # Write detailed JSONL outputs.
    ballots_path = output_dir / "pairwise_ballots.jsonl"
    with open(ballots_path, "w", encoding="utf-8") as f:
        for result in results:
            for ballot in result.ballots:
                row = {
                    "pair_key": result.pair_key,
                    "approach_left": result.approach_left,
                    "approach_right": result.approach_right,
                    "item_key": result.item_key,
                    "noise": result.noise,
                    "prompt_type": result.prompt_type,
                    **ballot,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    items_path = output_dir / "pairwise_item_judgments.jsonl"
    with open(items_path, "w", encoding="utf-8") as f:
        for result in results:
            row = {
                "pair_key": result.pair_key,
                "approach_left": result.approach_left,
                "approach_right": result.approach_right,
                "item_key": result.item_key,
                "story": result.story,
                "gold_story": result.gold_story,
                "noise": result.noise,
                "prompt_type": result.prompt_type,
                "final_winner": result.final_winner,
                "final_confidence": result.final_confidence,
                "vote_counts": result.vote_counts,
                "order_consistency": result.order_consistency,
                "run_consistency": result.run_consistency,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if errors:
        with open(output_dir / "errors.jsonl", "w", encoding="utf-8") as f:
            for err in errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")

    # Pair-level summary.
    pair_stats = defaultdict(lambda: {
        "approach_left": "",
        "approach_right": "",
        "wins_left": 0,
        "wins_right": 0,
        "ties": 0,
        "items": 0,
        "avg_confidence": 0.0,
        "avg_order_consistency": 0.0,
        "avg_run_consistency": 0.0,
    })

    for r in results:
        ps = pair_stats[r.pair_key]
        ps["approach_left"] = r.approach_left
        ps["approach_right"] = r.approach_right
        ps["items"] += 1
        ps["avg_confidence"] += r.final_confidence
        ps["avg_order_consistency"] += r.order_consistency
        ps["avg_run_consistency"] += r.run_consistency
        if r.final_winner == "left":
            ps["wins_left"] += 1
        elif r.final_winner == "right":
            ps["wins_right"] += 1
        else:
            ps["ties"] += 1

    pair_summary_rows = []
    for pair_key, ps in sorted(pair_stats.items()):
        items = max(ps["items"], 1)
        non_ties = max(ps["wins_left"] + ps["wins_right"], 1)
        row = {
            "pair_key": pair_key,
            "approach_left": ps["approach_left"],
            "approach_right": ps["approach_right"],
            "items": ps["items"],
            "wins_left": ps["wins_left"],
            "wins_right": ps["wins_right"],
            "ties": ps["ties"],
            "win_rate_left_excl_ties": ps["wins_left"] / non_ties,
            "win_rate_right_excl_ties": ps["wins_right"] / non_ties,
            "tie_rate": ps["ties"] / items,
            "avg_confidence": ps["avg_confidence"] / items,
            "avg_order_consistency": ps["avg_order_consistency"] / items,
            "avg_run_consistency": ps["avg_run_consistency"] / items,
        }
        pair_summary_rows.append(row)

    with open(output_dir / "pairwise_summary.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "pair_key",
            "approach_left",
            "approach_right",
            "items",
            "wins_left",
            "wins_right",
            "ties",
            "win_rate_left_excl_ties",
            "win_rate_right_excl_ties",
            "tie_rate",
            "avg_confidence",
            "avg_order_consistency",
            "avg_run_consistency",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_summary_rows)

    combined_rank_rows = []

    overall_lb = _build_leaderboard_by_approach(results)
    overall_rows = _compute_bradley_terry_rankings(
        results=results,
        approaches=approaches,
        output_dir=output_dir,
        leaderboard_by_approach=overall_lb,
        group_label="overall",
        noise_value="all",
        per_instance_filename="bradley_terry_per_instance.csv",
    )
    combined_rank_rows.extend(overall_rows)

    corrupted_results = [r for r in results if str(r.noise or "") != "original"]
    if corrupted_results:
        corrupted_lb = _build_leaderboard_by_approach(corrupted_results)
        corrupted_rows = _compute_bradley_terry_rankings(
            results=corrupted_results,
            approaches=approaches,
            output_dir=output_dir,
            leaderboard_by_approach=corrupted_lb,
            group_label="corrupted_only",
            noise_value="non_original",
            per_instance_filename="",
        )
        combined_rank_rows.extend(corrupted_rows)
    else:
        print("Bradley-Terry skipped for corrupted_only: no items with noise != original")

    noise_values = sorted({str(r.noise or "") for r in results})
    if "original" in noise_values:
        noise_values = ["original"] + [n for n in noise_values if n != "original"]

    for noise_value in noise_values:
        noise_results = [r for r in results if str(r.noise or "") == noise_value]
        if not noise_results:
            continue
        noise_lb = _build_leaderboard_by_approach(noise_results)
        noise_rows = _compute_bradley_terry_rankings(
            results=noise_results,
            approaches=approaches,
            output_dir=output_dir,
            leaderboard_by_approach=noise_lb,
            group_label=f"noise:{noise_value if noise_value else 'empty'}",
            noise_value=noise_value if noise_value else "empty",
            per_instance_filename="",
        )
        if noise_rows:
            combined_rank_rows.extend(noise_rows)
        else:
            label = noise_value if noise_value else "<empty>"
            print(f"Bradley-Terry skipped for noise group '{label}': no valid comparisons")

    if combined_rank_rows:
        wide_rows, fieldnames = _to_wide_rank_rows(combined_rank_rows, approaches)
        with open(output_dir / "bradley_terry_average_rank.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(wide_rows)
        print(f"Bradley-Terry average rank (wide): {output_dir / 'bradley_terry_average_rank.csv'}")
    else:
        print("Bradley-Terry average rank skipped: no valid grouped results")

    config = {
        "model": args.model,
        "input_dir": str(input_dir),
        "input_files": [str(p) for p in files],
        "approaches": approaches,
        "judge_instruction_config": str(Path(args.judge_instruction_config).resolve()),
        "aligned_items": len(item_keys),
        "runs": args.runs,
        "swap_orders": not args.no_swap_orders,
        "concurrency": args.concurrency,
        "max_retries": args.max_retries,
        "timeout_s": args.timeout_s,
        "errors": len(errors),
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Save API call logs (first 10 calls)
    if _api_logs:
        api_logs_path = output_dir / "api_call_logs.json"
        with open(api_logs_path, "w", encoding="utf-8") as f:
            json.dump(_api_logs, f, ensure_ascii=False, indent=2)
        print(f"API call logs (first {len(_api_logs)} calls): {api_logs_path}")

    print(f"Output directory: {output_dir}")
    print(f"Approaches: {len(approaches)}")
    print(f"Aligned items evaluated per pair: {len(item_keys)}")
    print(f"Total pair-item jobs: {len(jobs)}")
    print(f"Completed results: {len(results)}")
    print(f"Errors: {len(errors)}")


def _parse_args():
    parser = argparse.ArgumentParser(description="LLM-as-a-judge pairwise feedback evaluation.")
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR), help="Directory containing eval JSONL files.")
    parser.add_argument("--include_glob", type=str, default="eval_stories_test_*.jsonl", help="Glob to select input files.")
    parser.add_argument("--approaches", type=str, default="", help="Comma-separated subset of approach names.")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory. Default: eval/judge_results/<timestamp>")
    parser.add_argument(
        "--judge_instruction_config",
        type=str,
        default=str((SCRIPT_DIR / "judge_instructions.json").resolve()),
        help="JSON config with original_instruction and optionally corruption_instruction (optional when --assume_all_original is set).",
    )
    parser.add_argument(
        "--assume_all_original",
        action="store_true",
        help="Treat all items as original when building judge prompts. If set, configs may omit corruption_instruction.",
    )

    parser.add_argument("--model", type=str, default="llama-3.3-70b-instruct", help="Judge model ID.")
    parser.add_argument("--base_url", type=str, default="https://chat-ai.cluster.uni-hannover.de/v1", help="OpenAI-compatible backend base URL.")
    parser.add_argument("--api_key", type=str, default="", help="Optional API key override.")

    parser.add_argument("--runs", type=int, default=1, help="Number of repeated judge runs per item.")
    parser.add_argument("--no_swap_orders", action="store_true", help="Disable order-swapped judging.")
    parser.add_argument("--concurrency", type=int, default=16, help="Concurrent API requests.")
    parser.add_argument("--timeout_s", type=float, default=30.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per API call.")

    parser.add_argument("--max_items", type=int, default=0, help="If >0, sample this many aligned items.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for item sampling/order randomization.")

    parser.add_argument(
        "--recompute_bt_from_jsonl",
        type=str,
        default="",
        help="Optional path to existing pairwise_item_judgments.jsonl to recompute Bradley-Terry without API calls.",
    )
    parser.add_argument(
        "--exclude_approaches",
        type=str,
        default="",
        help="Comma-separated approach names to exclude when recomputing Bradley-Terry from saved JSONL.",
    )
    parser.add_argument(
        "--majority_vote_jsonls",
        type=str,
        default="",
        help="Comma-separated list of pairwise_item_judgments.jsonl files to merge by majority voting.",
    )
    parser.add_argument(
        "--majority_vote_output_name",
        type=str,
        default="pairwise_item_judgments_majority_vote.jsonl",
        help="Filename for merged majority-vote JSONL (written under --output_dir).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.majority_vote_jsonls:
        _majority_vote_from_saved(args)
    elif args.recompute_bt_from_jsonl:
        _recompute_bradley_terry_from_saved(args)
    else:
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
