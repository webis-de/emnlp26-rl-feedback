"""Shared JSONL read/write helpers used across data and evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def load_jsonl(path: str | Path, *, encoding: str = "utf-8") -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding=encoding) as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
) -> None:
    """Write dictionaries to a JSONL file (one JSON object per line)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding=encoding) as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=ensure_ascii))
            file.write("\n")
