"""Shared helpers for HF → JSONL prepare scripts."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def stratified_split(
    rows: list[dict],
    *,
    label_key: str,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[dict]]:
    """Split rows into train/val/test, stratified by a string label field when possible."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = row.get(label_key)
        if isinstance(key, list):
            # multi-hot: use first active name or "multi"
            active = [str(i) for i, v in enumerate(key) if v]
            bucket = active[0] if active else "empty"
        else:
            bucket = str(key)
        by_label[bucket].append(row)

    train, val, test = [], [], []
    for bucket_rows in by_label.values():
        rng.shuffle(bucket_rows)
        n = len(bucket_rows)
        n_test = max(1, int(n * test_ratio)) if n >= 10 else max(0, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio)) if n >= 10 else max(0, int(n * val_ratio))
        if n_test + n_val >= n:
            n_test = min(n_test, max(0, n // 5))
            n_val = min(n_val, max(0, (n - n_test) // 5))
        test.extend(bucket_rows[:n_test])
        val.extend(bucket_rows[n_test : n_test + n_val])
        train.extend(bucket_rows[n_test + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def take(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def template_signature(text: str) -> str:
    """Stable key for rows built from the same generator template.

    Synthetic corpora compose ``[opener][content][time phrase][closer]``, so the
    first and last few words identify the frame while the middle varies. Used to
    keep near-duplicate phrasings inside a single split — a plain row-level split
    leaks the template into val/test and inflates every metric.
    """
    words = re.findall(r"[a-z']+", (text or "").lower())
    return " ".join(words[:5]) + "||" + " ".join(words[-4:])


def grouped_split(
    rows: list[dict],
    *,
    group_key: str = "_group",
    seed: int = 42,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> dict[str, list[dict]]:
    """Split by group so no group's rows appear in more than one split.

    Groups are assigned whole to train/val/test by hashed shuffle, which keeps
    the split deterministic across runs and independent of row order.
    """
    rng = random.Random(seed)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get(group_key))].append(row)

    groups = sorted(by_group)
    rng.shuffle(groups)
    n_val = int(len(groups) * val_ratio)
    n_test = int(len(groups) * test_ratio)

    assignment = {}
    for i, group in enumerate(groups):
        if i < n_test:
            assignment[group] = "test"
        elif i < n_test + n_val:
            assignment[group] = "val"
        else:
            assignment[group] = "train"

    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for group, group_rows in by_group.items():
        out[assignment[group]].extend(group_rows)
    for split_rows in out.values():
        rng.shuffle(split_rows)
    return out


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def balance_by_label(
    rows: list[dict],
    *,
    label_key: str,
    max_per_label: int,
    seed: int = 42,
) -> list[dict]:
    """Cap rows per label so a dominant class cannot swamp the loss."""
    if max_per_label <= 0:
        return rows
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get(label_key))].append(row)
    kept: list[dict] = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        kept.extend(label_rows[:max_per_label])
    rng.shuffle(kept)
    return kept
