"""Shared helpers for HF → JSONL prepare scripts."""

from __future__ import annotations

import json
import random
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
