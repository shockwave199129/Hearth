#!/usr/bin/env python3
"""Convert ``data/hearth_relationship_understanding.jsonl`` into all five head formats.

One source row can feed five heads, since each field maps to a different task::

    emotional_state    -> EmotionHead      (multi-hot over 28 GoEmotions labels)
    intent             -> IntentHead       (10 companion needs)
    memory_worthy/topic-> MemoryHead       (store + type + importance)
    relationship_signal-> RelationshipHead (4 weak regression targets)
    suggested_stance   -> StrategyHead     (12 strategies)

Splits are grouped by template signature (see ``common.template_signature``) so
paraphrases of one generator template never straddle train/val/test. Rows whose
emotional_state has no honest GoEmotions equivalent are dropped from the emotion
output only — they still train the other four heads.

Usage (from ``hearth_ai/``)::

    python3 -m data.prepare.prepare_hearth_synthetic
    python3 -m data.prepare.prepare_hearth_synthetic --limit 50000 --strict-emotion-map
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hearth_ai.labels import (  # noqa: E402
    INTENT_LABELS,
    MEMORY_TYPES,
    RELATIONSHIP_SIGNALS,
    STRATEGY_LABELS,
    multi_hot,
)
from data.prepare.common import DATA_DIR, grouped_split, template_signature, write_jsonl  # noqa: E402
from data.prepare.hearth_synthetic_maps import (  # noqa: E402
    INTENT_MAP,
    MEMORY_NEGATIVE_IMPORTANCE,
    MEMORY_NEGATIVE_TYPE,
    MEMORY_TYPE_BY_TOPIC,
    emotion_names,
    memory_importance,
    relationship_signals,
    strategy_label,
)

SOURCE = "hearth_relationship_synthetic"
TASKS = ("emotion", "intent", "memory", "relationship", "strategy")


def _iter_source(path: Path, limit: int | None):
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit and i >= limit:
                return
            yield json.loads(line)


def convert(
    path: Path,
    *,
    limit: int | None,
    strict_emotion_map: bool,
    drop_memory_anomalies: bool,
) -> tuple[dict[str, list[dict]], collections.Counter]:
    """Build per-task row lists (each carrying ``_group``) plus a drop report."""
    per_task: dict[str, list[dict]] = {task: [] for task in TASKS}
    stats: collections.Counter = collections.Counter()

    for row in _iter_source(path, limit):
        text = (row.get("user_message") or "").strip()
        if not text:
            stats["skipped_empty_text"] += 1
            continue

        stats["rows_read"] += 1
        group = template_signature(text)
        state = str(row.get("emotional_state") or "").strip().lower()
        raw_intent = str(row.get("intent") or "").strip().lower()
        signal = str(row.get("relationship_signal") or "").strip().lower()
        closeness = str(row.get("closeness_delta") or "").strip().lower()
        topic = str(row.get("topic") or "").strip().lower()
        stance = str(row.get("suggested_stance") or "").strip().lower()
        memory_worthy = bool(row.get("memory_worthy"))
        memory_candidate = row.get("memory_candidate")

        # --- Emotion ---
        names = emotion_names(state, strict=strict_emotion_map)
        if names:
            per_task["emotion"].append(
                {
                    "text": text,
                    "labels": multi_hot(names),
                    "source": SOURCE,
                    "_group": group,
                    "_bucket": names[0],
                }
            )
        else:
            stats[f"emotion_dropped:{state or 'missing'}"] += 1

        # --- Intent ---
        intent = INTENT_MAP.get(raw_intent)
        if intent in INTENT_LABELS:
            per_task["intent"].append(
                {
                    "text": text,
                    "intent": intent,
                    "source": SOURCE,
                    "_group": group,
                    "_bucket": intent,
                }
            )
        else:
            stats[f"intent_dropped:{raw_intent or 'missing'}"] += 1

        # --- Memory ---
        if memory_worthy and memory_candidate in (None, ""):
            stats["memory_worthy_without_candidate"] += 1
            if drop_memory_anomalies:
                stats["memory_dropped_anomaly"] += 1
                memory_row = None
            else:
                memory_row = True
        else:
            memory_row = True

        if memory_row:
            if memory_worthy:
                mem_type = MEMORY_TYPE_BY_TOPIC.get(topic)
                importance = memory_importance(signal, topic)
                if mem_type is None or importance is None:
                    stats[f"memory_dropped_unmapped:{topic}|{signal}"] += 1
                    mem_type = None
            else:
                mem_type = MEMORY_NEGATIVE_TYPE
                importance = MEMORY_NEGATIVE_IMPORTANCE

            if mem_type is not None and mem_type in MEMORY_TYPES:
                per_task["memory"].append(
                    {
                        "text": text,
                        "store": int(memory_worthy),
                        "type": mem_type,
                        "importance": round(float(importance), 3),
                        "source": SOURCE,
                        "annotator": "synthetic_generator",
                        "_group": group,
                        "_bucket": mem_type if memory_worthy else "nostore",
                    }
                )

        # --- Relationship ---
        signals = relationship_signals(signal, closeness, state)
        if signals is not None:
            per_task["relationship"].append(
                {
                    "text": text,
                    "signals": [round(x, 3) for x in signals],
                    "source": SOURCE,
                    "annotator": "synthetic_weak_label",
                    "_group": group,
                    "_bucket": signal,
                }
            )
        else:
            stats[f"relationship_dropped:{signal or 'missing'}"] += 1

        # --- Strategy ---
        strategy = strategy_label(stance, raw_intent, state)
        if strategy in STRATEGY_LABELS:
            per_task["strategy"].append(
                {
                    "text": text,
                    "strategy": strategy,
                    "source": SOURCE,
                    "annotator": "synthetic_stance_map",
                    "_group": group,
                    "_bucket": strategy,
                }
            )
        else:
            stats[f"strategy_dropped:{stance or 'missing'}"] += 1

    return per_task, stats


_KEEP_FIELDS = {
    "emotion": ("text", "labels", "source"),
    "intent": ("text", "intent", "source"),
    "memory": ("text", "store", "type", "importance", "source", "annotator"),
    "relationship": ("text", "signals", "source", "annotator"),
    "strategy": ("text", "strategy", "source", "annotator"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "hearth_relationship_understanding.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR / "synthetic",
        help="Written as {out-dir}/{task}/{train,val,test}.jsonl",
    )
    parser.add_argument("--limit", type=int, default=0, help="Read only N source rows (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument(
        "--strict-emotion-map",
        action="store_true",
        help="Also drop low-confidence emotion mappings (nostalgic, warm, ...)",
    )
    parser.add_argument(
        "--drop-memory-anomalies",
        action="store_true",
        help="Drop rows where memory_worthy=true but memory_candidate is null",
    )
    parser.add_argument("--report", type=Path, default=None, help="Write drop/label report JSON")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Source not found: {args.input}")

    print(f"Reading {args.input}")
    per_task, stats = convert(
        args.input,
        limit=args.limit or None,
        strict_emotion_map=args.strict_emotion_map,
        drop_memory_anomalies=args.drop_memory_anomalies,
    )
    print(f"  read {stats['rows_read']} source rows")

    report: dict = {"source": str(args.input), "rows_read": stats["rows_read"], "tasks": {}}

    for task in TASKS:
        rows = per_task[task]
        if not rows:
            print(f"  {task}: no rows — check maps")
            continue
        splits = grouped_split(
            rows,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        keep = _KEEP_FIELDS[task]
        task_report: dict = {"total": len(rows), "splits": {}}
        for split_name, split_rows in splits.items():
            cleaned = [{k: r[k] for k in keep if k in r} for r in split_rows]
            n = write_jsonl(args.out_dir / task / f"{split_name}.jsonl", cleaned)
            task_report["splits"][split_name] = n
        label_field = {
            "emotion": "_bucket",
            "intent": "intent",
            "memory": "_bucket",
            "relationship": "_bucket",
            "strategy": "strategy",
        }[task]
        task_report["label_counts"] = dict(
            collections.Counter(str(r.get(label_field)) for r in rows).most_common()
        )
        report["tasks"][task] = task_report
        counts = task_report["splits"]
        print(
            f"  {task}: {len(rows)} rows -> train {counts['train']}, "
            f"val {counts['val']}, test {counts['test']}"
        )

    drops = {k: v for k, v in stats.items() if k != "rows_read"}
    report["drops"] = dict(sorted(drops.items(), key=lambda kv: -kv[1]))
    if drops:
        print("\nDropped / flagged (top 10):")
        for key, count in list(report["drops"].items())[:10]:
            print(f"  {key}: {count}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nReport -> {args.report}")

    assert RELATIONSHIP_SIGNALS == ["trust_delta", "vulnerability", "openness", "comfort"]


if __name__ == "__main__":
    main()
