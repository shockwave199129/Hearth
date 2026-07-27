#!/usr/bin/env python3
"""Download HF datasets and write Hearth IntentHead JSONL.

Output rows::
    {"text": "...", "intent": "comfort", "source": "empathetic_dialogues_v2"}

``intent`` is one of ``hearth_ai.labels.INTENT_LABELS`` (10 classes).

Sources:
  - Adapting/empathetic_dialogues_v2 (primary: behavior + question flag)
  - li2017dailydialog/daily_dialog (dialog acts)
  - tanaos/synthetic-intent-classifier-dataset-v1 (meta / small_talk)

Empty ``behavior`` rows: drop, unless the paired emotion is strongly negative
and there is no question — then label ``vent`` (documented heuristic).

Usage (from ``hearth_ai/``)::

    python3 -m data.prepare.prepare_intent
    python3 -m data.prepare.prepare_intent --max-per-source 500
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hearth_ai.labels import INTENT_LABELS  # noqa: E402
from data.prepare.common import DATA_DIR, stratified_split, write_jsonl  # noqa: E402

_BEHAVIOR_MAP = {
    "i'm in a negative mood, please comfort me.": "comfort",
    "i'm in a positive mood, please congratulate me and praise me.": "celebrate",
    "please give me some advices.": "advise",
}

_NEGATIVE_EMOTIONS = {
    "angry",
    "annoyed",
    "afraid",
    "terrified",
    "anxious",
    "apprehensive",
    "sad",
    "lonely",
    "guilty",
    "ashamed",
    "embarrassed",
    "disgusted",
    "furious",
    "devastated",
    "disappointed",
    "jealous",
}

_DAILY_ACT = {
    "inform": "vent",
    "question": "inquire",
    "directive": "advise",
    "commissive": "plan",
}

_TANAOS = [
    "greeting",
    "farewell",
    "thank_you",
    "affirmation",
    "negation",
    "small_talk",
    "bot_capabilities",
    "feedback_positive",
    "feedback_negative",
    "clarification",
    "suggestion",
    "language_change",
]
_TANAOS_TO_INTENT = {
    "greeting": "meta",
    "farewell": "meta",
    "thank_you": "meta",
    "affirmation": "meta",
    "negation": "meta",
    "small_talk": "small_talk",
    "bot_capabilities": "meta",
    "feedback_positive": "meta",
    "feedback_negative": "meta",
    "clarification": "inquire",
    "suggestion": "advise",
    "language_change": "meta",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _last_user_utterance(chat_history) -> str:
    if chat_history is None:
        return ""
    if isinstance(chat_history, str):
        try:
            chat_history = ast.literal_eval(chat_history)
        except (ValueError, SyntaxError):
            return chat_history.strip()
    if not isinstance(chat_history, (list, tuple)) or not chat_history:
        return ""
    # History alternates; last element is usually the latest user turn in this dataset
    return str(chat_history[-1]).strip()


def _map_empathetic_row(ex: dict) -> dict | None:
    behavior_raw = ex.get("behavior")
    question_raw = ex.get("question or not")
    emotion = str(ex.get("emotion") or "").strip().lower()

    behavior = None if behavior_raw in (None, "", "[None]", "None") else str(behavior_raw)
    question = None if question_raw in (None, "", "[None]", "None") else str(question_raw)

    intent = None
    if question and "ask me for further details" in _norm(question):
        intent = "inquire"
    if behavior:
        mapped = _BEHAVIOR_MAP.get(_norm(behavior))
        if mapped:
            intent = mapped

    if intent is None:
        # Heuristic: strong negative emotion + no question → vent
        if emotion in _NEGATIVE_EMOTIONS and not question:
            intent = "vent"
        else:
            return None

    text = _last_user_utterance(ex.get("chat_history")) or str(ex.get("situation") or "").strip()
    if not text:
        return None
    if intent not in INTENT_LABELS:
        return None
    return {"text": text, "intent": intent, "source": "empathetic_dialogues_v2"}


def load_empathetic_v2(max_per_source: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Adapting/empathetic_dialogues_v2")
    rows: list[dict] = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        for ex in ds[split]:
            row = _map_empathetic_row(ex)
            if not row:
                continue
            row["split_hint"] = (
                "train" if split == "train" else ("val" if split == "validation" else "test")
            )
            rows.append(row)
            if max_per_source and len(rows) >= max_per_source:
                return rows
    return rows


def load_daily_dialog(max_per_source: int | None) -> list[dict]:
    from datasets import load_dataset

    # Scripted daily_dialog is blocked on modern `datasets`; use Hub parquet convert.
    data_files = {
        "train": "hf://datasets/li2017dailydialog/daily_dialog@~parquet/default/train/*.parquet",
        "validation": "hf://datasets/li2017dailydialog/daily_dialog@~parquet/default/validation/*.parquet",
        "test": "hf://datasets/li2017dailydialog/daily_dialog@~parquet/default/test/*.parquet",
    }
    try:
        ds = load_dataset("parquet", data_files=data_files)
    except Exception:
        # Fallback: train-only parquet if split globs fail
        ds = load_dataset(
            "parquet",
            data_files={
                "train": "hf://datasets/li2017dailydialog/daily_dialog@~parquet/default/train/0000.parquet"
            },
        )

    act_names = {1: "inform", 2: "question", 3: "directive", 4: "commissive"}
    rows: list[dict] = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        for ex in ds[split]:
            dialog = ex.get("dialog") or []
            acts = ex.get("act") or []
            for turn, act in zip(dialog, acts):
                act_id = int(act)
                act_name = act_names.get(act_id)
                if not act_name:
                    continue
                intent = _DAILY_ACT[act_name]
                text = str(turn).strip()
                if not text:
                    continue
                rows.append(
                    {
                        "text": text,
                        "intent": intent,
                        "source": "daily_dialog",
                        "split_hint": (
                            "train"
                            if split == "train"
                            else ("val" if split == "validation" else "test")
                        ),
                    }
                )
                if max_per_source and len(rows) >= max_per_source:
                    return rows
    return rows


def load_tanaos(max_per_source: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("tanaos/synthetic-intent-classifier-dataset-v1")
    rows: list[dict] = []
    for split in ds:
        for ex in ds[split]:
            label_id = int(ex["labels"])
            if label_id < 0 or label_id >= len(_TANAOS):
                continue
            tanaos_name = _TANAOS[label_id]
            intent = _TANAOS_TO_INTENT[tanaos_name]
            text = str(ex.get("text") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "intent": intent,
                    "source": "tanaos",
                    "split_hint": "train",
                }
            )
            if max_per_source and len(rows) >= max_per_source:
                return rows
    return rows


def _preserve_hint_splits(rows: list[dict]) -> dict[str, list[dict]] | None:
    hinted = [r for r in rows if r.get("split_hint") in {"train", "val", "test"}]
    if len(hinted) < 0.9 * len(rows):
        return None
    out = {"train": [], "val": [], "test": []}
    for r in hinted:
        clean = {k: v for k, v in r.items() if k != "split_hint"}
        out[r["split_hint"]].append(clean)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "intent")
    parser.add_argument("--max-per-source", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["empathetic_dialogues_v2", "daily_dialog", "tanaos"],
        choices=["empathetic_dialogues_v2", "daily_dialog", "tanaos"],
    )
    args = parser.parse_args()
    limit = args.max_per_source or None

    loaders = {
        "empathetic_dialogues_v2": load_empathetic_v2,
        "daily_dialog": load_daily_dialog,
        "tanaos": load_tanaos,
    }

    all_rows: list[dict] = []
    for name in args.sources:
        print(f"Loading {name}...")
        part = loaders[name](limit)
        print(f"  -> {len(part)} rows")
        all_rows.extend(part)

    splits = _preserve_hint_splits(all_rows)
    if splits is None or not splits.get("val") or not splits.get("test"):
        pooled = []
        for r in all_rows:
            pooled.append({k: v for k, v in r.items() if k != "split_hint"})
        splits = stratified_split(pooled, label_key="intent", seed=args.seed)

    for split_name, rows in splits.items():
        kept = []
        for row in rows:
            if row.get("intent") not in INTENT_LABELS:
                continue
            if not row.get("text"):
                continue
            kept.append(
                {
                    "text": row["text"],
                    "intent": row["intent"],
                    "source": row.get("source", "unknown"),
                }
            )
        n = write_jsonl(args.out_dir / f"{split_name}.jsonl", kept)
        print(f"Wrote {n} -> {args.out_dir / f'{split_name}.jsonl'}")

    print(f"Intent classes ({len(INTENT_LABELS)}): {INTENT_LABELS}")


if __name__ == "__main__":
    main()
