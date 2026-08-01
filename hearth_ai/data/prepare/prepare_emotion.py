#!/usr/bin/env python3
"""Download HF emotion datasets and write Hearth EmotionHead JSONL.

Output rows::
    {"text": "...", "labels": [0,1,0,...], "source": "go_emotions"}

``labels`` is a length-28 multi-hot over ``hearth_ai.labels.EMOTION_LABELS``
(GoEmotions 27 + neutral).

Sources:
  - google-research-datasets/go_emotions (simplified)
  - dair-ai/emotion
  - bdotloh/empathetic-dialogues-contexts

Usage (from ``hearth_ai/``)::

    python3 -m data.prepare.prepare_emotion
    python3 -m data.prepare.prepare_emotion --max-per-source 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python3 data/prepare/prepare_emotion.py`` from hearth_ai/
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hearth_ai.labels import (  # noqa: E402
    EMOTION_GO_TO_HEARTH,
    EMOTION_LABELS,
    multi_hot,
)
from data.prepare.common import DATA_DIR, stratified_split, take, write_jsonl  # noqa: E402

# Empathetic Dialogues emotion → Hearth display (from labels/emotion.yaml)
_EMPATHETIC_TO_HEARTH = {
    "joyful": "joy",
    "sad": "sadness",
    "angry": "anger",
    "afraid": "fear",
    "terrified": "fear",
    "anxious": "anxiety",
    "apprehensive": "anxiety",
    "disgusted": "disgust",
    "surprised": "surprise",
    "excited": "joy",
    "grateful": "gratitude",
    "lonely": "loneliness",
    "guilty": "guilt",
    "ashamed": "guilt",
    "embarrassed": "guilt",
    "proud": "pride",
    "confident": "pride",
    "hopeful": "hope",
    "anticipating": "hope",
    "content": "joy",
    "sentimental": "sadness",
    "nostalgic": "sadness",
    "caring": "gratitude",
    "faithful": "gratitude",
    "trusting": "hope",
    "prepared": "hope",
    "impressed": "pride",
    "jealous": "anger",
    "furious": "anger",
    "annoyed": "anger",
    "devastated": "sadness",
    "disappointed": "sadness",
}

# dair-ai/emotion names → GoEmotions names (subset)
_DAIR_TO_GO = {
    "sadness": "sadness",
    "joy": "joy",
    "love": "love",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise",
}

# Pick one GoEmotions label that maps to each Hearth display (for weak sources)
_HEARTH_TO_GO: dict[str, str] = {}
for go_name, hearth_name in EMOTION_GO_TO_HEARTH.items():
    _HEARTH_TO_GO.setdefault(hearth_name, go_name)


def _goemotions_names_from_ids(ids: list[int], id2label: dict[int, str]) -> list[str]:
    names = []
    for i in ids:
        name = id2label.get(int(i))
        if name and name in EMOTION_LABELS:
            names.append(name)
    return names or ["neutral"]


def load_go_emotions(max_per_source: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    # ClassLabel names on the sequence feature
    feature = ds["train"].features["labels"].feature
    id2label = {i: feature.int2str(i) for i in range(feature.num_classes)}

    rows: list[dict] = []
    for split in ("train", "validation", "test"):
        for ex in ds[split]:
            names = _goemotions_names_from_ids(list(ex["labels"]), id2label)
            # Map dataset label names that might differ slightly
            names = [n if n in EMOTION_LABELS else "neutral" for n in names]
            text = (ex.get("text") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "labels": multi_hot(names),
                    "source": "go_emotions",
                    "split_hint": "train" if split == "train" else ("val" if split == "validation" else "test"),
                }
            )
            if max_per_source and len(rows) >= max_per_source:
                return rows
    return rows


def load_dair_emotion(max_per_source: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("dair-ai/emotion", "split")
    rows: list[dict] = []
    for split in ("train", "validation", "test"):
        for ex in ds[split]:
            name = ex["label"] if isinstance(ex["label"], str) else ds[split].features["label"].int2str(ex["label"])
            go = _DAIR_TO_GO.get(name)
            if not go:
                continue
            text = (ex.get("text") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "labels": multi_hot([go]),
                    "source": "dair_ai_emotion",
                    "split_hint": "train" if split == "train" else ("val" if split == "validation" else "test"),
                }
            )
            if max_per_source and len(rows) >= max_per_source:
                return rows
    return rows


def load_empathetic_contexts(max_per_source: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("bdotloh/empathetic-dialogues-contexts")
    rows: list[dict] = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        for ex in ds[split]:
            emotion = str(ex.get("emotion") or "").strip().lower()
            hearth = _EMPATHETIC_TO_HEARTH.get(emotion)
            if not hearth:
                continue
            go = _HEARTH_TO_GO.get(hearth)
            if not go:
                continue
            text = (ex.get("situation") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "labels": multi_hot([go]),
                    "source": "empathetic_contexts",
                    "split_hint": "train" if split == "train" else ("val" if split == "validation" else "test"),
                }
            )
            if max_per_source and len(rows) >= max_per_source:
                return rows
    return rows


def _preserve_hint_splits(rows: list[dict]) -> dict[str, list[dict]] | None:
    """If most rows carry split_hint, group by that instead of re-splitting."""
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
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "emotion")
    parser.add_argument("--max-per-source", type=int, default=0, help="Cap rows per HF source (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["go_emotions", "dair_ai_emotion", "empathetic_contexts"],
        choices=["go_emotions", "dair_ai_emotion", "empathetic_contexts"],
    )
    args = parser.parse_args()
    limit = args.max_per_source or None

    loaders = {
        "go_emotions": load_go_emotions,
        "dair_ai_emotion": load_dair_emotion,
        "empathetic_contexts": load_empathetic_contexts,
    }

    all_rows: list[dict] = []
    for name in args.sources:
        print(f"Loading {name}...")
        part = loaders[name](limit)
        print(f"  -> {len(part)} rows")
        all_rows.extend(part)

    splits = _preserve_hint_splits(all_rows)
    if splits is None or not splits.get("val") or not splits.get("test"):
        cleaned = []
        for r in all_rows:
            row = {k: v for k, v in r.items() if k != "split_hint"}
            active = [i for i, v in enumerate(row["labels"]) if v]
            row["_bucket"] = str(active[0]) if active else "0"
            cleaned.append(row)
        splits = stratified_split(cleaned, label_key="_bucket", seed=args.seed)
        for key in splits:
            for row in splits[key]:
                row.pop("_bucket", None)

    for split_name, rows in splits.items():
        # Drop empty texts / ensure label length
        kept = []
        for row in rows:
            if not row.get("text"):
                continue
            if len(row["labels"]) != len(EMOTION_LABELS):
                continue
            kept.append(
                {
                    "text": row["text"],
                    "labels": row["labels"],
                    "source": row.get("source", "unknown"),
                }
            )
        n = write_jsonl(args.out_dir / f"{split_name}.jsonl", kept)
        print(f"Wrote {n} -> {args.out_dir / f'{split_name}.jsonl'}")

    print(f"Emotion label dim = {len(EMOTION_LABELS)}")


if __name__ == "__main__":
    main()
