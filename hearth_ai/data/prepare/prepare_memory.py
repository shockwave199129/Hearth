#!/usr/bin/env python3
"""Generate synthetic MemoryHead JSONL (+ optional reuse of intent texts).

Output::
    {"text": "...", "store": 1, "type": "goal", "importance": 0.82, "source": "synthetic"}

``type`` is a string from MEMORY_TYPES (converted to index in label_fn at train time).

Usage::

    python3 -m data.prepare.prepare_memory
    python3 -m data.prepare.prepare_memory --n 400
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hearth_ai.labels import MEMORY_TYPES  # noqa: E402
from data.prepare.common import DATA_DIR, stratified_split, write_jsonl  # noqa: E402

# (text, store, type, importance)
_TEMPLATES: list[tuple[str, int, str, float]] = [
    ("My interview yesterday went terribly and I keep replaying it.", 1, "episodic", 0.75),
    ("Last Tuesday my sister called to say she got engaged.", 1, "episodic", 0.8),
    ("I broke up with Alex three weeks ago.", 1, "episodic", 0.9),
    ("Coffee helps me wake up better than tea.", 1, "preference", 0.45),
    ("I prefer texting over phone calls when I'm anxious.", 1, "preference", 0.55),
    ("I hate being interrupted when I'm focusing.", 1, "preference", 0.5),
    ("My goal this year is to save for a down payment.", 1, "goal", 0.85),
    ("I want to run a half marathon by October.", 1, "goal", 0.7),
    ("I'm trying to set a boundary: no work Slack after 8pm.", 1, "boundary", 0.8),
    ("Please don't push me to talk about my dad unless I bring him up.", 1, "boundary", 0.9),
    ("My coworker Priya has been covering for me during chemo weeks.", 1, "person", 0.85),
    ("Sam is my roommate and we share rent.", 1, "person", 0.4),
    ("I know that deep breathing can lower heart rate a bit.", 1, "semantic", 0.35),
    ("Grief isn't linear; waves come and go.", 1, "semantic", 0.4),
    ("I feel hollow and ashamed after yelling at my kid.", 1, "emotional", 0.8),
    ("I've been carrying this loneliness for months.", 1, "emotional", 0.85),
    ("Nice weather today.", 0, "other", 0.05),
    ("lol okay", 0, "other", 0.02),
    ("What time is it there?", 0, "other", 0.05),
    ("Can you hear me?", 0, "other", 0.05),
    ("hmm", 0, "other", 0.01),
    ("Thanks, that helps.", 0, "other", 0.1),
    ("I might move cities next year if I get the offer.", 1, "goal", 0.65),
    ("My therapist suggested journaling before bed.", 1, "preference", 0.5),
    ("I told my manager I won't take weekend shifts anymore.", 1, "boundary", 0.75),
    ("Remember when we talked about my fear of flying?", 1, "episodic", 0.6),
    ("Jordan is dating someone new and I'm weirdly okay with it.", 1, "person", 0.55),
    ("I keep thinking I'm a failure no matter what I achieve.", 1, "emotional", 0.88),
]


def _variants(rng: random.Random, n: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(n):
        text, store, typ, importance = _TEMPLATES[i % len(_TEMPLATES)]
        # light paraphrase noise
        suffix = rng.choice(["", " You know?", " That's where I'm at.", ""])
        imp = min(1.0, max(0.0, importance + rng.uniform(-0.05, 0.05)))
        if store == 0:
            imp = min(imp, 0.15)
            typ = "other"
        rows.append(
            {
                "text": text + suffix,
                "store": store,
                "type": typ,
                "importance": round(imp, 3),
                "source": "synthetic",
                "annotator": "synthetic",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "memory")
    parser.add_argument("--n", type=int, default=240, help="Total synthetic rows before split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = _variants(rng, args.n)
    # bucket by type for stratified split
    for r in rows:
        r["_bucket"] = r["type"] if r["store"] else "nostore"
    splits = stratified_split(rows, label_key="_bucket", seed=args.seed)
    for split_name, part in splits.items():
        kept = []
        for r in part:
            assert r["type"] in MEMORY_TYPES
            kept.append(
                {
                    "text": r["text"],
                    "store": int(r["store"]),
                    "type": r["type"],
                    "importance": float(r["importance"]),
                    "source": r["source"],
                    "annotator": r.get("annotator", "synthetic"),
                }
            )
        n = write_jsonl(args.out_dir / f"{split_name}.jsonl", kept)
        print(f"Wrote {n} -> {args.out_dir / f'{split_name}.jsonl'}")
    print(f"Memory types: {MEMORY_TYPES}")


if __name__ == "__main__":
    main()
