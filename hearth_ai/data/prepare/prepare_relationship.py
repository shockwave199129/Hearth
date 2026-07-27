#!/usr/bin/env python3
"""Generate synthetic RelationshipHead JSONL.

Output::
    {"text": "...", "signals": [trust_delta, vulnerability, openness, comfort], "source": "synthetic"}

Signal order matches RELATIONSHIP_SIGNALS.

Usage::

    python3 -m data.prepare.prepare_relationship
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hearth_ai.labels import RELATIONSHIP_SIGNALS  # noqa: E402
from data.prepare.common import DATA_DIR, stratified_split, write_jsonl  # noqa: E402

# (text, trust_delta, vulnerability, openness, comfort)
_TEMPLATES: list[tuple[str, float, float, float, float]] = [
    ("I've never told anyone this before, but I feel broken inside.", 0.2, 0.9, 0.95, 0.25),
    ("Thanks for listening; I feel a bit safer talking here.", 0.35, 0.4, 0.6, 0.7),
    ("I don't trust easily after what happened.", -0.2, 0.5, 0.35, 0.3),
    ("Whatever, I don't care.", -0.1, 0.1, 0.15, 0.4),
    ("I'm okay today, just checking in.", 0.05, 0.15, 0.3, 0.75),
    ("It means a lot that you remembered my exam.", 0.4, 0.3, 0.55, 0.8),
    ("I'm scared you'll judge me if I say more.", 0.0, 0.8, 0.5, 0.35),
    ("I can finally breathe after telling you.", 0.25, 0.6, 0.85, 0.65),
    ("Please don't share this with anyone.", 0.1, 0.7, 0.7, 0.45),
    ("I feel guarded; maybe another day.", -0.05, 0.35, 0.25, 0.5),
    ("You actually get it. That helps.", 0.3, 0.45, 0.7, 0.75),
    ("I'm shutting down a bit, sorry.", -0.15, 0.55, 0.2, 0.35),
    ("Hi — nothing deep, just saying hello.", 0.0, 0.05, 0.2, 0.7),
    ("I cried on the commute and then pretended I was fine at work.", 0.15, 0.85, 0.8, 0.3),
    ("I want to believe people can be kind again.", 0.2, 0.5, 0.6, 0.55),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "relationship")
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for i in range(args.n):
        text, td, vul, opn, com = _TEMPLATES[i % len(_TEMPLATES)]
        signals = [
            max(-1.0, min(1.0, td + rng.uniform(-0.05, 0.05))),
            max(0.0, min(1.0, vul + rng.uniform(-0.05, 0.05))),
            max(0.0, min(1.0, opn + rng.uniform(-0.05, 0.05))),
            max(0.0, min(1.0, com + rng.uniform(-0.05, 0.05))),
        ]
        rows.append(
            {
                "text": text,
                "signals": [round(x, 3) for x in signals],
                "source": "synthetic",
                "annotator": "synthetic",
                # bucket by comfort tertile for stratified split
                "_bucket": "hi" if signals[3] > 0.6 else ("lo" if signals[3] < 0.4 else "mid"),
            }
        )

    splits = stratified_split(rows, label_key="_bucket", seed=args.seed)
    assert RELATIONSHIP_SIGNALS == [
        "trust_delta",
        "vulnerability",
        "openness",
        "comfort",
    ]
    for split_name, part in splits.items():
        kept = [
            {
                "text": r["text"],
                "signals": r["signals"],
                "source": r["source"],
                "annotator": r["annotator"],
            }
            for r in part
        ]
        n = write_jsonl(args.out_dir / f"{split_name}.jsonl", kept)
        print(f"Wrote {n} -> {args.out_dir / f'{split_name}.jsonl'}")
    print(f"Relationship signals: {RELATIONSHIP_SIGNALS}")


if __name__ == "__main__":
    main()
