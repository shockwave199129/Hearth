#!/usr/bin/env python3
"""Generate synthetic StrategyHead JSONL (+ weak labels from intent priors).

Output::
    {"text": "...", "strategy": "validate", "source": "synthetic"}

If ``data/intent/train.jsonl`` exists, also emit weak labels via intent→strategy
priors from ``labels/strategy.yaml``.

Usage::

    python3 -m data.prepare.prepare_strategy
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hearth_ai.labels import INTENT_LABELS, STRATEGY_LABELS  # noqa: E402
from data.prepare.common import DATA_DIR, stratified_split, write_jsonl  # noqa: E402

_INTENT_PRIORS: dict[str, list[str]] = {
    "vent": ["listen", "validate", "comfort"],
    "validate": ["validate", "reflect"],
    "comfort": ["comfort", "ground"],
    "celebrate": ["celebrate", "encourage"],
    "advise": ["advise", "plan"],
    "inquire": ["ask_question", "listen"],
    "plan": ["plan", "advise"],
    "small_talk": ["listen", "ask_question"],
    "meta": ["listen"],
    "unknown": ["listen", "validate"],
}

_TEMPLATES: list[tuple[str, str]] = [
    ("I just need someone to hear me without fixing it.", "listen"),
    ("It makes sense that you'd feel overwhelmed after that week.", "validate"),
    ("So the core of it is feeling unseen at work?", "reflect"),
    ("I'm here with you; we can take this slowly.", "comfort"),
    ("You've handled hard things before — that courage is still there.", "encourage"),
    ("That's wonderful news about the offer!", "celebrate"),
    ("One option is to write down what you need before the talk.", "advise"),
    ("What feels most important to you about this situation?", "ask_question"),
    ("Let's sketch a small next step for tomorrow morning.", "plan"),
    ("Can you feel your feet on the floor and name five things you see?", "ground"),
    ("It's okay to say no to that request.", "boundary"),
    ("I'm really concerned about your safety — please reach out to local crisis support.", "defer_safety"),
    ("Tell me more whenever you're ready; no rush.", "listen"),
    ("Anyone would be shaken after a call like that.", "validate"),
    ("Congrats — you earned that celebration.", "celebrate"),
    ("If panic rises, try a longer exhale than inhale.", "ground"),
]


def _from_templates(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for i in range(n):
        text, strategy = _TEMPLATES[i % len(_TEMPLATES)]
        rows.append(
            {
                "text": text + rng.choice(["", " ", ""]),
                "strategy": strategy,
                "source": "synthetic",
                "annotator": "synthetic",
            }
        )
    return rows


def _from_intent_jsonl(path: Path, rng: random.Random, max_rows: int) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            intent = ex.get("intent")
            if intent not in _INTENT_PRIORS:
                continue
            strategy = rng.choice(_INTENT_PRIORS[intent])
            rows.append(
                {
                    "text": ex["text"],
                    "strategy": strategy,
                    "intent_hint": intent,
                    "source": "intent_prior_weak",
                    "annotator": "synthetic_prior",
                }
            )
            if len(rows) >= max_rows:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "strategy")
    parser.add_argument("--n-synthetic", type=int, default=180)
    parser.add_argument("--n-from-intent", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = _from_templates(rng, args.n_synthetic)
    rows.extend(_from_intent_jsonl(DATA_DIR / "intent" / "train.jsonl", rng, args.n_from_intent))

    for r in rows:
        assert r["strategy"] in STRATEGY_LABELS
        r["_bucket"] = r["strategy"]

    splits = stratified_split(rows, label_key="_bucket", seed=args.seed)
    for split_name, part in splits.items():
        kept = []
        for r in part:
            row = {
                "text": r["text"],
                "strategy": r["strategy"],
                "source": r["source"],
                "annotator": r.get("annotator", "synthetic"),
            }
            if "intent_hint" in r:
                row["intent_hint"] = r["intent_hint"]
            kept.append(row)
        n = write_jsonl(args.out_dir / f"{split_name}.jsonl", kept)
        print(f"Wrote {n} -> {args.out_dir / f'{split_name}.jsonl'}")
    print(f"Strategy labels ({len(STRATEGY_LABELS)}): {STRATEGY_LABELS}")
    print(f"Intent priors cover: {sorted(_INTENT_PRIORS)}")
    _ = INTENT_LABELS  # locked set documented for annotators


if __name__ == "__main__":
    main()
