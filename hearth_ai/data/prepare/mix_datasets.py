#!/usr/bin/env python3
"""Mix HF-derived ("real") and generator ("synthetic") JSONL into final train files.

500k synthetic rows against ~50k real rows would let the generator's templates
dominate every epoch, so the synthetic share is capped as a fraction of the
final train set rather than concatenated wholesale.

Outputs, per task, into ``--out``::

    train.jsonl            mixed (what the train scripts read)
    val.jsonl              mixed, used for checkpoint selection
    test.jsonl             mixed
    test_real.jsonl        real-only slice — report this separately
    test_synthetic.jsonl   synthetic-only slice

Never judge Emotion/Intent on ``test_synthetic.jsonl`` alone: it measures how
well the model learned the generator, not the language.

``--synthetic-share`` is an upper bound, not a target. ``--max-per-label`` runs
first and can shrink the synthetic pool below the requested share, so treat the
printed per-split counts as the real composition.

Usage (from ``hearth_ai/``)::

    python3 -m data.prepare.mix_datasets --task emotion \\
        --real data/base/emotion --synthetic data/synthetic/emotion \\
        --out data/emotion --synthetic-share 0.3
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.prepare.common import (  # noqa: E402
    DATA_DIR,
    balance_by_label,
    read_jsonl,
    write_jsonl,
)

# Field used for per-label balancing; None = no balancing available.
LABEL_KEY = {
    "emotion": None,  # multi-hot, balanced upstream by grouped split
    "intent": "intent",
    "memory": "type",
    "relationship": None,  # regression
    "strategy": "strategy",
}


def _mix_split(
    real: list[dict],
    synthetic: list[dict],
    *,
    synthetic_share: float,
    max_rows: int,
    seed: int,
) -> tuple[list[dict], int, int]:
    """Combine so synthetic is at most ``synthetic_share`` of the result."""
    rng = random.Random(seed)
    real = list(real)
    synthetic = list(synthetic)
    rng.shuffle(real)
    rng.shuffle(synthetic)

    if not real:
        kept_syn = synthetic[:max_rows] if max_rows else synthetic
        return kept_syn, 0, len(kept_syn)
    if not synthetic or synthetic_share <= 0:
        kept_real = real[:max_rows] if max_rows else real
        return kept_real, len(kept_real), 0

    # n_syn / (n_real + n_syn) == share  =>  n_syn = share/(1-share) * n_real
    if synthetic_share >= 1.0:
        n_syn = len(synthetic)
    else:
        n_syn = int(round(len(real) * synthetic_share / (1.0 - synthetic_share)))
    n_syn = min(n_syn, len(synthetic))

    mixed = real + synthetic[:n_syn]
    rng.shuffle(mixed)
    if max_rows and len(mixed) > max_rows:
        mixed = mixed[:max_rows]
    n_real_kept = sum(1 for r in mixed if r.get("source") != "hearth_relationship_synthetic")
    return mixed, n_real_kept, len(mixed) - n_real_kept


def mix_task(
    task: str,
    *,
    real_dir: Path | None,
    synthetic_dir: Path,
    out_dir: Path,
    synthetic_share: float,
    max_train_rows: int,
    max_eval_rows: int,
    max_per_label: int,
    seed: int,
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    label_key = LABEL_KEY.get(task)

    for split, max_rows in (
        ("train", max_train_rows),
        ("val", max_eval_rows),
        ("test", max_eval_rows),
    ):
        real = read_jsonl(real_dir / f"{split}.jsonl") if real_dir else []
        synthetic = read_jsonl(synthetic_dir / f"{split}.jsonl")

        if label_key and max_per_label and split == "train":
            synthetic = balance_by_label(
                synthetic, label_key=label_key, max_per_label=max_per_label, seed=seed
            )

        mixed, n_real, n_syn = _mix_split(
            real,
            synthetic,
            synthetic_share=synthetic_share,
            max_rows=max_rows,
            seed=seed,
        )
        written = write_jsonl(out_dir / f"{split}.jsonl", mixed)
        summary[split] = {"total": written, "real": n_real, "synthetic": n_syn}

        if split == "test":
            write_jsonl(out_dir / "test_real.jsonl", real[:max_eval_rows] if max_eval_rows else real)
            write_jsonl(
                out_dir / "test_synthetic.jsonl",
                synthetic[:max_eval_rows] if max_eval_rows else synthetic,
            )
            summary["test_real"] = {"total": len(real)}
            summary["test_synthetic"] = {"total": len(synthetic)}

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(LABEL_KEY))
    parser.add_argument("--real", type=Path, default=None, help="HF-derived dir (optional)")
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--synthetic-share",
        type=float,
        default=0.3,
        help="Max synthetic fraction of the mixed train set (1.0 = synthetic only)",
    )
    parser.add_argument("--max-train-rows", type=int, default=0, help="0 = no cap")
    parser.add_argument("--max-eval-rows", type=int, default=20000)
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=0,
        help="Cap synthetic train rows per label (0 = off)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.real and not args.real.is_dir():
        raise SystemExit(f"--real not found: {args.real}")
    if not args.synthetic.is_dir():
        raise SystemExit(f"--synthetic not found: {args.synthetic}")

    summary = mix_task(
        args.task,
        real_dir=args.real,
        synthetic_dir=args.synthetic,
        out_dir=args.out,
        synthetic_share=args.synthetic_share,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
        max_per_label=args.max_per_label,
        seed=args.seed,
    )

    print(f"{args.task} -> {args.out}")
    for split in ("train", "val", "test"):
        counts = summary[split]
        print(
            f"  {split}: {counts['total']} "
            f"(real {counts['real']}, synthetic {counts['synthetic']})"
        )
    print(
        f"  test_real: {summary['test_real']['total']}, "
        f"test_synthetic: {summary['test_synthetic']['total']}"
    )
    _ = DATA_DIR


if __name__ == "__main__":
    main()
