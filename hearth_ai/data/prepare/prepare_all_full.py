#!/usr/bin/env python3
"""Build the full training corpus for all five heads: HF datasets + synthetic 500k.

Pipeline::

    1. HF prepare  -> data/base/{emotion,intent}          (go_emotions, dair-ai,
                                                           empathetic v2, daily_dialog, tanaos)
    2. Convert     -> data/synthetic/{5 tasks}            (hearth_relationship_understanding)
    3. Seed        -> data/seed/{memory,relationship,strategy}  (template seeds, keeps defer_safety)
    4. Mix         -> data/{5 tasks}                      (what the train scripts read)

Emotion/Intent stay anchored on real HF text (synthetic capped at
``--emotion-share`` / ``--intent-share``); the other three heads have no real
corpus, so they run synthetic-heavy with the template seeds mixed in for label
coverage the generator lacks (notably ``defer_safety``).

Usage (from ``hearth_ai/``)::

    python3 -m data.prepare.prepare_all_full
    python3 -m data.prepare.prepare_all_full --limit 50000 --skip-hf   # quick rebuild
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.prepare.common import DATA_DIR  # noqa: E402
from data.prepare.mix_datasets import mix_task  # noqa: E402

PY = sys.executable


def _run(module: str, *cli: str) -> None:
    cmd = [PY, "-m", module, *cli]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "hearth_relationship_understanding.jsonl",
    )
    parser.add_argument("--limit", type=int, default=0, help="Cap synthetic source rows (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-hf", action="store_true", help="Reuse existing data/base/*")
    parser.add_argument("--skip-synthetic", action="store_true", help="Reuse data/synthetic/*")
    parser.add_argument("--skip-seed", action="store_true", help="Reuse data/seed/*")
    parser.add_argument(
        "--hf-max-per-source",
        type=int,
        default=0,
        help="Cap rows per HF dataset (0 = all)",
    )
    parser.add_argument("--emotion-share", type=float, default=0.30)
    parser.add_argument("--intent-share", type=float, default=0.50)
    parser.add_argument(
        "--seed-share",
        type=float,
        default=0.05,
        help="Template-seed share for memory/relationship/strategy",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=250_000,
        help="Per-task train cap — keeps a full 5-head run tractable on one GPU",
    )
    parser.add_argument("--max-eval-rows", type=int, default=20_000)
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=40_000,
        help="Cap synthetic train rows per label (0 = off)",
    )
    parser.add_argument("--strict-emotion-map", action="store_true")
    parser.add_argument("--drop-memory-anomalies", action="store_true")
    args = parser.parse_args()

    base = DATA_DIR / "base"
    synthetic = DATA_DIR / "synthetic"
    seed_dir = DATA_DIR / "seed"

    # 1. HF datasets — all five sources kept exactly as the existing loaders define them.
    if not args.skip_hf:
        hf_args = ["--max-per-source", str(args.hf_max_per_source), "--seed", str(args.seed)]
        _run("data.prepare.prepare_emotion", "--out-dir", str(base / "emotion"), *hf_args)
        _run("data.prepare.prepare_intent", "--out-dir", str(base / "intent"), *hf_args)
    else:
        print("Skipping HF prepare (reusing data/base/*)")

    # 2. Synthetic 500k → five task formats.
    if not args.skip_synthetic:
        syn_args = [
            "--input",
            str(args.input),
            "--out-dir",
            str(synthetic),
            "--seed",
            str(args.seed),
            "--report",
            str(DATA_DIR / "synthetic" / "conversion_report.json"),
        ]
        if args.limit:
            syn_args += ["--limit", str(args.limit)]
        if args.strict_emotion_map:
            syn_args.append("--strict-emotion-map")
        if args.drop_memory_anomalies:
            syn_args.append("--drop-memory-anomalies")
        _run("data.prepare.prepare_hearth_synthetic", *syn_args)
    else:
        print("Skipping synthetic conversion (reusing data/synthetic/*)")

    # 3. Template seeds — the only source of defer_safety and of hand-written
    #    memory/relationship examples, so they stay in the mix.
    if not args.skip_seed:
        _run("data.prepare.prepare_memory", "--out-dir", str(seed_dir / "memory"), "--n", "2000")
        _run(
            "data.prepare.prepare_relationship",
            "--out-dir",
            str(seed_dir / "relationship"),
            "--n",
            "2000",
        )
        _run(
            "data.prepare.prepare_strategy",
            "--out-dir",
            str(seed_dir / "strategy"),
            "--n-synthetic",
            "2400",
            "--n-from-intent",
            "0",
        )
    else:
        print("Skipping template seeds (reusing data/seed/*)")

    # 4. Mix into the dirs the train scripts read.
    plan = [
        ("emotion", base / "emotion", synthetic / "emotion", args.emotion_share),
        ("intent", base / "intent", synthetic / "intent", args.intent_share),
        # For these three "real" is the template seed set, so the share is inverted:
        # a small seed slice against a synthetic-dominant train set.
        ("memory", seed_dir / "memory", synthetic / "memory", 1.0 - args.seed_share),
        ("relationship", seed_dir / "relationship", synthetic / "relationship", 1.0 - args.seed_share),
        ("strategy", seed_dir / "strategy", synthetic / "strategy", 1.0 - args.seed_share),
    ]

    print("\n=== Mixing ===")
    for task, real_dir, syn_dir, share in plan:
        if not syn_dir.is_dir():
            print(f"  {task}: missing {syn_dir} — skipped")
            continue
        if task in {"emotion", "intent"} and not real_dir.is_dir():
            print(
                f"  !! {task}: {real_dir} missing — mixing SYNTHETIC ONLY. "
                f"Run without --skip-hf so {task} stays anchored on real text."
            )
        summary = mix_task(
            task,
            real_dir=real_dir if real_dir.is_dir() else None,
            synthetic_dir=syn_dir,
            out_dir=DATA_DIR / task,
            synthetic_share=share,
            max_train_rows=args.max_train_rows,
            max_eval_rows=args.max_eval_rows,
            max_per_label=args.max_per_label,
            seed=args.seed,
        )
        train = summary["train"]
        print(
            f"  {task}: train {train['total']} "
            f"(real/seed {train['real']}, synthetic {train['synthetic']}), "
            f"val {summary['val']['total']}, test {summary['test']['total']}"
        )

    print("\nDone. Next: python3 examples/build_tokenizer.py --full")


if __name__ == "__main__":
    main()
