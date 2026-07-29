#!/usr/bin/env python3
"""Full Hearth NLP run: preflight -> prepare -> tokenizer -> 5 heads -> ONNX -> eval.

Cross-platform driver for the whole pipeline. ``scripts/train_full_nlp.ps1`` is a
thin wrapper around this; the logic lives here so it is testable on any OS.

Combines the HF datasets (go_emotions, dair-ai/emotion, empathetic_dialogues_v2,
daily_dialog, tanaos) with ``hearth_ai/data/hearth_relationship_understanding.jsonl``.

Usage::

    python scripts/train_full_nlp.py --check-only
    python scripts/train_full_nlp.py
    python scripts/train_full_nlp.py --batch-size 16 --grad-accum 2
    python scripts/train_full_nlp.py --dry-run          # print commands only
    python scripts/train_full_nlp.py --skip-prepare --only memory strategy
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HEARTH_ROOT = REPO_ROOT / "hearth_ai"
BACKEND_ROOT = REPO_ROOT / "backend"

HEADS = ("emotion", "intent", "memory", "relationship", "strategy")


def resolve_python(override: str | None) -> tuple[str, str]:
    """Pick the interpreter for every stage; return (path, why).

    All deps are pip-installed into the repo's ``.venv``, so if we were launched
    by a bare system Python we prefer that venv instead of failing five minutes
    in with a missing torch.
    """
    if override:
        return override, "--python"
    if os.environ.get("HEARTH_PYTHON"):
        return os.environ["HEARTH_PYTHON"], "HEARTH_PYTHON"
    if sys.prefix != sys.base_prefix:
        return sys.executable, "active venv"

    for rel in ("Scripts/python.exe", "bin/python"):
        candidate = REPO_ROOT / ".venv" / rel
        if candidate.is_file():
            return str(candidate), "repo .venv"

    return sys.executable, "current interpreter (no .venv found)"


def run(cmd: list[str], *, cwd: Path, label: str, dry_run: bool) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"$ cd {cwd} && {' '.join(str(c) for c in cmd)}", flush=True)
    if dry_run:
        return
    started = time.perf_counter()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    print(f"-- {label} finished in {(time.perf_counter() - started) / 60:.1f} min", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", choices=["off", "auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--only", nargs="+", default=list(HEADS), choices=list(HEADS))
    parser.add_argument("--strict-gates", action="store_true")
    # Data
    parser.add_argument("--synthetic-limit", type=int, default=0)
    parser.add_argument("--emotion-share", type=float, default=0.30)
    parser.add_argument("--intent-share", type=float, default=0.50)
    # Stages
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", help="Interpreter to run every stage with")
    args = parser.parse_args()

    if not HEARTH_ROOT.is_dir():
        print(f"hearth_ai not found at {HEARTH_ROOT}", file=sys.stderr)
        return 1

    python, why = resolve_python(args.python)
    print(f"python   {python}  [{why}]")
    # tokenizers' Rust thread pool oversubscribes the CPU next to the DataLoader.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not args.skip_check:
        run(
            [python, str(REPO_ROOT / "scripts" / "check_gpu.py")],
            cwd=REPO_ROOT,
            label="Environment preflight",
            dry_run=args.dry_run,
        )
    if args.check_only:
        return 0

    if not args.skip_prepare:
        prepare = [
            python,
            "-m",
            "data.prepare.prepare_all_full",
            "--emotion-share",
            str(args.emotion_share),
            "--intent-share",
            str(args.intent_share),
        ]
        if args.synthetic_limit:
            prepare += ["--limit", str(args.synthetic_limit)]
        if args.skip_hf:
            prepare.append("--skip-hf")
        run(prepare, cwd=HEARTH_ROOT, label="Prepare corpus (HF + synthetic)", dry_run=args.dry_run)
    else:
        print("\nSkipping prepare (reusing hearth_ai/data/*)")

    train = [
        python,
        "examples/train_all_full.py",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
        "--max-seq",
        str(args.max_seq),
        "--vocab-size",
        str(args.vocab_size),
        "--num-workers",
        str(args.num_workers),
        "--amp",
        args.amp,
        "--only",
        *args.only,
    ]
    if args.max_train_rows:
        train += ["--max-train-rows", str(args.max_train_rows)]
    if args.skip_tokenizer:
        train.append("--skip-tokenizer")
    if args.skip_export:
        train.append("--skip-export")
    if args.strict_gates:
        train.append("--strict-gates")
    run(train, cwd=HEARTH_ROOT, label="Train all heads", dry_run=args.dry_run)

    if not args.skip_export and not args.skip_eval:
        run(
            [python, "-m", "app.eval.nlp_golden", "--update"],
            cwd=BACKEND_ROOT,
            label="Re-lock golden snapshots",
            dry_run=args.dry_run,
        )
        run(
            [python, "-m", "app.eval.nlp_golden"],
            cwd=BACKEND_ROOT,
            label="Golden eval",
            dry_run=args.dry_run,
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
