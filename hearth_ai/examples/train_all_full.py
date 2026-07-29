#!/usr/bin/env python3
"""Train all five heads end to end on the full corpus, then export ONNX.

Order matters: Emotion trains the encoder from scratch on the largest real
corpus, then every later head warm-starts from the previous checkpoint, so the
shared encoder keeps improving instead of five encoders diverging.

    build_tokenizer -> emotion -> intent -> memory -> relationship -> strategy -> export

Prerequisite (once)::

    python3 -m data.prepare.prepare_all_full

Usage (from ``hearth_ai/``)::

    python3 examples/train_all_full.py
    python3 examples/train_all_full.py --batch-size 24 --grad-accum 2 --epochs 3
    python3 examples/train_all_full.py --skip-tokenizer --only memory strategy
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

HEADS = ("emotion", "intent", "memory", "relationship", "strategy")
# Each head warm-starts from the head before it.
WARM_SOURCE = {
    "intent": "emotion",
    "memory": "intent",
    "relationship": "memory",
    "strategy": "relationship",
}


def _checkpoint(head: str) -> Path | None:
    for name in ("best.pt", "last.pt"):
        path = ROOT / "checkpoints" / head / name
        if path.is_file():
            return path
    return None


def _run(cmd: list[str], label: str) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"$ {' '.join(cmd)}", flush=True)
    started = time.perf_counter()
    subprocess.check_call(cmd, cwd=str(ROOT))
    print(f"-- {label} finished in {(time.perf_counter() - started) / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--amp", choices=["off", "auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--early-stop-patience", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Per-head cap on train rows (0 = use everything the mixer wrote)",
    )
    parser.add_argument("--only", nargs="+", default=list(HEADS), choices=list(HEADS))
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument(
        "--strict-gates",
        action="store_true",
        help="Fail the run when a head misses its eval gate",
    )
    parser.add_argument(
        "--export-out",
        type=Path,
        default=ROOT.parent / "models" / "nlp",
    )
    args = parser.parse_args()

    # Tokenizers' own thread pool fights the DataLoader and spams warnings.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not args.skip_tokenizer:
        _run(
            [
                PY,
                "examples/build_tokenizer.py",
                "--full",
                "--vocab-size",
                str(args.vocab_size),
                "--max-seq",
                str(args.max_seq),
            ],
            "Shared tokenizer",
        )

    shared = [
        "--full",
        "--keep-tokenizer",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--amp",
        args.amp,
        "--grad-accum",
        str(args.grad_accum),
        "--num-workers",
        str(args.num_workers),
        "--max-seq",
        str(args.max_seq),
        "--vocab-size",
        str(args.vocab_size),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--log-every",
        str(args.log_every),
    ]
    if args.max_train_rows:
        shared += ["--max-train-rows", str(args.max_train_rows)]

    for head in HEADS:
        if head not in args.only:
            continue
        cmd = [PY, f"examples/train_{head}.py", *shared]

        source = WARM_SOURCE.get(head)
        if source:
            warm = _checkpoint(source)
            if warm is None:
                print(f"\n!! no {source} checkpoint — {head} trains from scratch")
            else:
                cmd += ["--warm-start", str(warm)]

        if args.strict_gates and head in {"memory", "relationship", "strategy"}:
            cmd.append("--strict-gate")

        _run(cmd, f"Train {head}")

    if not args.skip_export:
        _run(
            [
                PY,
                "examples/export_onnx.py",
                "--full",
                "--out",
                str(args.export_out),
            ],
            "Export ONNX",
        )
        print(
            "\nNext (from repo root):\n"
            "  cd backend\n"
            "  python -m app.eval.nlp_golden --update\n"
            "  python -m app.eval.nlp_golden"
        )


if __name__ == "__main__":
    main()
