#!/usr/bin/env python3
"""Train one BPE tokenizer over every task's train text, before any head trains.

All five heads share an encoder, so they must share a vocabulary. If each
train_*.py builds its own tokenizer from its own rows, the checkpoints end up
with mismatched embeddings and the warm-start / ONNX export silently degrades.

Run this once, then pass ``--keep-tokenizer`` to every train script.

Usage (from ``hearth_ai/``)::

    python3 examples/build_tokenizer.py --full
    python3 examples/build_tokenizer.py --full --vocab-size 32000
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

from _train_common import ROOT, build_or_load_tokenizer

TASKS = ("emotion", "intent", "memory", "relationship", "strategy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="32k vocab / seq 128 defaults")
    parser.add_argument("--vocab-size", type=int, default=0)
    parser.add_argument("--max-seq", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "hearth_ai" / "tokenizer" / "emotion_intent.json",
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    args = parser.parse_args()

    vocab_size = args.vocab_size or (32000 if args.full else 4000)
    max_seq = args.max_seq or (128 if args.full else 64)

    paths: list[Path] = []
    for task in args.tasks:
        for split in ("train", "val"):
            path = args.data_dir / task / f"{split}.jsonl"
            if path.is_file():
                paths.append(path)
            elif split == "train":
                print(f"  missing {path} — run data.prepare.prepare_all_full first")

    if not paths:
        raise SystemExit("No task JSONL found; nothing to train a tokenizer on")

    print(f"Training tokenizer over {len(paths)} files (vocab {vocab_size}, seq {max_seq})")
    for path in paths:
        print(f"  {path}")

    tok = build_or_load_tokenizer(
        paths,
        args.out,
        vocab_size=vocab_size,
        max_seq_len=max_seq,
        force_retrain=True,
    )
    print(f"\nTokenizer -> {args.out} (vocab_size={tok.vocab_size})")
    print("Pass --keep-tokenizer to every train_*.py so this vocab is reused verbatim.")


if __name__ == "__main__":
    main()
