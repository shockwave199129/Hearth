#!/usr/bin/env python3
"""Train Emotion then Intent (encoder warm-start) — plan todo train-emotion-intent.

    python3 examples/train_emotion_intent.py --smoke
    python3 examples/train_emotion_intent.py --full
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", default=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--epochs", type=int, default=0)
    args = parser.parse_args()
    mode = ["--full"] if args.full else ["--smoke"]
    extra = ["--epochs", str(args.epochs)] if args.epochs else []

    env = os.environ.copy()
    py = sys.executable

    print("=== 1/2 Train Emotion ===")
    subprocess.check_call(
        [py, "examples/train_emotion.py", *mode, *extra],
        cwd=ROOT,
        env=env,
    )

    warm = os.path.join(ROOT, "checkpoints", "emotion", "best.pt")
    if not os.path.isfile(warm):
        warm = os.path.join(ROOT, "checkpoints", "emotion", "last.pt")

    print("=== 2/2 Train Intent (warm-start encoder) ===")
    subprocess.check_call(
        [
            py,
            "examples/train_intent.py",
            *mode,
            *extra,
            "--warm-start",
            warm,
            "--tokenizer",
            "hearth_ai/tokenizer/emotion_intent.json",
        ],
        cwd=ROOT,
        env=env,
    )
    print("Done. Checkpoints under checkpoints/emotion and checkpoints/intent")


if __name__ == "__main__":
    main()
