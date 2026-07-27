#!/usr/bin/env python3
"""Train Memory → Relationship → Strategy (plan todo train-remaining-heads).

    python3 examples/train_remaining_heads.py --smoke
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
    parser.add_argument("--strict-gate", action="store_true")
    args = parser.parse_args()
    mode = ["--full"] if args.full else ["--smoke"]
    extra = ["--epochs", str(args.epochs)] if args.epochs else []
    if args.strict_gate:
        extra.append("--strict-gate")

    warm = os.path.join(ROOT, "checkpoints", "emotion", "best.pt")
    if not os.path.isfile(warm):
        warm = os.path.join(ROOT, "checkpoints", "emotion", "last.pt")
    warm_args = ["--warm-start", warm] if os.path.isfile(warm) else []

    py = sys.executable
    for script in ("train_memory.py", "train_relationship.py", "train_strategy.py"):
        print(f"=== {script} ===")
        subprocess.check_call(
            [py, f"examples/{script}", *mode, *extra, *warm_args],
            cwd=ROOT,
        )
    print("Done. Checkpoints: checkpoints/{memory,relationship,strategy}/")


if __name__ == "__main__":
    main()
