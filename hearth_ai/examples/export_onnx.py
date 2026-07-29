#!/usr/bin/env python3
"""Export hearth_ai checkpoints to ONNX under models/nlp/ (with parity checks).

Examples::

    python3 examples/export_onnx.py --smoke
    python3 examples/export_onnx.py --full \\
        --checkpoint-root checkpoints --out ../../models/nlp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._train_common import make_config  # noqa: E402
from hearth_ai.export.onnx_export import (  # noqa: E402
    config_from_dict,
    export_all,
    load_checkpoint_state,
)
from hearth_ai.tokenizer.hearth_tokenizer import HearthTokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", default=True)
    parser.add_argument("--full", action="store_true", help="Use full-size HearthConfig")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "checkpoints",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT.parent / "models" / "nlp",
        help="Package root (default: <repo>/models/nlp)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "hearth_ai" / "tokenizer" / "emotion_intent.json",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--encoder-from",
        default="emotion",
        help="Which task checkpoint supplies the shared encoder.onnx weights",
    )
    parser.add_argument(
        "--ignore-checkpoint-config",
        action="store_true",
        help="Use --smoke/--full defaults instead of the config stored in the checkpoint",
    )
    args = parser.parse_args()
    smoke = not args.full

    if not args.tokenizer.is_file():
        raise SystemExit(f"Tokenizer not found: {args.tokenizer}")

    # Checkpoints written by Trainer embed their HearthConfig, which is the only
    # reliable source once --max-seq / --vocab-size have been overridden at
    # train time. Fall back to smoke/full defaults for older checkpoints.
    cfg = None
    if not args.ignore_checkpoint_config:
        for name in ("best.pt", "last.pt"):
            probe = args.checkpoint_root / args.encoder_from / name
            if probe.is_file():
                _, embedded = load_checkpoint_state(probe)
                if embedded:
                    cfg = config_from_dict(embedded)
                    print(f"Using config embedded in {probe}")
                break

    if cfg is None:
        max_seq = 64 if smoke else 128
        tok = HearthTokenizer(str(args.tokenizer), max_seq_len=max_seq)
        cfg = make_config(smoke=smoke, vocab_size=tok.vocab_size, max_seq_len=max_seq)

    print(f"Exporting smoke={smoke} vocab={cfg.vocab_size} seq={cfg.max_seq_len} → {args.out}")
    results = export_all(
        args.checkpoint_root,
        args.out,
        cfg,
        tokenizer_src=args.tokenizer,
        encoder_from=args.encoder_from,
        opset=args.opset,
        atol=args.atol,
    )
    for r in results:
        print(f"  {r.name:12s}  max|Δ|={r.max_abs_diff:.3e}  → {r.onnx_path}")
    print("OK — all artifacts passed parity")


if __name__ == "__main__":
    main()
