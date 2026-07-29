#!/usr/bin/env python3
"""Train IntentHead on prepared ``data/intent/*.jsonl``.

Prefer warm-starting the encoder from a prior emotion checkpoint::

    python3 examples/train_emotion.py --smoke
    python3 examples/train_intent.py --smoke --warm-start checkpoints/emotion/best.pt

Full::

    python3 examples/train_intent.py --full --warm-start checkpoints/emotion/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from hearth_ai.labels import INTENT_LABELS
from hearth_ai.models import HearthEncoder, HearthModel, IntentHead
from hearth_ai.trainer.dataset import HearthDataset
from hearth_ai.trainer.losses import intent_loss
from hearth_ai.trainer.train import Trainer

from _train_common import (
    ROOT,
    add_runtime_args,
    build_or_load_tokenizer,
    ensure_prepared_data,
    fit_kwargs,
    load_encoder_weights,
    make_config,
    resolve_sizing,
    trainer_kwargs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", default=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/intent")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="hearth_ai/tokenizer/emotion_intent.json",
        help="Share with emotion training when warm-starting",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        default="",
        help="Path to emotion (or prior) HearthModel checkpoint for encoder weights",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use tiny data/intent_sample.jsonl instead of prepared HF JSONL",
    )
    add_runtime_args(parser)
    args = parser.parse_args()
    smoke = not args.full

    if args.sample:
        train_path = ROOT / "data" / "intent_sample.jsonl"
        val_path = None
        paths = [train_path]
    else:
        data_dir = ensure_prepared_data("intent")
        train_path = data_dir / "train.jsonl"
        val_path = data_dir / "val.jsonl"
        paths = [train_path, val_path, data_dir / "test.jsonl"]

    vocab_size, max_seq = resolve_sizing(args, smoke=smoke)
    tok_path = ROOT / args.tokenizer
    # Reuse existing shared tokenizer if present (emotion run); else train.
    force = not tok_path.is_file()
    tok = build_or_load_tokenizer(
        paths,
        tok_path,
        vocab_size=vocab_size,
        max_seq_len=max_seq,
        force_retrain=force,
    )
    cfg = make_config(smoke=smoke, vocab_size=tok.vocab_size, max_seq_len=max_seq)

    encoder = HearthEncoder(cfg)
    model = HearthModel(encoder, IntentHead(cfg.hidden_size))
    print(f"Intent model params: {model.num_parameters() / 1e6:.2f}M  smoke={smoke}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    warm = args.warm_start
    if not warm:
        candidate = ROOT / "checkpoints" / "emotion" / "best.pt"
        if not candidate.is_file():
            candidate = ROOT / "checkpoints" / "emotion" / "last.pt"
        if candidate.is_file():
            warm = str(candidate)
    if warm:
        load_encoder_weights(model, warm, device)

    def label_fn(ex):
        return torch.tensor(INTENT_LABELS.index(ex["intent"]), dtype=torch.long)

    max_rows = args.max_train_rows or None
    train_ds = HearthDataset(str(train_path), tok, label_fn, max_examples=max_rows)
    val_ds = (
        HearthDataset(str(val_path), tok, label_fn)
        if val_path is not None and val_path.is_file()
        else None
    )
    print(f"train rows: {len(train_ds)}  val rows: {len(val_ds) if val_ds else 0}")

    epochs = args.epochs or (5 if smoke else 3)
    batch_size = args.batch_size or (8 if smoke else 16)
    lr = args.lr or (1e-3 if smoke else 3e-4)

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        loss_fn=intent_loss,
        pad_id=tok.pad_id,
        checkpoint_dir=args.checkpoint_dir,
        **trainer_kwargs(args),
    )
    trainer.fit(epochs=epochs, batch_size=batch_size, lr=lr, **fit_kwargs(args))

    model.eval()
    text = "I just got accepted into grad school!"
    ids, mask = tok.encode(text)
    with torch.no_grad():
        logits = model(torch.tensor([ids]), torch.tensor([mask]))
    pred = INTENT_LABELS[logits.argmax(dim=-1).item()]
    print(f"\n'{text}' -> predicted intent: {pred}")
    print(f"Checkpoint: {args.checkpoint_dir}/best.pt (or last.pt)")


if __name__ == "__main__":
    main()
