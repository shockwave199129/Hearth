#!/usr/bin/env python3
"""Train EmotionHead on prepared ``data/emotion/*.jsonl``.

Smoke (tiny model, fast)::

    python3 examples/train_emotion.py --smoke

Full default HearthConfig (~90M) after a full prepare::

    python3 -m data.prepare.prepare_emotion
    python3 examples/train_emotion.py --full
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from hearth_ai.labels import EMOTION_LABELS
from hearth_ai.models import EmotionHead, HearthEncoder, HearthModel
from hearth_ai.trainer.dataset import HearthDataset
from hearth_ai.trainer.losses import emotion_loss
from hearth_ai.trainer.train import Trainer

from _train_common import (
    ROOT,
    add_runtime_args,
    build_or_load_tokenizer,
    encode_for_model,
    ensure_prepared_data,
    fit_kwargs,
    make_config,
    resolve_sizing,
    trainer_kwargs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", default=True)
    parser.add_argument("--full", action="store_true", help="Use full HearthConfig (~90M)")
    parser.add_argument("--epochs", type=int, default=0, help="0 = smoke 5 / full 3")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/emotion")
    parser.add_argument("--tokenizer", type=str, default="hearth_ai/tokenizer/emotion_intent.json")
    add_runtime_args(parser)
    args = parser.parse_args()
    smoke = not args.full

    data_dir = ensure_prepared_data("emotion")
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    paths = [train_path, val_path, data_dir / "test.jsonl"]

    vocab_size, max_seq = resolve_sizing(args, smoke=smoke)
    tok_path = ROOT / args.tokenizer
    tok = build_or_load_tokenizer(
        paths,
        tok_path,
        vocab_size=vocab_size,
        max_seq_len=max_seq,
        # Emotion normally trains first and owns the shared vocab; --keep-tokenizer
        # is required when build_tokenizer.py already produced a cross-task one.
        force_retrain=not (args.keep_tokenizer and tok_path.is_file()),
    )
    cfg = make_config(smoke=smoke, vocab_size=tok.vocab_size, max_seq_len=max_seq)

    encoder = HearthEncoder(cfg)
    model = HearthModel(encoder, EmotionHead(cfg.hidden_size))
    print(f"Emotion model params: {model.num_parameters() / 1e6:.2f}M  smoke={smoke}")

    def label_fn(ex):
        labels = ex["labels"]
        if len(labels) != len(EMOTION_LABELS):
            raise ValueError(f"expected {len(EMOTION_LABELS)} labels, got {len(labels)}")
        return torch.tensor(labels, dtype=torch.float)

    max_rows = args.max_train_rows or None
    train_ds = HearthDataset(str(train_path), tok, label_fn, max_examples=max_rows)
    val_ds = HearthDataset(str(val_path), tok, label_fn) if val_path.is_file() else None
    print(f"train rows: {len(train_ds)}  val rows: {len(val_ds) if val_ds else 0}")

    epochs = args.epochs or (5 if smoke else 3)
    batch_size = args.batch_size or (8 if smoke else 16)
    lr = args.lr or (1e-3 if smoke else 3e-4)

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        loss_fn=emotion_loss,
        pad_id=tok.pad_id,
        checkpoint_dir=args.checkpoint_dir,
        **trainer_kwargs(args),
    )
    trainer.fit(epochs=epochs, batch_size=batch_size, lr=lr, **fit_kwargs(args))

    # Quick inference: top emotion name
    model.eval()
    text = "I am so happy and grateful today!"
    with torch.no_grad():
        logits = model(*encode_for_model(model, tok, text))
        probs = torch.sigmoid(logits)[0]
        top = int(probs.argmax().item())
    print(f"\n'{text}' -> top emotion: {EMOTION_LABELS[top]} ({probs[top]:.3f})")
    print(f"Checkpoint: {args.checkpoint_dir}/best.pt (or last.pt)")


if __name__ == "__main__":
    main()
