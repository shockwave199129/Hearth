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
    build_or_load_tokenizer,
    ensure_prepared_data,
    make_config,
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
    args = parser.parse_args()
    smoke = not args.full

    data_dir = ensure_prepared_data("emotion")
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    paths = [train_path, val_path, data_dir / "test.jsonl"]

    vocab_size = 4000 if smoke else 32000
    max_seq = 64 if smoke else 128
    tok = build_or_load_tokenizer(
        paths,
        ROOT / args.tokenizer,
        vocab_size=vocab_size,
        max_seq_len=max_seq,
        force_retrain=True,
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

    train_ds = HearthDataset(str(train_path), tok, label_fn)
    val_ds = HearthDataset(str(val_path), tok, label_fn) if val_path.is_file() else None

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
    )
    trainer.fit(epochs=epochs, batch_size=batch_size, lr=lr)

    # Quick inference: top emotion name
    model.eval()
    text = "I am so happy and grateful today!"
    ids, mask = tok.encode(text)
    with torch.no_grad():
        logits = model(torch.tensor([ids]), torch.tensor([mask]))
        probs = torch.sigmoid(logits)[0]
        top = int(probs.argmax().item())
    print(f"\n'{text}' -> top emotion: {EMOTION_LABELS[top]} ({probs[top]:.3f})")
    print(f"Checkpoint: {args.checkpoint_dir}/best.pt (or last.pt)")


if __name__ == "__main__":
    main()
