#!/usr/bin/env python3
"""Train RelationshipHead on ``data/relationship/*.jsonl``.

    python3 examples/train_relationship.py --smoke
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from hearth_ai.labels import RELATIONSHIP_SIGNALS, relationship_example_to_list
from hearth_ai.models import HearthEncoder, HearthModel, RelationshipHead
from hearth_ai.trainer.dataset import HearthDataset
from hearth_ai.trainer.losses import relationship_loss
from hearth_ai.trainer.train import Trainer

from _train_common import (
    ROOT,
    add_runtime_args,
    build_or_load_tokenizer,
    encode_for_model,
    ensure_prepared_data,
    fit_kwargs,
    load_encoder_weights,
    make_config,
    resolve_sizing,
    trainer_kwargs,
)

# Soft MAE gate for smoke/synthetic data (plan: "under tuned threshold")
MAE_GATE = 0.35


def _eval_mae(model, dataset, device: str) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            ids = item["input_ids"].unsqueeze(0).to(device)
            mask = item["attention_mask"].unsqueeze(0).to(device)
            pred = model(ids, mask)[0].cpu()
            gold = item["label"]
            total += torch.abs(pred - gold).mean().item()
            n += 1
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", default=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/relationship")
    parser.add_argument("--tokenizer", type=str, default="hearth_ai/tokenizer/emotion_intent.json")
    parser.add_argument("--warm-start", type=str, default="")
    parser.add_argument("--strict-gate", action="store_true")
    add_runtime_args(parser)
    args = parser.parse_args()
    smoke = not args.full

    data_dir = ensure_prepared_data("relationship")
    train_path, val_path = data_dir / "train.jsonl", data_dir / "val.jsonl"
    paths = [train_path, val_path, data_dir / "test.jsonl"]

    vocab_size, max_seq = resolve_sizing(args, smoke=smoke)
    tok_path = ROOT / args.tokenizer
    tok = build_or_load_tokenizer(
        paths, tok_path, vocab_size=vocab_size, max_seq_len=max_seq, force_retrain=not tok_path.is_file()
    )
    cfg = make_config(smoke=smoke, vocab_size=tok.vocab_size, max_seq_len=max_seq)
    model = HearthModel(HearthEncoder(cfg), RelationshipHead(cfg.hidden_size))
    print(f"Relationship model params: {model.num_parameters()/1e6:.2f}M")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    warm = args.warm_start or str(ROOT / "checkpoints/emotion/best.pt")
    if os.path.isfile(warm):
        load_encoder_weights(model, warm, device)

    def label_fn(ex):
        return torch.tensor(relationship_example_to_list(ex), dtype=torch.float)

    train_ds = HearthDataset(
        str(train_path), tok, label_fn, max_examples=args.max_train_rows or None
    )
    val_ds = HearthDataset(str(val_path), tok, label_fn) if val_path.is_file() else None

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        loss_fn=relationship_loss,
        pad_id=tok.pad_id,
        checkpoint_dir=args.checkpoint_dir,
        **trainer_kwargs(args),
    )
    epochs = args.epochs or (5 if smoke else 3)
    batch_size = args.batch_size or (8 if smoke else 16)
    lr = args.lr or (1e-3 if smoke else 3e-4)
    trainer.fit(epochs=epochs, batch_size=batch_size, lr=lr, **fit_kwargs(args))

    mae = _eval_mae(model, val_ds or train_ds, device)
    print(f"Eval MAE: {mae:.3f} (gate ≤ {MAE_GATE})")
    if args.strict_gate and mae > MAE_GATE:
        raise SystemExit(f"MAE gate failed: {mae:.3f} > {MAE_GATE}")

    model.eval()
    text = "I've never told anyone this before, but I feel broken inside."
    with torch.no_grad():
        pred = model(*encode_for_model(model, tok, text))[0]
    named = {n: float(pred[i]) for i, n in enumerate(RELATIONSHIP_SIGNALS)}
    print(f"'{text}' -> {named}")


if __name__ == "__main__":
    main()
