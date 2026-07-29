#!/usr/bin/env python3
"""Train StrategyHead on ``data/strategy/*.jsonl``.

    python3 examples/train_strategy.py --smoke
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from hearth_ai.labels import STRATEGY_LABELS, strategy_example_to_id
from hearth_ai.models import HearthEncoder, HearthModel, StrategyHead
from hearth_ai.trainer.dataset import HearthDataset
from hearth_ai.trainer.losses import strategy_loss
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


def _macro_f1(model, dataset, device: str) -> float:
    model.eval()
    # per-class tp/fp/fn
    stats = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(len(STRATEGY_LABELS))}
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            ids = item["input_ids"].unsqueeze(0).to(device)
            mask = item["attention_mask"].unsqueeze(0).to(device)
            pred = int(model(ids, mask).argmax(-1).item())
            gold = int(item["label"].item())
            if pred == gold:
                stats[gold]["tp"] += 1
            else:
                stats[pred]["fp"] += 1
                stats[gold]["fn"] += 1
    f1s = []
    for s in stats.values():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", default=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/strategy")
    parser.add_argument("--tokenizer", type=str, default="hearth_ai/tokenizer/emotion_intent.json")
    parser.add_argument("--warm-start", type=str, default="")
    parser.add_argument("--strict-gate", action="store_true")
    add_runtime_args(parser)
    args = parser.parse_args()
    smoke = not args.full

    data_dir = ensure_prepared_data("strategy")
    train_path, val_path = data_dir / "train.jsonl", data_dir / "val.jsonl"
    paths = [train_path, val_path, data_dir / "test.jsonl"]

    vocab_size, max_seq = resolve_sizing(args, smoke=smoke)
    tok_path = ROOT / args.tokenizer
    tok = build_or_load_tokenizer(
        paths, tok_path, vocab_size=vocab_size, max_seq_len=max_seq, force_retrain=not tok_path.is_file()
    )
    cfg = make_config(smoke=smoke, vocab_size=tok.vocab_size, max_seq_len=max_seq)
    model = HearthModel(HearthEncoder(cfg), StrategyHead(cfg.hidden_size))
    print(f"Strategy model params: {model.num_parameters()/1e6:.2f}M")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    warm = args.warm_start or str(ROOT / "checkpoints/emotion/best.pt")
    if os.path.isfile(warm):
        load_encoder_weights(model, warm, device)

    def label_fn(ex):
        return torch.tensor(strategy_example_to_id(ex), dtype=torch.long)

    train_ds = HearthDataset(
        str(train_path), tok, label_fn, max_examples=args.max_train_rows or None
    )
    val_ds = HearthDataset(str(val_path), tok, label_fn) if val_path.is_file() else None

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        loss_fn=strategy_loss,
        pad_id=tok.pad_id,
        checkpoint_dir=args.checkpoint_dir,
        **trainer_kwargs(args),
    )
    epochs = args.epochs or (5 if smoke else 3)
    batch_size = args.batch_size or (8 if smoke else 16)
    lr = args.lr or (1e-3 if smoke else 3e-4)
    trainer.fit(epochs=epochs, batch_size=batch_size, lr=lr, **fit_kwargs(args))

    macro = _macro_f1(model, val_ds or train_ds, device)
    print(f"Eval macro-F1: {macro:.3f} (gate ≥ 0.45)")
    if args.strict_gate and macro < 0.45:
        raise SystemExit(f"macro-F1 gate failed: {macro:.3f} < 0.45")

    model.eval()
    text = "I just need someone to hear me without fixing it."
    ids, mask = tok.encode(text)
    with torch.no_grad():
        pred = int(model(torch.tensor([ids]), torch.tensor([mask])).argmax(-1).item())
    print(f"'{text}' -> {STRATEGY_LABELS[pred]}")


if __name__ == "__main__":
    main()
