#!/usr/bin/env python3
"""Train MemoryHead on ``data/memory/*.jsonl``.

    python3 examples/train_memory.py --smoke
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from hearth_ai.labels import MEMORY_TYPES, memory_example_to_tensors
from hearth_ai.models import HearthEncoder, HearthModel, MemoryHead
from hearth_ai.trainer.dataset import HearthDataset
from hearth_ai.trainer.losses import memory_loss
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


def _eval_store_f1(model, dataset, device: str) -> float:
    """Binary F1 on store decision (threshold 0.5 on sigmoid)."""
    model.eval()
    tp = fp = fn = 0
    loader_items = range(len(dataset))
    with torch.no_grad():
        for i in loader_items:
            item = dataset[i]
            ids = item["input_ids"].unsqueeze(0).to(device)
            mask = item["attention_mask"].unsqueeze(0).to(device)
            out = model(ids, mask)
            pred = (torch.sigmoid(out["store_logit"]) >= 0.5).item()
            gold = int(item["label"]["store"].item() >= 0.5)
            if pred and gold:
                tp += 1
            elif pred and not gold:
                fp += 1
            elif not pred and gold:
                fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", default=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/memory")
    parser.add_argument("--tokenizer", type=str, default="hearth_ai/tokenizer/emotion_intent.json")
    parser.add_argument("--warm-start", type=str, default="")
    parser.add_argument("--strict-gate", action="store_true", help="Fail if store-F1 < 0.70")
    add_runtime_args(parser)
    args = parser.parse_args()
    smoke = not args.full

    data_dir = ensure_prepared_data("memory")
    train_path, val_path = data_dir / "train.jsonl", data_dir / "val.jsonl"
    paths = [train_path, val_path, data_dir / "test.jsonl"]

    vocab_size, max_seq = resolve_sizing(args, smoke=smoke)
    tok_path = ROOT / args.tokenizer
    tok = build_or_load_tokenizer(
        paths, tok_path, vocab_size=vocab_size, max_seq_len=max_seq, force_retrain=not tok_path.is_file()
    )
    cfg = make_config(smoke=smoke, vocab_size=tok.vocab_size, max_seq_len=max_seq)
    model = HearthModel(HearthEncoder(cfg), MemoryHead(cfg.hidden_size))
    print(f"Memory model params: {model.num_parameters()/1e6:.2f}M")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    warm = args.warm_start or str(ROOT / "checkpoints/emotion/best.pt")
    if os.path.isfile(warm):
        load_encoder_weights(model, warm, device)

    def label_fn(ex):
        d = memory_example_to_tensors(ex)
        return {
            "store": torch.tensor(d["store"], dtype=torch.float),
            "type": torch.tensor(d["type"], dtype=torch.long),
            "importance": torch.tensor(d["importance"], dtype=torch.float),
        }

    train_ds = HearthDataset(
        str(train_path), tok, label_fn, max_examples=args.max_train_rows or None
    )
    val_ds = HearthDataset(str(val_path), tok, label_fn) if val_path.is_file() else None

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        loss_fn=memory_loss,
        pad_id=tok.pad_id,
        checkpoint_dir=args.checkpoint_dir,
        **trainer_kwargs(args),
    )
    epochs = args.epochs or (5 if smoke else 3)
    batch_size = args.batch_size or (8 if smoke else 16)
    lr = args.lr or (1e-3 if smoke else 3e-4)
    trainer.fit(epochs=epochs, batch_size=batch_size, lr=lr, **fit_kwargs(args))

    eval_ds = val_ds or train_ds
    store_f1 = _eval_store_f1(model, eval_ds, device)
    print(f"Eval store-F1: {store_f1:.3f} (gate ≥ 0.70)")
    if args.strict_gate and store_f1 < 0.70:
        raise SystemExit(f"store-F1 gate failed: {store_f1:.3f} < 0.70")

    model.eval()
    text = "My goal this year is to save for a down payment."
    ids, mask = tok.encode(text)
    with torch.no_grad():
        out = model(torch.tensor([ids]), torch.tensor([mask]))
        store_p = torch.sigmoid(out["store_logit"]).item()
        typ = MEMORY_TYPES[int(out["type_logits"].argmax(-1).item())]
    print(f"'{text}' -> store_p={store_p:.2f} type={typ}")


if __name__ == "__main__":
    main()
