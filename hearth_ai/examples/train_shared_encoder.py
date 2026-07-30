#!/usr/bin/env python3
"""Jointly train one shared HearthEncoder + all five task heads together
(``MultiTaskTrainer``), instead of the sequential warm-start chain in
``train_all_full.py`` where each head only ever backprops its own task's
loss through the encoder it inherited — nothing anchors it back to the
earlier tasks, so the five encoders drift apart. Measured cosine similarity
between task-encoder pooled embeddings on the same input has been observed
as low as ~0.05 with the sequential approach; joint training keeps every
head's gradient landing on the *same* encoder every step, so the five
resulting checkpoints all embed byte-identical encoder weights and a single
shared ONNX encoder session becomes safe to run in production.

Smoke (small model, fast, local — proves the code path)::

    python3 examples/train_shared_encoder.py --smoke --max-train-rows 200 --epochs 1

Full (desktop GPU — real production training)::

    python3 examples/train_shared_encoder.py --full --epochs 3
    python3 examples/export_onnx.py --full --shared-encoder
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from hearth_ai.labels import (
    EMOTION_LABELS,
    INTENT_LABELS,
    memory_example_to_tensors,
    relationship_example_to_list,
    strategy_example_to_id,
)
from hearth_ai.models import build_multitask_model
from hearth_ai.trainer.dataset import HearthDataset
from hearth_ai.trainer.losses import (
    emotion_loss,
    intent_loss,
    memory_loss,
    relationship_loss,
    strategy_loss,
)
from hearth_ai.trainer.train import MultiTaskTrainer

from _train_common import ROOT, add_runtime_args, build_or_load_tokenizer, ensure_prepared_data, make_config

TASKS = ("emotion", "intent", "memory", "relationship", "strategy")

LOSS_FNS = {
    "emotion": emotion_loss,
    "intent": intent_loss,
    "memory": memory_loss,
    "relationship": relationship_loss,
    "strategy": strategy_loss,
}


def _label_fn(task: str):
    if task == "emotion":
        def fn(ex):
            labels = ex["labels"]
            if len(labels) != len(EMOTION_LABELS):
                raise ValueError(f"expected {len(EMOTION_LABELS)} labels, got {len(labels)}")
            return torch.tensor(labels, dtype=torch.float)
        return fn
    if task == "intent":
        return lambda ex: torch.tensor(INTENT_LABELS.index(ex["intent"]), dtype=torch.long)
    if task == "memory":
        def fn(ex):
            d = memory_example_to_tensors(ex)
            return {
                "store": torch.tensor(d["store"], dtype=torch.float),
                "type": torch.tensor(d["type"], dtype=torch.long),
                "importance": torch.tensor(d["importance"], dtype=torch.float),
            }
        return fn
    if task == "relationship":
        return lambda ex: torch.tensor(relationship_example_to_list(ex), dtype=torch.float)
    if task == "strategy":
        return lambda ex: torch.tensor(strategy_example_to_id(ex), dtype=torch.long)
    raise ValueError(task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", default=True)
    parser.add_argument("--full", action="store_true", help="Use full HearthConfig (~90M) + full 32k tokenizer")
    parser.add_argument("--epochs", type=int, default=0, help="0 = smoke 2 / full 3")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 = size of the largest task loader")
    parser.add_argument("--checkpoint-root", type=str, default="checkpoints")
    parser.add_argument(
        "--tokenizer", type=str, default="hearth_ai/tokenizer/emotion_intent.json",
        help="Shared tokenizer path — all five tasks must use the same one",
    )
    parser.add_argument("--only", nargs="+", default=list(TASKS), choices=list(TASKS))
    add_runtime_args(parser)
    args = parser.parse_args()
    smoke = not args.full

    data_dirs = {t: ensure_prepared_data(t) for t in args.only}
    paths = [data_dirs[t] / split for t in args.only for split in ("train.jsonl", "val.jsonl")]

    vocab_size = args.vocab_size or (4000 if smoke else 32000)
    max_seq = args.max_seq or (64 if smoke else 128)
    tok_path = ROOT / args.tokenizer
    tok = build_or_load_tokenizer(
        paths, tok_path, vocab_size=vocab_size, max_seq_len=max_seq,
        force_retrain=not (args.keep_tokenizer and tok_path.is_file()),
    )
    cfg = make_config(smoke=smoke, vocab_size=tok.vocab_size, max_seq_len=max_seq)

    model = build_multitask_model(cfg, include=tuple(args.only))
    print(f"Shared-encoder multi-task model params: {model.num_parameters() / 1e6:.2f}M  smoke={smoke}")

    train_datasets, val_datasets = {}, {}
    for t in args.only:
        train_path = data_dirs[t] / "train.jsonl"
        val_path = data_dirs[t] / "val.jsonl"
        label_fn = _label_fn(t)
        train_datasets[t] = HearthDataset(str(train_path), tok, label_fn, max_examples=args.max_train_rows or None)
        if val_path.is_file():
            val_datasets[t] = HearthDataset(str(val_path), tok, label_fn)
        print(f"  {t}: train rows={len(train_datasets[t])} val rows={len(val_datasets.get(t, []))}")

    trainer = MultiTaskTrainer(
        model=model,
        train_datasets=train_datasets,
        val_datasets=val_datasets or None,
        loss_fns=LOSS_FNS,
        pad_id=tok.pad_id,
        checkpoint_root=args.checkpoint_root,
        log_every=args.log_every,
    )
    epochs = args.epochs or (2 if smoke else 3)
    batch_size = args.batch_size or (8 if smoke else 16)
    lr = args.lr or (1e-3 if smoke else 3e-4)
    steps_per_epoch = args.steps_per_epoch or None
    result = trainer.fit(
        epochs=epochs, batch_size=batch_size, lr=lr, steps_per_epoch=steps_per_epoch,
        warmup_ratio=args.warmup_ratio, early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
    )
    print(f"\nDone. train_losses={result['train_losses']} best_val={result['best_val']}")
    print(f"Checkpoints written under {args.checkpoint_root}/<task>/{{best,last}}.pt — all sharing one encoder.")
    print(
        "\nNext:\n"
        f"  python3 examples/export_onnx.py {'--full' if args.full else ''} --shared-encoder\n"
        "  cd ../backend && python -m app.eval.nlp_golden --update && python -m app.eval.nlp_golden"
    )


if __name__ == "__main__":
    main()
