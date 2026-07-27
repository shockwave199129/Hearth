"""Generic training loop for any HearthModel(encoder, head).

Only three things change between the emotion, intent, memory, relationship,
and strategy models: the dataset, the head, and the loss function. Everything
else - the loop, optimizer, checkpointing - is identical, so it lives here once.

Example (see also examples/train_intent.py):

    from hearth_ai.config import HearthConfig
    from hearth_ai.models import HearthEncoder, HearthModel, IntentHead
    from hearth_ai.tokenizer.hearth_tokenizer import HearthTokenizer
    from hearth_ai.trainer.dataset import HearthDataset, make_collate_fn
    from hearth_ai.trainer.train import Trainer
    import torch.nn as nn
    import torch

    cfg = HearthConfig()
    tok = HearthTokenizer("hearth_ai/tokenizer/hearth_tokenizer.json", cfg.max_seq_len)
    encoder = HearthEncoder(cfg)
    model = HearthModel(encoder, IntentHead(cfg.hidden_size, num_intents=35))

    INTENT_LABELS = ["emotional_support", "goal_sharing", "celebration", ...]

    def label_fn(ex):
        return torch.tensor(INTENT_LABELS.index(ex["intent"]), dtype=torch.long)

    train_ds = HearthDataset("data/intent_train.jsonl", tok, label_fn)
    val_ds = HearthDataset("data/intent_val.jsonl", tok, label_fn)

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        loss_fn=nn.CrossEntropyLoss(),
        pad_id=tok.pad_id,
        checkpoint_dir="checkpoints/intent",
    )
    trainer.fit(epochs=10, batch_size=32, lr=3e-4)
"""
import os
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import make_collate_fn


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        loss_fn: Callable,
        val_dataset=None,
        pad_id: int = 0,
        device: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn
        self.collate_fn = make_collate_fn(pad_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = grad_clip

        self.model.to(self.device)
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _compute_loss(self, logits, label):
        # Handles both plain-tensor heads and dict-output heads (e.g. MemoryHead).
        if isinstance(logits, dict):
            return self.loss_fn(logits, label)
        return self.loss_fn(logits, label)

    def _run_epoch(self, loader, optimizer=None, scheduler=None):
        is_train = optimizer is not None
        self.model.train(is_train)
        total_loss, n_batches = 0.0, 0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            label = batch["label"]
            if isinstance(label, dict):
                label = {k: v.to(self.device) for k, v in label.items()}
            else:
                label = label.to(self.device)

            with torch.set_grad_enabled(is_train):
                logits = self.model(input_ids, attention_mask)
                loss = self._compute_loss(logits, label)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(
        self,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        num_workers: int = 0,
        save_best_only: bool = True,
    ):
        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=self.collate_fn, num_workers=num_workers,
        )
        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=self.collate_fn, num_workers=num_workers,
            )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        best_val = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train_loader, optimizer)
            log = f"epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}"

            val_loss = None
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader)
                log += f" - val_loss: {val_loss:.4f}"
            print(log)

            ckpt_path = os.path.join(self.checkpoint_dir, "last.pt")
            self.save_checkpoint(ckpt_path, epoch)

            if val_loss is not None and (not save_best_only or val_loss < best_val):
                best_val = val_loss
                self.save_checkpoint(os.path.join(self.checkpoint_dir, "best.pt"), epoch)

    def save_checkpoint(self, path: str, epoch: int):
        payload = {"model_state_dict": self.model.state_dict(), "epoch": epoch}
        # Embed HearthConfig when present so ONNX export does not need --smoke heuristics.
        encoder = getattr(self.model, "encoder", None)
        cfg = getattr(encoder, "config", None)
        if cfg is not None:
            from dataclasses import asdict

            payload["config"] = asdict(cfg)
        torch.save(payload, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        return ckpt.get("epoch", 0)
