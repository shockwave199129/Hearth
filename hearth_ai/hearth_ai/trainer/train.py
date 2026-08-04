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
import math
import os
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import make_collate_fn


def resolve_device() -> str:
    """Best available torch device: CUDA/ROCm > Apple MPS > CPU.

    ``torch.cuda.is_available()`` also reports True for ROCm builds (ROCm
    is exposed through the same ``cuda`` device namespace), so this covers
    NVIDIA and AMD-on-Linux. Apple Silicon has no CUDA, so without this
    check it silently fell back to CPU instead of using Metal via MPS.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_amp_dtype(amp: str, device: str) -> Optional[torch.dtype]:
    """Autocast dtype for ``amp`` in {"off", "bf16", "fp16", "auto"}, or None.

    ``auto`` picks bf16 on any CUDA device that reports support (Ampere and
    newer, which includes Blackwell/RTX 50-series) and disables AMP on CPU,
    where autocast costs more than it saves at this model size.
    """
    if device != "cuda" or amp == "off":
        return None
    if amp == "bf16":
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    if amp == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    raise ValueError(f"unknown amp mode: {amp!r}")


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
        amp: str = "off",
        grad_accum_steps: int = 1,
        log_every: int = 0,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn
        self.collate_fn = make_collate_fn(pad_id)
        self.device = device or resolve_device()
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = grad_clip
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.log_every = log_every
        self.amp_dtype = resolve_amp_dtype(amp, self.device)
        # fp16 needs loss scaling to avoid underflow; bf16 does not.
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_dtype is torch.float16
        )

        self.model.to(self.device)
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _compute_loss(self, logits, label):
        # Handles both plain-tensor heads and dict-output heads (e.g. MemoryHead).
        if isinstance(logits, dict):
            return self.loss_fn(logits, label)
        return self.loss_fn(logits, label)

    def _autocast(self):
        if self.amp_dtype is None:
            return torch.autocast("cpu", enabled=False)
        return torch.autocast("cuda", dtype=self.amp_dtype)

    def _run_epoch(self, loader, optimizer=None, scheduler=None, epoch_label: str = ""):
        is_train = optimizer is not None
        self.model.train(is_train)
        total_loss, n_batches = 0.0, 0
        accum = self.grad_accum_steps if is_train else 1
        started = time.perf_counter()

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            label = batch["label"]
            if isinstance(label, dict):
                label = {k: v.to(self.device, non_blocking=True) for k, v in label.items()}
            else:
                label = label.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(is_train), self._autocast():
                logits = self.model(input_ids, attention_mask)
                loss = self._compute_loss(logits, label)

            if is_train:
                # Scale so accumulated gradients match a single large batch.
                self.scaler.scale(loss / accum).backward()
                if (step + 1) % accum == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if is_train and self.log_every and n_batches % self.log_every == 0:
                elapsed = time.perf_counter() - started
                rate = n_batches / max(elapsed, 1e-9)
                total = len(loader)
                eta = (total - n_batches) / max(rate, 1e-9)
                print(
                    f"  {epoch_label}step {n_batches}/{total} "
                    f"loss {total_loss / n_batches:.4f} "
                    f"{rate:.1f} it/s eta {eta / 60:.1f}m",
                    flush=True,
                )

        # Flush a trailing partial accumulation window.
        if is_train and n_batches % accum != 0:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        return total_loss / max(n_batches, 1)

    def _build_scheduler(self, optimizer, total_steps: int, warmup_ratio: float):
        """Linear warmup then cosine decay — stabilises a from-scratch encoder."""
        warmup_steps = max(1, int(total_steps * warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def fit(
        self,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        num_workers: int = 0,
        save_best_only: bool = True,
        warmup_ratio: float = 0.0,
        early_stop_patience: int = 0,
    ):
        pin_memory = self.device == "cuda"
        loader_kwargs = {"collate_fn": self.collate_fn, "num_workers": num_workers}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            pin_memory=pin_memory, drop_last=False, **loader_kwargs,
        )
        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset, batch_size=batch_size, shuffle=False,
                pin_memory=pin_memory, **loader_kwargs,
            )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = None
        if warmup_ratio > 0:
            steps_per_epoch = max(1, math.ceil(len(train_loader) / self.grad_accum_steps))
            scheduler = self._build_scheduler(optimizer, steps_per_epoch * epochs, warmup_ratio)

        amp_name = "off" if self.amp_dtype is None else str(self.amp_dtype).replace("torch.", "")
        print(
            f"device={self.device} amp={amp_name} batch={batch_size} "
            f"grad_accum={self.grad_accum_steps} workers={num_workers} "
            f"train_batches={len(train_loader)}",
            flush=True,
        )

        best_val = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(
                train_loader, optimizer, scheduler, epoch_label=f"e{epoch} "
            )
            log = f"epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}"

            val_loss = None
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader)
                log += f" - val_loss: {val_loss:.4f}"
            print(log, flush=True)

            ckpt_path = os.path.join(self.checkpoint_dir, "last.pt")
            self.save_checkpoint(ckpt_path, epoch)

            if val_loss is not None:
                improved = val_loss < best_val
                if improved:
                    best_val = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if improved or not save_best_only:
                    self.save_checkpoint(os.path.join(self.checkpoint_dir, "best.pt"), epoch)

                # Checked outside the save branch on purpose: with
                # save_best_only=False the old code took the save path every
                # epoch and never reached this test, so early stopping silently
                # did nothing.
                if early_stop_patience and epochs_without_improvement >= early_stop_patience:
                    print(
                        f"early stop: no val improvement for {epochs_without_improvement} "
                        f"epoch(s) (patience {early_stop_patience})",
                        flush=True,
                    )
                    break

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


class MultiTaskTrainer:
    """Joint trainer for ``HearthMultiTaskModel`` — one encoder shared by all
    task heads, trained together so the encoder never diverges per task.

    This is the failure mode ``train_all_full.py``'s sequential warm-start
    chain has: each later head's fine-tuning only backprops its own task's
    loss through the (copied) encoder, so nothing anchors it back to the
    earlier tasks and the five encoders drift apart — measured cosine
    similarity between task-encoder pooled embeddings on the same input can
    be as low as ~0.05. Training every head against the *same* encoder
    instance every step, with every task's gradient landing on it every
    step, is what keeps the heads' embedding space aligned.

    Every step pulls one batch per task (from an infinitely-cycling loader
    per task, since dataset sizes differ), runs the shared encoder once per
    task-batch, computes that task's loss, and backprops the mean of all
    task losses in a single ``backward()`` before one optimizer step —
    so every parameter update already reflects every task's gradient.
    """

    def __init__(
        self,
        model,
        train_datasets: dict,
        loss_fns: dict,
        val_datasets: Optional[dict] = None,
        pad_id: int = 0,
        device: Optional[str] = None,
        checkpoint_root: str = "checkpoints",
        grad_clip: float = 1.0,
        log_every: int = 0,
    ):
        unknown = set(train_datasets) - set(model.heads)
        if unknown:
            raise ValueError(f"train_datasets has tasks not in model.heads: {sorted(unknown)}")
        self.model = model
        self.tasks = list(train_datasets.keys())
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets or {}
        self.loss_fns = loss_fns
        self.collate_fn = make_collate_fn(pad_id)
        self.device = device or resolve_device()
        self.checkpoint_root = checkpoint_root
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.model.to(self.device)
        os.makedirs(checkpoint_root, exist_ok=True)

    def _to_device(self, label):
        if isinstance(label, dict):
            return {k: v.to(self.device, non_blocking=True) for k, v in label.items()}
        return label.to(self.device, non_blocking=True)

    def _cycle(self, loader):
        while True:
            for batch in loader:
                yield batch

    def _forward_task(self, task: str, batch) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        label = self._to_device(batch["label"])
        _, pooled = self.model.encoder(input_ids, attention_mask)
        logits = self.model.heads[task](pooled)
        if isinstance(logits, dict):
            return self.loss_fns[task](logits, label)
        return self.loss_fns[task](logits, label)

    def _build_scheduler(self, optimizer, total_steps: int, warmup_ratio: float):
        warmup_steps = max(1, int(total_steps * warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def fit(
        self,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        steps_per_epoch: Optional[int] = None,
        warmup_ratio: float = 0.0,
        early_stop_patience: int = 0,
        num_workers: int = 0,
    ) -> dict:
        loaders = {
            t: DataLoader(
                ds, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn,
                num_workers=num_workers, drop_last=True,
            )
            for t, ds in self.train_datasets.items()
        }
        steps_per_epoch = steps_per_epoch or max(len(dl) for dl in loaders.values())
        iters = {t: self._cycle(dl) for t, dl in loaders.items()}

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = None
        if warmup_ratio > 0:
            scheduler = self._build_scheduler(optimizer, steps_per_epoch * epochs, warmup_ratio)

        print(
            f"device={self.device} tasks={self.tasks} batch={batch_size} "
            f"steps_per_epoch={steps_per_epoch}",
            flush=True,
        )

        best_val = float("inf")
        epochs_without_improvement = 0
        last_train_losses: dict = {}

        for epoch in range(1, epochs + 1):
            self.model.train()
            totals = {t: 0.0 for t in self.tasks}
            started = time.perf_counter()

            for step in range(1, steps_per_epoch + 1):
                optimizer.zero_grad(set_to_none=True)
                for t in self.tasks:
                    batch = next(iters[t])
                    loss = self._forward_task(t, batch)
                    (loss / len(self.tasks)).backward()
                    totals[t] += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                if self.log_every and step % self.log_every == 0:
                    elapsed = time.perf_counter() - started
                    rate = step / max(elapsed, 1e-9)
                    eta = (steps_per_epoch - step) / max(rate, 1e-9)
                    avg = ", ".join(f"{t}={totals[t] / step:.4f}" for t in self.tasks)
                    print(f"  e{epoch} step {step}/{steps_per_epoch} {avg} {rate:.2f} it/s eta {eta / 60:.1f}m", flush=True)

            last_train_losses = {t: totals[t] / steps_per_epoch for t in self.tasks}
            log = f"epoch {epoch}/{epochs} - train: " + ", ".join(f"{t}={v:.4f}" for t, v in last_train_losses.items())

            val_loss = None
            if self.val_datasets:
                val_loss = self._evaluate(num_workers=num_workers)
                log += f" - val_avg: {val_loss:.4f}"
            print(log, flush=True)

            self.save_checkpoints(epoch, best=False)

            if val_loss is not None:
                improved = val_loss < best_val
                if improved:
                    best_val = val_loss
                    epochs_without_improvement = 0
                    self.save_checkpoints(epoch, best=True)
                else:
                    epochs_without_improvement += 1
                if early_stop_patience and epochs_without_improvement >= early_stop_patience:
                    print(f"early stop: no val improvement for {epochs_without_improvement} epoch(s)", flush=True)
                    break
            else:
                self.save_checkpoints(epoch, best=True)

        return {"train_losses": last_train_losses, "best_val": best_val if self.val_datasets else None}

    def _evaluate(self, num_workers: int = 0) -> float:
        self.model.eval()
        per_task_avg = []
        with torch.no_grad():
            for t, ds in self.val_datasets.items():
                loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=self.collate_fn, num_workers=num_workers)
                total, n = 0.0, 0
                for batch in loader:
                    loss = self._forward_task(t, batch)
                    total += loss.item()
                    n += 1
                per_task_avg.append(total / max(n, 1))
        return sum(per_task_avg) / max(len(per_task_avg), 1)

    def save_checkpoints(self, epoch: int, *, best: bool) -> None:
        """One checkpoint per task, each shaped exactly like a single-task
        ``HearthModel(encoder, head)`` state dict (``encoder.*`` / ``head.*``
        keys) — every existing ``onnx_export.py`` loader works unmodified.
        All five embed byte-identical encoder weights, since every task's
        head was trained against the *same* encoder ``nn.Module`` instance.
        """
        from dataclasses import asdict

        name = "best.pt" if best else "last.pt"
        cfg = self.model.encoder.config
        enc_state = self.model.encoder.state_dict()
        for t in self.tasks:
            state = {f"encoder.{k}": v for k, v in enc_state.items()}
            state.update({f"head.{k}": v for k, v in self.model.heads[t].state_dict().items()})
            out_dir = os.path.join(self.checkpoint_root, t)
            os.makedirs(out_dir, exist_ok=True)
            torch.save({"model_state_dict": state, "epoch": epoch, "config": asdict(cfg)}, os.path.join(out_dir, name))
