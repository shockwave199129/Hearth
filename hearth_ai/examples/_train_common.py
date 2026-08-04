"""Shared helpers for emotion/intent training examples."""

from __future__ import annotations

import json
from pathlib import Path

from hearth_ai.config import HearthConfig
from hearth_ai.tokenizer.hearth_tokenizer import HearthTokenizer, train_tokenizer


ROOT = Path(__file__).resolve().parents[1]


def ensure_prepared_data(task: str, max_per_source: int = 500) -> Path:
    """Ensure ``data/{task}/{train,val,test}.jsonl`` exist; prepare if missing."""
    out = ROOT / "data" / task
    train = out / "train.jsonl"
    if train.is_file() and train.stat().st_size > 0:
        return out
    print(f"Prepared {task} data missing — running prepare…")
    import subprocess
    import sys

    mod = f"data.prepare.prepare_{task}"
    cmd = [sys.executable, "-m", mod]
    if task in {"emotion", "intent"}:
        cmd.extend(["--max-per-source", str(max_per_source)])
    subprocess.check_call(cmd, cwd=str(ROOT))
    return out


def jsonl_to_text_corpus(jsonl_paths: list[Path], corpus_path: Path) -> Path:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as out:
        for path in jsonl_paths:
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    text = json.loads(line).get("text") or ""
                    if text.strip():
                        out.write(text.replace("\n", " ") + "\n")
    return corpus_path


def build_or_load_tokenizer(
    jsonl_paths: list[Path],
    tokenizer_path: Path,
    *,
    vocab_size: int,
    max_seq_len: int,
    force_retrain: bool = False,
) -> HearthTokenizer:
    if force_retrain or not tokenizer_path.is_file():
        corpus = tokenizer_path.with_suffix(".corpus.txt")
        jsonl_to_text_corpus(jsonl_paths, corpus)
        train_tokenizer(
            [str(corpus)],
            str(tokenizer_path),
            vocab_size=vocab_size,
            min_frequency=1,
        )
    return HearthTokenizer(str(tokenizer_path), max_seq_len=max_seq_len)


def make_config(*, smoke: bool, vocab_size: int, max_seq_len: int | None = None) -> HearthConfig:
    if smoke:
        return HearthConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len or 64,
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            head_dim=32,
            ffn_hidden_size=256,
        )
    return HearthConfig(vocab_size=vocab_size, max_seq_len=max_seq_len or 128)


def add_runtime_args(parser) -> None:
    """GPU / dataloader flags shared by every train_*.py script."""
    parser.add_argument(
        "--amp",
        choices=["off", "auto", "bf16", "fp16"],
        default="auto",
        help="Mixed precision on CUDA (auto = bf16 where supported)",
    )
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; keep 0 on Windows unless verified (spawn + tokenizer pickling)",
    )
    parser.add_argument("--max-seq", type=int, default=0, help="Override sequence length")
    parser.add_argument("--vocab-size", type=int, default=0, help="Override tokenizer vocab target")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100, help="0 = only per-epoch logs")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Cap train rows loaded from JSONL (0 = all)",
    )
    parser.add_argument(
        "--keep-tokenizer",
        action="store_true",
        help="Never retrain the tokenizer — required when sharing one across heads",
    )


def trainer_kwargs(args) -> dict:
    """Trainer(...) kwargs from the shared runtime flags."""
    return {
        "amp": args.amp,
        "grad_accum_steps": args.grad_accum,
        "log_every": args.log_every,
    }


def fit_kwargs(args) -> dict:
    """Trainer.fit(...) kwargs from the shared runtime flags."""
    return {
        "num_workers": args.num_workers,
        "warmup_ratio": args.warmup_ratio,
        "early_stop_patience": args.early_stop_patience,
    }


def resolve_sizing(args, *, smoke: bool) -> tuple[int, int]:
    """(vocab_size, max_seq_len) honouring overrides, else smoke/full defaults."""
    vocab_size = args.vocab_size or (4000 if smoke else 32000)
    max_seq = args.max_seq or (64 if smoke else 128)
    return vocab_size, max_seq


def encode_for_model(model, tok, text: str):
    """Tokenize ``text`` as a batch of 1 on whatever device ``model`` is on.

    Trainer moves the model to CUDA, so the demo tensors built after ``fit()``
    have to follow it rather than defaulting to CPU. Read off the parameters
    instead of re-deriving from ``cuda.is_available()`` so an explicit
    ``Trainer(device=...)`` stays authoritative.
    """
    import torch

    ids, mask = tok.encode(text)
    device = next(model.parameters()).device
    return (
        torch.tensor([ids], device=device),
        torch.tensor([mask], device=device),
    )


def load_encoder_weights(model, checkpoint_path: str, device: str) -> None:
    """Load encoder weights from a prior HearthModel checkpoint (warm start)."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"]
    enc_state = {
        k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")
    }
    missing, unexpected = model.encoder.load_state_dict(enc_state, strict=False)
    if missing:
        print(f"warm-start missing keys: {missing[:5]}...")
    if unexpected:
        print(f"warm-start unexpected keys: {unexpected[:5]}...")
    print(f"Warm-started encoder from {checkpoint_path}")
