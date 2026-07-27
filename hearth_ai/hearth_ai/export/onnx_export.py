"""Export HearthEncoder + task heads to ONNX with PyTorch↔ORT parity checks.

Package layout (plan ``models/nlp/``)::

    models/nlp/
      tokenizer.json
      manifest.json
      encoder/model.onnx + config.json
      emotion|intent|memory|relationship|strategy/
        model.onnx + config.json + labels.json
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..config import HearthConfig
from ..labels import (
    EMOTION_GO_TO_HEARTH,
    EMOTION_HEARTH_DISPLAY,
    EMOTION_LABELS,
    INTENT_LABELS,
    MEMORY_TYPES,
    RELATIONSHIP_SIGNALS,
    STRATEGY_LABELS,
)
from ..models import (
    EmotionHead,
    HearthEncoder,
    HearthModel,
    IntentHead,
    MemoryHead,
    RelationshipHead,
    StrategyHead,
)

PARITY_ATOL = 1e-4
DEFAULT_OPSET = 17

TASK_HEADS: dict[str, type[nn.Module]] = {
    "emotion": EmotionHead,
    "intent": IntentHead,
    "memory": MemoryHead,
    "relationship": RelationshipHead,
    "strategy": StrategyHead,
}

TASK_LABEL_PAYLOAD: dict[str, dict[str, Any]] = {
    "emotion": {
        "task": "emotion",
        "labels": EMOTION_LABELS,
        "hearth_display": EMOTION_HEARTH_DISPLAY,
        "go_to_hearth": EMOTION_GO_TO_HEARTH,
        "num_labels": len(EMOTION_LABELS),
        "output": "logits",
        "loss": "BCEWithLogits",
    },
    "intent": {
        "task": "intent",
        "labels": INTENT_LABELS,
        "num_labels": len(INTENT_LABELS),
        "output": "logits",
        "loss": "CrossEntropy",
    },
    "memory": {
        "task": "memory",
        "types": MEMORY_TYPES,
        "num_types": len(MEMORY_TYPES),
        "outputs": ["store_logit", "type_logits", "importance_logit"],
        "loss": "MemoryLoss",
    },
    "relationship": {
        "task": "relationship",
        "signals": RELATIONSHIP_SIGNALS,
        "num_signals": len(RELATIONSHIP_SIGNALS),
        "output": "signals",
        "loss": "SmoothL1",
    },
    "strategy": {
        "task": "strategy",
        "labels": STRATEGY_LABELS,
        "num_labels": len(STRATEGY_LABELS),
        "output": "logits",
        "loss": "CrossEntropy",
    },
}


class _EncoderPooled(nn.Module):
    def __init__(self, encoder: HearthEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        _, pooled = self.encoder(input_ids, attention_mask)
        return pooled


class _MemoryOnnx(nn.Module):
    """MemoryHead returns a dict; ONNX needs a fixed tuple of tensors."""

    def __init__(self, model: HearthModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.model(input_ids, attention_mask)
        return out["store_logit"], out["type_logits"], out["importance_logit"]


@dataclass
class ExportResult:
    name: str
    onnx_path: Path
    max_abs_diff: float
    passed: bool


def config_to_dict(config: HearthConfig) -> dict[str, Any]:
    return config.to_dict()


def config_from_dict(data: dict[str, Any]) -> HearthConfig:
    return HearthConfig.from_dict(data)


def load_checkpoint_state(path: Path | str) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    cfg = ckpt.get("config")
    return state, cfg if isinstance(cfg, dict) else None


def build_task_model(task: str, config: HearthConfig) -> HearthModel:
    if task not in TASK_HEADS:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASK_HEADS)}")
    head_cls = TASK_HEADS[task]
    return HearthModel(HearthEncoder(config), head_cls(config.hidden_size))


def load_task_model(
    task: str,
    checkpoint: Path | str,
    config: HearthConfig | None = None,
) -> HearthModel:
    state, ckpt_cfg = load_checkpoint_state(checkpoint)
    if config is None:
        if ckpt_cfg is None:
            raise ValueError(
                f"Checkpoint {checkpoint} has no embedded config; pass an explicit HearthConfig "
                "(e.g. smoke sizing matching training)."
            )
        config = config_from_dict(ckpt_cfg)
    model = build_task_model(task, config)
    model.load_state_dict(state)
    model.eval()
    return model


def _dynamic_axes(output_names: list[str]) -> dict[str, dict[int, str]]:
    axes: dict[str, dict[int, str]] = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
    }
    for name in output_names:
        axes[name] = {0: "batch"}
    return axes


def _export_module(
    module: nn.Module,
    path: Path,
    *,
    sample_ids: torch.Tensor,
    sample_mask: torch.Tensor,
    output_names: list[str],
    opset: int = DEFAULT_OPSET,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            (sample_ids, sample_mask),
            str(path),
            input_names=["input_ids", "attention_mask"],
            output_names=output_names,
            dynamic_axes=_dynamic_axes(output_names),
            opset_version=opset,
            dynamo=False,
        )


def _ort_run(onnx_path: Path, input_ids: np.ndarray, attention_mask: np.ndarray) -> list[np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess.run(
        None,
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )


def parity_check(
    pytorch_outputs: list[np.ndarray],
    onnx_outputs: list[np.ndarray],
    *,
    atol: float = PARITY_ATOL,
) -> float:
    if len(pytorch_outputs) != len(onnx_outputs):
        raise ValueError(
            f"output count mismatch: pytorch={len(pytorch_outputs)} onnx={len(onnx_outputs)}"
        )
    max_diff = 0.0
    for pt, ox in zip(pytorch_outputs, onnx_outputs):
        max_diff = max(max_diff, float(np.max(np.abs(pt - ox))))
    if max_diff > atol:
        raise AssertionError(f"ONNX parity failed: max abs diff {max_diff:.6g} > atol {atol}")
    return max_diff


def package_labels(task: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "labels.json"
    path.write_text(json.dumps(TASK_LABEL_PAYLOAD[task], indent=2) + "\n", encoding="utf-8")
    return path


def write_config(out_dir: Path, config: HearthConfig, **extra: Any) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"hearth_config": config_to_dict(config), **extra}
    path = out_dir / "config.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def export_encoder(
    encoder: HearthEncoder,
    out_dir: Path,
    *,
    sample_ids: torch.Tensor,
    sample_mask: torch.Tensor,
    opset: int = DEFAULT_OPSET,
    atol: float = PARITY_ATOL,
) -> ExportResult:
    out_dir = Path(out_dir)
    onnx_path = out_dir / "model.onnx"
    wrap = _EncoderPooled(encoder)
    wrap.eval()
    with torch.no_grad():
        pt = [wrap(sample_ids, sample_mask).detach().cpu().numpy()]
    _export_module(
        wrap,
        onnx_path,
        sample_ids=sample_ids,
        sample_mask=sample_mask,
        output_names=["pooled"],
        opset=opset,
    )
    ort_out = _ort_run(onnx_path, sample_ids.numpy(), sample_mask.numpy())
    max_diff = parity_check(pt, ort_out, atol=atol)
    write_config(
        out_dir,
        encoder.config,
        artifact="encoder",
        inputs=["input_ids", "attention_mask"],
        outputs=["pooled"],
        opset=opset,
        parity_max_abs_diff=max_diff,
    )
    return ExportResult("encoder", onnx_path, max_diff, True)


def export_task(
    task: str,
    model: HearthModel,
    out_dir: Path,
    *,
    sample_ids: torch.Tensor,
    sample_mask: torch.Tensor,
    opset: int = DEFAULT_OPSET,
    atol: float = PARITY_ATOL,
) -> ExportResult:
    out_dir = Path(out_dir)
    onnx_path = out_dir / "model.onnx"
    model.eval()

    if task == "memory":
        wrap: nn.Module = _MemoryOnnx(model)
        output_names = ["store_logit", "type_logits", "importance_logit"]
    else:
        wrap = model
        output_names = ["logits"]

    wrap.eval()
    with torch.no_grad():
        raw = wrap(sample_ids, sample_mask)
        if isinstance(raw, tuple):
            pt = [t.detach().cpu().numpy() for t in raw]
        else:
            pt = [raw.detach().cpu().numpy()]

    _export_module(
        wrap,
        onnx_path,
        sample_ids=sample_ids,
        sample_mask=sample_mask,
        output_names=output_names,
        opset=opset,
    )
    ort_out = _ort_run(onnx_path, sample_ids.numpy(), sample_mask.numpy())
    max_diff = parity_check(pt, ort_out, atol=atol)

    write_config(
        out_dir,
        model.encoder.config,
        artifact=task,
        inputs=["input_ids", "attention_mask"],
        outputs=output_names,
        opset=opset,
        parity_max_abs_diff=max_diff,
    )
    package_labels(task, out_dir)
    return ExportResult(task, onnx_path, max_diff, True)


def export_all(
    checkpoint_root: Path | str,
    out_root: Path | str,
    config: HearthConfig,
    *,
    tokenizer_src: Path | str | None = None,
    tasks: tuple[str, ...] = (
        "emotion",
        "intent",
        "memory",
        "relationship",
        "strategy",
    ),
    encoder_from: str = "emotion",
    batch_size: int = 2,
    seq_len: int = 24,
    opset: int = DEFAULT_OPSET,
    atol: float = PARITY_ATOL,
    seed: int = 0,
) -> list[ExportResult]:
    """Load best checkpoints, export ONNX package, run parity for each artifact."""
    checkpoint_root = Path(checkpoint_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = torch.Generator().manual_seed(seed)
    sample_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), dtype=torch.long, generator=rng
    )
    sample_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    results: list[ExportResult] = []
    loaded: dict[str, HearthModel] = {}

    for task in tasks:
        ckpt = checkpoint_root / task / "best.pt"
        if not ckpt.is_file():
            ckpt = checkpoint_root / task / "last.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"No checkpoint for {task} under {checkpoint_root / task}")
        model = load_task_model(task, ckpt, config)
        loaded[task] = model
        results.append(
            export_task(
                task,
                model,
                out_root / task,
                sample_ids=sample_ids,
                sample_mask=sample_mask,
                opset=opset,
                atol=atol,
            )
        )

    if encoder_from not in loaded:
        raise ValueError(f"encoder_from={encoder_from!r} not in exported tasks {list(loaded)}")
    results.insert(
        0,
        export_encoder(
            loaded[encoder_from].encoder,
            out_root / "encoder",
            sample_ids=sample_ids,
            sample_mask=sample_mask,
            opset=opset,
            atol=atol,
        ),
    )

    if tokenizer_src is not None:
        src = Path(tokenizer_src)
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, out_root / "tokenizer.json")

    manifest = {
        "opset": opset,
        "parity_atol": atol,
        "hearth_config": config_to_dict(config),
        "artifacts": [
            {
                "name": r.name,
                "onnx": str(r.onnx_path.relative_to(out_root)),
                "max_abs_diff": r.max_abs_diff,
                "passed": r.passed,
            }
            for r in results
        ],
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return results
