"""ONNX classifier runtime for hearth_ai exported heads.

Each task artifact under ``NLP_MODELS_DIR/{task}/model.onnx`` is a full
encoder+head graph (inputs: ``input_ids``, ``attention_mask``). Missing
models → ``available=False`` and workers fail-soft to unknown / zeros.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import NLP_MODELS_DIR, resolve_nlp_models_dir
from app.nlp import labels as label_defaults

logger = logging.getLogger(__name__)

EMOTION_THRESHOLD = 0.35
INTENT_MIN_CONFIDENCE = 0.25
STRATEGY_MIN_CONFIDENCE = 0.25
MEMORY_STORE_THRESHOLD = 0.5


@dataclass(frozen=True)
class EmotionPrediction:
    emotion: str
    confidence: float
    raw_labels: list[str]
    scores: dict[str, float]


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float
    scores: dict[str, float]


@dataclass(frozen=True)
class MemoryPrediction:
    store: bool
    memory_type: str
    importance: float
    store_prob: float
    type_scores: dict[str, float]


@dataclass(frozen=True)
class RelationshipPrediction:
    trust_delta: float
    vulnerability: float
    openness: float
    comfort: float


@dataclass(frozen=True)
class StrategyPrediction:
    strategy: str
    confidence: float
    scores: dict[str, float]


NLP_TASKS = ("emotion", "intent", "memory", "relationship", "strategy")


def _read_graph_kind(root: Path) -> str:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return "full"
    return str(manifest.get("graph_kind", "full"))


def classify_available(models_dir: Path | None = None) -> bool:
    """True only when ``models_dir`` can actually run a head.

    Metadata presence is deliberately not sufficient. A source checkout
    tracks manifest.json, tokenizer.json and every per-head config/labels
    file while .gitignore excludes all ``*.onnx``, so a weights-less clone
    would otherwise call itself available, then fail-soft to zeros on every
    turn — and CI would run the golden suite against nothing and report the
    empty result as a model regression.

    The conditions here mirror what ``OnnxClassifier._init`` needs to reach
    ``available = True``, so the two never disagree.
    """
    root = models_dir or NLP_MODELS_DIR or resolve_nlp_models_dir()
    if root is None:
        return False
    if not ((root / "manifest.json").is_file() and (root / "tokenizer.json").is_file()):
        return False
    if _read_graph_kind(root) == "head_only" and not (root / "encoder" / "model.onnx").is_file():
        return False
    return any((root / task / "model.onnx").is_file() for task in NLP_TASKS)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.clip(e.sum(), 1e-12, None)


class OnnxClassifier:
    """Load tokenizer + per-task ONNX sessions; run classification."""

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or NLP_MODELS_DIR or resolve_nlp_models_dir()
        self.available = False
        self._tokenizer = None
        self._max_seq_len = 64
        self._pad_id = 0
        self._sessions: dict[str, Any] = {}
        self._label_meta: dict[str, dict[str, Any]] = {}
        # "full": each task session is a full encoder+head graph (input_ids,
        # attention_mask). "head_only": one shared encoder session (output
        # "pooled") + per-task head sessions (input "pooled") — see
        # hearth_ai's export_all_shared_encoder / nlp-track plan item.
        self._graph_kind = "full"
        self._encoder_session: Any = None
        self._init()

    def _init(self) -> None:
        if self.models_dir is None or not classify_available(self.models_dir):
            logger.info("NLP models not found — classifiers disabled (fail-soft)")
            return
        try:
            from tokenizers import Tokenizer
            import onnxruntime as ort
        except ImportError as exc:
            logger.warning("NLP deps missing (%s) — classifiers disabled", exc)
            return

        manifest_path = self.models_dir / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                logger.exception("Failed to parse manifest.json at %s", manifest_path)
        self._graph_kind = manifest.get("graph_kind", "full")

        tok_path = self.models_dir / "tokenizer.json"
        expected_sha = manifest.get("tokenizer_sha256")
        if expected_sha:
            actual_sha = hashlib.sha256(tok_path.read_bytes()).hexdigest() if tok_path.is_file() else None
            if actual_sha != expected_sha:
                logger.error(
                    "tokenizer.json sha256 mismatch (manifest=%s, on-disk=%s) — "
                    "models_dir may be a partial/corrupt export bundle",
                    expected_sha, actual_sha,
                )
        try:
            self._tokenizer = Tokenizer.from_file(str(tok_path))
            self._pad_id = self._tokenizer.token_to_id("<pad>") or 0
            # Prefer max_seq from any head config.
            cfg_path = self.models_dir / "emotion" / "config.json"
            if cfg_path.is_file():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                self._max_seq_len = int(cfg.get("hearth_config", {}).get("max_seq_len", 64))
            self._tokenizer.enable_truncation(max_length=self._max_seq_len)
            self._tokenizer.enable_padding(
                pad_id=self._pad_id, pad_token="<pad>", length=None
            )
        except Exception:
            logger.exception("Failed to load NLP tokenizer from %s", tok_path)
            return

        if self._graph_kind == "head_only":
            encoder_path = self.models_dir / "encoder" / "model.onnx"
            if not encoder_path.is_file():
                logger.warning("graph_kind=head_only but missing shared encoder at %s", encoder_path)
                return
            try:
                self._encoder_session = ort.InferenceSession(
                    str(encoder_path), providers=["CPUExecutionProvider"]
                )
            except Exception:
                logger.exception("Failed to load shared encoder ONNX session")
                return

        for task in NLP_TASKS:
            onnx_path = self.models_dir / task / "model.onnx"
            labels_path = self.models_dir / task / "labels.json"
            if not onnx_path.is_file():
                logger.warning("Missing ONNX for %s at %s", task, onnx_path)
                continue
            try:
                self._sessions[task] = ort.InferenceSession(
                    str(onnx_path), providers=["CPUExecutionProvider"]
                )
                if labels_path.is_file():
                    self._label_meta[task] = json.loads(labels_path.read_text(encoding="utf-8"))
            except Exception:
                logger.exception("Failed to load ONNX session for %s", task)

        self.available = bool(self._sessions) and (self._graph_kind != "head_only" or self._encoder_session is not None)
        if self.available:
            logger.info(
                "NLP classifiers ready from %s graph_kind=%s (%s)",
                self.models_dir,
                self._graph_kind,
                ", ".join(sorted(self._sessions)),
            )

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        if self._tokenizer is None:
            raise RuntimeError("tokenizer not loaded")
        enc = self._tokenizer.encode(text or "")
        ids = np.asarray([enc.ids], dtype=np.int64)
        mask = np.asarray([enc.attention_mask], dtype=np.int64)
        return ids, mask

    def _encode_pooled(self, text: str) -> np.ndarray | None:
        if self._encoder_session is None or self._tokenizer is None:
            return None
        try:
            ids, mask = self.encode(text)
            return self._encoder_session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        except Exception:
            logger.exception("Shared encoder inference failed")
            return None

    def _run(self, task: str, text: str, *, pooled: np.ndarray | None = None) -> list[np.ndarray] | None:
        session = self._sessions.get(task)
        if session is None or self._tokenizer is None:
            return None
        try:
            if self._graph_kind == "head_only":
                if pooled is None:
                    pooled = self._encode_pooled(text)
                if pooled is None:
                    return None
                return session.run(None, {"pooled": pooled})
            ids, mask = self.encode(text)
            return session.run(None, {"input_ids": ids, "attention_mask": mask})
        except Exception:
            logger.exception("ONNX inference failed for %s", task)
            return None

    def predict_all(self, text: str, tasks: list[str]) -> dict[str, Any]:
        """Run every requested task off ONE shared-encoder pass when
        graph_kind=='head_only' (the nlp-track perf win); falls back to one
        independent full-graph pass per task otherwise — same results
        either way, just not the encoder-sharing speedup."""
        pooled = self._encode_pooled(text) if self._graph_kind == "head_only" else None
        dispatch = {
            "emotion": self.predict_emotion,
            "intent": self.predict_intent,
            "memory": self.predict_memory,
            "relationship": self.predict_relationship,
            "strategy": self.predict_strategy,
        }
        results: dict[str, Any] = {}
        for task in tasks:
            fn = dispatch.get(task)
            if fn is None:
                continue
            results[task] = fn(text, pooled=pooled) if self._graph_kind == "head_only" else fn(text)
        return results

    def predict_emotion(self, text: str, *, pooled: np.ndarray | None = None) -> EmotionPrediction:
        outs = self._run("emotion", text, pooled=pooled)
        if outs is None:
            return EmotionPrediction("unknown", 0.0, [], {})
        logits = np.asarray(outs[0][0], dtype=np.float64)
        meta = self._label_meta.get("emotion", {})
        vocab = list(meta.get("labels") or label_defaults.EMOTION_LABELS)
        go_map = dict(meta.get("go_to_hearth") or label_defaults.EMOTION_GO_TO_HEARTH)
        probs = _sigmoid(logits)
        scores = {lab: float(probs[i]) for i, lab in enumerate(vocab) if i < len(probs)}
        active = [(lab, p) for lab, p in scores.items() if p >= EMOTION_THRESHOLD]
        if not active:
            # Fall back to strongest non-neutral signal, else unknown.
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            if ranked and ranked[0][0] != "neutral" and ranked[0][1] >= 0.2:
                lab, conf = ranked[0]
                return EmotionPrediction(go_map.get(lab, "unknown"), conf, [lab], scores)
            return EmotionPrediction("unknown", float(ranked[0][1]) if ranked else 0.0, [], scores)

        hearth_scores: dict[str, float] = {}
        raw: list[str] = []
        for lab, p in active:
            raw.append(lab)
            hearth = go_map.get(lab, "unknown")
            hearth_scores[hearth] = max(hearth_scores.get(hearth, 0.0), p)
        best = max(hearth_scores, key=hearth_scores.get)
        return EmotionPrediction(best, hearth_scores[best], raw, scores)

    def predict_intent(self, text: str, *, pooled: np.ndarray | None = None) -> IntentPrediction:
        outs = self._run("intent", text, pooled=pooled)
        if outs is None:
            return IntentPrediction("unknown", 0.0, {})
        logits = np.asarray(outs[0][0], dtype=np.float64)
        vocab = list(
            self._label_meta.get("intent", {}).get("labels") or label_defaults.INTENT_LABELS
        )
        probs = _softmax(logits)
        scores = {lab: float(probs[i]) for i, lab in enumerate(vocab) if i < len(probs)}
        best = max(scores, key=scores.get) if scores else "unknown"
        conf = scores.get(best, 0.0)
        if conf < INTENT_MIN_CONFIDENCE:
            return IntentPrediction("unknown", conf, scores)
        return IntentPrediction(best, conf, scores)

    def predict_memory(self, text: str, *, pooled: np.ndarray | None = None) -> MemoryPrediction:
        outs = self._run("memory", text, pooled=pooled)
        types = list(
            self._label_meta.get("memory", {}).get("types") or label_defaults.MEMORY_TYPES
        )
        if outs is None or len(outs) < 3:
            return MemoryPrediction(False, "other", 0.0, 0.0, {})
        store_logit = float(np.asarray(outs[0]).reshape(-1)[0])
        type_logits = np.asarray(outs[1][0], dtype=np.float64)
        importance_logit = float(np.asarray(outs[2]).reshape(-1)[0])
        store_prob = float(_sigmoid(np.asarray([store_logit]))[0])
        type_probs = _softmax(type_logits)
        type_scores = {t: float(type_probs[i]) for i, t in enumerate(types) if i < len(type_probs)}
        mem_type = max(type_scores, key=type_scores.get) if type_scores else "other"
        importance = float(_sigmoid(np.asarray([importance_logit]))[0])
        return MemoryPrediction(
            store=store_prob >= MEMORY_STORE_THRESHOLD,
            memory_type=mem_type,
            importance=importance,
            store_prob=store_prob,
            type_scores=type_scores,
        )

    def predict_relationship(self, text: str, *, pooled: np.ndarray | None = None) -> RelationshipPrediction:
        outs = self._run("relationship", text, pooled=pooled)
        if outs is None:
            return RelationshipPrediction(0.0, 0.0, 0.0, 0.0)
        signals = np.asarray(outs[0][0], dtype=np.float64).reshape(-1)
        # Soft clamp to a gentle delta range for MindState (durable writer elsewhere).
        vals = [float(np.clip(v, -1.0, 1.0)) for v in signals.tolist()]
        while len(vals) < 4:
            vals.append(0.0)
        return RelationshipPrediction(vals[0], vals[1], vals[2], vals[3])

    def predict_strategy(self, text: str, *, pooled: np.ndarray | None = None) -> StrategyPrediction:
        outs = self._run("strategy", text, pooled=pooled)
        if outs is None:
            return StrategyPrediction("listen", 0.0, {})
        logits = np.asarray(outs[0][0], dtype=np.float64)
        vocab = list(
            self._label_meta.get("strategy", {}).get("labels") or label_defaults.STRATEGY_LABELS
        )
        probs = _softmax(logits)
        scores = {lab: float(probs[i]) for i, lab in enumerate(vocab) if i < len(probs)}
        best = max(scores, key=scores.get) if scores else "listen"
        conf = scores.get(best, 0.0)
        if conf < STRATEGY_MIN_CONFIDENCE:
            return StrategyPrediction("listen", conf, scores)
        return StrategyPrediction(best, conf, scores)
