"""Locked label schemas for all five Hearth task heads.

Canonical human-readable specs: ``hearth_ai/labels/*.yaml``.
This module is the importable source of truth for training code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_LABELS_DIR = Path(__file__).resolve().parents[2] / "labels"

# --- Emotion (GoEmotions 27 + neutral = 28, multi-label BCE) ---

EMOTION_LABELS: list[str] = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

EMOTION_HEARTH_DISPLAY: list[str] = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "anxiety",
    "disgust",
    "surprise",
    "gratitude",
    "loneliness",
    "guilt",
    "pride",
    "hope",
    "neutral",
    "unknown",
]

EMOTION_GO_TO_HEARTH: dict[str, str] = {
    "admiration": "pride",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "pride",
    "caring": "gratitude",
    "confusion": "anxiety",
    "curiosity": "hope",
    "desire": "hope",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "disgust",
    "embarrassment": "guilt",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "gratitude",
    "grief": "sadness",
    "joy": "joy",
    "love": "joy",
    "nervousness": "anxiety",
    "optimism": "hope",
    "pride": "pride",
    "realization": "surprise",
    "relief": "joy",
    "remorse": "guilt",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}

# --- Intent (10 companion needs, single-label CE) ---

INTENT_LABELS: list[str] = [
    "vent",
    "validate",
    "comfort",
    "celebrate",
    "advise",
    "inquire",
    "plan",
    "small_talk",
    "meta",
    "unknown",
]

# --- Memory (composite: store + type + importance) ---

MEMORY_TYPES: list[str] = [
    "episodic",
    "semantic",
    "emotional",
    "preference",
    "goal",
    "boundary",
    "person",
    "other",
]

# --- Relationship (4 regression signals) ---

RELATIONSHIP_SIGNALS: list[str] = [
    "trust_delta",
    "vulnerability",
    "openness",
    "comfort",
]

# --- Strategy (12 intervention suggestions, single-label CE) ---

STRATEGY_LABELS: list[str] = [
    "listen",
    "validate",
    "reflect",
    "comfort",
    "encourage",
    "celebrate",
    "advise",
    "ask_question",
    "plan",
    "ground",
    "boundary",
    "defer_safety",
]


def labels_dir() -> Path:
    return _LABELS_DIR


def label_to_id(labels: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(labels)}


def id_to_label(labels: list[str]) -> dict[int, str]:
    return {i: name for i, name in enumerate(labels)}


def emotion_num_labels() -> int:
    return len(EMOTION_LABELS)


def intent_num_labels() -> int:
    return len(INTENT_LABELS)


def memory_num_types() -> int:
    return len(MEMORY_TYPES)


def relationship_num_signals() -> int:
    return len(RELATIONSHIP_SIGNALS)


def strategy_num_labels() -> int:
    return len(STRATEGY_LABELS)


def multi_hot(active: list[str], vocabulary: list[str] | None = None) -> list[float]:
    """Build a multi-hot vector for EmotionHead targets."""
    vocabulary = vocabulary or EMOTION_LABELS
    index = label_to_id(vocabulary)
    vec = [0.0] * len(vocabulary)
    for name in active:
        if name not in index:
            raise KeyError(f"Unknown label {name!r}; not in vocabulary")
        vec[index[name]] = 1.0
    return vec


def map_goemotions_to_hearth(active: list[str]) -> str:
    """Map active GoEmotions labels to a single Hearth display emotion."""
    if not active:
        return "unknown"
    for name in active:
        if name == "neutral":
            continue
        mapped = EMOTION_GO_TO_HEARTH.get(name)
        if mapped:
            return mapped
    if "neutral" in active:
        return "neutral"
    return "unknown"


def load_schema(task: str) -> dict[str, Any]:
    """Load ``labels/<task>.yaml`` when PyYAML is installed; else return a summary dict."""
    path = _LABELS_DIR / f"{task}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing label schema: {path}")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} did not parse to a mapping")
        return data
    except ImportError:
        summaries = {
            "emotion": {
                "task": "emotion",
                "labels": EMOTION_LABELS,
                "hearth_display": EMOTION_HEARTH_DISPLAY,
                "num_labels": len(EMOTION_LABELS),
            },
            "intent": {
                "task": "intent",
                "labels": INTENT_LABELS,
                "num_labels": len(INTENT_LABELS),
            },
            "memory": {
                "task": "memory",
                "types": MEMORY_TYPES,
                "num_types": len(MEMORY_TYPES),
            },
            "relationship": {
                "task": "relationship",
                "signals": RELATIONSHIP_SIGNALS,
                "num_signals": len(RELATIONSHIP_SIGNALS),
            },
            "strategy": {
                "task": "strategy",
                "labels": STRATEGY_LABELS,
                "num_labels": len(STRATEGY_LABELS),
            },
        }
        if task not in summaries:
            raise KeyError(task)
        return summaries[task]


def assert_schema_files_exist() -> None:
    for name in ("emotion", "intent", "memory", "relationship", "strategy"):
        path = _LABELS_DIR / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Missing locked schema file: {path}")


def assert_counts() -> None:
    """Hard-lock head sizes from the plan."""
    assert len(EMOTION_LABELS) == 28, len(EMOTION_LABELS)
    assert len(EMOTION_HEARTH_DISPLAY) == 14, len(EMOTION_HEARTH_DISPLAY)
    assert len(INTENT_LABELS) == 10, len(INTENT_LABELS)
    assert len(MEMORY_TYPES) == 8, len(MEMORY_TYPES)
    assert len(RELATIONSHIP_SIGNALS) == 4, len(RELATIONSHIP_SIGNALS)
    assert len(STRATEGY_LABELS) == 12, len(STRATEGY_LABELS)
    assert_schema_files_exist()


assert_counts()


def memory_example_to_tensors(ex: dict):
    """Convert a memory JSONL row to the dict MemoryLoss expects (as Python floats/ints).

    Trainers should wrap with torch.tensor — kept free of torch here for light imports.
    """
    type_name = ex["type"]
    if isinstance(type_name, int):
        type_id = type_name
    else:
        type_id = MEMORY_TYPES.index(type_name)
    return {
        "store": float(ex["store"]),
        "type": int(type_id),
        "importance": float(ex["importance"]),
    }


def relationship_example_to_list(ex: dict) -> list[float]:
    signals = ex["signals"]
    if len(signals) != len(RELATIONSHIP_SIGNALS):
        raise ValueError(f"expected {len(RELATIONSHIP_SIGNALS)} signals, got {len(signals)}")
    return [float(x) for x in signals]


def strategy_example_to_id(ex: dict) -> int:
    return STRATEGY_LABELS.index(ex["strategy"])


__all__ = [
    "labels_dir",
    "load_schema",
    "assert_schema_files_exist",
    "assert_counts",
    "EMOTION_LABELS",
    "EMOTION_HEARTH_DISPLAY",
    "EMOTION_GO_TO_HEARTH",
    "INTENT_LABELS",
    "MEMORY_TYPES",
    "RELATIONSHIP_SIGNALS",
    "STRATEGY_LABELS",
    "label_to_id",
    "id_to_label",
    "emotion_num_labels",
    "intent_num_labels",
    "memory_num_types",
    "relationship_num_signals",
    "strategy_num_labels",
    "multi_hot",
    "map_goemotions_to_hearth",
    "memory_example_to_tensors",
    "relationship_example_to_list",
    "strategy_example_to_id",
]
