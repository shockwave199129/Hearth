"""Fallback label vocabularies when ``labels.json`` is missing from a head package.

Canonical source of truth for training is ``hearth_ai/labels/``; runtime prefers
the JSON shipped next to each ONNX artifact.
"""

from __future__ import annotations

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

# Map intent → MindState.goal (book-facing companion goal string).
INTENT_TO_GOAL: dict[str, str] = {
    "vent": "listen",
    "validate": "listen",
    "comfort": "support",
    "celebrate": "celebrate",
    "advise": "advise",
    "inquire": "understand",
    "plan": "plan",
    "small_talk": "listen",
    "meta": "listen",
    "unknown": "listen",
}

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

RELATIONSHIP_SIGNALS: list[str] = [
    "trust_delta",
    "vulnerability",
    "openness",
    "comfort",
]

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
