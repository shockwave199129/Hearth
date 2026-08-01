"""Rule-based memory formation (Book Vol 4 Ch 8) — five triggers, entirely
Python/regex/classifier-based, no LLM judgment call:

  1. new entity detected
  2. high emotional intensity (real classifier score, not a keyword guess)
  3. explicit significance markers
  4. repetition of a topic across the conversation
  5. explicit request to remember

Silence is the default: if no trigger fires for a message, nothing is
stored — this module only produces candidates; app.memory2.store persists
them."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from app.memory2.entities import extract_entities, extract_significance_markers
from app.memory2.models import EmotionalMetadata
from app.nlp.runtime import OnnxClassifier

EMOTIONAL_INTENSITY_THRESHOLD = 0.55
REPETITION_THRESHOLD = 2  # same entity mentioned 2+ times in one session

_POSITIVE_EMOTIONS = {"joy", "pride", "gratitude", "hope", "love"}
_NEGATIVE_EMOTIONS = {"sadness", "fear", "anger", "disgust", "guilt", "grief", "anxiety"}
_SENSITIVE_INTENSITY_FLOOR = 0.6


@dataclass(frozen=True)
class FormationCandidate:
    text: str
    entities: list[str] = field(default_factory=list)
    markers: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    emotion_category: str = "neutral"
    emotion_intensity: float = 0.0


def _classify_emotion(classifier: OnnxClassifier, text: str) -> tuple[str, float]:
    if not classifier.available:
        return "neutral", 0.0
    pred = classifier.predict_emotion(text)
    return pred.emotion, pred.confidence


def find_candidates(
    user_messages: list[str], classifier: OnnxClassifier | None = None
) -> list[FormationCandidate]:
    """Evaluates each user message in a session against the five formation
    triggers. Repetition is evaluated across the whole session, so an
    entity mentioned earlier makes a later mention trigger even if that
    later mention wouldn't otherwise stand out."""
    classifier = classifier or OnnxClassifier()
    per_message_entities = [extract_entities(text) for text in user_messages]
    entity_counts = Counter(e for ents in per_message_entities for e in ents)
    seen_entities: set[str] = set()

    candidates: list[FormationCandidate] = []
    for text, entities in zip(user_messages, per_message_entities):
        markers = extract_significance_markers(text)
        emotion, confidence = _classify_emotion(classifier, text)
        triggers: list[str] = []

        new_entities = [e for e in entities if e not in seen_entities]
        if new_entities:
            triggers.append("new_entity")
        if confidence >= EMOTIONAL_INTENSITY_THRESHOLD and emotion not in ("neutral", "unknown"):
            triggers.append("high_emotional_intensity")
        if markers:
            triggers.append("significance_marker")
        if any(entity_counts[e] >= REPETITION_THRESHOLD for e in entities):
            triggers.append("repetition")
        if "explicit_request" in markers:
            triggers.append("explicit_request")

        seen_entities.update(entities)
        if triggers:
            candidates.append(
                FormationCandidate(
                    text=text,
                    entities=entities,
                    markers=markers,
                    triggers=triggers,
                    emotion_category=emotion,
                    emotion_intensity=confidence,
                )
            )
    return candidates


def _primary_entity_or_topic(candidate: FormationCandidate) -> str:
    if candidate.entities:
        return candidate.entities[0]
    if candidate.markers:
        return candidate.markers[0].replace("_", " ")
    return "conversation"


def _primary_marker(candidate: FormationCandidate) -> str:
    if candidate.markers:
        return candidate.markers[0].replace("_", " ")
    return candidate.triggers[0].replace("_", " ")


def build_summary(candidate: FormationCandidate) -> str:
    """Templated, never free-form LLM prose (Vol 4 Ch 5):
    '{entity_or_topic} — {emotion_label} — {significance_marker}'."""
    entity = _primary_entity_or_topic(candidate)
    emotion = candidate.emotion_category if candidate.emotion_category not in ("neutral", "unknown") else "neutral"
    marker = _primary_marker(candidate)
    return f"{entity} — {emotion} — {marker}"


def build_emotional_metadata(candidate: FormationCandidate) -> EmotionalMetadata:
    if candidate.emotion_category in _POSITIVE_EMOTIONS:
        valence = "positive"
    elif candidate.emotion_category in _NEGATIVE_EMOTIONS:
        valence = "negative"
    elif candidate.emotion_category in ("neutral", "unknown"):
        valence = "neutral"
    else:
        valence = "mixed"
    sensitivity = candidate.emotion_category in _NEGATIVE_EMOTIONS and candidate.emotion_intensity >= _SENSITIVE_INTENSITY_FLOOR
    return EmotionalMetadata(
        valence=valence,
        intensity=round(candidate.emotion_intensity, 4),
        category=candidate.emotion_category,
        sensitivity_flag=sensitivity,
    )
