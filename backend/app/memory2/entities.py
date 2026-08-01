"""Rule-based entity + significance-marker extraction (Book Vol 4 Ch 5) —
no NER model, no LLM call. Covers the two shapes Hearth actually needs to
recognize: an explicit relationship noun ("my sister") and a proper noun
used mid-sentence. Volume 3 Ch 8's Life Model consumes the same extracted
entities this module produces."""
from __future__ import annotations

import re

_RELATION_NOUNS = (
    "sister", "brother", "mother", "father", "mom", "dad", "wife", "husband",
    "partner", "boyfriend", "girlfriend", "friend", "manager", "boss",
    "therapist", "doctor", "coworker", "colleague", "roommate", "daughter", "son",
)
_RELATION_PATTERN = re.compile(r"\bmy (" + "|".join(_RELATION_NOUNS) + r")\b", re.IGNORECASE)

_COMMON_STARTERS = {
    "I", "The", "My", "This", "That", "It", "We", "They", "You", "He", "She",
    "What", "Why", "How", "When", "Where", "Please", "Yesterday", "Today", "Tomorrow",
}

_VULNERABLE = re.compile(
    r"\b(scared|afraid|overwhelmed|ashamed|lonely|hurt|panic|depressed|anxious|stressed|worried|grief|grieving)\b",
    re.IGNORECASE,
)
_LIFE_EVENT = re.compile(
    r"\b(new job|moved|breakup|divorce|promotion|fired|laid off|diagnosed|hospital|graduation|wedding|engaged|pregnant)\b",
    re.IGNORECASE,
)
_GOAL_OR_PLAN = re.compile(r"\b(i'm going to|i plan to|my goal is|i promise|i will)\b", re.IGNORECASE)
_REQUEST_REMEMBER = re.compile(r"\b(remember this|please remember|don't forget)\b", re.IGNORECASE)


def extract_entities(text: str) -> list[str]:
    """Rule-based, not NER/LLM (Vol 4 Ch 5) — relationship nouns lowercased
    (so "my Sister" and "my sister" collapse to the same entity), proper
    nouns kept as-is."""
    entities: set[str] = set()
    for match in _RELATION_PATTERN.finditer(text):
        entities.add(match.group(1).lower())
    for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", text):
        word = match.group(0)
        if word not in _COMMON_STARTERS:
            entities.add(word)
    return sorted(entities)


def extract_significance_markers(text: str) -> list[str]:
    """Explicit significance markers (Vol 4 Ch 4/8) — flags that a segment
    of conversation may be worth remembering independent of entity/emotion
    signal."""
    markers: list[str] = []
    if _LIFE_EVENT.search(text):
        markers.append("life_event")
    if _GOAL_OR_PLAN.search(text):
        markers.append("stated_goal_or_plan")
    if _REQUEST_REMEMBER.search(text):
        markers.append("explicit_request")
    if _VULNERABLE.search(text):
        markers.append("vulnerable_disclosure")
    return markers
