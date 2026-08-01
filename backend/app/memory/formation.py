"""Rule-based memory formation for phase 2."""

from dataclasses import dataclass
import re

from app.memory import long_term
from app.memory.short_term import ShortTermMemory


_FIRST_PERSON_FACT = re.compile(r"\b(i|i'm|i am|my)\b", re.IGNORECASE)
_VULNERABLE = re.compile(r"\b(scared|afraid|overwhelmed|ashamed|lonely|hurt|panic|depressed|anxious|stressed|worried)\b", re.IGNORECASE)
_LIFE_EVENT = re.compile(r"\b(new job|moved|breakup|divorce|promotion|fired|laid off|diagnosed|hospital|graduation|wedding)\b", re.IGNORECASE)
_PREFERENCES = re.compile(r"\b(i like|i prefer|i love|i hate|i usually|i'm not into)\b", re.IGNORECASE)
_REQUEST_REMEMBER = re.compile(r"\b(remember this|please remember|don't forget)\b", re.IGNORECASE)


@dataclass(frozen=True)
class MemoryFormationResult:
    created: int
    updated: int
    candidates: list[str]


def _canonical_category(text: str) -> str:
    if _PREFERENCES.search(text):
        return "preference"
    if _LIFE_EVENT.search(text):
        return "life_event"
    if _VULNERABLE.search(text):
        return "stressor"
    return "other"


def _should_form(text: str) -> bool:
    return bool(_FIRST_PERSON_FACT.search(text) and (_VULNERABLE.search(text) or _LIFE_EVENT.search(text) or _PREFERENCES.search(text) or _REQUEST_REMEMBER.search(text)))


def _normalise_fact(text: str) -> str:
    return text.strip().rstrip(".")


def process_session_memory(user_id: str, memory: ShortTermMemory) -> MemoryFormationResult:
    candidates: list[str] = []
    created = 0
    updated = 0
    for message in memory.messages:
        if message.get("role") != "user":
            continue
        text = str(message.get("content", "")).strip()
        if not text or not _should_form(text):
            continue
        candidates.append(text)
        category = _canonical_category(text)
        fact = _normalise_fact(text)
        similar = long_term.search(fact, user_id, k=3)
        if similar:
            target = similar[0]
            long_term.update(target["id"], fact, user_id)
            updated += 1
        else:
            long_term.create(fact, category, user_id)
            created += 1
    return MemoryFormationResult(created=created, updated=updated, candidates=candidates)

