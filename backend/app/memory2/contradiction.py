"""Contradiction & Conflict Resolution (Book Vol 4 Ch 13) — detects when a
new episodic memory disagrees with an existing semantic fact, using
similarity/entity overlap plus a rule-based opposite-valence check. Never an
LLM judgment call. A detected contradiction reduces the existing fact's
confidence; it never silently overwrites it — Promotion (Ch 9) will
naturally produce an updated fact once enough new episodes support the
changed pattern."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from app.memory2.models import EpisodicMemory, SemanticMemory
from app.memory2.store import MemoryStore

logger = logging.getLogger(__name__)

_OPPOSITE_VALENCE = {"positive": "negative", "negative": "positive"}
CONTRADICTION_CONFIDENCE_PENALTY = 0.25
MIN_CONFIDENCE = 0.05


@dataclass(frozen=True)
class Contradiction:
    semantic_id: str
    episodic_id: str
    reason: str


def detect_contradictions(store: MemoryStore, new_episode: EpisodicMemory) -> list[Contradiction]:
    """Rule-based: a semantic fact sharing an entity with the new episode,
    whose valence is the opposite of the new episode's valence, is a
    contradiction candidate (e.g. an established 'anxiety' fact about a
    person vs. a new, strongly 'calm'/'confident' episode about the same
    person)."""
    if not new_episode.entities:
        return []
    candidates = store.semantic_sharing_entities(new_episode.user_id, new_episode.entities)
    found: list[Contradiction] = []
    episode_valence = new_episode.emotional_metadata.valence
    for semantic in candidates:
        semantic_valence = semantic.emotional_metadata.valence
        if semantic_valence in _OPPOSITE_VALENCE and episode_valence == _OPPOSITE_VALENCE[semantic_valence]:
            found.append(
                Contradiction(
                    semantic_id=semantic.id,
                    episodic_id=new_episode.id,
                    reason=f"existing fact valence={semantic_valence!r} vs new episode valence={episode_valence!r}",
                )
            )
    return found


def resolve_contradiction(
    store: MemoryStore, semantic: SemanticMemory, *, now: datetime | None = None
) -> SemanticMemory:
    """Reduces confidence — never deletes or overwrites the fact outright
    (Vol 4 Ch 13). A single contradicting episode is a data point, not
    proof the old pattern is gone."""
    now = now or datetime.now(timezone.utc)
    updated = semantic.model_copy(
        update={"confidence": max(MIN_CONFIDENCE, round(semantic.confidence - CONTRADICTION_CONFIDENCE_PENALTY, 4))}
    )
    store.update_semantic(updated)
    return updated


def check_and_resolve(
    store: MemoryStore, new_episode: EpisodicMemory, *, now: datetime | None = None
) -> list[Contradiction]:
    """Detects contradictions against the new episode and immediately
    applies the confidence-reduction resolution policy, logging each one so
    it's visible for the next periodic Growth Engine recomputation rather
    than resolved silently."""
    contradictions = detect_contradictions(store, new_episode)
    for contradiction in contradictions:
        semantic = store.get_semantic(contradiction.semantic_id, new_episode.user_id)
        if semantic is not None:
            logger.info("memory2.contradiction: %s", contradiction.reason)
            resolve_contradiction(store, semantic, now=now)
    return contradictions
