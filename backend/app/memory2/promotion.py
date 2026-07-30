"""Promotion: Episodic -> Semantic (Book Vol 4 Ch 9). Clustering is pure
vector math in Chroma; most clusters fit a recognizable template and are
promoted deterministically. Only genuinely irregular clusters may use the
one narrow, background-only LLM fallback this chapter defends — disabled
unless a caller explicitly supplies one, and logged distinctly when used, so
it never quietly becomes the default path."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Callable

from app.memory2.decay import recency_factor
from app.memory2.models import EmotionalMetadata, EpisodicMemory, MemoryStatus, SemanticMemory
from app.memory2.store import MemoryStore

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.6  # max Chroma distance to be considered "close"
MIN_CLUSTER_SIZE = 3

_POSITIVE_EMOTIONS = {"joy", "pride", "gratitude", "hope", "love"}
_NEGATIVE_EMOTIONS = {"sadness", "fear", "anger", "disgust", "guilt", "grief", "anxiety"}

LlmFallback = Callable[[list[str]], str]


def find_clusters(store: MemoryStore, user_id: str) -> list[list[EpisodicMemory]]:
    """Groups active episodic memories into candidate clusters by embedding
    similarity plus shared entity/emotion-category overlap — pure vector
    math and set comparisons, no LLM (Vol 4 Ch 9, Step 1)."""
    episodics = [m for m in store.list_episodic(user_id) if m.status == MemoryStatus.active]
    clustered_ids: set[str] = set()
    clusters: list[list[EpisodicMemory]] = []
    for mem in episodics:
        if mem.id in clustered_ids:
            continue
        neighbors = store.neighbors_episodic(mem, k=8)
        group = [mem]
        for candidate, distance in neighbors:
            if candidate.id in clustered_ids or candidate.status != MemoryStatus.active:
                continue
            if distance > SIMILARITY_THRESHOLD:
                continue
            shares_entity = bool(set(candidate.entities) & set(mem.entities))
            shares_emotion = candidate.emotional_metadata.category == mem.emotional_metadata.category
            if shares_entity or shares_emotion:
                group.append(candidate)
        if len(group) >= MIN_CLUSTER_SIZE:
            for member in group:
                clustered_ids.add(member.id)
            clusters.append(group)
    return clusters


def _dominant_entity(cluster: list[EpisodicMemory]) -> str | None:
    counts: dict[str, int] = {}
    for mem in cluster:
        for entity in mem.entities:
            counts[entity] = counts.get(entity, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _dominant_emotion(cluster: list[EpisodicMemory]) -> str:
    counts: dict[str, int] = {}
    for mem in cluster:
        category = mem.emotional_metadata.category
        counts[category] = counts.get(category, 0) + 1
    return max(counts, key=counts.get) if counts else "neutral"


def fits_template(cluster: list[EpisodicMemory]) -> bool:
    """A cluster fits the recognizable "recurring emotion in response to a
    recurring entity/theme" shape (Vol 4 Ch 9, Step 2) whenever it has a
    dominant entity to anchor the fact to."""
    return _dominant_entity(cluster) is not None


def build_semantic_fact(cluster: list[EpisodicMemory]) -> str:
    """Templated, never LLM-generated: 'User tends to experience {emotion}
    in response to {recurring_entity}.'"""
    entity = _dominant_entity(cluster) or "this topic"
    emotion = _dominant_emotion(cluster)
    return f"User tends to experience {emotion} in response to {entity}."


def compute_confidence(cluster: list[EpisodicMemory], *, now: datetime | None = None) -> float:
    """Confidence from cluster size, recency, and (via the shared entity
    requirement above) embedding tightness — never an LLM's self-reported
    confidence (Vol 4 Ch 9)."""
    now = now or datetime.now(timezone.utc)
    size_factor = min(1.0, len(cluster) / 6.0)
    recency = sum(recency_factor(m.last_reinforced, now) for m in cluster) / len(cluster)
    return round(min(1.0, 0.3 + 0.4 * size_factor + 0.3 * recency), 4)


def _emotional_metadata_for(cluster: list[EpisodicMemory]) -> EmotionalMetadata:
    dominant_emotion = _dominant_emotion(cluster)
    if dominant_emotion in _NEGATIVE_EMOTIONS:
        valence = "negative"
    elif dominant_emotion in _POSITIVE_EMOTIONS:
        valence = "positive"
    else:
        valence = "neutral"
    max_intensity = max((m.emotional_metadata.intensity for m in cluster), default=0.0)
    sensitivity = any(m.emotional_metadata.sensitivity_flag for m in cluster)
    return EmotionalMetadata(valence=valence, intensity=max_intensity, category=dominant_emotion, sensitivity_flag=sensitivity)


def promote_cluster(
    cluster: list[EpisodicMemory],
    user_id: str,
    *,
    llm_fallback: LlmFallback | None = None,
    now: datetime | None = None,
) -> SemanticMemory | None:
    now = now or datetime.now(timezone.utc)
    if fits_template(cluster):
        fact = build_semantic_fact(cluster)
    elif llm_fallback is not None:
        # Rare, background-only, logged distinctly (Vol 4 Ch 9, Step 3) —
        # never the default path, only reached when Step 2 genuinely fails.
        logger.info("memory2.promotion: irregular cluster (%d entries) — using narrow LLM fallback", len(cluster))
        try:
            fact = llm_fallback([m.summary for m in cluster]).strip()
        except Exception:
            logger.exception("memory2.promotion: LLM fallback failed — cluster left unpromoted")
            return None
        if not fact:
            return None
    else:
        logger.info(
            "memory2.promotion: irregular cluster (%d entries) left unpromoted — no fallback configured", len(cluster)
        )
        return None

    return SemanticMemory(
        id=str(uuid.uuid4()),
        user_id=user_id,
        fact=fact,
        source_episodes=[m.id for m in cluster],
        confidence=compute_confidence(cluster, now=now),
        emotional_metadata=_emotional_metadata_for(cluster),
        last_reinforced=now,
    )


def run_promotion(store: MemoryStore, user_id: str, *, llm_fallback: LlmFallback | None = None) -> list[SemanticMemory]:
    """Finds clusters, promotes each, and persists the resulting semantic
    facts. Returns the newly created SemanticMemory objects."""
    promoted: list[SemanticMemory] = []
    for cluster in find_clusters(store, user_id):
        semantic = promote_cluster(cluster, user_id, llm_fallback=llm_fallback)
        if semantic is not None:
            store.save_semantic(semantic)
            promoted.append(semantic)
    return promoted
