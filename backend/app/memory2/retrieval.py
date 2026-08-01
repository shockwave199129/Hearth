"""Deterministic, explainable memory retrieval (Book Vol 4 Ch 11) — Chroma
similarity search followed by a transparent, additive re-ranking step. This
module stops at producing a ranked candidate set; whether/how a memory
actually gets mentioned is a Volume 2 (Ch 13) communication-level decision,
made by the Prompt Builder, not here."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.memory2.decay import compute_priority
from app.memory2.models import EpisodicMemory, MemoryStatus, SemanticMemory
from app.memory2.store import MemoryStore

# Relationship Development levels, ordered low -> high (Vol 3 Ch 5). A
# memory can be gated to only surface once a sufficient level is reached.
DEVELOPMENT_LEVELS = ("stranger", "acquaintance", "familiar", "trusted_companion", "deep_long_term_companion")


@dataclass(frozen=True)
class RankedMemory:
    kind: str  # "episodic" | "semantic"
    memory: EpisodicMemory | SemanticMemory
    text: str
    score: float
    distance: float


def _development_index(level: str) -> int:
    return DEVELOPMENT_LEVELS.index(level) if level in DEVELOPMENT_LEVELS else 0


def retrieve(
    store: MemoryStore,
    query: str,
    user_id: str,
    *,
    k: int = 8,
    top_n: int = 3,
    development_level: str = "stranger",
    now: datetime | None = None,
) -> list[RankedMemory]:
    """Retrieval pipeline: embed query -> Chroma similarity search over
    episodic + semantic -> filter/re-rank by decay-adjusted priority,
    (semantic) confidence, emotional-sensitivity, and relationship-level
    gating -> top_n candidates. Named, inspectable factors combined
    additively, never an opaque single score (Vol 4 Ch 11)."""
    now = now or datetime.now(timezone.utc)
    candidates: list[RankedMemory] = []

    for mem, distance in store.search_episodic(query, user_id, k=k):
        if mem.status not in (MemoryStatus.active, MemoryStatus.consolidated):
            continue
        similarity = max(0.0, 1.0 - distance)
        priority = compute_priority(mem, now=now)
        score = similarity + 0.3 * priority
        if mem.emotional_metadata.sensitivity_flag:
            score -= 0.25  # deprioritized unless directly relevant (high similarity carries it anyway)
        candidates.append(RankedMemory(kind="episodic", memory=mem, text=mem.summary, score=score, distance=distance))

    for mem, distance in store.search_semantic(query, user_id, k=k):
        if mem.status != MemoryStatus.active:
            continue
        similarity = max(0.0, 1.0 - distance)
        score = similarity + 0.3 * mem.confidence
        if mem.emotional_metadata.sensitivity_flag:
            score -= 0.25
        candidates.append(RankedMemory(kind="semantic", memory=mem, text=mem.fact, score=score, distance=distance))

    return _finalize(candidates, top_n=top_n, development_level=development_level)


def _finalize(candidates: list[RankedMemory], *, top_n: int, development_level: str) -> list[RankedMemory]:
    dev_index = _development_index(development_level)
    gated = [c for c in candidates if _min_level_index(c) <= dev_index]
    gated.sort(key=lambda c: c.score, reverse=True)
    return gated[:top_n]


def _min_level_index(candidate: RankedMemory) -> int:
    # Only Semantic facts referencing highly sensitive content require a
    # deeper relationship before they're eligible; episodic memories and
    # ordinary semantic facts have no gating requirement.
    if candidate.kind == "semantic" and candidate.memory.emotional_metadata.sensitivity_flag:
        return _development_index("familiar")
    return 0
