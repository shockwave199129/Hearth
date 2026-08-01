"""Privacy & User Control over Memory (Book Vol 4 Ch 15) — legibility
(plain-language account, not a raw record dump), direct correction (treated
as strong evidence, applied immediately), and hard deletion (never a soft
decay-to-zero) with confidence-reduction cascading into dependent semantic
facts."""
from __future__ import annotations

from datetime import datetime, timezone

from app.memory2.models import EpisodicMemory, MemoryStatus, SemanticMemory
from app.memory2.store import MemoryStore

DELETION_CASCADE_PENALTY_PER_EPISODE = 0.15
MIN_CONFIDENCE_AFTER_CASCADE = 0.05


def plain_language_summary(store: MemoryStore, user_id: str) -> dict:
    """A readable account grouped by theme — never a raw model dump (Vol 4
    Ch 15). Uses the same templated text the storage already produces."""
    episodics = [m for m in store.list_episodic(user_id) if m.status != MemoryStatus.contradicted]
    semantics = [m for m in store.list_semantic(user_id) if m.status == MemoryStatus.active]
    themes: dict[str, list[str]] = {}
    for mem in episodics:
        for entity in mem.entities or ["general"]:
            themes.setdefault(entity, []).append(mem.summary)
    return {
        "remembered_moments_by_theme": themes,
        "general_facts": [m.fact for m in semantics],
        "total_moments_remembered": len(episodics),
        "total_general_facts": len(semantics),
    }


def correct_episodic(
    store: MemoryStore, mem_id: str, user_id: str, *, corrected_summary: str, now: datetime | None = None
) -> EpisodicMemory | None:
    """A direct user correction is strong, immediate evidence — applied
    right away, not queued for slow, evidence-based reconciliation the way
    an inferred contradiction is (Vol 3 Ch 12; Vol 4 Ch 15)."""
    mem = store.get_episodic(mem_id, user_id)
    if mem is None:
        return None
    now = now or datetime.now(timezone.utc)
    corrected = mem.model_copy(update={"summary": corrected_summary, "last_reinforced": now})
    store.update_episodic(corrected, re_embed=True)
    return corrected


def correct_semantic(
    store: MemoryStore, mem_id: str, user_id: str, *, corrected_fact: str | None = None, mark_inaccurate: bool = False
) -> SemanticMemory | None:
    mem = store.get_semantic(mem_id, user_id)
    if mem is None:
        return None
    updates: dict = {}
    if corrected_fact is not None:
        updates["fact"] = corrected_fact
    if mark_inaccurate:
        updates["confidence"] = 0.0
        updates["status"] = MemoryStatus.contradicted
    if not updates:
        return mem
    corrected = mem.model_copy(update=updates)
    store.update_semantic(corrected)
    return corrected


def delete_episodic_with_cascade(store: MemoryStore, mem_id: str, user_id: str) -> list[SemanticMemory]:
    """Hard delete (never decay-to-zero, Vol 4 Ch 15). If the deleted
    episode was a source for any semantic fact, that fact's confidence is
    reduced proportionally rather than left standing on a partially deleted
    foundation; if too few supporting episodes remain, it's flagged for
    re-evaluation by marking it for the next promotion cycle (status stays
    active but confidence trends toward the floor)."""
    store.delete_episodic(mem_id, user_id)
    affected: list[SemanticMemory] = []
    for semantic in store.list_semantic(user_id):
        if mem_id not in semantic.source_episodes:
            continue
        remaining_sources = [sid for sid in semantic.source_episodes if sid != mem_id]
        new_confidence = max(MIN_CONFIDENCE_AFTER_CASCADE, round(semantic.confidence - DELETION_CASCADE_PENALTY_PER_EPISODE, 4))
        updated = semantic.model_copy(update={"source_episodes": remaining_sources, "confidence": new_confidence})
        store.update_semantic(updated)
        affected.append(updated)
    return affected


def delete_semantic(store: MemoryStore, mem_id: str, user_id: str) -> None:
    store.delete_semantic(mem_id, user_id)


def delete_all_memory(store: MemoryStore, user_id: str) -> None:
    """Deletes the entire memory store for a user outright (Vol 4 Ch 15) —
    used for account-level deletion, not per-memory correction."""
    store.delete_all_for_user(user_id)
