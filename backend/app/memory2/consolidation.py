"""Consolidation (Book Vol 4 Ch 12) — merges near-duplicate episodic
memories using the same clustering approach as Promotion (Ch 9), rather than
introducing a second mechanism. Produces one consolidated episodic entry
that retains references to every source entry — nothing is destroyed, only
compacted — and inherits the strongest attributes rather than averaging
them down. Distinct from Promotion: this stays at the episodic level
(efficiency), it does not produce a new, higher-abstraction Semantic fact."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.memory2.models import EpisodicMemory, MemoryStatus
from app.memory2.promotion import SIMILARITY_THRESHOLD
from app.memory2.store import MemoryStore

CONSOLIDATION_WINDOW_DAYS = 14
MIN_MERGE_SIZE = 2


def find_merge_candidates(
    store: MemoryStore, user_id: str, *, now: datetime | None = None
) -> list[list[EpisodicMemory]]:
    now = now or datetime.now(timezone.utc)
    window_start = now - timedelta(days=CONSOLIDATION_WINDOW_DAYS)
    episodics = [
        m for m in store.list_episodic(user_id) if m.status == MemoryStatus.active and m.timestamp >= window_start
    ]
    clustered_ids: set[str] = set()
    groups: list[list[EpisodicMemory]] = []
    for mem in episodics:
        if mem.id in clustered_ids:
            continue
        neighbors = store.neighbors_episodic(mem, k=8)
        group = [mem]
        for candidate, distance in neighbors:
            if candidate.id in clustered_ids or candidate.status != MemoryStatus.active:
                continue
            if candidate.timestamp < window_start:
                continue
            if distance > SIMILARITY_THRESHOLD:
                continue
            if set(candidate.entities) & set(mem.entities):
                group.append(candidate)
        if len(group) >= MIN_MERGE_SIZE:
            for member in group:
                clustered_ids.add(member.id)
            groups.append(group)
    return groups


def merge(cluster: list[EpisodicMemory], *, now: datetime | None = None) -> EpisodicMemory:
    now = now or datetime.now(timezone.utc)
    strongest = max(cluster, key=lambda m: m.emotional_metadata.intensity)
    total_references = sum(m.reference_count for m in cluster)
    all_entities = sorted({e for m in cluster for e in m.entities})
    all_markers = sorted({marker for m in cluster for marker in m.significance_markers})
    base = min(cluster, key=lambda m: m.timestamp)
    merged_from = sorted({mid for m in cluster for mid in ([m.id] + m.merged_from) if mid != base.id})
    return base.model_copy(
        update={
            "summary": strongest.summary,
            "entities": all_entities,
            "significance_markers": all_markers,
            "emotional_metadata": strongest.emotional_metadata,
            "reference_count": total_references,
            "status": MemoryStatus.consolidated,
            "last_reinforced": now,
            "merged_from": merged_from,
        }
    )


def run_consolidation(store: MemoryStore, user_id: str, *, now: datetime | None = None) -> list[EpisodicMemory]:
    now = now or datetime.now(timezone.utc)
    consolidated: list[EpisodicMemory] = []
    for cluster in find_merge_candidates(store, user_id, now=now):
        merged = merge(cluster, now=now)
        store.save_episodic(merged)
        for mem in cluster:
            if mem.id != merged.id:
                store.delete_episodic(mem.id, mem.user_id)
        store.log_consolidation(merged.id, merged.merged_from, now.isoformat())
        consolidated.append(merged)
    return consolidated
