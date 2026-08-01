"""Pure arithmetic memory decay (Book Vol 4 Ch 10) — reduces retrieval
priority over time, never deletes. Deletion is a separate, explicit,
user-controlled action (see privacy.py)."""
from __future__ import annotations

import math
from datetime import datetime, timezone

from app.memory2.models import EpisodicMemory

# Half-life-style decay: priority halves roughly every DECAY_HALF_LIFE_DAYS
# of no reinforcement, before the emotional-weight and reference-count
# factors are applied.
DECAY_HALF_LIFE_DAYS = 30.0
MIN_RECENCY_FACTOR = 0.05
REFERENCE_BOOST_CAP = 1.5
RECENT_OVERUSE_WINDOW_DAYS = 1.0
RECENT_OVERUSE_PENALTY = 0.6


def recency_factor(last_reinforced: datetime, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    days = max(0.0, (now - last_reinforced).total_seconds() / 86400.0)
    factor = math.pow(0.5, days / DECAY_HALF_LIFE_DAYS)
    return max(MIN_RECENCY_FACTOR, factor)


def emotional_weight(intensity: float) -> float:
    """Flattens the decay curve for high-intensity memories (Vol 4 Ch 7) —
    a deeply significant memory persists longer than an incidental one of
    the same age."""
    return 1.0 + intensity  # ranges 1.0 (neutral) .. 2.0 (max intensity)


def reference_factor(reference_count: int, last_reinforced: datetime, now: datetime | None = None) -> float:
    """Rewards memories that have genuinely been useful, while applying the
    same repetition penalty used for Skills (Vol 1 Ch 8) and Shared History
    (Vol 3 Ch 9) — referenced too recently and too often is temporarily
    deprioritized to avoid repetitive-feeling conversation."""
    now = now or datetime.now(timezone.utc)
    boost = min(REFERENCE_BOOST_CAP, 1.0 + 0.05 * reference_count)
    days_since = (now - last_reinforced).total_seconds() / 86400.0
    if reference_count > 0 and days_since < RECENT_OVERUSE_WINDOW_DAYS:
        boost *= RECENT_OVERUSE_PENALTY
    return boost


def compute_priority(mem: EpisodicMemory, *, base_significance: float = 1.0, now: datetime | None = None) -> float:
    """priority = base_significance x recency_factor x emotional_weight x reference_factor"""
    now = now or datetime.now(timezone.utc)
    return (
        base_significance
        * recency_factor(mem.last_reinforced, now)
        * emotional_weight(mem.emotional_metadata.intensity)
        * reference_factor(mem.reference_count, mem.last_reinforced, now)
    )


def reinforce(mem: EpisodicMemory, *, now: datetime | None = None) -> EpisodicMemory:
    """A new episodic memory or conversation touching the same entity/theme
    resets decay (Vol 4 Ch 10) — this is what lets a recurring theme stay
    highly retrievable indefinitely, while a one-off mention fades."""
    now = now or datetime.now(timezone.utc)
    return mem.model_copy(update={"last_reinforced": now, "reference_count": mem.reference_count + 1})


SEMANTIC_CONFIDENCE_HALF_LIFE_DAYS = 60.0
MIN_SEMANTIC_CONFIDENCE = 0.1


def decay_semantic_confidence(confidence: float, last_reinforced: datetime, now: datetime | None = None) -> float:
    """A semantic fact decays based on whether its pattern is still being
    reinforced by new episodes, not merely its own age (Vol 4 Ch 10) — this
    reduces `confidence` independently of any episodic priority."""
    now = now or datetime.now(timezone.utc)
    days = max(0.0, (now - last_reinforced).total_seconds() / 86400.0)
    factor = math.pow(0.5, days / SEMANTIC_CONFIDENCE_HALF_LIFE_DAYS)
    return max(MIN_SEMANTIC_CONFIDENCE, confidence * max(factor, MIN_SEMANTIC_CONFIDENCE))
