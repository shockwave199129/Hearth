"""The Recomputation Pipeline (Book Vol 7 Ch 4) — one generic mechanism
every specific computation (Communication Traits Ch 5, Skill Affinity
Ch 6, Trust Ch 7) configures rather than reimplements:

  query observations -> fold a moving average -> compare to cache ->
  write only if meaningfully changed.

Every computation in this module is SQL/arithmetic over the observation
store — no LLM involvement (Design Goal 3)."""
from __future__ import annotations

from dataclasses import dataclass

from app.learning.observation_store import Observation, ObservationStore

CHANGE_TOLERANCE = 0.001


@dataclass(frozen=True)
class PipelineResult:
    value: float
    changed: bool
    sample_size: int


def fold_ewma(current: float, observations: list[Observation], alpha: float) -> float:
    """Folds EVERY fetched observation, oldest first — not just the single
    most recent one. Reading only `observations[0]` (the bug this
    replaces) makes recomputation a two-point blend regardless of how many
    new observations accumulated since the last cycle; folding the whole
    window is what makes this a genuine moving average (Ch 4)."""
    value = current
    for obs in sorted(observations, key=lambda o: o.timestamp):
        value = alpha * obs.value + (1 - alpha) * value
    return value


def recompute_value(
    store: ObservationStore,
    observation_type: str,
    subject_id: str,
    *,
    current: float,
    alpha: float,
    limit: int = 20,
    clamp: tuple[float, float] = (0.0, 1.0),
) -> PipelineResult:
    """The generic pipeline: query -> fold -> compare -> (maybe) write. The
    caller decides whether/where to persist; this function only computes."""
    observations = store.latest(observation_type, subject_id, limit)
    if not observations:
        return PipelineResult(value=current, changed=False, sample_size=0)
    folded = fold_ewma(current, observations, alpha)
    folded = max(clamp[0], min(clamp[1], folded))
    changed = abs(folded - current) > CHANGE_TOLERANCE
    return PipelineResult(value=round(folded, 4) if changed else round(current, 4), changed=changed, sample_size=len(observations))
