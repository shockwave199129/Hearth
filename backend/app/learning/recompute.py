"""EWMA recomputation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.learning.observation_store import ObservationStore
from app.onboarding.profile_store import get_profile, update_learning_state, update_relationship_state


@dataclass(frozen=True)
class RecomputeResult:
    communication_traits: dict[str, float]
    skill_affinity: dict[str, float]
    trust: dict[str, float]
    development_level: str
    attachment_signal: float
    updated_at: datetime


def _ewma(current: float, latest: float, alpha: float) -> float:
    return round(alpha * latest + (1.0 - alpha) * current, 3)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def recompute_all(user_id: str, store: ObservationStore | None = None) -> RecomputeResult:
    store = store or ObservationStore()
    profile = get_profile(user_id)
    if profile is None:
        raise ValueError("profile not found")

    communication_traits = dict(profile.communication_traits)
    for trait in ("likes_reflection", "likes_direct_advice", "prefers_questions", "interruption_tolerance", "emotional_openness"):
        current = communication_traits.get(trait, 0.5 if trait != "interruption_tolerance" else 0.2)
        observations = store.latest("communication", trait, 20)
        if observations:
            latest = observations[0].value
            communication_traits[trait] = _ewma(current, latest, 0.05)

    skill_affinity = dict(profile.skill_affinity)
    for skill_id in ["validation", "grounding", "journaling", "cognitive_reframing", "boundary_setting", "sleep_hygiene", "crisis_support"]:
        current = skill_affinity.get(skill_id, 0.5)
        observations = store.latest("skill", skill_id, 20)
        if observations:
            latest = observations[0].value
            skill_affinity[skill_id] = _ewma(current, latest, 0.1)

    trust = {
        "general_trust": profile.relationship_general_trust,
        "vulnerability_trust": profile.relationship_vulnerability_trust,
        "advice_trust": profile.relationship_advice_trust,
        "consistency_confidence": profile.relationship_consistency_confidence,
    }
    for key in list(trust):
        observations = store.latest("relationship", key, 20)
        if observations:
            trust[key] = _ewma(trust[key], observations[0].value, 0.06)
            trust[key] = _clamp(trust[key])

    attachment_signal = profile.relationship_general_trust * 0.2
    attachment_obs = store.latest("relationship", "attachment_signal", 20)
    if attachment_obs:
        attachment_signal = _clamp(_ewma(attachment_signal, attachment_obs[0].value, 0.2))

    conversation_count = max(1, len(store.latest("communication", "likes_reflection", 100)))
    depth_score = trust["general_trust"] + trust["vulnerability_trust"] + trust["consistency_confidence"]
    if depth_score > 1.8 and conversation_count >= 12:
        development_level = "deep"
    elif depth_score > 1.2 and conversation_count >= 6:
        development_level = "growing"
    elif depth_score > 0.6:
        development_level = "forming"
    else:
        development_level = "early"

    update_learning_state(user_id, communication_traits=communication_traits, skill_affinity=skill_affinity, evaluation_last_run_at=datetime.now(timezone.utc))
    update_relationship_state(
        user_id,
        relationship_general_trust=trust["general_trust"],
        relationship_vulnerability_trust=trust["vulnerability_trust"],
        relationship_advice_trust=trust["advice_trust"],
        relationship_consistency_confidence=trust["consistency_confidence"],
        relationship_boundaries=profile.relationship_boundaries,
        relationship_life_model=profile.relationship_life_model,
    )
    return RecomputeResult(
        communication_traits=communication_traits,
        skill_affinity=skill_affinity,
        trust=trust,
        development_level=development_level,
        attachment_signal=attachment_signal,
        updated_at=datetime.now(timezone.utc),
    )

