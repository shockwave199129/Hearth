"""Applies Book Volume 7's generic Recomputation Pipeline (Chapter 4) to
Communication Traits (Ch 5), Skill Affinity (Ch 6), and Trust (Ch 7),
writing results to Profile's cached fields — the only interface the live
runtime ever reads; it never queries the observation store directly.

Development level and Attachment signals are NOT computed here — those are
computed and persisted directly against `RelationshipProfile` by
`app.growth.engine` (using this same pipeline), which is the single source
of truth Phase 3's Intervention Engine and Phase 4's Safety Worker both
read. This module writes only the flat, fast-access Profile fields Volume 3
Ch 3 describes as the runtime cache; RelationshipProfile is the fuller,
versioned object."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.cognitive.communication import NEUTRAL_TRAIT, TRAIT_KEYS
from app.learning.coldstart import blend
from app.learning.observation_store import ObservationStore
from app.learning.pipeline import recompute_value
from app.onboarding.profile_store import get_profile, update_learning_state, update_relationship_state

COMMUNICATION_TRAIT_ALPHA = 0.05  # Ch 5: low, for long-term stability.
SKILL_AFFINITY_ALPHA = 0.1  # Ch 6: moderate — skill fit can genuinely shift.
TRUST_ALPHA = 0.06
NEUTRAL_AFFINITY = 0.5
COLD_START_SAMPLE_THRESHOLD = 8  # Ch 11's confidence_curve `k`.

TRUST_SUBSCORES = ("general_trust", "vulnerability_trust", "advice_trust", "consistency_confidence")

# Ch 6's cross-affinity cold-start prior: which Communication Trait informs
# a skill's *initial* affinity estimate when it has too little observation
# history of its own. Skills with no strong related trait (grounding,
# sleep_hygiene, crisis_support) fall back to the plain neutral population
# default instead.
SKILL_AFFINITY_TRAIT_PRIOR: dict[str, str] = {
    "cognitive_reframing": "likes_direct_advice",
    "boundary_setting": "likes_direct_advice",
    "validation": "likes_reflection",
    "journaling": "likes_reflection",
}


@dataclass(frozen=True)
class RecomputeResult:
    communication_traits: dict[str, float]
    skill_affinity: dict[str, float]
    trust: dict[str, float]
    updated_at: datetime


def _trait_prior_to_affinity_prior(trait_value: float) -> float:
    """Maps a 0..1 trait value onto a plausible 0.3..0.7 starting affinity
    range — a strong prior nudges the cold-start estimate, but even a
    maximal trait value shouldn't assert near-certain affinity for a skill
    that's never actually been used."""
    return 0.3 + 0.4 * trait_value


def recompute_all(user_id: str, store: ObservationStore | None = None) -> RecomputeResult:
    store = store or ObservationStore()
    profile = get_profile(user_id)
    if profile is None:
        raise ValueError("profile not found")

    communication_traits = dict(profile.communication_traits)
    for trait in TRAIT_KEYS:
        current = communication_traits.get(trait, NEUTRAL_TRAIT)
        result = recompute_value(store, "communication", trait, current=current, alpha=COMMUNICATION_TRAIT_ALPHA)
        if result.changed:
            communication_traits[trait] = result.value

    skill_affinity = dict(profile.skill_affinity)
    for skill_id in ["validation", "grounding", "journaling", "cognitive_reframing", "boundary_setting", "sleep_hygiene", "crisis_support"]:
        current = skill_affinity.get(skill_id, NEUTRAL_AFFINITY)
        result = recompute_value(store, "skill", skill_id, current=current, alpha=SKILL_AFFINITY_ALPHA)
        if result.sample_size == 0:
            continue
        if result.sample_size < COLD_START_SAMPLE_THRESHOLD:
            # Ch 11's cold-start blend: weight given to the actually-observed
            # value grows smoothly with sample_size, rather than switching
            # abruptly from "prior" to "observed" at a fixed cutoff.
            trait_id = SKILL_AFFINITY_TRAIT_PRIOR.get(skill_id)
            prior = _trait_prior_to_affinity_prior(communication_traits[trait_id]) if trait_id else None
            skill_affinity[skill_id] = blend(NEUTRAL_AFFINITY, prior, result.value, result.sample_size, k=COLD_START_SAMPLE_THRESHOLD)
        elif result.changed:
            skill_affinity[skill_id] = result.value

    trust = {
        "general_trust": profile.relationship_general_trust,
        "vulnerability_trust": profile.relationship_vulnerability_trust,
        "advice_trust": profile.relationship_advice_trust,
        "consistency_confidence": profile.relationship_consistency_confidence,
    }
    for subscore in TRUST_SUBSCORES:
        # Trust genuinely starts at zero and has no prior (Ch 11) — no
        # cold-start blend here, just the plain pipeline fold.
        result = recompute_value(store, "relationship", subscore, current=trust[subscore], alpha=TRUST_ALPHA)
        if result.changed:
            trust[subscore] = result.value

    update_learning_state(
        user_id,
        communication_traits=communication_traits,
        skill_affinity=skill_affinity,
        evaluation_last_run_at=datetime.now(timezone.utc),
    )
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
        updated_at=datetime.now(timezone.utc),
    )
