"""Intervention Engine (Book Vol 1 Ch 8, Vol 5). Decides a support
*strategy* first, skill(s) second — sometimes the strategy is simply
"listen", no skill at all.

Crisis Support (Vol 5 Ch 14, Invariant 8) is structurally separate: no
candidate retrieval, no competitive ranking, never composed. It is reached
only when a caller (Phase 4's Safety Worker) has already determined a
crisis exists and passes `crisis=True` — this module never detects a
crisis itself."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.intervention.ranking import MIN_PRIMARY_SCORE, RankedSkill, compose_skills, rank_candidates
from app.intervention.retrieval import SkillRetriever, retrieve_candidates
from app.onboarding.profile_schema import UserProfile
from app.skills.loader import Skill, get_skill

CRISIS_SKILL_ID = "crisis_support"

# Maps the chosen primary skill to a short strategy label the Prompt
# Builder/response path can reason about — a plain lookup table, not the
# hardcoded trigger-phrase override this replaces (see module history:
# that override re-decided the primary skill after ranking had already
# picked one, silently discarding the additive score).
_SKILL_STRATEGY: dict[str, str] = {
    "validation": "validate",
    "grounding": "ground",
    "journaling": "reflect",
    "cognitive_reframing": "reframe",
    "boundary_setting": "boundary",
    "sleep_hygiene": "sleep",
    CRISIS_SKILL_ID: "crisis_support",
}


@dataclass(frozen=True)
class InterventionContext:
    """Volume 1 Ch 8's "Intervention Context" — everything Dynamic Scoring
    needs beyond the transcript/profile themselves."""

    stage: str
    emotion: str = "unknown"
    emotion_confidence: float = 0.0
    goal: str = "listen"
    development_level: str = "stranger"
    skill_affinity: dict[str, float] = field(default_factory=dict)
    recent_skill_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class InterventionPlan:
    strategy: str
    primary_skill: RankedSkill | None
    secondary_skill: RankedSkill | None
    candidate_ids: list[str]


def _wrap(skill: Skill, *, reason: str, score: float = 1.0) -> RankedSkill:
    return RankedSkill(skill=skill, score=score, reason=reason)


class InterventionEngine:
    def __init__(self, *, retriever: SkillRetriever | None = None):
        self.retriever = retriever

    def _retrieve(self, query: str) -> list[Skill]:
        return retrieve_candidates(query, retriever=self.retriever)

    def plan(self, transcript: str, profile: UserProfile, context: InterventionContext, crisis: bool = False) -> InterventionPlan:
        if crisis:
            return self._plan_crisis()
        return self._plan_ordinary(transcript, profile, context)

    def _plan_crisis(self) -> InterventionPlan:
        """Vol 5 Ch 14: no permission step, no skill scoring delay — a
        fixed, dedicated path, never a competitively-ranked candidate, and
        never composed with anything else."""
        crisis_skill = get_skill(CRISIS_SKILL_ID)
        primary = _wrap(crisis_skill, reason="crisis path (dedicated, unscored)") if crisis_skill is not None else None
        return InterventionPlan(
            strategy=_SKILL_STRATEGY[CRISIS_SKILL_ID],
            primary_skill=primary,
            secondary_skill=None,
            candidate_ids=[CRISIS_SKILL_ID] if primary else [],
        )

    def _plan_ordinary(self, transcript: str, profile: UserProfile, context: InterventionContext) -> InterventionPlan:
        # Crisis Support never competes in ordinary scoring (Vol 5 Ch 14) —
        # detection is not this engine's job, so it's excluded from the
        # candidate set outright rather than relying on it simply losing.
        candidates = [s for s in self._retrieve(transcript) if s.id != CRISIS_SKILL_ID]
        ranked = rank_candidates(transcript, candidates, profile, context)

        if not ranked or ranked[0].score < MIN_PRIMARY_SCORE:
            return InterventionPlan(
                strategy="listen",
                primary_skill=None,
                secondary_skill=None,
                candidate_ids=[item.skill.id for item in ranked],
            )

        primary = ranked[0]
        composed = compose_skills(primary, ranked[1:])
        secondary = composed[1] if len(composed) > 1 else None

        return InterventionPlan(
            strategy=_SKILL_STRATEGY.get(primary.skill.id, "listen"),
            primary_skill=primary,
            secondary_skill=secondary,
            candidate_ids=[item.skill.id for item in ranked],
        )
