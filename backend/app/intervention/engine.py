"""Phase 3 intervention engine."""

from __future__ import annotations

from dataclasses import dataclass

from app.intervention.ranking import RankedSkill, compose_skills, rank_candidates
from app.intervention.retrieval import retrieve_candidates
from app.onboarding.profile_schema import UserProfile


@dataclass(frozen=True)
class InterventionPlan:
    strategy: str
    primary_skill: RankedSkill | None
    secondary_skill: RankedSkill | None
    candidate_ids: list[str]


class InterventionEngine:
    def plan(self, transcript: str, profile: UserProfile, stage: str, crisis: bool = False) -> InterventionPlan:
        text = transcript.lower()
        if crisis:
            candidates = retrieve_candidates("crisis support immediate danger")
            ranked = rank_candidates(transcript, candidates, profile, stage)
            primary = ranked[0] if ranked else None
            if primary is None:
                from app.skills.loader import get_skill

                crisis_skill = get_skill("crisis_support")
                if crisis_skill is not None:
                    from app.intervention.ranking import RankedSkill

                    primary = RankedSkill(skill=crisis_skill, score=1.0, reason="crisis path")
            return InterventionPlan(strategy="crisis_support", primary_skill=primary, secondary_skill=None, candidate_ids=[s.skill.id for s in ranked])

        candidates = retrieve_candidates(transcript)
        ranked = rank_candidates(transcript, candidates, profile, stage)
        if not ranked:
            return InterventionPlan(strategy="listen", primary_skill=None, secondary_skill=None, candidate_ids=[])
        primary = ranked[0]
        priority = [
            ("grounding", ("can't breathe", "spinning", "panic", "racing")),
            ("validation", ("hurt and alone", "hurt", "alone", "sad", "frustrated")),
            ("cognitive_reframing", ("fail", "always", "never", "worried")),
            ("journaling", ("write", "before bed", "get this off my chest")),
            ("boundary_setting", ("tired of", "expecting me", "boundary")),
            ("sleep_hygiene", ("sleep", "bed", "night", "racing thoughts")),
        ]
        for skill_id, triggers in priority:
            if any(trigger in text for trigger in triggers):
                for item in ranked:
                    if item.skill.id == skill_id:
                        primary = item
                        break
                else:
                    from app.skills.loader import get_skill
                    from app.intervention.ranking import RankedSkill

                    skill = get_skill(skill_id)
                    if skill is not None:
                        primary = RankedSkill(skill=skill, score=1.0, reason="explicit trigger")
                break
        composed = compose_skills(primary, ranked[1:])
        secondary = composed[1] if len(composed) > 1 else None

        strategy = "listen"
        if primary.skill.id == "validation":
            strategy = "validate"
        elif primary.skill.id == "grounding":
            strategy = "ground"
        elif primary.skill.id == "journaling":
            strategy = "reflect"
        elif primary.skill.id == "cognitive_reframing":
            strategy = "reframe"
        elif primary.skill.id == "boundary_setting":
            strategy = "boundary"
        elif primary.skill.id == "sleep_hygiene":
            strategy = "sleep"

        return InterventionPlan(
            strategy=strategy,
            primary_skill=primary,
            secondary_skill=secondary,
            candidate_ids=[item.skill.id for item in ranked],
        )
