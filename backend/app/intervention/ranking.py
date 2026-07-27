"""Skill ranking and compatibility handling."""

from __future__ import annotations

from dataclasses import dataclass

from app.onboarding.profile_schema import UserProfile
from app.skills.loader import Skill


@dataclass(frozen=True)
class RankedSkill:
    skill: Skill
    score: float
    reason: str


def _compatibility_table() -> set[tuple[str, str]]:
    return {("grounding", "cognitive_reframing"), ("cognitive_reframing", "grounding")}


def rank_candidates(transcript: str, skills: list[Skill], profile: UserProfile, stage: str) -> list[RankedSkill]:
    text = transcript.lower()
    trigger_map = {
        "validation": ("hurt", "alone", "sad", "frustrated", "exhausted", "overwhelmed"),
        "grounding": ("can't breathe", "spinning", "panic", "racing", "overwhelmed"),
        "journaling": ("write", "get this off my chest", "before bed", "process"),
        "cognitive_reframing": ("fail", "always", "never", "worried", "stuck", "ruminating"),
        "boundary_setting": ("tired of", "expecting me", "pushback", "boundary"),
        "sleep_hygiene": ("sleep", "bed", "night", "racing thoughts", "insomnia"),
        "crisis_support": ("hurt myself", "kill myself", "suicide", "self harm"),
    }
    ranked: list[RankedSkill] = []
    for skill in skills:
        score = 0.0
        reasons: list[str] = []
        for trigger in trigger_map.get(skill.id, ()):
            if trigger in text:
                score += 0.4
                reasons.append("trigger match")
                break
        if any(token in text for token in skill.manifest.when_use):
            score += 0.5
            reasons.append("when_use match")
        if profile.response_length == "short" and skill.id in {"grounding", "validation"}:
            score += 0.15
            reasons.append("short preference")
        if stage in {"listening", "understanding"} and skill.id in {"validation", "journaling"}:
            score += 0.15
            reasons.append("stage fit")
        if stage in {"supporting", "planning"} and skill.id in {"cognitive_reframing", "boundary_setting", "sleep_hygiene"}:
            score += 0.15
            reasons.append("stage fit")
        if skill.id == "crisis_support":
            score += 1.0
            reasons.append("crisis path")
        ranked.append(RankedSkill(skill=skill, score=round(score, 3), reason=", ".join(reasons) or "baseline"))
    ranked.sort(key=lambda r: (-r.score, r.skill.id))
    return ranked


def compose_skills(primary: RankedSkill, others: list[RankedSkill]) -> list[RankedSkill]:
    out = [primary]
    forbidden = {pair for pair in _compatibility_table() if primary.skill.id in pair}
    for item in others:
        if item.skill.id == primary.skill.id:
            continue
        if any(tuple(sorted((primary.skill.id, item.skill.id))) == tuple(sorted(pair)) for pair in forbidden):
            continue
        out.append(item)
        break
    return out
