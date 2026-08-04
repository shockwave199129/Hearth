"""Skill ranking and composition (Book Vol 1 Ch 8, Vol 5 Ch 13).

Implements the book's additive Dynamic Intervention Score:

    Final Score = Conversation Score + Profile Score + Relationship Score
                + Historical Effectiveness + Current Emotion + Current Goal
                + Context Penalty

recalculated fresh on every call — never a fixed, stored priority — so the
same conversation can rank differently ten minutes later as the moment
actually changes (the book's GPS-navigation analogy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import yaml

from app.onboarding.profile_schema import UserProfile
from app.skills.loader import SKILLS_ROOT, Skill

if TYPE_CHECKING:
    # engine.py imports this module, so a runtime import here would be
    # circular — hence the string annotation on rank_candidates(). Guarding
    # it under TYPE_CHECKING keeps that property while letting type
    # checkers and linters actually resolve the name.
    from app.intervention.engine import InterventionContext

COMPATIBILITY_PATH = SKILLS_ROOT / "_compatibility.yaml"

# -- Dynamic Intervention Score component weights ---------------------------
CONVERSATION_TRIGGER_WEIGHT = 0.35
WHEN_USE_WEIGHT = 0.2
PROFILE_WEIGHT = 0.1
RELATIONSHIP_WEIGHT = 0.1
AFFINITY_WEIGHT = 0.15
EMOTION_WEIGHT = 0.2
GOAL_WEIGHT = 0.1
CONTEXT_PENALTY_PER_RECENT_USE = 0.25
NEUTRAL_AFFINITY = 0.5

# A skill needs at least this much score to be selected at all — otherwise
# the Intervention Engine's strategy is simply "listen", no skill (Vol 1
# Ch 8: "an Intervention Strategy is not limited to picking a skill").
MIN_PRIMARY_SCORE = 0.3

# Composing a secondary skill requires it to clear this floor too — the fix
# for the previously-unconditional `break`, which attached the first
# *compatible* candidate as secondary regardless of how irrelevant it was.
MIN_SECONDARY_SCORE = 0.3

CRISIS_SKILL_ID = "crisis_support"

_TRIGGER_MAP: dict[str, tuple[str, ...]] = {
    "validation": ("hurt", "alone", "sad", "frustrated", "exhausted", "overwhelmed"),
    # "racing" alone used to collide with sleep_hygiene's "racing thoughts at
    # night" — narrowed to phrasing that's specifically panic/body-symptom
    # language, distinct from the sleep-context "racing thoughts".
    "grounding": ("can't breathe", "spinning", "panic", "racing heart", "heart racing", "chest is tight", "overwhelmed"),
    "journaling": ("write", "get this off my chest", "before bed", "process"),
    "cognitive_reframing": ("fail", "always", "never", "worried", "stuck", "ruminating"),
    "boundary_setting": ("tired of", "expecting me", "pushback", "boundary"),
    "sleep_hygiene": ("sleep", "bed", "night", "racing thoughts", "insomnia"),
}

# Relationship Score uses the already-derived Development level (Vol 3
# Ch 5) rather than a raw trust score, matching what's actually exposed to
# downstream components.
_RELATIONSHIP_LEVEL_SKILL_BOOST: dict[str, set[str]] = {
    "stranger": {"validation"},
    "acquaintance": {"validation", "grounding"},
    "familiar": {"validation", "journaling", "sleep_hygiene"},
    "trusted_companion": {"cognitive_reframing", "boundary_setting"},
    "deep_long_term_companion": {"cognitive_reframing", "boundary_setting"},
}

_EMOTION_SKILL_BOOST: dict[str, set[str]] = {
    "grounding": {"fear", "anxiety"},
    "validation": {"sadness", "grief", "guilt", "anxiety"},
    "cognitive_reframing": {"anger", "disgust"},
    "boundary_setting": {"anger"},
    "journaling": {"sadness"},
}
EMOTION_CONFIDENCE_FLOOR = 0.3

_GOAL_SKILL_BOOST: dict[str, set[str]] = {
    "listen": {"validation"},
    "support": {"validation", "grounding"},
    "advise": {"cognitive_reframing", "boundary_setting"},
    "plan": {"boundary_setting", "sleep_hygiene"},
    "understand": {"journaling"},
}


@dataclass(frozen=True)
class RankedSkill:
    skill: Skill
    score: float
    reason: str
    components: dict[str, float] = field(default_factory=dict)


@lru_cache(maxsize=1)
def _compatibility_table_cached(mtime: float) -> frozenset[tuple[str, str]]:
    if not COMPATIBILITY_PATH.is_file():
        return frozenset()
    data = yaml.safe_load(COMPATIBILITY_PATH.read_text(encoding="utf-8")) or {}
    pairs = data.get("incompatible", [])
    return frozenset(tuple(sorted(pair)) for pair in pairs)


def _compatibility_table() -> frozenset[tuple[str, str]]:
    """Loads `_compatibility.yaml` (Vol 5 Ch 4/13), not a hardcoded pair —
    cached per file mtime so edits to the file (skills are files, not code)
    take effect without a process restart."""
    try:
        mtime = COMPATIBILITY_PATH.stat().st_mtime
    except FileNotFoundError:
        mtime = 0.0
    return _compatibility_table_cached(mtime)


def rank_candidates(
    transcript: str,
    skills: list[Skill],
    profile: UserProfile,
    context: "InterventionContext",
) -> list[RankedSkill]:
    """Dynamic Intervention Score (Vol 1 Ch 8) — every component is named
    and inspectable via `RankedSkill.components`, combined additively
    rather than as an opaque single score, so a bad ranking result can be
    traced to a specific cause."""
    text = transcript.lower()
    ranked: list[RankedSkill] = []
    for skill in skills:
        components: dict[str, float] = {}

        conversation = 0.0
        for trigger in _TRIGGER_MAP.get(skill.id, ()):
            if trigger in text:
                conversation += CONVERSATION_TRIGGER_WEIGHT
                break
        if any(token.lower() in text for token in skill.manifest.when_use):
            conversation += WHEN_USE_WEIGHT
        components["conversation"] = round(conversation, 3)

        profile_score = 0.0
        if profile.response_length == "short" and skill.id in {"grounding", "validation"}:
            profile_score += PROFILE_WEIGHT
        components["profile"] = round(profile_score, 3)

        relationship_score = (
            RELATIONSHIP_WEIGHT
            if skill.id in _RELATIONSHIP_LEVEL_SKILL_BOOST.get(context.development_level, set())
            else 0.0
        )
        components["relationship"] = round(relationship_score, 3)

        # Historical Effectiveness (Vol 3 Ch 6 / Vol 1 Ch 8's Skill
        # Affinity) — read from the profile, never assumed; neutral (0.5)
        # contributes nothing either way.
        affinity = context.skill_affinity.get(skill.id, NEUTRAL_AFFINITY)
        components["historical_effectiveness"] = round((affinity - NEUTRAL_AFFINITY) * 2 * AFFINITY_WEIGHT, 3)

        emotion_score = 0.0
        if context.emotion_confidence >= EMOTION_CONFIDENCE_FLOOR and context.emotion in _EMOTION_SKILL_BOOST.get(
            skill.id, set()
        ):
            emotion_score = EMOTION_WEIGHT
        components["emotion"] = round(emotion_score, 3)

        goal_score = GOAL_WEIGHT if skill.id in _GOAL_SKILL_BOOST.get(context.goal, set()) else 0.0
        components["goal"] = round(goal_score, 3)

        recent_uses = context.recent_skill_ids.count(skill.id)
        components["context_penalty"] = round(-CONTEXT_PENALTY_PER_RECENT_USE * recent_uses, 3)

        score = round(sum(components.values()), 3)
        reason = ", ".join(f"{name}={value:+.2f}" for name, value in components.items() if value) or "baseline"
        ranked.append(RankedSkill(skill=skill, score=score, reason=reason, components=components))

    ranked.sort(key=lambda r: (-r.score, r.skill.id))
    return ranked


def compose_skills(primary: RankedSkill, others: list[RankedSkill]) -> list[RankedSkill]:
    """Vol 5 Ch 13's composition, plus Ch 14/Invariant 8: Crisis Support is
    never composed with another skill, in either direction."""
    if primary.skill.id == CRISIS_SKILL_ID:
        return [primary]

    out = [primary]
    forbidden = {pair for pair in _compatibility_table() if primary.skill.id in pair}
    for item in others:
        if item.skill.id == primary.skill.id or item.skill.id == CRISIS_SKILL_ID:
            continue
        if item.score < MIN_SECONDARY_SCORE:
            continue
        if tuple(sorted((primary.skill.id, item.skill.id))) in forbidden:
            continue
        out.append(item)
        break
    return out
