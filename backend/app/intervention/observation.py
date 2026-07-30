"""Skill Effectiveness Feedback (Book Vol 5 Ch 16) — records an immutable
SkillObservation for every skill use. This module defines only what gets
*reported*; computing effectiveness from accumulated observations is
Volume 6's Learning & Growth Engine's job, not this module's.

`user_reaction_signal` and `emotional_shift` are both computed from
existing, non-LLM signals (the real Emotion classifier already running for
the next turn) — no new "was this skill effective?" LLM judgment call is
introduced here, consistent with Volume 4's minimal-LLM-use discipline.

Because effectiveness depends on what happens *after* a skill is used, this
is a two-step, cross-turn protocol:
  1. `mark_skill_used` — called the turn a skill is chosen; records what the
     conversation looked like right before the reply (emotion, stage).
  2. `resolve_pending_observation` — called at the start of the *next*
     turn, once that turn's own signals exist, comparing before/after and
     appending the actual SkillObservation. Never asserted from a single
     conversation; Volume 6 infers effectiveness from many of these."""
from __future__ import annotations

from app.cognitive.mind_state import MindState
from app.learning.observation_store import ObservationStore
from app.skills.loader import Skill

_EMOTION_VALENCE: dict[str, int] = {
    "joy": 1,
    "pride": 1,
    "gratitude": 1,
    "hope": 1,
    "sadness": -1,
    "fear": -1,
    "anger": -1,
    "disgust": -1,
    "guilt": -1,
    "grief": -1,
    "anxiety": -1,
    "neutral": 0,
    "unknown": 0,
    "surprise": 0,
}

_DISENGAGED_MESSAGES = {"ok", "okay", "fine", "sure", "whatever", "nevermind", "nvm", "k"}
CONTINUED_ENGAGEMENT_WORD_FLOOR = 6


def _valence(emotion: str) -> int:
    return _EMOTION_VALENCE.get(emotion, 0)


def mark_skill_used(mind_state: MindState, *, skill: Skill, composed_with: str | None) -> None:
    """Called the turn a skill is actually used as `primary_skill` — its
    version and the emotion/stage right before the reply are the "before"
    half of the before/after comparison Ch 16 asks for."""
    mind_state.pending_skill_id = skill.id
    mind_state.pending_skill_version = skill.manifest.version
    mind_state.pending_skill_composed_with = composed_with
    mind_state.pending_skill_emotion_before = mind_state.emotion
    mind_state.pending_skill_stage = mind_state.stage


def _user_reaction_signal(new_user_message: str) -> str:
    normalized = new_user_message.strip().lower()
    word_count = len(normalized.split())
    if normalized in _DISENGAGED_MESSAGES:
        return "disengaged"
    if word_count >= CONTINUED_ENGAGEMENT_WORD_FLOOR:
        return "continued_engagement"
    return "neutral_continuation"


def resolve_pending_observation(
    mind_state: MindState, *, new_user_message: str, store: ObservationStore
) -> None:
    """Call at the start of the next turn, after that turn's NLP workers
    (if any ran) have updated `mind_state.emotion` — compares it against
    the emotion recorded when the skill was used, and records the
    SkillObservation. A no-op if no skill is pending."""
    if not mind_state.pending_skill_id:
        return

    emotion_after = mind_state.emotion if mind_state.nlp_available else mind_state.pending_skill_emotion_before
    before = _valence(mind_state.pending_skill_emotion_before)
    after = _valence(emotion_after)
    if after > before:
        emotional_shift = "improved"
    elif after < before:
        emotional_shift = "worsened"
    else:
        emotional_shift = "unchanged"

    user_reaction_signal = _user_reaction_signal(new_user_message)

    value = {"improved": 1.0, "unchanged": 0.6, "worsened": 0.2}[emotional_shift]
    if user_reaction_signal == "disengaged":
        value = min(value, 0.3)

    # Templated, not LLM prose (Vol 4's non-LLM approach, referenced
    # directly by Ch 16 for this exact field).
    context_summary = f"{mind_state.pending_skill_stage} — {mind_state.pending_skill_emotion_before}"

    store.append(
        "skill",
        mind_state.pending_skill_id,
        value,
        {
            "skill_version": mind_state.pending_skill_version,
            "conversation_context_summary": context_summary,
            "composed_with": mind_state.pending_skill_composed_with,
            "user_reaction_signal": user_reaction_signal,
            "emotional_shift": emotional_shift,
        },
        "intervention_engine",
    )

    mind_state.pending_skill_id = None
    mind_state.pending_skill_version = None
    mind_state.pending_skill_composed_with = None
