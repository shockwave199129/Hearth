"""Book Volume 2's Communication Model (Chapter 7) and the mechanics that
support it — the three-layer split (explicit Preferences, learned Traits,
per-conversation Mode), the communication-lifecycle stage/mode inference
(Chapter 2), and the mechanical question-frequency rule (Chapter 6).

Stage/mode inference used to live as private keyword `if` chains inside
CognitiveScheduler; it lives here instead so the Communication Model has one
home, and so prompt_builder.py can reason about the same stage taxonomy
without importing the scheduler."""
from __future__ import annotations

from dataclasses import dataclass

from app.onboarding.profile_schema import UserProfile

# Neutral midpoint for every learned trait until Phase 5's real EWMA
# recomputation (learning/recompute.py) has actual signal — Book Vol 2 Ch 7
# explicitly defers learning, so an un-learned trait must read as "unknown,
# assume nothing" rather than defaulting toward any particular style.
NEUTRAL_TRAIT = 0.5
NEUTRAL_PACE = "moderate"

TRAIT_KEYS = (
    "likes_reflection",
    "likes_direct_advice",
    "prefers_questions",
    "humor_receptiveness",
    "interruption_tolerance",
    "challenge_tolerance",
    "emotional_openness",
)

# Vol 2 Ch 6: after this many consecutive assistant turns ending in a
# question, the next turn must default to reflection/validation/support
# instead of another question.
MAX_CONSECUTIVE_QUESTIONS = 2


@dataclass(frozen=True)
class CommunicationPreferences:
    """Explicit, user-owned (Book Vol 2 Ch 7) — set at onboarding or in
    Settings, never inferred, and never silently overridden by anything
    Hearth learns."""

    preferred_name: str
    voice: str
    emoji_usage: str
    response_length: str
    formality: str

    @classmethod
    def from_profile(cls, profile: UserProfile) -> "CommunicationPreferences":
        return cls(
            preferred_name=profile.name,
            voice=profile.preferred_voice,
            emoji_usage=profile.emoji_usage,
            response_length=profile.response_length,
            formality=profile.communication_formality,
        )


@dataclass(frozen=True)
class CommunicationTraits:
    """Learned/implicit (Book Vol 2 Ch 7) — never asked directly, updated
    only by learning/recompute.py's moving average (Phase 5). Any trait not
    yet present in `profile.communication_traits` stays at the neutral
    midpoint here, which is what "stubbed neutral until Phase 5" means in
    practice: this layer carries no opinion until there's real signal."""

    likes_reflection: float = NEUTRAL_TRAIT
    likes_direct_advice: float = NEUTRAL_TRAIT
    prefers_questions: float = NEUTRAL_TRAIT
    humor_receptiveness: float = NEUTRAL_TRAIT
    interruption_tolerance: float = NEUTRAL_TRAIT
    challenge_tolerance: float = NEUTRAL_TRAIT
    emotional_openness: float = NEUTRAL_TRAIT
    conversation_pace: str = NEUTRAL_PACE

    @classmethod
    def from_profile(cls, profile: UserProfile) -> "CommunicationTraits":
        learned = profile.communication_traits or {}
        overrides = {key: float(learned[key]) for key in TRAIT_KEYS if key in learned}
        return cls(**overrides)


_GREETING_TOKENS = ("good morning", "good evening", "hi", "hello", "hey")
_CLOSING_TOKENS = ("bye", "goodbye", "talk later", "see you")
_PLANNING_TOKENS = ("what should", "what do i do", "help me plan", "next step", "should i")
_SUPPORTING_TOKENS = ("need", "want", "should", "help", "please")
_DISTRESS_TOKENS = ("panic", "overwhelmed", "can't", "cannot", "stuck", "lost")

# hearth_ai's intent classifier (app/nlp/labels.py's INTENT_LABELS) mapped
# onto the communication-lifecycle stage taxonomy — used when the NLP
# workers actually ran this turn (full_path) and returned a confident
# label; keyword heuristics below remain the fallback for fast_path turns
# and for whenever the classifier itself is unavailable or unsure.
_PLANNING_INTENTS = {"plan"}
_UNDERSTANDING_INTENTS = {"inquire"}
_SUPPORTING_INTENTS = {"comfort", "advise", "celebrate"}
_LISTENING_INTENTS = {"vent", "validate", "small_talk", "meta"}
INTENT_CONFIDENCE_FLOOR = 0.25

# hearth_ai's emotion classifier returns Hearth's own small vocabulary
# (app/nlp/labels.py's EMOTION_GO_TO_HEARTH values), not raw GoEmotions —
# these drive CommunicationMode the same way the keyword distress list did.
_DISTRESS_EMOTIONS = {"fear", "sadness", "anxiety", "disgust", "guilt", "grief", "anger"}
_POSITIVE_EMOTIONS = {"joy", "pride", "gratitude", "hope"}
EMOTION_CONFIDENCE_FLOOR = 0.4


def infer_stage(
    transcript: str,
    complexity_level: str,
    *,
    intent: str | None = None,
    intent_confidence: float = 0.0,
) -> str:
    """Communication-lifecycle stage (Book Vol 2 Ch 2): Greeting ->
    Listening -> Understanding -> Exploring -> Supporting -> Planning ->
    Closing. Not strictly linear — a conversation can move backward turn to
    turn, so this only classifies the current transcript, never a running
    position in a fixed sequence.

    `intent`/`intent_confidence` come from hearth_ai's trained intent head
    (MindState.intent, populated only when the NLP workers ran this turn —
    see app/workers/runner.py) and take priority over the keyword fallback
    when confident; greeting/closing detection stays keyword-based since the
    intent vocabulary has no such categories."""
    text = transcript.strip().lower()
    if not text:
        return "listening"
    if any(greet in text for greet in _GREETING_TOKENS) and len(text.split()) <= 5:
        return "greeting"
    if any(end in text for end in _CLOSING_TOKENS):
        return "closing"

    if intent and intent != "unknown" and intent_confidence >= INTENT_CONFIDENCE_FLOOR:
        if intent in _PLANNING_INTENTS:
            return "planning"
        if intent in _UNDERSTANDING_INTENTS:
            return "understanding"
        if intent in _SUPPORTING_INTENTS:
            return "supporting"
        if intent in _LISTENING_INTENTS:
            return "understanding" if "?" in text else "listening"

    if complexity_level == "fast_path":
        return "listening"
    if "?" in text:
        return "understanding"
    if any(token in text for token in _PLANNING_TOKENS):
        return "planning"
    if any(token in text for token in _SUPPORTING_TOKENS):
        return "supporting"
    return "exploring"


def infer_mode(
    transcript: str,
    stage: str,
    *,
    emotion: str | None = None,
    emotion_confidence: float = 0.0,
) -> str:
    """Per-conversation CommunicationMode.current_style (Book Vol 2 Ch 7) —
    overrides long-term traits whenever the immediate moment calls for it.
    `emotion`/`emotion_confidence` come from hearth_ai's trained emotion
    head (MindState.emotion), populated only on turns where the NLP workers
    ran; the keyword distress list remains the fallback."""
    if stage in {"greeting", "closing"}:
        return "gentle"
    if emotion and emotion_confidence >= EMOTION_CONFIDENCE_FLOOR and emotion in _DISTRESS_EMOTIONS:
        return "calm"
    if any(token in transcript.lower() for token in _DISTRESS_TOKENS):
        return "calm"
    return "warm"


def is_celebration_moment(intent: str | None, emotion: str | None, nlp_available: bool) -> bool:
    """Book Vol 2 Ch 17/24: flat responses to good news are a named
    anti-pattern — only trust this off real classifier signal, never a
    keyword guess, since false positives here read as forced enthusiasm."""
    if not nlp_available:
        return False
    return intent == "celebrate" or emotion in _POSITIVE_EMOTIONS


def update_question_streak(prior_reply: str | None, streak: int) -> int:
    """Mechanical enforcement of Vol 2 Ch 6's question rule: after one or two
    questions asked without a substantive statement in between, Hearth
    should default to reflection/validation/support rather than another
    question. `prior_reply` is Hearth's own previous turn — if it ended in a
    question the streak grows, otherwise a substantive (non-question)
    statement resets it to zero."""
    if not prior_reply:
        return 0
    return streak + 1 if prior_reply.rstrip().endswith("?") else 0


def must_suppress_question(streak: int) -> bool:
    return streak >= MAX_CONSECUTIVE_QUESTIONS
