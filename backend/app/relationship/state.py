"""Book Volume 3 — the consolidated `RelationshipState` object (Ch 10).

Named `RelationshipProfile` here to avoid colliding with the existing
`app.relationship.engine.RelationshipState` dataclass, which stays exactly
as-is: it's the fast, per-turn *cached* view (general_trust/
vulnerability_trust/advice_trust/consistency_confidence + boundaries/
life_model strings) stored as flat columns on Profile — precisely the "the
current, cached trust values live in Profile for fast runtime access" role
Chapter 3 describes. `RelationshipProfile` is the fuller, versioned object
Chapter 10 consolidates from Chapters 3-9 (Trust, Attachment, Development,
Affinity, Boundaries, Life Model, Shared History); only the Growth Engine
(`app.growth.engine`) writes to it — everything else only reads."""
from __future__ import annotations

import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field

RELATIONSHIP_SCHEMA_VERSION = 1

# Volume 2 Ch 11's five Relationship Development levels, low -> high.
DEVELOPMENT_LEVELS = ("stranger", "acquaintance", "familiar", "trusted_companion", "deep_long_term_companion")


class TrustModel(BaseModel):
    """Book Vol 3 Ch 3 — trust across several distinct dimensions, never a
    single number; never computed from session length or message frequency."""

    general_trust: float = 0.0
    vulnerability_trust: float = 0.0
    advice_trust: float = 0.0
    consistency_confidence: float = 0.0


class AttachmentSignals(BaseModel):
    """Book Vol 3 Ch 4 — computed and logged only. These are a *check*, not
    an optimization target, and never trigger escalation on their own
    (Phase 4's Safety Worker is a separate, independent gate)."""

    escalating_contact_frequency: bool = False
    replacement_language_detected: bool = False
    distress_about_unavailability: bool = False
    healthy_engagement_with_others: bool = True
    # Book Vol 7 Ch 8 — the three streams (contact-urgency trend,
    # replacement language, unavailability distress) folded into one
    # higher-alpha moving average by app.learning.attachment. Never
    # optimized to increase; purely a check feeding Phase 4's escalation.
    combined_score: float = 0.0
    last_evaluated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def has_warning_signal(self) -> bool:
        return (
            self.escalating_contact_frequency
            or self.replacement_language_detected
            or self.distress_about_unavailability
        )


class UserBoundaries(BaseModel):
    """Book Vol 3 Ch 7 — populated conservatively; flagged, not permanently
    forbidden. Never loosens as Development level increases."""

    avoid_topics: list[str] = Field(default_factory=list)
    comfort_topics: list[str] = Field(default_factory=list)
    disclosed_boundaries: list[str] = Field(default_factory=list)
    sensitivity_flags: list[str] = Field(default_factory=list)


class HearthBoundaryState(BaseModel):
    """Which of Hearth's own boundaries (Vol 0 Non-Goals) have already been
    tested/established in this relationship — tracked so a boundary, once
    warmly established, doesn't need re-explaining from scratch."""

    asked_to_be_therapist: bool = False
    asked_to_be_decision_maker: bool = False
    asked_to_replace_human_relationship: bool = False
    notes: list[str] = Field(default_factory=list)


class ImportantPerson(BaseModel):
    name: str
    relationship_type: str = "unknown"
    sentiment_context: str = "neutral"
    last_mentioned: datetime


class OngoingSituation(BaseModel):
    topic: str
    status: str = "ongoing"
    first_mentioned: datetime
    last_mentioned: datetime


class LifeModel(BaseModel):
    """Book Vol 3 Ch 8 — "who and what matters in this person's life", a
    structural layer distinct from Volume 4's general memory (which answers
    "what was said"). Entries deprioritize over time; they are never
    deleted here (that's Volume 4's memory-deletion path, not this one's)."""

    important_people: list[ImportantPerson] = Field(default_factory=list)
    ongoing_situations: list[OngoingSituation] = Field(default_factory=list)
    recurring_themes: list[str] = Field(default_factory=list)


class SharedHistoryEntry(BaseModel):
    """Book Vol 3 Ch 9 — a curated, high-confidence subset of the past,
    distinct from raw stored memory. `times_referenced` tracks reuse so
    eligibility decreases the more it's invoked."""

    summary: str
    emotional_significance: str
    times_referenced: int = 0
    relationship_level_required: str = "familiar"


class RelationshipProfile(BaseModel):
    """Book Vol 3 Ch 10 — consolidates every concept in this volume into one
    versioned, persistent object. Lives in Profile; read-only during a live
    conversation; only the Growth Engine writes to it (Ch 13, Invariant 6)."""

    schema_version: int = RELATIONSHIP_SCHEMA_VERSION
    user_id: str
    trust: TrustModel = Field(default_factory=TrustModel)
    attachment_signals: AttachmentSignals = Field(default_factory=AttachmentSignals)
    development_level: str = "stranger"
    skill_affinity: dict[str, float] = Field(default_factory=dict)
    communication_traits: dict[str, float] = Field(default_factory=dict)
    boundaries: UserBoundaries = Field(default_factory=UserBoundaries)
    hearth_boundary_state: HearthBoundaryState = Field(default_factory=HearthBoundaryState)
    life_model: LifeModel = Field(default_factory=LifeModel)
    shared_history: list[SharedHistoryEntry] = Field(default_factory=list)
    conversation_count: int = 0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------
# Development level (Vol 3 Ch 5) — derived, never a counter; can move
# backward, never a stale high-water mark.
# --------------------------------------------------------------------------


def _traits_calibration(traits: dict[str, float]) -> float:
    """How far learned Communication Traits have moved from the neutral
    midpoint — a proxy for "has learning had enough signal to be reliable
    yet", per Vol 3 Ch 5's "communication familiarity" requirement."""
    if not traits:
        return 0.0
    deviations = [abs(v - 0.5) for v in traits.values()]
    return sum(deviations) / len(deviations)


def compute_development_level(
    trust: TrustModel,
    *,
    conversation_count: int,
    disclosure_depth: float = 0.0,
    communication_traits: dict[str, float] | None = None,
    days_since_last_contact: float = 0.0,
) -> str:
    """Level is a derived value (Vol 3 Ch 5): trust as a floor requirement,
    disclosure depth, communication familiarity, and duration/count as a
    floor (never a driver). A long contact gap moves the level back down."""
    communication_traits = communication_traits or {}
    general = trust.general_trust
    calibration = _traits_calibration(communication_traits)

    if days_since_last_contact >= 180:
        return "acquaintance" if general >= 0.3 else "stranger"

    if general >= 0.75 and disclosure_depth >= 0.7 and conversation_count >= 30 and calibration >= 0.15:
        return "deep_long_term_companion"
    if general >= 0.55 and disclosure_depth >= 0.5 and conversation_count >= 12:
        return "trusted_companion"
    if general >= 0.35 and conversation_count >= 6 and calibration >= 0.05:
        return "familiar"
    if general >= 0.15 and conversation_count >= 2:
        return "acquaintance"
    return "stranger"


# --------------------------------------------------------------------------
# Attachment signals (Vol 3 Ch 4) — rule-based, computed + logged only.
# --------------------------------------------------------------------------

_REPLACEMENT_LANGUAGE = re.compile(
    r"\b(you'?re all i have|only one who understands me|don'?t need anyone else|"
    r"replace my (friends|family|therapist))\b",
    re.IGNORECASE,
)
_UNAVAILABILITY_DISTRESS = re.compile(
    r"\b(why weren'?t you (here|there)|you were gone|i hate when you'?re not here|"
    r"i panicked when i couldn'?t reach you)\b",
    re.IGNORECASE,
)


def evaluate_attachment_signals(
    recent_user_messages: list[str], *, message_frequency_ratio: float = 1.0
) -> AttachmentSignals:
    """Vol 3 Ch 4 — never used to change how Hearth talks directly; surfaced
    for the Intervention Engine (Phase 3) to weight autonomy-reinforcing
    interventions more highly, and never to unilaterally withdraw warmth."""
    joined = " ".join(recent_user_messages).lower()
    replacement = bool(_REPLACEMENT_LANGUAGE.search(joined))
    unavailability_distress = bool(_UNAVAILABILITY_DISTRESS.search(joined))
    escalating = message_frequency_ratio >= 2.0
    return AttachmentSignals(
        escalating_contact_frequency=escalating,
        replacement_language_detected=replacement,
        distress_about_unavailability=unavailability_distress,
        healthy_engagement_with_others=not replacement,
    )


# --------------------------------------------------------------------------
# Boundaries (Vol 3 Ch 7) — populated conservatively, rule-based.
# --------------------------------------------------------------------------

_AVOID_SIGNAL = re.compile(
    r"\b(please don'?t (ask|bring that up)|i'?d rather not talk about|stop asking about)\b", re.IGNORECASE
)
_COMFORT_SIGNAL = re.compile(r"\b(i love talking about|feel free to ask about|happy to discuss)\b", re.IGNORECASE)


def update_boundaries(existing: UserBoundaries, transcript: str, topic: str | None) -> UserBoundaries:
    avoid = list(existing.avoid_topics)
    comfort = list(existing.comfort_topics)
    if topic and _AVOID_SIGNAL.search(transcript) and topic not in avoid:
        avoid.append(topic)
    if topic and _COMFORT_SIGNAL.search(transcript) and topic not in comfort:
        comfort.append(topic)
    return existing.model_copy(update={"avoid_topics": avoid, "comfort_topics": comfort})


# --------------------------------------------------------------------------
# Shared History (Vol 3 Ch 9) — curated subset of memory2 episodics.
# --------------------------------------------------------------------------

SHARED_HISTORY_INTENSITY_FLOOR = 0.6
SHARED_HISTORY_MARKER_ALLOWLIST = {"vulnerable_disclosure", "life_event", "stated_goal_or_plan"}


def derive_shared_history_candidates(episodic_summaries: list[tuple[str, float, list[str]]]) -> list[SharedHistoryEntry]:
    """`episodic_summaries` is (summary, emotional_intensity, significance_markers)
    tuples. Only meaningfully significant moments become eligible — this is
    curation, not a mirror of everything stored in Volume 4's memory."""
    entries = []
    for summary, intensity, markers in episodic_summaries:
        if intensity < SHARED_HISTORY_INTENSITY_FLOOR:
            continue
        if not (set(markers) & SHARED_HISTORY_MARKER_ALLOWLIST):
            continue
        entries.append(
            SharedHistoryEntry(
                summary=summary,
                emotional_significance=f"intensity={intensity:.2f}",
                relationship_level_required="familiar",
            )
        )
    return entries
