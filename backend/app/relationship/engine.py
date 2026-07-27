"""Simple phase 2 relationship tracking."""

from dataclasses import dataclass

from app.onboarding.profile_schema import UserProfile


@dataclass(frozen=True)
class RelationshipState:
    general_trust: float
    vulnerability_trust: float
    advice_trust: float
    consistency_confidence: float
    boundaries: str
    life_model: str


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def update_relationship(profile: UserProfile, transcript: str, reply_text: str, memory_formed: int = 0) -> RelationshipState:
    text = transcript.lower()
    reply = reply_text.lower()
    general = profile.relationship_general_trust
    vulnerability = profile.relationship_vulnerability_trust
    advice = profile.relationship_advice_trust
    consistency = profile.relationship_consistency_confidence

    if any(token in text for token in ("i feel", "i’m feeling", "i am feeling", "i'm scared", "i'm overwhelmed", "i need to tell you")):
        vulnerability = _clamp(vulnerability + 0.05)
        general = _clamp(general + 0.03)
    if any(token in text for token in ("what should i do", "help me decide", "should i", "any advice")):
        advice = _clamp(advice + 0.04)
        general = _clamp(general + 0.02)
    if memory_formed:
        consistency = _clamp(consistency + min(0.04 * memory_formed, 0.12))
        general = _clamp(general + min(0.02 * memory_formed, 0.06))
    if any(token in reply for token in ("i'm here", "i’m here", "that sounds", "it makes sense")):
        consistency = _clamp(consistency + 0.01)

    boundaries = profile.relationship_boundaries
    if any(token in text for token in ("please don't", "not ready", "too much", "stop asking")):
        boundaries = "firm"
    elif any(token in text for token in ("maybe later", "not today", "space")):
        boundaries = "gentle"

    life_model = profile.relationship_life_model
    if any(token in text for token in ("my family", "my job", "my partner", "my friend", "my doctor", "my manager")):
        life_model = "contextualized"

    return RelationshipState(
        general_trust=round(general, 3),
        vulnerability_trust=round(vulnerability, 3),
        advice_trust=round(advice, 3),
        consistency_confidence=round(consistency, 3),
        boundaries=boundaries,
        life_model=life_model,
    )

