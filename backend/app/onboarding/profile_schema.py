"""See project-plan.md §4 — short, mostly-optional profile, bucketed age
range rather than raw DOB to minimize sensitive data at rest."""
from datetime import datetime

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    user_id: str
    name: str
    age_range: str | None = None       # e.g. "18-24", "25-34" — never exact DOB
    gender: str | None = None          # optional, used for voice/pronoun defaults
    profession: str | None = None
    stressors: list[str] = Field(default_factory=list)
    preferred_voice: str = "female"    # "male" | "female"
    companion_name: str = "Companion"
    # Whether replies get synthesized as audio at all — on by default; the
    # one exception is the crisis-response path (safety/escalation.py's
    # sibling, main.py's _respond_to_crisis), which always speaks regardless
    # of this, since the pre-synthesized safety audio existing at all was a
    # deliberate safety-critical design choice, not something this
    # convenience toggle should be able to silence.
    speak_replies: bool = True
    # Opt-in emergency contact for the safety escalation path (project-plan.md
    # §9) — off by default, and escalation.py additionally requires a
    # repeated crisis pattern before ever acting on this. See
    # app/safety/escalation.py's module docstring for the full gating logic.
    emergency_contact_consent: bool = False
    emergency_contact_name: str | None = None
    emergency_contact_method: str | None = None    # "sms" | "email"
    emergency_contact_value: str | None = None
    created_at: datetime


class OnboardingRequest(BaseModel):
    """Same fields as UserProfile minus created_at, which the server stamps."""
    name: str
    age_range: str | None = None
    gender: str | None = None
    profession: str | None = None
    stressors: list[str] = Field(default_factory=list)
    preferred_voice: str = "female"
    companion_name: str = "Companion"
    speak_replies: bool = True
    emergency_contact_consent: bool = False
    emergency_contact_name: str | None = None
    emergency_contact_method: str | None = None
    emergency_contact_value: str | None = None
