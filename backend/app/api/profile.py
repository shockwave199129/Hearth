"""Profile / onboarding / multi-profile routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.checkin.state import delete_checkin
from app.deps import Pipeline, get_pipeline, get_pipeline_optional
from app.memory import chat_history, long_term
from app.memory2 import privacy as memory2_privacy
from app.onboarding.active_profile import clear_active_user_id, get_active_user_id, set_active_user_id
from app.onboarding.profile_schema import OnboardingRequest, UserProfile
from app.onboarding.profile_store import (
    create_profile,
    delete_profile,
    get_profile,
    list_profiles,
    update_communication_preferences,
    update_region,
    update_speak_replies,
    update_voice_preferences,
)
from app.pipeline import DEFAULT_PROFILE
from app.relationship.profile_store import delete_relationship_profile
from app.safety import crisis_detector, escalation
from app.tts.voice_styles import VOICE_STYLE_IDS, VOICES

router = APIRouter()


@router.get("/api/profile")
def api_get_profile() -> UserProfile:
    """404 (no active profile) is how the frontend tells 'never onboarded'
    apart from 'onboarded with default-ish answers'."""
    user_id = get_active_user_id()
    profile = get_profile(user_id) if user_id else None
    if profile is None:
        raise HTTPException(status_code=404, detail="no profile saved yet")
    return profile


class ProfileSettingsUpdate(BaseModel):
    speak_replies: bool
    communication_formality: str | None = None
    response_length: str | None = None
    emoji_usage: str | None = None
    preferred_voice: str | None = None
    voice_style: str | None = None
    region: str | None = None


@router.put("/api/profile")
def api_update_profile(
    payload: ProfileSettingsUpdate,
    pipeline: Pipeline = Depends(get_pipeline),
) -> UserProfile:
    """Lets Settings flip lightweight preferences (speak_replies, the
    spoken voice + speaking-style preset, plus the explicit
    CommunicationPreferences from Book Vol 2 Ch 7 — formality, response
    length, emoji usage) without redoing the whole onboarding flow. These
    are user-owned and never silently overridden by anything Hearth learns
    (Book Vol 2 Ch 7)."""
    user_id = get_active_user_id()
    profile = get_profile(user_id) if user_id else None
    if profile is None:
        raise HTTPException(status_code=404, detail="no profile saved yet")
    update_speak_replies(user_id, payload.speak_replies)
    if payload.preferred_voice is not None or payload.voice_style is not None:
        preferred_voice = payload.preferred_voice or profile.preferred_voice
        voice_style = payload.voice_style or profile.voice_style
        # Strict here even though the TTS path falls back — a rejected write
        # is a bug the caller can see, a silently coerced one isn't.
        if preferred_voice not in VOICES:
            raise HTTPException(status_code=400, detail=f"unknown voice {preferred_voice!r}")
        if voice_style not in VOICE_STYLE_IDS:
            raise HTTPException(status_code=400, detail=f"unknown voice style {voice_style!r}")
        update_voice_preferences(user_id, preferred_voice=preferred_voice, voice_style=voice_style)
    if payload.communication_formality is not None and payload.response_length is not None:
        update_communication_preferences(
            user_id,
            communication_formality=payload.communication_formality,
            response_length=payload.response_length,
            emoji_usage=payload.emoji_usage,
        )
    if payload.region is not None:
        update_region(user_id, payload.region)
    updated = get_profile(user_id)
    # A plain attribute swap, not set_profile() — these preferences only
    # affect prompt shaping and the per-call TTS arguments, not the runtime
    # tier or profile identity. No engine reload: voice and style are read
    # off the profile at each synthesize() call, so the next reply already
    # speaks the new way.
    pipeline.profile = updated
    return updated


@router.post("/api/onboarding")
def api_complete_onboarding(payload: OnboardingRequest) -> UserProfile:
    """Creates a new profile and activates it — used for first-run
    onboarding AND for adding another profile later (Settings → Profiles →
    Add another profile reuses this same form/endpoint).

    Profile + active_user_id are persisted first so a later launch still
    skips onboarding even if wiring the live Pipeline fails mid-request.
    """
    profile = create_profile(payload)
    set_active_user_id(profile.user_id)
    pipeline = get_pipeline_optional()
    if pipeline is not None:
        pipeline.set_profile(profile)
    return profile


@router.get("/api/profiles")
def api_list_profiles() -> list[UserProfile]:
    return list_profiles()


@router.post("/api/profiles/{user_id}/activate")
def api_activate_profile(user_id: str, pipeline: Pipeline = Depends(get_pipeline)) -> UserProfile:
    profile = get_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="profile not found")
    set_active_user_id(user_id)
    pipeline.set_profile(profile)
    return profile


@router.delete("/api/profiles/{user_id}")
def api_delete_profile(user_id: str, pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    """Cascades across every user_id-scoped table — memories, checkin,
    crisis/escalation history, and chat history — never a partial delete."""
    if get_profile(user_id) is None:
        raise HTTPException(status_code=404, detail="profile not found")
    was_active = get_active_user_id() == user_id

    delete_profile(user_id)
    long_term.delete_all_for_user(user_id)
    delete_checkin(user_id)
    crisis_detector.delete_events(user_id)
    escalation.delete_escalations(user_id)
    chat_history.delete_all_for_user(user_id)
    delete_relationship_profile(user_id)

    memory2_privacy.delete_all_memory(pipeline.growth_engine.store, user_id)
    if was_active:
        remaining = list_profiles()
        if remaining:
            set_active_user_id(remaining[0].user_id)
            pipeline.set_profile(remaining[0])
        else:
            clear_active_user_id()
            pipeline.set_profile(DEFAULT_PROFILE)
    return {"ok": True}
