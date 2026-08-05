"""CRUD for the `profiles` table — multi-profile support (docs/project-plan.md
§4 as extended for real profile switching). Each profile is injected into
the system prompt every session as static context, not something to
re-retrieve via search. Cross-module cascade deletion (memory, checkin,
crisis/escalation, chat_history) is orchestrated by main.py's
DELETE /api/profiles/{user_id} handler, not here, to avoid a circular
import (safety/escalation.py already imports this module)."""
import json
import uuid
from datetime import datetime, timezone

from app.config import DATA_DIR
from app.db.sqlite_models import get_connection
from app.onboarding.profile_schema import OnboardingRequest, UserProfile

PROFILE_DB_PATH = DATA_DIR / "profile.db"

_COLUMNS = (
    "user_id, name, age_range, gender, profession, stressors, preferred_voice, voice_style, companion_name, "
    "communication_formality, response_length, emoji_usage, speak_replies, emergency_contact_consent, emergency_contact_name, emergency_contact_method, "
    "emergency_contact_value, relationship_general_trust, relationship_vulnerability_trust, "
    "relationship_advice_trust, relationship_consistency_confidence, relationship_boundaries, relationship_life_model, "
    "communication_traits_json, skill_affinity_json, evaluation_last_run_at, region, created_at"
)


def _row_to_profile(row) -> UserProfile:
    (
        user_id,
        name,
        age_range,
        gender,
        profession,
        stressors_json,
        preferred_voice,
        voice_style,
        companion_name,
        communication_formality,
        response_length,
        emoji_usage,
        speak_replies,
        emergency_contact_consent,
        emergency_contact_name,
        emergency_contact_method,
        emergency_contact_value,
        relationship_general_trust,
        relationship_vulnerability_trust,
        relationship_advice_trust,
        relationship_consistency_confidence,
        relationship_boundaries,
        relationship_life_model,
        communication_traits_json,
        skill_affinity_json,
        evaluation_last_run_at,
        region,
        created_at,
    ) = row
    return UserProfile(
        user_id=user_id,
        name=name,
        age_range=age_range,
        gender=gender,
        profession=profession,
        stressors=json.loads(stressors_json),
        preferred_voice=preferred_voice,
        voice_style=voice_style,
        companion_name=companion_name,
        communication_formality=communication_formality,
        response_length=response_length,
        emoji_usage=emoji_usage,
        speak_replies=bool(speak_replies),
        emergency_contact_consent=bool(emergency_contact_consent),
        emergency_contact_name=emergency_contact_name,
        emergency_contact_method=emergency_contact_method,
        emergency_contact_value=emergency_contact_value,
        relationship_general_trust=float(relationship_general_trust or 0.0),
        relationship_vulnerability_trust=float(relationship_vulnerability_trust or 0.0),
        relationship_advice_trust=float(relationship_advice_trust or 0.0),
        relationship_consistency_confidence=float(relationship_consistency_confidence or 0.0),
        relationship_boundaries=relationship_boundaries,
        relationship_life_model=relationship_life_model,
        communication_traits=json.loads(communication_traits_json or "{}"),
        skill_affinity=json.loads(skill_affinity_json or "{}"),
        evaluation_last_run_at=datetime.fromisoformat(evaluation_last_run_at) if evaluation_last_run_at else None,
        region=region,
        created_at=datetime.fromisoformat(created_at),
    )


def create_profile(payload: OnboardingRequest) -> UserProfile:
    profile = UserProfile(
        user_id=str(uuid.uuid4()), created_at=datetime.now(timezone.utc), **payload.model_dump()
    )
    conn = get_connection(PROFILE_DB_PATH)
    try:
        conn.execute(
            f"INSERT INTO profiles ({_COLUMNS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                profile.user_id,
                profile.name,
                profile.age_range,
                profile.gender,
                profile.profession,
                json.dumps(profile.stressors),
                profile.preferred_voice,
                profile.voice_style,
                profile.companion_name,
                profile.communication_formality,
                profile.response_length,
                profile.emoji_usage,
                int(profile.speak_replies),
                int(profile.emergency_contact_consent),
                profile.emergency_contact_name,
                profile.emergency_contact_method,
                profile.emergency_contact_value,
                profile.relationship_general_trust,
                profile.relationship_vulnerability_trust,
                profile.relationship_advice_trust,
                profile.relationship_consistency_confidence,
                profile.relationship_boundaries,
                profile.relationship_life_model,
                json.dumps(profile.communication_traits),
                json.dumps(profile.skill_affinity),
                profile.evaluation_last_run_at.isoformat() if profile.evaluation_last_run_at else None,
                profile.region,
                profile.created_at.isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return profile


def update_speak_replies(user_id: str, value: bool) -> None:
    conn = get_connection(PROFILE_DB_PATH)
    try:
        conn.execute("UPDATE profiles SET speak_replies = ? WHERE user_id = ?", (int(value), user_id))
        conn.commit()
    finally:
        conn.close()


def update_voice_preferences(user_id: str, *, preferred_voice: str, voice_style: str) -> None:
    """Which voice speaks, and how. Callers validate against
    tts.voice_styles first — this writes whatever it's given."""
    conn = get_connection(PROFILE_DB_PATH)
    try:
        conn.execute(
            "UPDATE profiles SET preferred_voice = ?, voice_style = ? WHERE user_id = ?",
            (preferred_voice, voice_style, user_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_communication_preferences(
    user_id: str, *, communication_formality: str, response_length: str, emoji_usage: str | None = None
) -> None:
    conn = get_connection(PROFILE_DB_PATH)
    try:
        if emoji_usage is None:
            conn.execute(
                "UPDATE profiles SET communication_formality = ?, response_length = ? WHERE user_id = ?",
                (communication_formality, response_length, user_id),
            )
        else:
            conn.execute(
                "UPDATE profiles SET communication_formality = ?, response_length = ?, emoji_usage = ? WHERE user_id = ?",
                (communication_formality, response_length, emoji_usage, user_id),
            )
        conn.commit()
    finally:
        conn.close()


def update_region(user_id: str, region: str | None) -> None:
    conn = get_connection(PROFILE_DB_PATH)
    try:
        conn.execute("UPDATE profiles SET region = ? WHERE user_id = ?", (region, user_id))
        conn.commit()
    finally:
        conn.close()


def update_relationship_state(
    user_id: str,
    *,
    relationship_general_trust: float,
    relationship_vulnerability_trust: float,
    relationship_advice_trust: float,
    relationship_consistency_confidence: float,
    relationship_boundaries: str,
    relationship_life_model: str,
) -> None:
    conn = get_connection(PROFILE_DB_PATH)
    try:
        conn.execute(
            """
            UPDATE profiles
            SET relationship_general_trust = ?,
                relationship_vulnerability_trust = ?,
                relationship_advice_trust = ?,
                relationship_consistency_confidence = ?,
                relationship_boundaries = ?,
                relationship_life_model = ?
            WHERE user_id = ?
            """,
            (
                relationship_general_trust,
                relationship_vulnerability_trust,
                relationship_advice_trust,
                relationship_consistency_confidence,
                relationship_boundaries,
                relationship_life_model,
                user_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def update_learning_state(
    user_id: str,
    *,
    communication_traits: dict[str, float],
    skill_affinity: dict[str, float],
    evaluation_last_run_at: datetime | None,
) -> None:
    conn = get_connection(PROFILE_DB_PATH)
    try:
        conn.execute(
            """
            UPDATE profiles
            SET communication_traits_json = ?,
                skill_affinity_json = ?,
                evaluation_last_run_at = ?
            WHERE user_id = ?
            """,
            (
                json.dumps(communication_traits),
                json.dumps(skill_affinity),
                evaluation_last_run_at.isoformat() if evaluation_last_run_at else None,
                user_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_profile(user_id: str) -> UserProfile | None:
    conn = get_connection(PROFILE_DB_PATH)
    try:
        row = conn.execute(f"SELECT {_COLUMNS} FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
    finally:
        conn.close()
    return _row_to_profile(row) if row is not None else None


def list_profiles() -> list[UserProfile]:
    conn = get_connection(PROFILE_DB_PATH)
    try:
        rows = conn.execute(f"SELECT {_COLUMNS} FROM profiles ORDER BY created_at ASC").fetchall()
    finally:
        conn.close()
    return [_row_to_profile(row) for row in rows]


def delete_profile(user_id: str) -> None:
    conn = get_connection(PROFILE_DB_PATH)
    try:
        conn.execute("DELETE FROM profiles WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()
