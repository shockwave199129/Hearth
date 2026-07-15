"""CRUD for the `profiles` table — multi-profile support (project-plan.md
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
    "user_id, name, age_range, gender, profession, stressors, preferred_voice, companion_name, "
    "speak_replies, emergency_contact_consent, emergency_contact_name, emergency_contact_method, "
    "emergency_contact_value, created_at"
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
        companion_name,
        speak_replies,
        emergency_contact_consent,
        emergency_contact_name,
        emergency_contact_method,
        emergency_contact_value,
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
        companion_name=companion_name,
        speak_replies=bool(speak_replies),
        emergency_contact_consent=bool(emergency_contact_consent),
        emergency_contact_name=emergency_contact_name,
        emergency_contact_method=emergency_contact_method,
        emergency_contact_value=emergency_contact_value,
        created_at=datetime.fromisoformat(created_at),
    )


def create_profile(payload: OnboardingRequest) -> UserProfile:
    profile = UserProfile(
        user_id=str(uuid.uuid4()), created_at=datetime.now(timezone.utc), **payload.model_dump()
    )
    conn = get_connection(PROFILE_DB_PATH)
    try:
        conn.execute(
            f"INSERT INTO profiles ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                profile.user_id,
                profile.name,
                profile.age_range,
                profile.gender,
                profile.profession,
                json.dumps(profile.stressors),
                profile.preferred_voice,
                profile.companion_name,
                int(profile.speak_replies),
                int(profile.emergency_contact_consent),
                profile.emergency_contact_name,
                profile.emergency_contact_method,
                profile.emergency_contact_value,
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
