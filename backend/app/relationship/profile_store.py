"""CRUD for `relationship_profiles` — the persisted store for Book Volume
3's full `RelationshipProfile` (app.relationship.state). Same db file as
`onboarding/profile_store.py` (profile.db), separate table, since this is a
distinct versioned object rather than more flat columns on `profiles`.
Only the Growth Engine (`app.growth.engine`) should call `save`."""
from __future__ import annotations

from datetime import datetime, timezone

from app.config import DATA_DIR
from app.db.sqlite_models import get_connection
from app.relationship.state import RelationshipProfile

RELATIONSHIP_PROFILE_DB_PATH = DATA_DIR / "profile.db"


def get_relationship_profile(user_id: str) -> RelationshipProfile | None:
    conn = get_connection(RELATIONSHIP_PROFILE_DB_PATH)
    try:
        row = conn.execute(
            "SELECT profile_json FROM relationship_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return RelationshipProfile.model_validate_json(row[0])


def save_relationship_profile(profile: RelationshipProfile) -> None:
    conn = get_connection(RELATIONSHIP_PROFILE_DB_PATH)
    try:
        conn.execute(
            """INSERT INTO relationship_profiles (user_id, profile_json, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET profile_json=excluded.profile_json, updated_at=excluded.updated_at""",
            (profile.user_id, profile.model_dump_json(), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def delete_relationship_profile(user_id: str) -> None:
    """Cascade helper for profile deletion — see main.py's
    DELETE /api/profiles/{user_id} handler."""
    conn = get_connection(RELATIONSHIP_PROFILE_DB_PATH)
    try:
        conn.execute("DELETE FROM relationship_profiles WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def get_or_create_relationship_profile(user_id: str) -> RelationshipProfile:
    existing = get_relationship_profile(user_id)
    if existing is not None:
        return existing
    return RelationshipProfile(user_id=user_id)
