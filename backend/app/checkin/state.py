"""Tracks when the companion last checked in on each profile — one row per
user_id in profile.db, same pattern as onboarding/profile_store.py. See
project-plan.md §8."""
from datetime import datetime

from app.config import DATA_DIR
from app.db.sqlite_models import get_connection

CHECKIN_DB_PATH = DATA_DIR / "profile.db"


def get_last_checkin(user_id: str) -> datetime | None:
    conn = get_connection(CHECKIN_DB_PATH)
    try:
        row = conn.execute("SELECT last_checkin_at FROM checkin WHERE user_id = ?", (user_id,)).fetchone()
    finally:
        conn.close()
    if row is None or row[0] is None:
        return None
    return datetime.fromisoformat(row[0])


def set_last_checkin(user_id: str, ts: datetime) -> None:
    conn = get_connection(CHECKIN_DB_PATH)
    try:
        conn.execute(
            """INSERT INTO checkin (user_id, last_checkin_at) VALUES (?, ?)
               ON CONFLICT(user_id) DO UPDATE SET last_checkin_at=excluded.last_checkin_at""",
            (user_id, ts.isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def delete_checkin(user_id: str) -> None:
    """Cascade helper for profile deletion — see main.py's
    DELETE /api/profiles/{user_id} handler."""
    conn = get_connection(CHECKIN_DB_PATH)
    try:
        conn.execute("DELETE FROM checkin WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()
