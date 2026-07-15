"""Tracks which of the (possibly several) local profiles this app instance
is currently using — a single-process desktop app runs one conversation at
a time, so this is a deliberate switch, not per-request routing."""
from app.config import DATA_DIR
from app.db.sqlite_models import get_connection

ACTIVE_PROFILE_DB_PATH = DATA_DIR / "profile.db"


def get_active_user_id() -> str | None:
    conn = get_connection(ACTIVE_PROFILE_DB_PATH)
    try:
        row = conn.execute("SELECT user_id FROM active_profile WHERE id = 1").fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def set_active_user_id(user_id: str) -> None:
    conn = get_connection(ACTIVE_PROFILE_DB_PATH)
    try:
        conn.execute(
            """INSERT INTO active_profile (id, user_id) VALUES (1, ?)
               ON CONFLICT(id) DO UPDATE SET user_id=excluded.user_id""",
            (user_id,),
        )
        conn.commit()
    finally:
        conn.close()


def clear_active_user_id() -> None:
    conn = get_connection(ACTIVE_PROFILE_DB_PATH)
    try:
        conn.execute("DELETE FROM active_profile WHERE id = 1")
        conn.commit()
    finally:
        conn.close()
