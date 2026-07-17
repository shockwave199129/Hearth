"""Persisted first-run setup flag in profile.db.

Once setup finishes and the app can start, we write complete=1 so the next
launch skips the Setup UI without re-deriving "done" only from imports.
See orchestrator.detect_status() for how the flag is combined with model
presence checks.
"""
from datetime import datetime, timezone

from app.config import DATA_DIR
from app.db.sqlite_models import get_connection

SETUP_STATE_DB_PATH = DATA_DIR / "profile.db"


def is_setup_complete() -> bool:
    conn = get_connection(SETUP_STATE_DB_PATH)
    try:
        row = conn.execute("SELECT complete FROM setup_state WHERE id = 1").fetchone()
    finally:
        conn.close()
    return bool(row and row[0])


def mark_setup_complete() -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection(SETUP_STATE_DB_PATH)
    try:
        conn.execute(
            """INSERT INTO setup_state (id, complete, completed_at) VALUES (1, 1, ?)
               ON CONFLICT(id) DO UPDATE SET complete=1, completed_at=excluded.completed_at""",
            (now,),
        )
        conn.commit()
    finally:
        conn.close()


def clear_setup_complete() -> None:
    conn = get_connection(SETUP_STATE_DB_PATH)
    try:
        conn.execute(
            """INSERT INTO setup_state (id, complete, completed_at) VALUES (1, 0, NULL)
               ON CONFLICT(id) DO UPDATE SET complete=0, completed_at=NULL"""
        )
        conn.commit()
    finally:
        conn.close()
