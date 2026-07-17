"""Persisted, encrypted conversation history — project-plan.md §1's
chat_history.db (folded into the same profile.db file as everything else,
same convention as checkin/crisis_events/escalations). Content is Fernet-
encrypted at rest, same pattern as long_term.py's Chroma documents.

Replay re-synthesizes stored text through the live TTS engine with the
profile's preferred voice (no audio files on disk).
"""
from datetime import datetime, timezone

from app.config import DATA_DIR
from app.db.sqlite_models import get_connection
from app.security.crypto import decrypt, encrypt

CHAT_HISTORY_DB_PATH = DATA_DIR / "profile.db"


def record_turn(user_id: str, session_id: str, turn_id: int, role: str, content: str) -> int:
    conn = get_connection(CHAT_HISTORY_DB_PATH)
    try:
        cursor = conn.execute(
            """INSERT INTO chat_history (user_id, session_id, turn_id, role, content, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                session_id,
                turn_id,
                role,
                encrypt(content).decode("latin1"),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def list_turns(
    user_id: str, limit: int = 40, before_id: int | None = None
) -> dict:
    """Newest page of encrypted chat rows for a profile.

    Returns ``{"items": [...], "has_more": bool}``. ``before_id`` loads the
    next older page (rows with id < before_id) for Talk-page scroll-up.
    Each conversation turn is two rows (user + assistant), so a limit of
    ~40 rows is roughly 20 visible exchanges.
    """
    limit = max(1, min(limit, 200))
    conn = get_connection(CHAT_HISTORY_DB_PATH)
    try:
        if before_id is None:
            rows = conn.execute(
                """SELECT id, session_id, turn_id, role, content, created_at FROM chat_history
                   WHERE user_id = ? ORDER BY id DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, session_id, turn_id, role, content, created_at FROM chat_history
                   WHERE user_id = ? AND id < ? ORDER BY id DESC LIMIT ?""",
                (user_id, before_id, limit),
            ).fetchall()
        items = [
            {
                "id": r[0],
                "session_id": r[1],
                "turn_id": r[2],
                "role": r[3],
                "content": decrypt(r[4].encode("latin1")),
                "created_at": r[5],
            }
            for r in rows
        ]
        has_more = False
        if items:
            oldest_id = min(item["id"] for item in items)
            has_more = (
                conn.execute(
                    "SELECT 1 FROM chat_history WHERE user_id = ? AND id < ? LIMIT 1",
                    (user_id, oldest_id),
                ).fetchone()
                is not None
            )
        return {"items": items, "has_more": has_more}
    finally:
        conn.close()


def get_turn(user_id: str, row_id: int) -> dict | None:
    conn = get_connection(CHAT_HISTORY_DB_PATH)
    try:
        row = conn.execute(
            "SELECT id, session_id, turn_id, role, content, created_at FROM chat_history WHERE id = ? AND user_id = ?",
            (row_id, user_id),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "id": row[0],
        "session_id": row[1],
        "turn_id": row[2],
        "role": row[3],
        "content": decrypt(row[4].encode("latin1")),
        "created_at": row[5],
    }


def delete_turn(user_id: str, row_id: int) -> None:
    conn = get_connection(CHAT_HISTORY_DB_PATH)
    try:
        conn.execute(
            "DELETE FROM chat_history WHERE id = ? AND user_id = ?", (row_id, user_id)
        )
        conn.commit()
    finally:
        conn.close()


def delete_all_for_user(user_id: str) -> None:
    """Cascade helper for profile deletion — see main.py's
    DELETE /api/profiles/{user_id} handler."""
    conn = get_connection(CHAT_HISTORY_DB_PATH)
    try:
        conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()
