"""Safety audit log store."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import DATA_DIR
from app.db.sqlite_models import get_connection

SAFETY_AUDIT_DB_PATH = DATA_DIR / "profile.db"


@dataclass(frozen=True)
class SafetyAuditEntry:
    user_id: str
    timestamp: str
    category: str
    confidence_signals: dict
    response_taken: str
    outcome_notes: str
    retention_expiry: str


def record(entry: SafetyAuditEntry) -> None:
    conn = get_connection(SAFETY_AUDIT_DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS safety_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence_signals TEXT NOT NULL,
                response_taken TEXT NOT NULL,
                outcome_notes TEXT NOT NULL,
                retention_expiry TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO safety_audit
                (user_id, timestamp, category, confidence_signals, response_taken, outcome_notes, retention_expiry)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.user_id,
                entry.timestamp,
                entry.category,
                json.dumps(entry.confidence_signals),
                entry.response_taken,
                entry.outcome_notes,
                entry.retention_expiry,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def purge_expired(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    conn = get_connection(SAFETY_AUDIT_DB_PATH)
    try:
        cur = conn.execute("DELETE FROM safety_audit WHERE retention_expiry <= ?", (now.isoformat(),))
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()

