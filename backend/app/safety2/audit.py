"""Safety audit log store."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from app.config import DATA_DIR
from app.db.sqlite_models import get_connection

SAFETY_AUDIT_DB_PATH = DATA_DIR / "profile.db"

# Book Vol 6 Ch12: this exemption from Volume 4's general deletion rights
# must be disclosed to the user explicitly, in plain language — never a
# silent carve-out. Surfaced via /api/safety/status.
RETENTION_POLICY_DISCLOSURE = (
    "Messages that trigger a safety check (for example, signs of crisis or acute distress) are "
    "kept in a separate, limited safety record for up to 30 days, even if you delete your other "
    "memories or chat history. This is so a safety response can be reviewed for quality and "
    "improved over time. After 30 days, that record is deleted. Nothing else about your "
    "conversations is treated this way."
)


def retention_policy_disclosure() -> str:
    return RETENTION_POLICY_DISCLOSURE


def pending_entry_count(user_id: str) -> int:
    """How many of this user's safety-audit entries are currently retained
    (not yet purged) — surfaced for transparency, same principle as
    /api/memories."""
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
        row = conn.execute("SELECT COUNT(*) FROM safety_audit WHERE user_id = ?", (user_id,)).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


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


def delete_entries(user_id: str) -> int:
    """Deletes this user's safety-audit entries outright, ahead of expiry.

    Called when a profile is deleted (api/profile.py). The 30-day retention
    carve-out in RETENTION_POLICY_DISCLOSURE is scoped to deleting
    *memories or chat history* — deleting the whole profile is a stronger,
    unambiguous "erase me", and honouring it here keeps the disclosure
    literally true rather than quietly broader than it reads.

    The Vol 6 Ch12 audit purpose survives this because these rows hold no
    message content — only category, signal flags, and which response was
    taken. Retaining rows keyed to a profile that no longer exists buys
    almost nothing for safety-quality review and costs exactly the kind of
    "deleted but not really" behaviour Volume 4's deletion rights exist to
    prevent.

    If aggregate safety metrics across deleted profiles are wanted later,
    anonymise (null the user_id) here instead of deleting — the schema
    already stores nothing else identifying. That is a deliberate policy
    change, not a refactor; see docs/compliance.md."""
    conn = get_connection(SAFETY_AUDIT_DB_PATH)
    try:
        cur = conn.execute("DELETE FROM safety_audit WHERE user_id = ?", (user_id,))
        conn.commit()
        return cur.rowcount or 0
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

