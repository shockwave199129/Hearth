"""Regex-based crisis detection — runs before every LLM call (see main.py's
Pipeline.respond). Deliberately narrow and phrase-based rather than
keyword-based: single ambiguous words ("die", "kill") false-positive
constantly ("that meeting killed me"), so patterns require a clearer
first-person crisis framing. This is a blunt safety net, not a clinical
screening tool — it needs review by a licensed mental health professional
before being relied on, same as the skills library. See project-plan.md §9.
"""
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.config import DATA_DIR
from app.db.sqlite_models import get_connection

CRISIS_DB_PATH = DATA_DIR / "profile.db"

# Ordered by severity; first match wins. Word-boundary, case-insensitive.
# Deliberately not exhaustive — a narrow, high-precision list that should
# grow only with real review, not speculative expansion.
_PATTERNS: list[tuple[str, str]] = [
    ("acute", r"\b(kill|hurt|harm)\s+myself\b"),
    ("acute", r"\bsuicid(e|al)\b"),
    ("acute", r"\b(want|going|planning|thinking about)\s+to\s+(die|end (it|my life)|not (be|exist) anymore)\b"),
    ("acute", r"\bno reason to (live|go on living)\b"),
    ("acute", r"\bbetter off (dead|without me|if i (were|was)n'?t (here|around))\b"),
    ("acute", r"\bend(ing)? (it all|my life)\b"),
]
_COMPILED = [(severity, re.compile(pattern, re.IGNORECASE)) for severity, pattern in _PATTERNS]


@dataclass
class CrisisMatch:
    severity: str
    matched_pattern: str


def detect(text: str) -> CrisisMatch | None:
    for severity, compiled in _COMPILED:
        if compiled.search(text):
            return CrisisMatch(severity=severity, matched_pattern=compiled.pattern)
    return None


def record_event(match: CrisisMatch, user_id: str, occurred_at: datetime | None = None) -> None:
    occurred_at = occurred_at or datetime.now(timezone.utc)
    conn = get_connection(CRISIS_DB_PATH)
    try:
        conn.execute(
            "INSERT INTO crisis_events (user_id, occurred_at, severity, matched_pattern) VALUES (?, ?, ?, ?)",
            (user_id, occurred_at.isoformat(), match.severity, match.matched_pattern),
        )
        conn.commit()
    finally:
        conn.close()


def recent_events(user_id: str, within_days: int) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=within_days)).isoformat()
    conn = get_connection(CRISIS_DB_PATH)
    try:
        rows = conn.execute(
            "SELECT occurred_at, severity, matched_pattern FROM crisis_events "
            "WHERE user_id = ? AND occurred_at >= ? ORDER BY occurred_at DESC",
            (user_id, cutoff),
        ).fetchall()
    finally:
        conn.close()
    return [{"occurred_at": r[0], "severity": r[1], "matched_pattern": r[2]} for r in rows]


def event_count(user_id: str, within_days: int) -> int:
    return len(recent_events(user_id, within_days))


def delete_events(user_id: str) -> None:
    """Cascade helper for profile deletion — see main.py's
    DELETE /api/profiles/{user_id} handler."""
    conn = get_connection(CRISIS_DB_PATH)
    try:
        conn.execute("DELETE FROM crisis_events WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()
