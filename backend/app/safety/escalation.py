"""The only network-facing module in the app — see project-plan.md §9. The
plan flags this as needing its own dedicated design pass (exact trigger,
channel, consent UI) rather than being bolted on; this implementation is a
conservative scaffold pending that pass:

- The real consent capture (onboarding opt-in + emergency contact) and the
  trigger/pattern logic are implemented for real.
- The actual "send" is a logged stub (LoggedNotifier) — nothing here ever
  contacts anyone. Wiring a real provider (Twilio, SMTP, etc.) is explicit
  follow-up work once a channel is chosen and the message/consent language
  is reviewed.
- Interpretation of "gated on repeated/escalating pattern or explicit
  onboarding consent": implemented as requiring BOTH consent AND a
  repeated-pattern threshold, never either alone — the more conservative
  reading, so a single ambiguous crisis phrase never triggers outreach to a
  real (if currently stubbed) contact without a sustained pattern.
"""
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from app.config import DATA_DIR, ESCALATION_TRIGGER_COUNT, ESCALATION_WINDOW_DAYS
from app.db.sqlite_models import get_connection
from app.onboarding.profile_store import get_profile
from app.safety.crisis_detector import event_count

logger = logging.getLogger("hearth.escalation")

ESCALATION_DB_PATH = DATA_DIR / "profile.db"

ESCALATION_MESSAGE_TEMPLATE = (
    "This is an automated message from {companion_name}, {name}'s companion app. "
    "{name} has been going through a difficult time recently and may need support. "
    "This message was sent because {name} opted in to this during setup."
)


class Notifier(ABC):
    @abstractmethod
    def send(self, message: str, method: str, value: str) -> dict:
        ...


class LoggedNotifier(Notifier):
    """Stub — logs what WOULD be sent instead of actually contacting anyone.
    Replace with a real provider once a channel has been chosen and
    reviewed. See module docstring."""

    def send(self, message: str, method: str, value: str) -> dict:
        logger.warning("ESCALATION STUB (no real message sent) — via %s to %s: %s", method, value, message)
        return {"ok": True, "stub": True}


def record_escalation(
    user_id: str, reason: str, method: str | None, result: dict, occurred_at: datetime | None = None
) -> None:
    occurred_at = occurred_at or datetime.now(timezone.utc)
    conn = get_connection(ESCALATION_DB_PATH)
    try:
        conn.execute(
            "INSERT INTO escalations (user_id, occurred_at, reason, method, result_json) VALUES (?, ?, ?, ?, ?)",
            (user_id, occurred_at.isoformat(), reason, method, json.dumps(result)),
        )
        conn.commit()
    finally:
        conn.close()


def last_escalation(user_id: str) -> datetime | None:
    conn = get_connection(ESCALATION_DB_PATH)
    try:
        row = conn.execute(
            "SELECT occurred_at FROM escalations WHERE user_id = ? ORDER BY occurred_at DESC LIMIT 1", (user_id,)
        ).fetchone()
    finally:
        conn.close()
    return datetime.fromisoformat(row[0]) if row else None


def delete_escalations(user_id: str) -> None:
    """Cascade helper for profile deletion — see main.py's
    DELETE /api/profiles/{user_id} handler."""
    conn = get_connection(ESCALATION_DB_PATH)
    try:
        conn.execute("DELETE FROM escalations WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def maybe_escalate(user_id: str, reason: str, notifier: Notifier | None = None) -> None:
    """No-op unless the user explicitly consented AND gave a contact value
    AND a repeated/escalating pattern has actually occurred — see module
    docstring for why both are required."""
    profile = get_profile(user_id)
    if profile is None or not profile.emergency_contact_consent or not profile.emergency_contact_value:
        return
    if event_count(user_id, ESCALATION_WINDOW_DAYS) < ESCALATION_TRIGGER_COUNT:
        return

    notifier = notifier or LoggedNotifier()
    message = ESCALATION_MESSAGE_TEMPLATE.format(
        companion_name=profile.companion_name, name=profile.name
    )
    result = notifier.send(message, profile.emergency_contact_method or "sms", profile.emergency_contact_value)
    record_escalation(user_id, reason, profile.emergency_contact_method, result)
