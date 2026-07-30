"""The only network-facing module in the app — see project-plan.md §9.

- The real consent capture (onboarding opt-in + emergency contact) and the
  trigger/pattern logic are implemented for real.
- "email"-method escalation actually sends, via `EmailNotifier` (stdlib
  `smtplib`, no new dependency), when SMTP is configured (see
  `app.config.SAFETY_SMTP_*` — unset by default). Unconfigured, it logs
  instead of failing the turn.
- "sms"-method escalation stays a logged stub (`LoggedNotifier`): SMS needs
  a chosen, paid third-party provider (e.g. Twilio) and its own credentials
  this codebase has no basis to assume — wiring one is explicit follow-up
  work once a provider is chosen and the message/consent language is
  reviewed. Hearth never implies it has contacted anyone it hasn't (Book
  Vol 6 Ch11) — `get_notifier` makes this an explicit selection, not a
  silent fallback that could be mistaken for "it worked."
- Interpretation of "gated on repeated/escalating pattern or explicit
  onboarding consent": implemented as requiring BOTH consent AND a
  repeated-pattern threshold, never either alone — the more conservative
  reading, so a single ambiguous crisis phrase never triggers outreach to a
  real contact without a sustained pattern.
"""
import json
import logging
import smtplib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from email.message import EmailMessage

from app.config import (
    DATA_DIR,
    ESCALATION_TRIGGER_COUNT,
    ESCALATION_WINDOW_DAYS,
    SAFETY_SMTP_FROM_ADDRESS,
    SAFETY_SMTP_HOST,
    SAFETY_SMTP_PASSWORD,
    SAFETY_SMTP_PORT,
    SAFETY_SMTP_USERNAME,
)
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

ESCALATION_EMAIL_SUBJECT = "A message from {companion_name}"


class Notifier(ABC):
    @abstractmethod
    def send(self, message: str, method: str, value: str) -> dict:
        ...


class LoggedNotifier(Notifier):
    """Stub — logs what WOULD be sent instead of actually contacting anyone.
    Used for any channel with no configured real provider. See module
    docstring."""

    def send(self, message: str, method: str, value: str) -> dict:
        logger.warning("ESCALATION STUB (no real message sent) — via %s to %s: %s", method, value, message)
        return {"ok": True, "stub": True}


class EmailNotifier(Notifier):
    """Real channel — sends via stdlib `smtplib` using SMTP credentials
    from `app.config.SAFETY_SMTP_*`. Only usable when those are configured
    (see `is_configured`); callers should check that and fall back to
    `LoggedNotifier` otherwise (see `get_notifier`), never silently no-op
    behind a "sent" result."""

    def __init__(
        self,
        *,
        host: str = SAFETY_SMTP_HOST,
        port: int = SAFETY_SMTP_PORT,
        username: str = SAFETY_SMTP_USERNAME,
        password: str = SAFETY_SMTP_PASSWORD,
        from_address: str = SAFETY_SMTP_FROM_ADDRESS,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.from_address = from_address

    def is_configured(self) -> bool:
        return bool(self.host and self.from_address)

    def send(self, message: str, method: str, value: str) -> dict:
        if method != "email":
            raise ValueError(f"EmailNotifier only handles method='email', got {method!r}")
        if not self.is_configured():
            raise RuntimeError("EmailNotifier is not configured (SAFETY_SMTP_HOST/SAFETY_SMTP_FROM_ADDRESS unset)")

        email_message = EmailMessage()
        email_message["Subject"] = ESCALATION_EMAIL_SUBJECT.format(companion_name="Hearth")
        email_message["From"] = self.from_address
        email_message["To"] = value
        email_message.set_content(message)

        with smtplib.SMTP(self.host, self.port, timeout=10) as smtp:
            smtp.starttls()
            if self.username:
                smtp.login(self.username, self.password)
            smtp.send_message(email_message)
        return {"ok": True, "stub": False, "channel": "email"}


def get_notifier(method: str) -> Notifier:
    """Explicit selection, never a silent fallback dressed up as success:
    email uses the real `EmailNotifier` when configured, otherwise the
    logged stub; every other method is the logged stub until a provider is
    chosen (see module docstring)."""
    if method == "email":
        email_notifier = EmailNotifier()
        if email_notifier.is_configured():
            return email_notifier
    return LoggedNotifier()


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
    docstring for why both are required.

    Also no-ops if an escalation was already recorded within
    ESCALATION_WINDOW_DAYS — keeps the stub (and a future real provider)
    from re-firing on every subsequent crisis hit in the same window.
    """
    profile = get_profile(user_id)
    if profile is None or not profile.emergency_contact_consent or not profile.emergency_contact_value:
        return
    if event_count(user_id, ESCALATION_WINDOW_DAYS) < ESCALATION_TRIGGER_COUNT:
        return

    last = last_escalation(user_id)
    if last is not None:
        age = datetime.now(timezone.utc) - last
        if age.total_seconds() < ESCALATION_WINDOW_DAYS * 86400:
            return

    method = profile.emergency_contact_method or "sms"
    notifier = notifier or get_notifier(method)
    message = ESCALATION_MESSAGE_TEMPLATE.format(
        companion_name=profile.companion_name, name=profile.name
    )
    try:
        result = notifier.send(message, method, profile.emergency_contact_value)
    except Exception:
        logger.exception("escalation send failed — falling back to logged stub result")
        result = {"ok": False, "stub": True, "error": "send_failed"}
    record_escalation(user_id, reason, profile.emergency_contact_method, result)
