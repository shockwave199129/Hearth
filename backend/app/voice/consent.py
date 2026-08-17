"""Written consent for biometric collection — docs/compliance.md §6.

BIPA-style statutes require informed written consent *before* a biometric
identifier is collected. `app.voice.verification.enroll` refuses without a
record here, so consent is a precondition of collection rather than a
checkbox next to it.

Shaped after `profile_store.record_attestation`, for the same reasons: the
timestamp is stamped server-side so it cannot be backdated by a client, and
the endpoint that records it takes no body — there is nothing for the caller
to assert beyond "the user agreed", and the wording they agreed to is
identified by version rather than sent up and trusted.

Revocation is a row delete. That leaves no state in which a stale timestamp
could be misread as live consent, and it means deleting a voiceprint also
withdraws permission to collect the next one: re-enrolling asks again, which
is the conservative reading and costs the user one extra tap.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.config import VOICE_BIOMETRIC_CONSENT_VERSION, DATA_DIR
from app.db.sqlite_models import get_connection

CONSENT_DB_PATH = DATA_DIR / "profile.db"


@dataclass(frozen=True)
class ConsentRecord:
    user_id: str
    consented_at: str
    consent_version: str

    @property
    def is_current(self) -> bool:
        """False when the wording has changed since this was agreed to.

        Treated as no consent by `has_current_consent`: agreement to earlier
        text is not agreement to this text.
        """
        return self.consent_version == VOICE_BIOMETRIC_CONSENT_VERSION


def record(user_id: str, *, version: str | None = None) -> ConsentRecord:
    """Record consent to the current wording. Overwrites any earlier record."""
    now = datetime.now(timezone.utc).isoformat()
    resolved = version or VOICE_BIOMETRIC_CONSENT_VERSION
    conn = get_connection(CONSENT_DB_PATH)
    try:
        conn.execute(
            """INSERT INTO voiceprint_consent (user_id, consented_at, consent_version)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   consented_at = excluded.consented_at,
                   consent_version = excluded.consent_version""",
            (user_id, now, resolved),
        )
        conn.commit()
    finally:
        conn.close()
    return ConsentRecord(user_id, now, resolved)


def get(user_id: str) -> ConsentRecord | None:
    conn = get_connection(CONSENT_DB_PATH)
    try:
        row = conn.execute(
            "SELECT consented_at, consent_version FROM voiceprint_consent WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    finally:
        conn.close()
    return ConsentRecord(user_id, str(row[0]), str(row[1])) if row else None


def has_current_consent(user_id: str) -> bool:
    existing = get(user_id)
    return existing is not None and existing.is_current


def revoke(user_id: str) -> None:
    """Withdraw consent. Idempotent."""
    conn = get_connection(CONSENT_DB_PATH)
    try:
        conn.execute("DELETE FROM voiceprint_consent WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def status(user_id: str) -> dict:
    """Consent state for the API and the enrollment UI."""
    existing = get(user_id)
    return {
        "consent_recorded": existing is not None,
        "consent_current": existing is not None and existing.is_current,
        "consented_at": existing.consented_at if existing else None,
        "consent_version": existing.consent_version if existing else None,
        "current_consent_version": VOICE_BIOMETRIC_CONSENT_VERSION,
    }
