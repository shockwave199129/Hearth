"""Biometric consent and the retention schedule — docs/compliance.md §6.

These pin the two BIPA-shaped mechanisms rather than the audio maths:
informed consent recorded *before* collection, and destruction at the earlier
of purpose-satisfied or the retention deadline.

They are written as compliance assertions on purpose. If one of these fails,
the correct response is not to relax the test — it is that Hearth is
collecting or keeping a biometric identifier it has no record of permission
for.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app import config
from app.db import sqlite_models
from app.memory import chat_history
from app.voice import consent, retention
from app.voice import store as voiceprint_store
from app.voice import verification

SR = 16000


class FakeEmbedder:
    available = True

    def embed(self, audio, sample_rate: int = SR):
        vector = np.zeros(8, dtype=np.float32)
        vector[0] = 1.0
        return vector


@pytest.fixture
def voice_db(tmp_path, monkeypatch):
    db = tmp_path / "profile.db"
    sqlite_models.close_pooled_connections()
    monkeypatch.setattr(voiceprint_store, "VOICEPRINT_DB_PATH", db)
    monkeypatch.setattr(consent, "CONSENT_DB_PATH", db)
    monkeypatch.setattr(chat_history, "CHAT_HISTORY_DB_PATH", db)
    yield db
    sqlite_models.close_pooled_connections()


def _samples(n: int = 3, seconds: float = 4.0) -> list[np.ndarray]:
    return [np.zeros(int(SR * seconds), dtype=np.float32) for _ in range(n)]


# --- Consent precedes collection --------------------------------------------


def test_enrollment_is_refused_without_consent(voice_db):
    """The whole point: BIPA-style statutes bind *collection*, so no code path
    may produce a stored template before consent is on record."""
    result = verification.enroll("u1", _samples(), FakeEmbedder())

    assert not result.ok
    assert result.needs_consent is True
    assert voiceprint_store.get("u1") is None


def test_enrollment_proceeds_once_consent_is_recorded(voice_db):
    consent.record("u1")

    result = verification.enroll("u1", _samples(), FakeEmbedder())

    assert result.ok
    assert voiceprint_store.get("u1") is not None


def test_consent_is_checked_in_the_logic_not_only_the_route(voice_db):
    """Calling enroll() directly — bypassing the API entirely — must still be
    refused. If this ever passes without consent, the gate has moved into the
    route and can be sidestepped."""
    assert verification.enroll("u1", _samples(), FakeEmbedder()).needs_consent is True


def test_recording_consent_collects_nothing_on_its_own(voice_db):
    consent.record("u1")

    assert consent.has_current_consent("u1") is True
    assert voiceprint_store.get("u1") is None


def test_consent_timestamp_is_server_stamped(voice_db):
    before = datetime.now(timezone.utc)
    record = consent.record("u1")

    stamped = datetime.fromisoformat(record.consented_at)

    assert stamped >= before.replace(microsecond=0) - timedelta(seconds=1)
    assert stamped.tzinfo is not None


def test_stale_consent_version_is_not_current_consent(voice_db):
    """Agreement to earlier wording is not agreement to this wording."""
    consent.record("u1", version="2020-01-01.0")

    assert consent.get("u1") is not None
    assert consent.has_current_consent("u1") is False
    assert verification.enroll("u1", _samples(), FakeEmbedder()).needs_consent is True


def test_consent_status_reports_the_current_version_for_comparison(voice_db):
    status = consent.status("u1")

    assert status["consent_recorded"] is False
    assert status["current_consent_version"] == config.VOICE_BIOMETRIC_CONSENT_VERSION


def test_revoking_consent_blocks_further_collection(voice_db):
    consent.record("u1")
    consent.revoke("u1")

    assert consent.has_current_consent("u1") is False
    assert verification.enroll("u1", _samples(), FakeEmbedder()).needs_consent is True
    # Idempotent: "no consent on record" is the requested state either way.
    consent.revoke("u1")


def test_consent_wording_is_a_single_reviewable_string():
    """Counsel reviews one constant. If the copy is ever duplicated into the
    frontend, the version recorded against a profile stops corresponding to
    what the user actually read."""
    text = config.VOICE_BIOMETRIC_CONSENT_TEXT

    assert "{companion}" in text
    assert "{retention_years}" in text
    for required in ("biometric", "never uploaded", "delete", "do not have to"):
        assert required in text, f"consent copy no longer mentions {required!r}"


# --- Retention and destruction ----------------------------------------------


def test_voiceprint_survives_inside_the_retention_window(voice_db):
    consent.record("u1")
    verification.enroll("u1", _samples(), FakeEmbedder())

    destroyed = retention.enforce("u1", now=datetime.now(timezone.utc) + timedelta(days=30))

    assert destroyed is False
    assert voiceprint_store.get("u1") is not None


def test_voiceprint_is_destroyed_after_the_retention_window(voice_db):
    consent.record("u1")
    verification.enroll("u1", _samples(), FakeEmbedder())

    later = datetime.now(timezone.utc) + timedelta(days=config.VOICEPRINT_RETENTION_DAYS + 1)
    destroyed = retention.enforce("u1", now=later)

    assert destroyed is True
    assert voiceprint_store.get("u1") is None


def test_destruction_also_withdraws_consent(voice_db):
    """Permission to hold a biometric identifier does not outlive the
    identifier — re-enrolling later has to ask again."""
    consent.record("u1")
    verification.enroll("u1", _samples(), FakeEmbedder())

    later = datetime.now(timezone.utc) + timedelta(days=config.VOICEPRINT_RETENTION_DAYS + 1)
    retention.enforce("u1", now=later)

    assert consent.has_current_consent("u1") is False


def test_conversation_activity_defers_the_deadline(voice_db):
    """"Three years from the last interaction", not from enrollment."""
    consent.record("u1")
    verification.enroll("u1", _samples(), FakeEmbedder())
    # A conversation turn two years after enrollment pushes the deadline out.
    recent = datetime.now(timezone.utc) + timedelta(days=730)
    conn = sqlite_models.get_connection(voice_db)
    try:
        conn.execute(
            """INSERT INTO chat_history (user_id, session_id, turn_id, role, content, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("u1", "s1", 1, "user", "x", recent.isoformat()),
        )
        conn.commit()
    finally:
        conn.close()

    just_past_enrollment = datetime.now(timezone.utc) + timedelta(
        days=config.VOICEPRINT_RETENTION_DAYS + 1
    )

    assert retention.enforce("u1", now=just_past_enrollment) is False
    assert voiceprint_store.get("u1") is not None


def test_an_abandoned_voiceprint_still_expires_without_any_conversation(voice_db):
    """A voiceprint enrolled and then never used must not wait forever for
    activity that never comes — enrollment time is the floor."""
    consent.record("u1")
    verification.enroll("u1", _samples(), FakeEmbedder())
    assert chat_history.last_activity_at("u1") is None

    later = datetime.now(timezone.utc) + timedelta(days=config.VOICEPRINT_RETENTION_DAYS + 1)

    assert retention.enforce("u1", now=later) is True


def test_expiry_is_reportable_so_the_schedule_can_be_shown(voice_db):
    consent.record("u1")
    verification.enroll("u1", _samples(), FakeEmbedder())

    due = retention.expiry_for("u1")

    assert due is not None
    assert due > datetime.now(timezone.utc)


def test_expiry_is_none_when_nothing_is_stored(voice_db):
    assert retention.expiry_for("u1") is None
    assert retention.enforce("u1") is False


def test_retention_sweep_never_raises(voice_db, monkeypatch):
    """It runs on profile activation. A sweep that could throw would stop the
    app opening, which is a worse failure than a late deletion."""
    monkeypatch.setattr(
        chat_history, "last_activity_at", lambda uid: (_ for _ in ()).throw(RuntimeError("db gone"))
    )

    assert retention.enforce("u1") is False


def test_last_activity_reads_the_most_recent_turn(voice_db):
    for day, text in ((1, "older"), (5, "newest"), (3, "middle")):
        stamp = (datetime(2026, 1, day, tzinfo=timezone.utc)).isoformat()
        conn = sqlite_models.get_connection(voice_db)
        try:
            conn.execute(
                """INSERT INTO chat_history (user_id, session_id, turn_id, role, content, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("u1", "s", 1, "user", text, stamp),
            )
            conn.commit()
        finally:
            conn.close()

    assert chat_history.last_activity_at("u1") == datetime(2026, 1, 5, tzinfo=timezone.utc)


def test_last_activity_is_scoped_to_one_profile(voice_db):
    conn = sqlite_models.get_connection(voice_db)
    try:
        conn.execute(
            """INSERT INTO chat_history (user_id, session_id, turn_id, role, content, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("someone-else", "s", 1, "user", "x", "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()

    assert chat_history.last_activity_at("u1") is None
