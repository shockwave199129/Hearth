"""VAD and speaker verification — app/voice/.

The tests that matter here are not "does it recognise a voice" (that needs
real weights and a real corpus; see test_voice_models.py, which skips
without them). They are the *consequence* tests: what a negative or absent
verdict is allowed to do to a turn.

The invariant under test throughout: a speaker score may suppress **memory
formation** and nothing else. It may never suppress safety, never discard a
turn, and an unchecked turn must behave exactly as this app behaved before
verification existed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from app.db import sqlite_models
from app.memory.formation import process_session_memory
from app.memory.short_term import ShortTermMemory
from app.onboarding.profile_schema import UserProfile
from app.voice import consent as voice_consent
from app.voice import store as voiceprint_store
from app.voice import verification
from app.voice.vad import MIN_SPEECH_WINDOWS, SileroVad, VadResult

SR = 16000


class FakeEmbedder:
    """Deterministic stand-in: each distinct `speaker` label gets its own
    orthogonal unit vector, so cosine similarity is exactly 1.0 within a
    speaker and 0.0 across speakers. Lets the decision logic be tested
    without weights, a corpus, or onnxruntime.

    Labels are assigned sequential basis vectors from a shared registry
    rather than hashed into `dim` slots. `hash()` on a str is salted per
    process, so a hashed index made "different speaker" tests pass alone and
    fail in the full suite whenever two labels happened to collide — a real
    flake, not a tuning issue.
    """

    available = True
    _indices: dict[str, int] = {}

    def __init__(self, speaker: str = "a", dim: int = 16):
        self.speaker = speaker
        self.dim = dim

    def _vector(self, speaker: str) -> np.ndarray:
        index = self._indices.setdefault(speaker, len(self._indices))
        assert index < self.dim, "more distinct fake speakers than basis vectors"
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[index] = 1.0
        return vector

    def embed(self, audio, sample_rate: int = SR):
        if len(np.asarray(audio).reshape(-1)) < 100:
            return None
        return self._vector(self.speaker)


class UnavailableEmbedder:
    available = False

    def embed(self, audio, sample_rate: int = SR):
        return None


@pytest.fixture
def voiceprint_db(tmp_path, monkeypatch):
    """Isolated profile.db with biometric consent already on record.

    Consent is granted here so each test below exercises the thing it is
    named after. That enrollment *refuses* without consent is pinned
    separately, in tests/test_voice_compliance.py — don't reintroduce it as
    an incidental assertion here.
    """
    db = tmp_path / "profile.db"
    sqlite_models.close_pooled_connections()
    monkeypatch.setattr(voiceprint_store, "VOICEPRINT_DB_PATH", db)
    monkeypatch.setattr(voice_consent, "CONSENT_DB_PATH", db)
    for user_id in ("u1", "nobody"):
        voice_consent.record(user_id)
    yield db
    sqlite_models.close_pooled_connections()


def _speech(seconds: float = 4.0) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def _samples(n: int = 3, seconds: float = 4.0) -> list[np.ndarray]:
    return [_speech(seconds) for _ in range(n)]


# --- Enrollment -------------------------------------------------------------


def test_enrollment_stores_a_voiceprint(voiceprint_db):
    result = verification.enroll("u1", _samples(), FakeEmbedder("ada"))

    assert result.ok
    assert result.sample_count == 3
    assert voiceprint_store.get("u1") is not None


def test_enrollment_refuses_too_few_usable_samples(voiceprint_db):
    """A weak enrollment is not a smaller version of a good one — it is a
    verifier that is wrong on every later turn. Better to refuse."""
    result = verification.enroll("u1", _samples(n=2), FakeEmbedder("ada"))

    assert not result.ok
    assert voiceprint_store.get("u1") is None


def test_enrollment_ignores_samples_that_are_too_short(voiceprint_db):
    short = [_speech(0.5) for _ in range(5)]

    result = verification.enroll("u1", short, FakeEmbedder("ada"))

    assert not result.ok
    assert result.sample_count == 0


def test_enrollment_refuses_samples_that_disagree_with_each_other(voiceprint_db):
    """Two different people during enrollment, or a broken mic. The centroid
    would be meaningless, so this is refused with an actionable message."""

    class TwoSpeakers(FakeEmbedder):
        def __init__(self):
            super().__init__("a")
            self.calls = 0

        def embed(self, audio, sample_rate: int = SR):
            self.calls += 1
            return self._vector("a" if self.calls % 2 else "b")

    result = verification.enroll("u1", _samples(n=4), TwoSpeakers())

    assert not result.ok
    assert result.cohesion is not None and result.cohesion < verification.MIN_ENROLLMENT_COHESION
    assert voiceprint_store.get("u1") is None


def test_re_enrollment_replaces_rather_than_blends(voiceprint_db):
    verification.enroll("u1", _samples(), FakeEmbedder("ada"))
    first = voiceprint_store.get("u1").embedding.copy()

    verification.enroll("u1", _samples(), FakeEmbedder("bo"))
    second = voiceprint_store.get("u1").embedding

    assert not np.allclose(first, second)
    # A user redoing enrollment is correcting it; averaging the bad attempt in
    # would preserve the very thing they were fixing.
    assert verification.verify("u1", _speech(), FakeEmbedder("bo")).decision == "match"


def test_enrollment_requires_the_model(voiceprint_db):
    result = verification.enroll("u1", _samples(), UnavailableEmbedder())

    assert not result.ok
    assert voiceprint_store.get("u1") is None


# --- Verification decisions -------------------------------------------------


def test_matching_voice_verifies(voiceprint_db):
    verification.enroll("u1", _samples(), FakeEmbedder("ada"))

    verdict = verification.verify("u1", _speech(), FakeEmbedder("ada"))

    assert verdict.decision == "match"
    assert verdict.verified is True
    assert verdict.score is not None and verdict.score >= verdict.threshold


def test_different_voice_is_unrecognized(voiceprint_db):
    verification.enroll("u1", _samples(), FakeEmbedder("ada"))

    verdict = verification.verify("u1", _speech(), FakeEmbedder("stranger"))

    assert verdict.decision == "unrecognized"
    assert verdict.verified is False


def test_unchecked_cases_are_none_not_false(voiceprint_db):
    """The tri-state is the whole point: None means "we did not check" and
    must not suppress memory formation the way False does."""
    no_model = verification.verify("u1", _speech(), UnavailableEmbedder())
    assert (no_model.decision, no_model.verified) == ("unavailable", None)

    not_enrolled = verification.verify("nobody", _speech(), FakeEmbedder("ada"))
    assert (not_enrolled.decision, not_enrolled.verified) == ("not_enrolled", None)

    verification.enroll("u1", _samples(), FakeEmbedder("ada"))
    brief = verification.verify("u1", _speech(0.4), FakeEmbedder("ada"))
    assert (brief.decision, brief.verified) == ("too_short", None)


def test_short_utterances_are_not_scored_at_all(voiceprint_db):
    """Below ~1.5s of speech the same/different-speaker margin collapses and
    at 0.5s it inverts, so a score there is worse than no score. Measured on
    the LibriSpeech probe set — see app/voice/embedder.py."""
    verification.enroll("u1", _samples(), FakeEmbedder("ada"))

    verdict = verification.verify("u1", _speech(30.0), FakeEmbedder("stranger"), speech_seconds=0.5)

    assert verdict.decision == "too_short"
    assert verdict.score is None


def test_speech_seconds_beats_buffer_length(voiceprint_db):
    """A 30-second recording holding 0.5s of speech is a short sample. Scoring
    it on buffer length is how a verifier starts rejecting its own user."""
    verification.enroll("u1", _samples(), FakeEmbedder("ada"))

    on_speech = verification.verify("u1", _speech(30.0), FakeEmbedder("ada"), speech_seconds=0.5)
    on_buffer = verification.verify("u1", _speech(30.0), FakeEmbedder("ada"))

    assert on_speech.decision == "too_short"
    assert on_buffer.decision == "match"


def test_a_voiceprint_from_another_model_is_not_compared(voiceprint_db):
    """A template only means something to the model that made it. Comparing
    across models yields a plausible number with no meaning."""
    voiceprint_store.save("u1", np.array([1.0, 0.0], dtype=np.float32),
                          sample_count=3, model_id="some-older-model")

    verdict = verification.verify("u1", _speech(), FakeEmbedder("ada"))

    assert verdict.decision == "not_enrolled"


# --- What a verdict is allowed to do ----------------------------------------


def _session(verified_flags: list[bool | None]) -> ShortTermMemory:
    """Each turn is phrased to trip `formation._should_form` (first person +
    a stressor/preference/life-event keyword), so a zero count means the
    speaker gate suppressed it rather than the trigger never firing."""
    memory = ShortTermMemory(llm=None)
    for i, flag in enumerate(verified_flags):
        memory.add_turn(
            f"I am really anxious about the new job in week {i}", "mm", speaker_verified=flag
        )
    return memory


def test_unverified_turns_do_not_form_memories(voiceprint_db, monkeypatch):
    """The one and only consequence of a negative verdict."""
    created = []
    monkeypatch.setattr("app.memory.formation.long_term.search", lambda *a, **k: [])
    monkeypatch.setattr(
        "app.memory.formation.long_term.create",
        lambda fact, category, user_id: created.append(fact),
    )

    result = process_session_memory("u1", _session([False, False]))

    assert result.created == 0
    assert created == []


def test_unchecked_turns_still_form_memories(voiceprint_db, monkeypatch):
    """No model, no enrollment, or typed input must behave exactly as before
    this feature existed. If None ever starts suppressing formation, every
    install without the optional weights silently stops remembering."""
    created = []
    monkeypatch.setattr("app.memory.formation.long_term.search", lambda *a, **k: [])
    monkeypatch.setattr(
        "app.memory.formation.long_term.create",
        lambda fact, category, user_id: created.append(fact),
    )

    result = process_session_memory("u1", _session([None, None]))

    assert result.created > 0
    assert created


def test_typed_turns_are_unchecked_by_default():
    memory = ShortTermMemory(llm=None)
    memory.add_turn("typed this", "reply")

    assert memory.messages[0]["speaker_verified"] is None


def test_a_verdict_never_removes_the_turn_from_the_transcript(voiceprint_db):
    """Unrecognised speech is still answered and still stored — the user can
    see and delete it. Silently dropping audio is the failure mode this
    design exists to avoid."""
    memory = _session([False])

    assert len(memory.messages) == 2
    assert memory.messages[0]["content"].startswith("I am really anxious")
    assert memory.messages[1]["role"] == "assistant"


# --- VAD --------------------------------------------------------------------


def test_vad_fails_open_when_the_model_is_absent(tmp_path):
    """A missing optional model must never make Hearth ignore someone who is
    talking to it."""
    vad = SileroVad(model_path=tmp_path / "nope.onnx")

    result = vad.assess(np.zeros(SR, dtype=np.float32))

    assert result.available is False
    assert result.has_speech is True


def test_vad_fails_open_on_inference_error(tmp_path, monkeypatch):
    model = tmp_path / "silero_vad.onnx"
    model.write_bytes(b"not a real onnx file")
    vad = SileroVad(model_path=model)
    assert vad.available  # the file exists; loading it will fail

    result = vad.assess(np.zeros(SR, dtype=np.float32))

    assert result.has_speech is True
    assert result.available is False


def test_vad_requires_a_run_of_speech_not_scattered_windows(tmp_path, monkeypatch):
    """Three isolated windows across a long recording is a noise floor
    twitching over the threshold, not somebody speaking."""
    vad = SileroVad(model_path=tmp_path / "x.onnx")
    monkeypatch.setattr(vad, "available", True)
    scattered = [0.9, 0.0, 0.9, 0.0, 0.9, 0.0] * 4
    monkeypatch.setattr(vad, "probabilities", lambda *a, **k: scattered)

    assert vad.assess(np.zeros(SR, dtype=np.float32)).has_speech is False

    monkeypatch.setattr(vad, "probabilities", lambda *a, **k: [0.9] * MIN_SPEECH_WINDOWS)
    assert vad.assess(np.zeros(SR, dtype=np.float32)).has_speech is True


def test_vad_reports_speech_seconds_for_the_verifier(tmp_path, monkeypatch):
    vad = SileroVad(model_path=tmp_path / "x.onnx")
    monkeypatch.setattr(vad, "available", True)
    # 32 windows of 512 samples at 16 kHz = 1.024s of speech
    monkeypatch.setattr(vad, "probabilities", lambda *a, **k: [0.9] * 32)

    result = vad.assess(np.zeros(SR, dtype=np.float32))

    assert result.speech_seconds == pytest.approx(32 * 512 / SR)
    assert result.speech_ratio == 1.0


def test_vad_rejects_a_buffer_shorter_than_one_window(tmp_path, monkeypatch):
    vad = SileroVad(model_path=tmp_path / "x.onnx")
    monkeypatch.setattr(vad, "available", True)
    monkeypatch.setattr(vad, "probabilities", lambda *a, **k: [])

    assert vad.assess(np.zeros(100, dtype=np.float32)).has_speech is False


# --- Storage / deletion -----------------------------------------------------


def test_voiceprint_metadata_never_includes_the_template(voiceprint_db):
    verification.enroll("u1", _samples(), FakeEmbedder("ada"))

    meta = voiceprint_store.metadata("u1")

    assert meta["enrolled"] is True
    assert "embedding" not in meta
    assert not any(isinstance(v, (list, tuple)) for v in meta.values())


def test_voiceprint_can_be_deleted_on_its_own(voiceprint_db):
    verification.enroll("u1", _samples(), FakeEmbedder("ada"))

    voiceprint_store.delete("u1")

    assert voiceprint_store.get("u1") is None
    assert voiceprint_store.metadata("u1") == {"enrolled": False}
    # Idempotent: "my voice is not stored" is the requested state either way.
    voiceprint_store.delete("u1")


def test_voiceprint_is_encrypted_at_rest(voiceprint_db):
    verification.enroll("u1", _samples(), FakeEmbedder("ada"))

    conn = sqlite_models.get_connection(voiceprint_db)
    try:
        raw = conn.execute("SELECT embedding FROM voiceprints WHERE user_id = ?", ("u1",)).fetchone()[0]
    finally:
        conn.close()

    assert "1.0" not in raw and "[" not in raw


def test_an_undecryptable_voiceprint_reads_as_absent(voiceprint_db):
    """Treated as "not enrolled" rather than raising: that is a safe state,
    where raising would take out every voice turn."""
    verification.enroll("u1", _samples(), FakeEmbedder("ada"))
    conn = sqlite_models.get_connection(voiceprint_db)
    try:
        conn.execute("UPDATE voiceprints SET embedding = ? WHERE user_id = ?", ("garbage", "u1"))
        conn.commit()
    finally:
        conn.close()

    assert voiceprint_store.get("u1") is None
    assert verification.verify("u1", _speech(), FakeEmbedder("ada")).decision == "not_enrolled"


def test_enrolled_at_survives_re_enrollment(voiceprint_db):
    verification.enroll("u1", _samples(), FakeEmbedder("ada"))
    first = voiceprint_store.get("u1")

    verification.enroll("u1", _samples(), FakeEmbedder("ada"))
    second = voiceprint_store.get("u1")

    assert second.enrolled_at == first.enrolled_at
    assert second.updated_at >= first.updated_at


def test_profile_helper_shape_is_json_safe(voiceprint_db):
    """metadata() goes straight into an API response and the data export."""
    verification.enroll("u1", _samples(), FakeEmbedder("ada"))
    import json

    json.dumps(voiceprint_store.metadata("u1"))


def _unused_profile() -> UserProfile:
    return UserProfile(
        user_id="u1", name="Ada", companion_name="Hearth",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
