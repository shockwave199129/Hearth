"""API-level integration tests against Depends-injected routes.

These exist because of ARCH-1: while Pipeline lived inside main.py with a
module global, route handlers could not be exercised without constructing
real STT/LLM/TTS engines. Now routers take ``Depends(get_pipeline)``, so a
fake pipeline unblocks coverage of the HTTP surface without a GPU or model
weights.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import deps, main
from app.db import sqlite_models
from app.memory import chat_history
from app.memory.short_term import ShortTermMemory
from app.onboarding import active_profile, profile_store
from app.onboarding.profile_schema import UserProfile
from app.setup import orchestrator


class FakeTts:
    sample_rate = 16000

    def synthesize(self, text: str, voice: str = "female", style: str = "grounded"):
        # A few samples of silence — enough for pcm_to_wav_bytes to succeed.
        return np.zeros(32, dtype=np.float32)


class FakeGrowthStore:
    """memory2 privacy helpers accept any object with the methods they call;
    for routes that only delete/summarize we stub the minimal surface."""

    def list_by_user(self, *args, **kwargs):
        return []

    def delete_all_for_user(self, *args, **kwargs):
        return None

    # Read by POST /api/data/export — an empty tiered store is the honest
    # answer for a fake, and distinguishes "no memories" from the
    # unreadable-store case tests/test_data_export.py covers.
    def list_episodic(self, *args, **kwargs):
        return []

    def list_semantic(self, *args, **kwargs):
        return []


class FakePipeline:
    def __init__(self, profile: UserProfile):
        self.profile = profile
        self.tier = SimpleNamespace(
            tier="C",
            llm_gguf="fake.gguf",
            stt_model="fake-stt",
            tts_engine="kokoro",
            n_gpu_layers=0,
            ctx_size=2048,
        )
        self.tts = FakeTts()
        # Voice-input analysis is optional; a fake pipeline reports it absent,
        # which is the state every install without the weights is in.
        self.speaker_embedder = SimpleNamespace(available=False)
        self.vad = SimpleNamespace(available=False)
        self.growth_engine = SimpleNamespace(store=FakeGrowthStore())
        self._llm = SimpleNamespace(complete=lambda *a, **k: "summary")

    def set_profile(self, profile: UserProfile) -> None:
        self.profile = profile

    def new_session_memory(self) -> ShortTermMemory:
        return ShortTermMemory(self._llm)

    def respond_to_text(self, text: str, memory: ShortTermMemory):
        reply = f"heard: {text}"
        memory.add_turn(text, reply)
        return text, reply, None, 0, 1

    async def run_maintenance(self, memory: ShortTermMemory) -> None:
        return None

    def shutdown(self) -> None:
        return None


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Point every profile.db consumer at a throwaway path and clear the pool."""
    db = tmp_path / "profile.db"
    sqlite_models.close_pooled_connections()
    for mod_path, attr in (
        ("app.onboarding.profile_store", "PROFILE_DB_PATH"),
        ("app.onboarding.active_profile", "ACTIVE_PROFILE_DB_PATH"),
        ("app.memory.chat_history", "CHAT_HISTORY_DB_PATH"),
        ("app.checkin.state", "CHECKIN_DB_PATH"),
        ("app.safety.crisis_detector", "CRISIS_DB_PATH"),
        ("app.safety.escalation", "ESCALATION_DB_PATH"),
        ("app.safety2.audit", "SAFETY_AUDIT_DB_PATH"),
        ("app.relationship.profile_store", "RELATIONSHIP_PROFILE_DB_PATH"),
        ("app.voice.store", "VOICEPRINT_DB_PATH"),
        ("app.voice.consent", "CONSENT_DB_PATH"),
        ("app.setup.state", "SETUP_STATE_DB_PATH"),
    ):
        monkeypatch.setattr(f"{mod_path}.{attr}", db)
    yield db
    deps.set_pipeline(None)
    sqlite_models.close_pooled_connections()


@pytest.fixture
def profile(isolated_db) -> UserProfile:
    from app.onboarding.profile_schema import OnboardingRequest

    return profile_store.create_profile(
        OnboardingRequest(name="Ada", companion_name="Hearth", speak_replies=False)
    )


@pytest.fixture
def client(isolated_db, profile, monkeypatch):
    """TestClient with setup marked incomplete (so startup skips real Pipeline)
    and a FakePipeline injected afterwards."""
    monkeypatch.setattr(
        orchestrator,
        "detect_status",
        lambda: {
            "hardware": {},
            "tier": "C",
            "tts_engine": "kokoro",
            "gpu_vendor": "none",
            "complete": False,
        },
    )
    active_profile.set_active_user_id(profile.user_id)
    with TestClient(main.app) as c:
        deps.set_pipeline(FakePipeline(profile))
        yield c


def test_status_returns_tier_from_pipeline(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    body = res.json()
    assert body["tier"] == "C"
    assert body["tts_engine"] == "kokoro"


def test_status_503_without_pipeline(isolated_db, monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "detect_status",
        lambda: {
            "hardware": {},
            "tier": "C",
            "tts_engine": "kokoro",
            "gpu_vendor": "none",
            "complete": False,
        },
    )
    with TestClient(main.app) as c:
        deps.set_pipeline(None)
        assert c.get("/api/status").status_code == 503


def test_reset_data_stops_pipeline_and_reports_retained_profile(client, monkeypatch):
    stopped = []
    pipeline = deps.get_pipeline_optional()
    monkeypatch.setattr(pipeline, "shutdown", lambda: stopped.append(True))
    monkeypatch.setattr("app.api.data.reset_local_data", lambda: True)

    response = client.post("/api/data/reset")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "profile_retained": True}
    assert stopped == [True]
    assert deps.get_pipeline_optional() is None


def test_export_data_writes_a_folder_and_returns_its_path(client, monkeypatch, tmp_path):
    # exports_root() is ~/Hearth-exports; a test must never write there.
    monkeypatch.setattr("app.data_export.exports_root", lambda: tmp_path / "exports")
    chat_history.record_turn(
        deps.get_pipeline().profile.user_id, "session", 1, "user", "exported turn"
    )

    response = client.post("/api/data/export")

    assert response.status_code == 200
    body = response.json()
    folder = Path(body["path"])
    assert folder.is_dir()
    assert (folder / "transcript.json").is_file()
    assert body["counts"]["transcript_messages"] == 1


def test_export_data_reports_a_write_failure_instead_of_a_partial_success(
    client, monkeypatch, tmp_path
):
    def unwritable():
        raise OSError("No space left on device")

    monkeypatch.setattr("app.data_export.exports_root", unwritable)

    response = client.post("/api/data/export")

    assert response.status_code == 500
    assert "No space left on device" in response.json()["detail"]


def test_export_data_503_without_pipeline(isolated_db, monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "detect_status",
        lambda: {
            "hardware": {},
            "tier": "C",
            "tts_engine": "kokoro",
            "gpu_vendor": "none",
            "complete": False,
        },
    )
    with TestClient(main.app) as c:
        deps.set_pipeline(None)
        assert c.post("/api/data/export").status_code == 503


# --- Voice enrollment (app/api/voice.py) ------------------------------------


def _pcm_bundle(samples: list[np.ndarray]) -> bytes:
    """Mirror of frontend/src/lib/voiceEnrollment.ts encodeSamples, so this
    test fails if either side of the wire format drifts."""
    header = np.array([len(samples)], dtype="<u4").tobytes()
    header += np.array([len(s) for s in samples], dtype="<u4").tobytes()
    return header + b"".join(np.asarray(s, dtype="<f4").tobytes() for s in samples)


def test_enrollment_status_reports_unavailable_without_the_model(client, monkeypatch):
    monkeypatch.setattr(deps.get_pipeline().speaker_embedder, "available", False)

    body = client.get("/api/voice/enrollment").json()

    assert body["model_available"] is False
    assert body["enrolled"] is False


def test_enrollment_status_never_returns_the_template(client, monkeypatch):
    """A biometric template must not be reachable from the API at all."""
    monkeypatch.setattr(
        "app.voice.store.metadata",
        lambda user_id: {"enrolled": True, "sample_count": 3, "enrolled_at": "2026-08-17T00:00:00Z"},
    )

    body = client.get("/api/voice/enrollment").json()

    assert body["enrolled"] is True
    assert "embedding" not in body


def test_enroll_rejects_a_malformed_bundle(client):
    res = client.post("/api/voice/enrollment", content=b"\x03\x00\x00\x00short")

    assert res.status_code == 422
    assert "Malformed" in res.json()["detail"]


def test_enroll_rejects_an_implausible_sample_count(client):
    res = client.post("/api/voice/enrollment", content=np.array([99], dtype="<u4").tobytes())

    assert res.status_code == 422


def test_enroll_passes_decoded_samples_through_and_surfaces_refusals(client, monkeypatch):
    """The route must not turn a recoverable enrollment refusal into a 500 —
    the usual causes (too short, more than one voice) are user-fixable."""
    seen = {}

    def fake_enroll(user_id, samples, embedder):
        seen["lengths"] = [len(s) for s in samples]
        from app.voice.verification import EnrollmentResult

        return EnrollmentResult(ok=False, error="Those recordings didn't match.")

    monkeypatch.setattr("app.api.voice.verification.enroll", fake_enroll)
    bundle = _pcm_bundle([np.zeros(4000, dtype=np.float32), np.zeros(6000, dtype=np.float32)])

    res = client.post("/api/voice/enrollment", content=bundle)

    assert seen["lengths"] == [4000, 6000]
    assert res.status_code == 422
    assert res.json()["detail"] == "Those recordings didn't match."


def test_forget_voice_is_idempotent(client, monkeypatch):
    deleted = []
    monkeypatch.setattr("app.api.voice.voiceprint_store.delete", lambda uid: deleted.append(uid))

    first = client.delete("/api/voice/enrollment")
    second = client.delete("/api/voice/enrollment")

    assert first.status_code == second.status_code == 200
    # consent_recorded is part of the response because deleting the template
    # also withdraws permission to collect the next one.
    assert first.json() == {"ok": True, "enrolled": False, "consent_recorded": False}
    assert len(deleted) == 2


def test_enrollment_status_serves_the_consent_text_and_retention_window(client):
    """The wording is served, never duplicated in the frontend, so the version
    recorded against a profile corresponds to text the user actually read."""
    body = client.get("/api/voice/enrollment").json()

    assert "biometric" in body["consent_text"]
    assert body["consent_recorded"] is False
    assert body["consent_current"] is False
    assert body["retention_days"] > 0
    # No voiceprint yet, so nothing is scheduled for destruction.
    assert body["expires_at"] is None


def test_consent_route_records_a_server_stamped_agreement(client):
    res = client.post("/api/voice/consent")

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["consent_version"] == body["current_consent_version"]
    assert client.get("/api/voice/enrollment").json()["consent_current"] is True


def test_enroll_is_forbidden_until_consent_is_recorded(client, monkeypatch):
    """403 rather than 422: missing consent is a permission state, and the UI
    shows the consent step instead of a "try recording again" hint."""
    monkeypatch.setattr(deps.get_pipeline().speaker_embedder, "available", True)
    monkeypatch.setattr(
        "app.api.voice.verification.enroll",
        lambda user_id, samples, embedder: __import__(
            "app.voice.verification", fromlist=["EnrollmentResult"]
        ).EnrollmentResult(ok=False, error="needs agreement", needs_consent=True),
    )
    bundle = _pcm_bundle([np.zeros(4000, dtype=np.float32)])

    assert client.post("/api/voice/enrollment", content=bundle).status_code == 403


def test_forget_voice_also_withdraws_consent(client):
    client.post("/api/voice/consent")
    assert client.get("/api/voice/enrollment").json()["consent_current"] is True

    body = client.delete("/api/voice/enrollment").json()

    assert body["consent_recorded"] is False
    assert client.get("/api/voice/enrollment").json()["consent_current"] is False


# --- About / attribution (app/api/about.py) ----------------------------------


def test_about_reports_version_and_components(client):
    body = client.get("/api/about").json()

    assert body["version"]
    assert any(c["name"].startswith("Moonshine") for c in body["components"])
    assert all({"name", "purpose", "license"} <= set(c) for c in body["components"])


def test_about_credits_the_cc_by_model_only_when_it_is_installed(client, monkeypatch):
    """CC-BY attribution attaches to what is distributed. A build with no
    speaker model must not credit one, and a build with it must."""
    monkeypatch.setattr("app.api.about.SPEAKER_MODEL_PATH", Path("/nonexistent/model.onnx"))
    absent = client.get("/api/about").json()
    assert absent["required_attributions"] == []

    monkeypatch.setattr(
        "app.api.about.SPEAKER_MODEL_PATH", SimpleNamespace(is_file=lambda: True)
    )
    present = client.get("/api/about").json()
    assert any("WeSpeaker" in line and "CC-BY-4.0" in line for line in present["required_attributions"])


def test_about_exposes_no_profile_data(client):
    body = client.get("/api/about").json()

    assert set(body) == {"version", "components", "required_attributions"}
    assert "Ada" not in str(body)


def test_voice_routes_503_without_pipeline(isolated_db, monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "detect_status",
        lambda: {
            "hardware": {},
            "tier": "C",
            "tts_engine": "kokoro",
            "gpu_vendor": "none",
            "complete": False,
        },
    )
    with TestClient(main.app) as c:
        deps.set_pipeline(None)
        assert c.get("/api/voice/enrollment").status_code == 503


def test_get_profile_404_when_none_active(client, monkeypatch):
    # profile.py binds get_active_user_id at import time — patch there.
    monkeypatch.setattr("app.api.profile.get_active_user_id", lambda: None)
    assert client.get("/api/profile").status_code == 404


def test_get_and_update_profile(client, profile):
    res = client.get("/api/profile")
    assert res.status_code == 200
    assert res.json()["name"] == "Ada"

    res = client.put("/api/profile", json={"speak_replies": False, "region": "us"})
    assert res.status_code == 200
    assert res.json()["speak_replies"] is False
    assert res.json()["region"] == "us"


def test_onboarding_creates_and_activates(client):
    res = client.post(
        "/api/onboarding",
        # adult_attested is required — see the 18+ gate test below. This
        # request predated that gate and asserted 200 while the route
        # returned 422, so it was failing rather than covering anything.
        json={
            "name": "Bo",
            "companion_name": "Ember",
            "speak_replies": False,
            "adult_attested": True,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "Bo"
    assert active_profile.get_active_user_id() == body["user_id"]
    assert deps.get_pipeline_optional().profile.user_id == body["user_id"]


def test_onboarding_rejects_a_profile_that_has_not_attested_to_being_an_adult(client):
    """The 18+ gate is a launch-gate item (docs/compliance.md) whose whole
    point is that it cannot be bypassed by calling the API directly instead
    of using the UI. It had no test until now."""
    res = client.post(
        "/api/onboarding",
        json={"name": "Bo", "companion_name": "Ember", "adult_attested": False},
    )

    assert res.status_code == 422
    assert active_profile.get_active_user_id() != "Bo"


def test_onboarding_treats_a_missing_attestation_as_absent_not_as_consent(client):
    """`adult_attested` defaults to False in the schema; a client that simply
    omits the field must be refused rather than defaulting into attested."""
    res = client.post("/api/onboarding", json={"name": "Bo", "companion_name": "Ember"})

    assert res.status_code == 422


def test_onboarding_records_when_the_attestation_happened(client):
    """Stamped server-side so it cannot be backdated by a client, and stored
    per profile so a later policy change can tell who was onboarded under
    which wording."""
    res = client.post(
        "/api/onboarding",
        json={"name": "Bo", "companion_name": "Ember", "adult_attested": True},
    )

    body = res.json()
    assert body["adult_attested"] is True
    assert body["adult_attested_at"] is not None
    assert body["ai_disclosure_ack_at"] is not None


def test_list_profiles(client, profile):
    res = client.get("/api/profiles")
    assert res.status_code == 200
    ids = {p["user_id"] for p in res.json()}
    assert profile.user_id in ids


def test_skills_list_and_get(client):
    res = client.get("/api/skills")
    assert res.status_code == 200
    skills = res.json()
    assert isinstance(skills, list)
    assert len(skills) > 0
    skill_id = skills[0]["id"]

    detail = client.get(f"/api/skills/{skill_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == skill_id
    assert "content" in detail.json()

    assert client.get("/api/skills/does-not-exist").status_code == 404


def test_memories_crud(client, profile, monkeypatch, tmp_path):
    # long_term uses chromadb under DATA_DIR — point it at tmp via the
    # store's own path knobs if present; otherwise exercise list (empty).
    res = client.get("/api/memories")
    assert res.status_code == 200
    body = res.json()
    assert body["items"] == []
    assert body["has_more"] is False
    assert body["offset"] == 0
    assert body["limit"] == 50

    page = client.get("/api/memories", params={"limit": 10, "offset": 0})
    assert page.status_code == 200
    assert page.json()["limit"] == 10


def test_chat_history_empty_then_replay_rejects_missing(client):
    res = client.get("/api/chat_history")
    assert res.status_code == 200
    assert res.json()["items"] == []

    assert client.get("/api/chat_history/999/audio").status_code == 404


def test_chat_history_replay_synthesizes_wav(client, profile):
    row_id = chat_history.record_turn(profile.user_id, "sess", 1, "assistant", "Hello there.")
    res = client.get(f"/api/chat_history/{row_id}/audio")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("audio/wav")
    assert res.content[:4] == b"RIFF"


def test_checkin_and_safety_status(client):
    assert client.get("/api/checkin").status_code == 200
    safety = client.get("/api/safety/status")
    assert safety.status_code == 200
    assert "recent_crisis_events" in safety.json()
    assert "safety_log_retention_policy" in safety.json()


def test_setup_status_and_progress(client):
    status = client.get("/api/setup/status")
    assert status.status_code == 200
    assert status.json()["complete"] is False

    progress = client.get("/api/setup/progress")
    assert progress.status_code == 200
    assert "step" in progress.json()


def test_websocket_text_turn(client, profile):
    # speak_replies is False on the fixture profile so the server skips audio.
    with client.websocket_connect("/ws") as ws:
        ws.send_text('{"type": "text", "text": "hi"}')
        meta = ws.receive_json()
        assert meta["transcript"] == "hi"
        assert "heard: hi" in meta["reply_text"]
        assert meta["has_audio"] is False
        # Deliberately silent, not a synthesis failure.
        assert meta["voice_failed"] is False


def test_websocket_text_turn_flags_failed_voice(client, profile):
    """speak_replies on with no audio means synthesis failed. The turn must
    still reach the client, flagged — dropping it hid the user's own message
    too, since the transcript only renders turns that arrive."""
    pipeline = deps.get_pipeline_optional()
    pipeline.set_profile(profile.model_copy(update={"speak_replies": True}))
    with client.websocket_connect("/ws") as ws:
        ws.send_text('{"type": "text", "text": "hi"}')
        meta = ws.receive_json()
        assert meta.get("type") != "error"
        assert meta["transcript"] == "hi"
        assert "heard: hi" in meta["reply_text"]
        assert meta["has_audio"] is False
        assert meta["voice_failed"] is True
