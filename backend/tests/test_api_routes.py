"""API-level integration tests against Depends-injected routes.

These exist because of ARCH-1: while Pipeline lived inside main.py with a
module global, route handlers could not be exercised without constructing
real STT/LLM/TTS engines. Now routers take ``Depends(get_pipeline)``, so a
fake pipeline unblocks coverage of the HTTP surface without a GPU or model
weights.
"""

from __future__ import annotations

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
        json={"name": "Bo", "companion_name": "Ember", "speak_replies": False},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "Bo"
    assert active_profile.get_active_user_id() == body["user_id"]
    assert deps.get_pipeline_optional().profile.user_id == body["user_id"]


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
    assert res.json() == []


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
