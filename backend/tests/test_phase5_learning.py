from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.worker import EvaluationWorker
from app.learning.observation_store import ObservationStore
from app.learning.recompute import recompute_all
from app.onboarding.profile_schema import UserProfile
from app.onboarding.profile_store import create_profile
from app.onboarding.profile_store import get_profile
from app.onboarding.profile_schema import OnboardingRequest


def _profile(user_id: str = "u1") -> UserProfile:
    return UserProfile(
        user_id=user_id,
        name="A",
        companion_name="Companion",
        created_at=datetime.now(timezone.utc),
    )


def test_observation_store_append_and_read(tmp_path):
    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    store.append("communication", "likes_reflection", 1.0, {"signal": "explicit"}, "growth_engine")
    obs = store.latest("communication", "likes_reflection")
    assert obs and obs[0].value == 1.0


def test_recompute_returns_shape(tmp_path, monkeypatch):
    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    monkeypatch.setattr("app.learning.recompute.ObservationStore", lambda: store)
    monkeypatch.setattr("app.learning.recompute.get_profile", lambda user_id: _profile(user_id))
    monkeypatch.setattr(
        "app.learning.recompute.update_learning_state",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "app.learning.recompute.update_relationship_state",
        lambda *args, **kwargs: None,
    )
    store.append("communication", "likes_reflection", 1.0, {"signal": "explicit"}, "growth_engine")
    result = recompute_all("u1", store)
    assert "likes_reflection" in result.communication_traits


def test_evaluation_worker_writes_observations(tmp_path):
    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    worker = EvaluationWorker(store)
    result = worker.evaluate("u1", "hello", "I’m here with you.")
    assert result.invariant_score > 0.0
    assert store.latest("communication", "evaluation_invariant")

