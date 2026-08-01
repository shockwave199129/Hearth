"""Shared test fixtures for the Hearth backend test suite."""

from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cognitive.mind_state import MindState
from app.onboarding.profile_schema import UserProfile


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test isolation."""
    return tmp_path


@pytest.fixture
def profile():
    """Minimal valid UserProfile for testing."""
    return UserProfile(
        user_id="test-user",
        name="Test",
        companion_name="Hearth",
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mind_state():
    """Fresh MindState instance."""
    return MindState()


class FakeLlm:
    """Configurable fake LLM adapter for tests."""

    def __init__(self, response: str = "I'm here with you."):
        self.response = response
        self.calls: list[str] = []

    def complete(self, prompt: str, max_tokens: int = 220, **kwargs) -> str:
        self.calls.append(prompt)
        return self.response


@pytest.fixture
def fake_llm():
    """A fake LLM that records calls and returns a fixed response."""
    return FakeLlm()
