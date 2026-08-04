"""Shared test fixtures for the Hearth backend test suite."""

from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile

import keyring
import pytest
from keyring.backend import KeyringBackend

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cognitive.mind_state import MindState
from app.onboarding.profile_schema import UserProfile


class _InMemoryKeyring(KeyringBackend):
    """Process-local stand-in for the OS keychain."""

    priority = 1  # type: ignore[assignment]

    def __init__(self):
        super().__init__()
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self._store.pop((service, username), None)


@pytest.fixture(scope="session", autouse=True)
def isolate_keyring():
    """Route app.security.crypto's key storage into memory for the suite.

    Two reasons this is autouse rather than opt-in. Any test that touches an
    encrypted store transitively calls ``get_or_create_key``, which on a
    headless machine (CI, a container, a bare SSH session) has no viable
    backend and raises NoKeyringError — the suite was failing 22 tests on
    exactly that. Where a backend *does* exist, the unpatched code writes a
    real secret into the developer's login keychain under the same service
    name the shipped app uses, so running the tests mutated live user state.

    Session-scoped: a keychain persists across a process, and tests like
    test_phase2_memory's restart cases depend on the key staying stable
    across the reopens they simulate.
    """
    previous = keyring.get_keyring()
    keyring.set_keyring(_InMemoryKeyring())
    yield
    keyring.set_keyring(previous)


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
