"""The per-launch local API token (see desktop/src-tauri/src/main.rs).

Loopback binding keeps the network out; this keeps other processes running
as the same user out. `/api/setup/progress` is the probe route throughout:
it answers from memory before the pipeline exists, so neither a 503 nor the
developer's real profile.db can mask an auth regression.
"""

import pytest
from fastapi.testclient import TestClient

from app import main

TOKEN = "test-token-value"


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def secured(monkeypatch):
    """Token set, as the packaged app runs. The middleware reads the module
    global at request time, so patching it is equivalent to launching with
    HEARTH_API_TOKEN set."""
    monkeypatch.setattr(main, "API_TOKEN", TOKEN)


def test_unauthenticated_when_no_token_is_configured(client):
    """A bare `python -m app.main` (dev, --cli, Docker) has no token to
    share, so requiring one would just break it."""
    assert client.get("/api/setup/progress").status_code == 200


def test_request_without_the_token_is_rejected(secured, client):
    response = client.get("/api/setup/progress")
    assert response.status_code == 401
    assert "token" in response.json()["detail"]


def test_request_with_the_wrong_token_is_rejected(secured, client):
    response = client.get("/api/setup/progress", headers={"X-Hearth-Token": "not-it"})
    assert response.status_code == 401


def test_non_ascii_token_is_rejected_not_crashed(secured):
    """Starlette decodes header bytes as latin-1, so a raw high byte on the
    wire arrives as a non-ASCII str — which `secrets.compare_digest` raises
    TypeError on unless both sides are encoded first, turning an attacker's
    junk header into a 500. Checked at the function rather than through the
    client, because httpx refuses to send such a header at all."""
    assert main._token_matches(b"\xf6\xf6".decode("latin-1")) is False


def test_request_with_the_token_passes_through(secured, client):
    response = client.get("/api/setup/progress", headers={"X-Hearth-Token": TOKEN})
    assert response.status_code == 200


def test_websocket_without_the_token_is_refused(secured, client):
    """The browser WebSocket constructor cannot set headers, so /ws checks a
    query parameter instead — and rejects before accept(), so the handshake
    itself fails rather than the socket opening and closing."""
    with pytest.raises(Exception):
        with client.websocket_connect("/ws"):
            pass
