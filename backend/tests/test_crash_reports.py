"""Opt-in crash report buffer + S3 upload (app.diagnostics.crash_log)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app.diagnostics import crash_log


@pytest.fixture
def crash_dir(tmp_path, monkeypatch):
    pending = tmp_path / "crash-logs"
    monkeypatch.setattr(crash_log, "CRASH_LOGS_DIR", pending)
    # Re-import path used inside helpers reads the module attribute.
    monkeypatch.setattr("app.config.CRASH_LOGS_DIR", pending)
    return pending


def test_record_and_list_pending(crash_dir):
    summary = crash_log.record_crash(
        source="python",
        message="RuntimeError: boom",
        stack="Traceback...",
        extra={"tier": "C", "secret_should_drop": "nope"},
    )
    assert summary["id"]
    pending = crash_log.list_pending()
    assert len(pending) == 1
    assert pending[0]["message"] == "RuntimeError: boom"

    stored = json.loads((crash_dir / "pending" / f"{summary['id']}.json").read_text())
    assert stored["tier"] == "C"
    assert "secret_should_drop" not in stored
    assert "transcript" not in stored


def test_dismiss_deletes_pending(crash_dir):
    summary = crash_log.record_crash(source="frontend", message="TypeError: x")
    assert crash_log.dismiss_report(summary["id"]) is True
    assert crash_log.list_pending() == []
    assert crash_log.dismiss_report(summary["id"]) is False


def test_send_uploads_and_removes(crash_dir, monkeypatch):
    summary = crash_log.record_crash(source="python", message="boom", stack="tb")
    monkeypatch.setattr(crash_log, "internet_reachable", lambda timeout=3.0: True)

    put_calls = []

    class FakeResp:
        status = 200

        def read(self, n=-1):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=30):
        put_calls.append((req.full_url, req.data, req.get_method()))
        return FakeResp()

    monkeypatch.setattr(crash_log, "urlopen", fake_urlopen)

    result = crash_log.send_report(summary["id"])
    assert result["id"] == summary["id"]
    assert "uploaded_as" in result
    assert crash_log.list_pending() == []
    assert len(put_calls) == 1
    url, body, method = put_calls[0]
    assert method == "PUT"
    assert "/hearth_ai/crash-logs/" in url
    assert json.loads(body)["message"] == "boom"


def test_send_requires_internet(crash_dir, monkeypatch):
    summary = crash_log.record_crash(source="python", message="boom")
    monkeypatch.setattr(crash_log, "internet_reachable", lambda timeout=3.0: False)
    with pytest.raises(ConnectionError):
        crash_log.send_report(summary["id"])
    assert len(crash_log.list_pending()) == 1


def test_api_pending_send_dismiss(crash_dir, monkeypatch):
    client = TestClient(main.app)
    created = client.post(
        "/api/crash-reports",
        json={"message": "ReferenceError: x", "stack": "at App", "component": "window.onerror"},
    )
    assert created.status_code == 200
    report_id = created.json()["id"]

    listed = client.get("/api/crash-reports/pending")
    assert listed.status_code == 200
    assert listed.json()["reports"][0]["id"] == report_id

    monkeypatch.setattr(crash_log, "internet_reachable", lambda timeout=3.0: True)

    class FakeResp:
        status = 200

        def read(self, n=-1):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(crash_log, "urlopen", lambda *a, **k: FakeResp())

    sent = client.post(f"/api/crash-reports/{report_id}/send")
    assert sent.status_code == 200
    assert client.get("/api/crash-reports/pending").json()["reports"] == []

    # Second report — dismiss path
    created2 = client.post("/api/crash-reports", json={"message": "again"})
    rid2 = created2.json()["id"]
    dismissed = client.post(f"/api/crash-reports/{rid2}/dismiss")
    assert dismissed.status_code == 200
    assert client.get("/api/crash-reports/pending").json()["reports"] == []


def test_api_send_offline_returns_503(crash_dir, monkeypatch):
    client = TestClient(main.app)
    created = client.post("/api/crash-reports", json={"message": "offline"})
    rid = created.json()["id"]
    monkeypatch.setattr(crash_log, "internet_reachable", lambda timeout=3.0: False)
    res = client.post(f"/api/crash-reports/{rid}/send")
    assert res.status_code == 503
