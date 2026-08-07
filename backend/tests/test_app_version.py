"""App version resolution (tag v0.3.4 → 0.3.4)."""

from __future__ import annotations

from app import version as version_mod


def test_strip_v_prefix():
    assert version_mod.strip_v_prefix("v0.3.4") == "0.3.4"
    assert version_mod.strip_v_prefix("V0.3.4") == "0.3.4"
    assert version_mod.strip_v_prefix("0.3.4") == "0.3.4"


def test_env_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTH_APP_VERSION", "v0.3.4")
    monkeypatch.setattr(version_mod, "_VERSION_FILE", tmp_path / "missing")
    monkeypatch.setattr(version_mod, "_from_git", lambda: "9.9.9")
    assert version_mod.resolve_app_version() == "0.3.4"


def test_baked_file_when_no_env(monkeypatch, tmp_path):
    baked = tmp_path / "VERSION"
    baked.write_text("v0.3.4\n", encoding="utf-8")
    monkeypatch.delenv("HEARTH_APP_VERSION", raising=False)
    monkeypatch.setattr(version_mod, "_VERSION_FILE", baked)
    monkeypatch.setattr(version_mod, "_from_git", lambda: "9.9.9")
    assert version_mod.resolve_app_version() == "0.3.4"


def test_git_then_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("HEARTH_APP_VERSION", raising=False)
    monkeypatch.setattr(version_mod, "_VERSION_FILE", tmp_path / "missing")
    monkeypatch.setattr(version_mod, "_from_git", lambda: "0.3.3")
    assert version_mod.resolve_app_version() == "0.3.3"

    monkeypatch.setattr(version_mod, "_from_git", lambda: None)
    assert version_mod.resolve_app_version() == "0.0.0"
