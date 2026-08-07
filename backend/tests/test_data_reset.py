"""Regression tests for profile-preserving local-data cleanup."""

from pathlib import Path

from app.db.sqlite_models import close_pooled_connections, get_connection
from app.setup import data_reset


def _insert_profile(db: Path) -> None:
    conn = get_connection(db)
    try:
        conn.execute(
            """INSERT INTO profiles (
                user_id, name, stressors, preferred_voice, companion_name, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("profile-1", "Riki", "[]", "default", "Hearth", "2026-08-07T00:00:00+00:00"),
        )
        conn.execute("INSERT INTO active_profile (id, user_id) VALUES (1, ?)", ("profile-1",))
        conn.execute(
            """INSERT INTO chat_history (
                user_id, session_id, turn_id, role, content, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("profile-1", "session", 1, "user", "encrypted chat", "2026-08-07T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()


def _configure_roots(tmp_path, monkeypatch):
    from app import config

    install_root = tmp_path / "install-userdata"
    os_root = tmp_path / "os-data" / "Hearth"
    monkeypatch.setattr(config, "USER_DATA_DIR", install_root)
    monkeypatch.setattr(config, "DATA_DIR", install_root / "data")
    monkeypatch.setattr(config, "_os_app_data_hearth", lambda: os_root)
    return install_root, os_root


def test_uninstall_cleanup_wipes_heavy_data_and_retains_only_identity(tmp_path, monkeypatch):
    install_root, os_root = _configure_roots(tmp_path, monkeypatch)
    profile_db = install_root / "data" / "profile.db"
    _insert_profile(profile_db)

    for relative in (
        "models/llm/model.gguf",
        "backend-deps/package.py",
        "setup-python/python",
        "hf-home/cache",
        "crash-logs/pending/report.json",
        "data/vector_store/chroma.sqlite3",
        "data/memory2_index.sqlite3",
        "data/hearth_learning.duckdb",
        "data/tier_cache.json",
        "data/runtime_snapshot.json",
        ".env",
    ):
        path = install_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("disposable", encoding="utf-8")

    assert data_reset.uninstall_cleanup() is True
    assert not profile_db.exists()
    assert not (install_root / "models").exists()
    assert not (install_root / "data" / "vector_store").exists()
    assert not (install_root / ".env").exists()

    retained = os_root / "retained" / "profile.db"
    assert retained.exists()
    conn = data_reset._connect(retained)  # pylint: disable=protected-access
    try:
        assert conn.execute("SELECT name FROM profiles").fetchone()[0] == "Riki"
        assert conn.execute("SELECT user_id FROM active_profile").fetchone()[0] == "profile-1"
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables == {"profiles", "active_profile"}
    finally:
        conn.close()


def test_reset_rebuilds_profile_db_without_history_and_consumes_retained(tmp_path, monkeypatch):
    install_root, os_root = _configure_roots(tmp_path, monkeypatch)
    profile_db = install_root / "data" / "profile.db"
    _insert_profile(profile_db)

    assert data_reset.reset_local_data() is True
    assert not (os_root / "retained" / "profile.db").exists()

    # get_connection applies the normal schema after restore; identity remains,
    # while the conversation table is fresh and setup begins from incomplete.
    close_pooled_connections()
    conn = get_connection(profile_db)
    try:
        assert conn.execute("SELECT name FROM profiles").fetchone()[0] == "Riki"
        assert conn.execute("SELECT COUNT(*) FROM chat_history").fetchone()[0] == 0
        assert conn.execute("SELECT complete FROM setup_state WHERE id = 1").fetchone() is None
    finally:
        conn.close()
