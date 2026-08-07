"""Local data cleanup that retains onboarding/profile identity.

The desktop installers call this module before removing an install.  The
Settings reset endpoint uses the same code, so every platform has one
well-tested definition of which local data is disposable.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from app.db.sqlite_models import (
    ACTIVE_PROFILE_SCHEMA,
    PROFILES_SCHEMA,
    _PROFILES_TEXT_INPUT_COLUMNS,
    close_pooled_connections,
)
from app.security.crypto import get_or_create_sqlcipher_key_hex

_PROFILE_DB_NAME = "profile.db"
_RETAINED_DIR_NAME = "retained"
_HEAVY_DIRECTORIES = (
    "models",
    "backend-deps",
    "setup-python",
    "hf-home",
    "crash-logs",
    "data/vector_store",
)
_HEAVY_FILES = (
    "data/memory2_index.sqlite3",
    "data/hearth_learning.duckdb",
    "data/tier_cache.json",
    "data/runtime_snapshot.json",
    ".env",
)


def _config():
    # config imports this module during startup to restore a retained profile,
    # so importing it at module load would create a circular import.
    from app import config

    return config


def _profile_db(root: Path) -> Path:
    return root / "data" / _PROFILE_DB_NAME


def retained_profile_db() -> Path:
    """The OS-owned location that survives removal of the app install tree."""
    config = _config()
    return config._os_app_data_hearth() / _RETAINED_DIR_NAME / _PROFILE_DB_NAME


def _connect(path: Path):
    """Open a short-lived SQLCipher connection without creating every schema."""
    from sqlcipher3 import dbapi2 as sqlcipher

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlcipher.connect(str(path))
    conn.execute(f"PRAGMA key = \"x'{get_or_create_sqlcipher_key_hex()}'\"")
    return conn


def _ensure_identity_schema(conn) -> None:
    conn.execute(PROFILES_SCHEMA)
    conn.execute(ACTIVE_PROFILE_SCHEMA)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(profiles)")}
    for name, declaration in _PROFILES_TEXT_INPUT_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE profiles ADD COLUMN {name} {declaration}")
    conn.commit()


def _remove_db(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)


def _has_profiles(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        conn = _connect(path)
        try:
            tables = {
                row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            return "profiles" in tables and bool(conn.execute("SELECT 1 FROM profiles LIMIT 1").fetchone())
        finally:
            conn.close()
    except Exception:
        return False


def export_profiles_to_retained(source_root: Path | None = None) -> bool:
    """Export only ``profiles`` and ``active_profile`` to OS app data.

    Returns False for a first-run install with no profile.  The export is
    rebuilt atomically enough for this local, best-effort uninstall path:
    the previous retained copy remains until a fresh copy is complete.
    """
    config = _config()
    source = _profile_db(source_root or config.USER_DATA_DIR)
    if not _has_profiles(source):
        return False

    target = retained_profile_db()
    temporary = target.with_suffix(".tmp")
    _remove_db(temporary)
    source_conn = _connect(source)
    target_conn = _connect(temporary)
    try:
        _ensure_identity_schema(target_conn)
        columns = [row[1] for row in source_conn.execute("PRAGMA table_info(profiles)")]
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        placeholders = ", ".join("?" for _ in columns)
        rows = source_conn.execute(f"SELECT {quoted_columns} FROM profiles").fetchall()
        target_conn.executemany(
            f"INSERT INTO profiles ({quoted_columns}) VALUES ({placeholders})", rows
        )
        active = source_conn.execute(
            "SELECT user_id FROM active_profile WHERE id = 1"
        ).fetchone()
        if active is not None:
            target_conn.execute("INSERT INTO active_profile (id, user_id) VALUES (1, ?)", active)
        target_conn.commit()
    finally:
        source_conn.close()
        target_conn.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(target)
    _remove_db(Path(f"{temporary}-wal"))
    _remove_db(Path(f"{temporary}-shm"))
    return True


def _delete_heavy_paths(root: Path) -> None:
    for relative in _HEAVY_DIRECTORIES:
        shutil.rmtree(root / relative, ignore_errors=True)
    for relative in _HEAVY_FILES:
        (root / relative).unlink(missing_ok=True)


def wipe_heavy_userdata(roots: tuple[Path, ...] | None = None, *, remove_profile_db: bool) -> None:
    """Delete regenerable/runtime data from each known Hearth user-data root."""
    config = _config()
    if roots is None:
        roots = (config.USER_DATA_DIR, config._os_app_data_hearth())
    for root in {path.resolve() for path in roots}:
        _delete_heavy_paths(root)
        if remove_profile_db:
            _remove_db(_profile_db(root))


def reset_local_data() -> bool:
    """Settings reset: retain identity, erase all conversations and assets."""
    config = _config()
    close_pooled_connections()
    exported = export_profiles_to_retained()
    wipe_heavy_userdata(remove_profile_db=True)
    restore_retained_profiles_if_needed(config.USER_DATA_DIR, consume=True)
    return exported


def uninstall_cleanup() -> bool:
    """Installer entry point. Never removes OS keychain credentials."""
    close_pooled_connections()
    exported = export_profiles_to_retained()
    wipe_heavy_userdata(remove_profile_db=True)
    return exported


def restore_retained_profiles_if_needed(root: Path | None = None, *, consume: bool = True) -> bool:
    """Restore a retained identity DB when a fresh install has no profile."""
    config = _config()
    destination = _profile_db(root or config.USER_DATA_DIR)
    retained = retained_profile_db()
    if not retained.is_file() or _has_profiles(destination):
        return False

    close_pooled_connections()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _remove_db(destination)
    shutil.copy2(retained, destination)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{retained}{suffix}")
        if sidecar.exists():
            shutil.copy2(sidecar, Path(f"{destination}{suffix}"))
    if consume:
        _remove_db(retained)
        if retained.parent.exists() and not any(retained.parent.iterdir()):
            retained.parent.rmdir()
    return True
