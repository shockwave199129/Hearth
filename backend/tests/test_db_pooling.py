"""Connection pooling (`app.db.sqlite_models`) and the SQLCipher key format
(`app.security.crypto`).

Both are changes to code every encrypted store depends on, and both have a
silent failure mode the rest of the suite would not catch: a pooled
connection that leaks uncommitted writes across borrows, and a key
derivation change that makes existing databases undecryptable.
"""

import threading

import keyring
import pytest

from app.db import sqlite_models
from app.security import crypto

# A real Fernet key as stored by installs predating raw-key mode: 44
# characters of url-safe base64, hex-encoded to 88 characters at open time.
LEGACY_FERNET_KEY = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWY="

_INSERT_PROFILE = """
INSERT INTO profiles (user_id, name, stressors, preferred_voice, companion_name, created_at)
VALUES (?, 'n', '[]', 'v', 'c', '2026-01-01T00:00:00+00:00')
"""


@pytest.fixture
def db_path(tmp_path):
    """A throwaway database, with the pool cleared on both sides so neither
    a previous test's cached handle nor this one's outlives the test."""
    sqlite_models.close_pooled_connections()
    yield tmp_path / "profile.db"
    sqlite_models.close_pooled_connections()


def _count_profiles(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]


def test_close_releases_to_the_pool_instead_of_closing(db_path):
    first = sqlite_models.get_connection(db_path)
    underlying = first._conn
    first.execute(_INSERT_PROFILE, ("u1",))
    first.commit()
    first.close()

    second = sqlite_models.get_connection(db_path)
    assert second._conn is underlying
    assert _count_profiles(second) == 1
    second.close()


def test_release_discards_uncommitted_writes(db_path):
    """What a real `close()` would have done. Without this, a store function
    that forgets to commit leaves a write transaction open on a connection
    every later caller borrows."""
    conn = sqlite_models.get_connection(db_path)
    conn.execute(_INSERT_PROFILE, ("committed",))
    conn.commit()
    conn.execute(_INSERT_PROFILE, ("never-committed",))
    conn.close()

    conn = sqlite_models.get_connection(db_path)
    assert _count_profiles(conn) == 1
    conn.close()


def test_schema_is_initialized_once_per_database(db_path, monkeypatch):
    sqlite_models.get_connection(db_path).close()

    calls = []
    real_init = sqlite_models.init_schema
    monkeypatch.setattr(
        sqlite_models, "init_schema", lambda conn: calls.append(conn) or real_init(conn)
    )

    sqlite_models.get_connection(db_path).close()
    assert calls == []


def test_each_thread_gets_its_own_connection(db_path):
    """sqlite3 objects are not safe to share across threads, and FastAPI
    runs sync route handlers on a worker pool."""
    main_conn = sqlite_models.get_connection(db_path)
    main_conn.execute(_INSERT_PROFILE, ("u1",))
    main_conn.commit()

    seen = {}

    def worker():
        conn = sqlite_models.get_connection(db_path)
        seen["handle"] = conn._conn
        seen["rows"] = _count_profiles(conn)
        conn.close()
        sqlite_models.close_pooled_connections()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert seen["handle"] is not main_conn._conn
    assert seen["rows"] == 1
    main_conn.close()


def test_new_installs_get_a_real_sqlcipher_raw_key():
    keyring.delete_password(crypto.SERVICE_NAME, crypto.SQLCIPHER_KEY_ACCOUNT)
    try:
        key = crypto.get_or_create_sqlcipher_key_hex()
        assert crypto.is_sqlcipher_raw_key(key)
        assert crypto.get_or_create_sqlcipher_key_hex() == key
    finally:
        keyring.delete_password(crypto.SERVICE_NAME, crypto.SQLCIPHER_KEY_ACCOUNT)


def test_legacy_passphrase_key_is_preserved_byte_for_byte():
    """The regression guard that matters: these databases were encrypted
    under SQLCipher's PBKDF2 passphrase path, so returning anything other
    than the original 88-hex derivation makes them undecryptable."""
    previous = keyring.get_password(crypto.SERVICE_NAME, crypto.SQLCIPHER_KEY_ACCOUNT)
    keyring.set_password(crypto.SERVICE_NAME, crypto.SQLCIPHER_KEY_ACCOUNT, LEGACY_FERNET_KEY)
    try:
        key = crypto.get_or_create_sqlcipher_key_hex()
        assert key == LEGACY_FERNET_KEY.encode().hex()
        assert not crypto.is_sqlcipher_raw_key(key)
    finally:
        if previous is None:
            keyring.delete_password(crypto.SERVICE_NAME, crypto.SQLCIPHER_KEY_ACCOUNT)
        else:
            keyring.set_password(crypto.SERVICE_NAME, crypto.SQLCIPHER_KEY_ACCOUNT, previous)
