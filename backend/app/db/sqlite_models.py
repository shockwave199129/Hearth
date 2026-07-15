"""SQLCipher connection helper + schema for every table in profile.db.
Multi-profile: each local install can hold several named profiles
(`profiles`, keyed by `user_id`), with exactly one marked active at a time
(`active_profile`) — this is still a single-process desktop app (one
conversation at a time), not concurrent multi-tenant serving; switching
profiles is a deliberate user action. See project-plan.md §1/§4."""
import uuid
from pathlib import Path

PROFILES_SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    user_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    age_range TEXT,
    gender TEXT,
    profession TEXT,
    stressors TEXT NOT NULL,        -- JSON-encoded list[str]
    preferred_voice TEXT NOT NULL,
    companion_name TEXT NOT NULL,
    emergency_contact_consent INTEGER NOT NULL DEFAULT 0,
    emergency_contact_name TEXT,
    emergency_contact_method TEXT,
    emergency_contact_value TEXT,
    created_at TEXT NOT NULL
);
"""

# Tracks which profile this app instance is currently using.
ACTIVE_PROFILE_SCHEMA = """
CREATE TABLE IF NOT EXISTS active_profile (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    user_id TEXT NOT NULL
);
"""

# One row per profile tracking when the companion last checked in on them
# (project-plan.md §8).
CHECKIN_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkin (
    user_id TEXT PRIMARY KEY,
    last_checkin_at TEXT
);
"""

# Crisis/escalation history (project-plan.md §9) — append-only, unlike the
# single-row tables above, since the pattern-detection logic in
# safety/escalation.py needs to look back over multiple events per profile.
CRISIS_EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS crisis_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    severity TEXT NOT NULL,
    matched_pattern TEXT NOT NULL
);
"""

ESCALATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS escalations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    method TEXT,
    result_json TEXT NOT NULL
);
"""

# Persisted, encrypted conversation history (project-plan.md §1's
# chat_history.db) — content is Fernet-encrypted before insert, same
# pattern as long_term.py's Chroma documents. Backs the "replay a past
# reply" feature (memory/chat_history.py).
CHAT_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

# --- Legacy schema, pre-multi-profile (Phases 1-5) — kept only so
# _migrate_legacy_singleton_profile can read out of it once. Never written
# to again; not dropped, so the migration is reversible / inspectable.
_LEGACY_PROFILE_SCHEMA = """
CREATE TABLE IF NOT EXISTS profile (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    name TEXT NOT NULL,
    age_range TEXT,
    gender TEXT,
    profession TEXT,
    stressors TEXT NOT NULL,
    preferred_voice TEXT NOT NULL,
    companion_name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_LEGACY_PROFILE_SAFETY_COLUMNS = {
    "emergency_contact_consent": "INTEGER NOT NULL DEFAULT 0",
    "emergency_contact_name": "TEXT",
    "emergency_contact_method": "TEXT",
    "emergency_contact_value": "TEXT",
}


def _ensure_columns(conn, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, decl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


# speak_replies added after `profiles` already existed for earlier-onboarded
# local installs — applied via _ensure_columns, same reasoning as the
# emergency-contact columns above.
_PROFILES_TEXT_INPUT_COLUMNS = {
    "speak_replies": "INTEGER NOT NULL DEFAULT 1",
}


def _migrate_legacy_singleton_profile(conn) -> None:
    """One-time, idempotent: if an install still has the old single-row
    `profile` table (Phases 1-5, before multi-profile support) and no
    profile has been migrated into `profiles` yet, copy it over under a
    freshly generated user_id and mark it active. Leaves the legacy table
    in place untouched — this only ever reads from it."""
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "profile" not in tables:
        return
    already_migrated = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    if already_migrated:
        return
    _ensure_columns(conn, "profile", _LEGACY_PROFILE_SAFETY_COLUMNS)
    old = conn.execute(
        """SELECT name, age_range, gender, profession, stressors, preferred_voice, companion_name,
                  emergency_contact_consent, emergency_contact_name, emergency_contact_method,
                  emergency_contact_value, created_at
           FROM profile WHERE id = 1"""
    ).fetchone()
    if old is None:
        return
    user_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO profiles (user_id, name, age_range, gender, profession, stressors,
               preferred_voice, companion_name, speak_replies, emergency_contact_consent,
               emergency_contact_name, emergency_contact_method, emergency_contact_value, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)""",
        (user_id, *old),
    )
    conn.execute(
        """INSERT INTO active_profile (id, user_id) VALUES (1, ?)
           ON CONFLICT(id) DO UPDATE SET user_id=excluded.user_id""",
        (user_id,),
    )
    conn.commit()


def get_connection(db_path: Path):
    """Returns a sqlcipher3 connection (dbapi2-compatible with stdlib
    sqlite3), keyed from the OS-keychain-backed secret in security/crypto.py.
    Deferred import: sqlcipher3 pulls in a compiled libsqlcipher, only
    needed once encryption is actually exercised."""
    from sqlcipher3 import dbapi2 as sqlcipher

    from app.security.crypto import get_or_create_sqlcipher_key_hex

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlcipher.connect(str(db_path))
    conn.execute(f"PRAGMA key = \"x'{get_or_create_sqlcipher_key_hex()}'\"")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(_LEGACY_PROFILE_SCHEMA)
    conn.execute(PROFILES_SCHEMA)
    conn.execute(ACTIVE_PROFILE_SCHEMA)
    conn.execute(CHECKIN_SCHEMA)
    conn.execute(CRISIS_EVENTS_SCHEMA)
    conn.execute(ESCALATIONS_SCHEMA)
    conn.execute(CHAT_HISTORY_SCHEMA)
    _ensure_columns(conn, "profiles", _PROFILES_TEXT_INPUT_COLUMNS)
    conn.commit()
    _migrate_legacy_singleton_profile(conn)
    return conn
