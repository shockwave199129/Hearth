"""SQLCipher connection helper + schema for every table in profile.db.
Multi-profile: each local install can hold several named profiles
(`profiles`, keyed by `user_id`), with exactly one marked active at a time
(`active_profile`) — this is still a single-process desktop app (one
conversation at a time), not concurrent multi-tenant serving; switching
profiles is a deliberate user action. See docs/project-plan.md §1/§4."""
import threading
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
    voice_style TEXT NOT NULL DEFAULT 'grounded',
    companion_name TEXT NOT NULL,
    communication_formality TEXT NOT NULL DEFAULT 'casual',
    response_length TEXT NOT NULL DEFAULT 'balanced',
    emoji_usage TEXT NOT NULL DEFAULT 'minimal',
    relationship_general_trust REAL NOT NULL DEFAULT 0.0,
    relationship_vulnerability_trust REAL NOT NULL DEFAULT 0.0,
    relationship_advice_trust REAL NOT NULL DEFAULT 0.0,
    relationship_consistency_confidence REAL NOT NULL DEFAULT 0.0,
    relationship_boundaries TEXT NOT NULL DEFAULT 'normal',
    relationship_life_model TEXT NOT NULL DEFAULT 'unknown',
    communication_traits_json TEXT NOT NULL DEFAULT '{}',
    skill_affinity_json TEXT NOT NULL DEFAULT '{}',
    evaluation_last_run_at TEXT,
    emergency_contact_consent INTEGER NOT NULL DEFAULT 0,
    emergency_contact_name TEXT,
    emergency_contact_method TEXT,
    emergency_contact_value TEXT,
    region TEXT,
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

# Install-wide first-run setup gate (packages + models). Singleton row —
# once complete=1, subsequent launches skip the Setup UI. Cleared if the
# required model files disappear so the user can re-run setup.
SETUP_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS setup_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    complete INTEGER NOT NULL DEFAULT 0,
    completed_at TEXT
);
"""

# One row per profile tracking when the companion last checked in on them
# (docs/project-plan.md §8).
CHECKIN_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkin (
    user_id TEXT PRIMARY KEY,
    last_checkin_at TEXT
);
"""

# Book Volume 3's full RelationshipProfile (app/relationship/state.py) — the
# versioned, consolidated object (Trust, Attachment, Development, Boundaries,
# Life Model, Shared History), distinct from the flat cached trust columns
# on `profiles` above. One row per profile; only the Growth Engine writes.
RELATIONSHIP_PROFILES_SCHEMA = """
CREATE TABLE IF NOT EXISTS relationship_profiles (
    user_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# Crisis/escalation history (docs/project-plan.md §9) — append-only, unlike the
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

# Persisted, encrypted conversation history (docs/project-plan.md §1's
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
    communication_formality TEXT NOT NULL DEFAULT 'casual',
    response_length TEXT NOT NULL DEFAULT 'balanced',
    relationship_general_trust REAL NOT NULL DEFAULT 0.0,
    relationship_vulnerability_trust REAL NOT NULL DEFAULT 0.0,
    relationship_advice_trust REAL NOT NULL DEFAULT 0.0,
    relationship_consistency_confidence REAL NOT NULL DEFAULT 0.0,
    relationship_boundaries TEXT NOT NULL DEFAULT 'normal',
    relationship_life_model TEXT NOT NULL DEFAULT 'unknown',
    communication_traits_json TEXT NOT NULL DEFAULT '{}',
    skill_affinity_json TEXT NOT NULL DEFAULT '{}',
    evaluation_last_run_at TEXT,
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
    "voice_style": "TEXT NOT NULL DEFAULT 'grounded'",
    "communication_formality": "TEXT NOT NULL DEFAULT 'casual'",
    "response_length": "TEXT NOT NULL DEFAULT 'balanced'",
    "emoji_usage": "TEXT NOT NULL DEFAULT 'minimal'",
    "relationship_general_trust": "REAL NOT NULL DEFAULT 0.0",
    "relationship_vulnerability_trust": "REAL NOT NULL DEFAULT 0.0",
    "relationship_advice_trust": "REAL NOT NULL DEFAULT 0.0",
    "relationship_consistency_confidence": "REAL NOT NULL DEFAULT 0.0",
    "relationship_boundaries": "TEXT NOT NULL DEFAULT 'normal'",
    "relationship_life_model": "TEXT NOT NULL DEFAULT 'unknown'",
    "communication_traits_json": "TEXT NOT NULL DEFAULT '{}'",
    "skill_affinity_json": "TEXT NOT NULL DEFAULT '{}'",
    "evaluation_last_run_at": "TEXT",
    "region": "TEXT",
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
                  communication_formality, response_length, relationship_general_trust, relationship_vulnerability_trust,
                  relationship_advice_trust, relationship_consistency_confidence, relationship_boundaries, relationship_life_model,
                  communication_traits_json, skill_affinity_json, evaluation_last_run_at,
                  emergency_contact_consent, emergency_contact_name, emergency_contact_method,
                  emergency_contact_value, created_at
           FROM profile WHERE id = 1"""
    ).fetchone()
    if old is None:
        return
    user_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO profiles (user_id, name, age_range, gender, profession, stressors,
               preferred_voice, companion_name, communication_formality, response_length,
               relationship_general_trust, relationship_vulnerability_trust, relationship_advice_trust,
               relationship_consistency_confidence, relationship_boundaries, relationship_life_model,
               communication_traits_json, skill_affinity_json, evaluation_last_run_at,
               speak_replies, emergency_contact_consent,
               emergency_contact_name, emergency_contact_method, emergency_contact_value, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)""",
        (user_id, *old),
    )
    conn.execute(
        """INSERT INTO active_profile (id, user_id) VALUES (1, ?)
           ON CONFLICT(id) DO UPDATE SET user_id=excluded.user_id""",
        (user_id,),
    )
    conn.commit()


def init_schema(conn) -> None:
    """Create/migrate every table. Idempotent, but ~15 statements plus a
    `PRAGMA table_info` probe and a `sqlite_master` scan, so it runs once
    per database file per process (see `get_connection`) rather than on
    every open."""
    conn.execute(_LEGACY_PROFILE_SCHEMA)
    conn.execute(PROFILES_SCHEMA)
    conn.execute(ACTIVE_PROFILE_SCHEMA)
    conn.execute(SETUP_STATE_SCHEMA)
    conn.execute(CHECKIN_SCHEMA)
    conn.execute(CRISIS_EVENTS_SCHEMA)
    conn.execute(ESCALATIONS_SCHEMA)
    conn.execute(CHAT_HISTORY_SCHEMA)
    conn.execute(RELATIONSHIP_PROFILES_SCHEMA)
    _ensure_columns(conn, "profiles", _PROFILES_TEXT_INPUT_COLUMNS)
    conn.commit()
    _migrate_legacy_singleton_profile(conn)


class PooledConnection:
    """Borrowed handle onto a cached sqlcipher connection.

    `close()` returns it to the pool instead of closing it — the same
    contract as a pooled connection in SQLAlchemy/psycopg, and the reason
    the store modules' existing open→query→`close()` shape needs no
    changes. The rollback on release matches what a real `close()` would
    have done with uncommitted work, so a store function that forgets to
    commit still discards its writes rather than leaving a write
    transaction open across the pool.
    """

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self) -> None:
        try:
            self._conn.rollback()
        except Exception:
            # A connection too broken to roll back is not reusable; drop it
            # so the next borrow opens a fresh one instead of handing back
            # a dead handle forever.
            _discard(self._conn)


# One connection per (thread, database file). Threads are the unit because
# sqlite3 objects are not safe to share across them by default, and FastAPI
# runs sync route handlers on a worker pool — a process-wide singleton would
# need `check_same_thread=False` plus external serialization.
_local = threading.local()
_schema_lock = threading.Lock()
_initialized: set[str] = set()


def _discard(conn) -> None:
    pool = getattr(_local, "connections", None)
    if pool:
        for key, cached in list(pool.items()):
            if cached is conn:
                del pool[key]
    try:
        conn.close()
    except Exception:
        pass


def get_connection(db_path: Path) -> PooledConnection:
    """Returns a sqlcipher connection (dbapi2-compatible with stdlib
    sqlite3), keyed from the OS-keychain-backed secret in security/crypto.py.
    Deferred import: sqlcipher3 pulls in a compiled libsqlcipher, only
    needed once encryption is actually exercised.

    Connections are pooled per thread rather than opened per call. Keying a
    connection makes SQLCipher run PBKDF2 — deliberately expensive — so the
    old open-per-call shape charged a key derivation plus the full schema
    DDL to every single store function; a settings write alone opens four
    connections. Callers still `close()`, which now releases back to the
    pool (see `PooledConnection`).
    """
    from sqlcipher3 import dbapi2 as sqlcipher

    from app.security.crypto import get_or_create_sqlcipher_key_hex

    key = str(db_path)
    pool = getattr(_local, "connections", None)
    if pool is None:
        pool = _local.connections = {}
    cached = pool.get(key)
    if cached is not None:
        return PooledConnection(cached)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlcipher.connect(str(db_path))
    conn.execute(f"PRAGMA key = \"x'{get_or_create_sqlcipher_key_hex()}'\"")
    conn.execute("PRAGMA journal_mode = WAL")
    # Guarded across threads, not just within one: two threads opening the
    # same file for the first time would otherwise run the migration
    # concurrently.
    with _schema_lock:
        if key not in _initialized:
            init_schema(conn)
            _initialized.add(key)
    pool[key] = conn
    return PooledConnection(conn)


def close_pooled_connections() -> None:
    """Drops this thread's pooled connections and forgets which files have
    been initialized. Tests that point a store at a fresh temp database, or
    delete one mid-run, need this; nothing in the app calls it."""
    pool = getattr(_local, "connections", None) or {}
    for conn in pool.values():
        try:
            conn.close()
        except Exception:
            pass
    pool.clear()
    with _schema_lock:
        _initialized.clear()
