"""Append-only observation store with DuckDB when available."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config import DATA_DIR

LEARNING_DB_PATH = DATA_DIR / "hearth_learning.duckdb"
LEARNING_SQLITE_FALLBACK_SUFFIX = ".sqlite3"

# observation_type is interpolated into a per-type table name (DuckDB has no
# parameter binding for identifiers) — restrict it to a safe identifier
# shape so it can never be used to inject arbitrary SQL.
_SAFE_OBSERVATION_TYPE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _table_name(observation_type: str) -> str:
    if not _SAFE_OBSERVATION_TYPE.match(observation_type):
        raise ValueError(f"unsafe observation_type: {observation_type!r}")
    return f"{observation_type}_observations"


# Book Vol 7 Ch3/Ch7, Invariant 4: "no Trust observation may derive from
# message frequency or session length" — enforced here, at the schema
# level, not merely documented. Every "relationship" observation tagged to
# one of these subject_ids must carry a `derivation` in its context drawn
# from this allowlist; anything else (including a bare "message_frequency"
# or "session_length" tag) is a hard append-time failure.
TRUST_SUBJECTS = {"general_trust", "vulnerability_trust", "advice_trust", "consistency_confidence"}
VALID_TRUST_DERIVATIONS = {
    "consistency_observation",
    "disclosure_depth",
    "repair_outcome",
    "return_behavior",
    "explicit_correction",
}


class InvalidTrustObservationError(ValueError):
    pass


@dataclass(frozen=True)
class Observation:
    id: str
    timestamp: str
    subject_id: str
    value: float
    context: dict
    source_module: str
    observation_type: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ObservationStore:
    def __init__(self, path: Path = LEARNING_DB_PATH):
        self.path = path
        self._duckdb = None
        try:
            import duckdb  # type: ignore

            self._duckdb = duckdb
        except Exception:
            self._duckdb = None

    def _fallback_path(self) -> Path:
        return self.path.with_suffix(LEARNING_SQLITE_FALLBACK_SUFFIX)

    def _connect(self):
        if self._duckdb is not None:
            return self._duckdb.connect(str(self.path))
        fallback_path = self._fallback_path()
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(fallback_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                observation_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                value REAL NOT NULL,
                context TEXT NOT NULL,
                source_module TEXT NOT NULL
            )
            """
        )
        conn.commit()
        return conn

    def append(self, observation_type: str, subject_id: str, value: float, context: dict | None, source_module: str) -> Observation:
        if observation_type == "relationship" and subject_id in TRUST_SUBJECTS:
            derivation = (context or {}).get("derivation")
            if derivation not in VALID_TRUST_DERIVATIONS:
                raise InvalidTrustObservationError(
                    f"Trust observation for {subject_id!r} must carry a context['derivation'] from "
                    f"{sorted(VALID_TRUST_DERIVATIONS)} (got {derivation!r}) — Book Vol 7 Ch3/Invariant 4: "
                    "no Trust observation may derive from message frequency or session length."
                )
        obs = Observation(
            id=str(uuid.uuid4()),
            timestamp=_now_iso(),
            subject_id=subject_id,
            value=float(value),
            context=context or {},
            source_module=source_module,
            observation_type=observation_type,
        )
        conn = self._connect()
        try:
            if self._duckdb is not None:
                table = _table_name(observation_type)
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id VARCHAR PRIMARY KEY,
                        timestamp TIMESTAMP,
                        subject_id VARCHAR,
                        value DOUBLE,
                        context JSON,
                        source_module VARCHAR
                    )
                    """
                )
                conn.execute(
                    f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?)",
                    [obs.id, obs.timestamp, obs.subject_id, obs.value, json.dumps(obs.context), obs.source_module],
                )
            else:
                conn.execute(
                    "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (obs.id, obs.timestamp, observation_type, obs.subject_id, obs.value, json.dumps(obs.context), obs.source_module),
                )
                conn.commit()
        finally:
            conn.close()
        return obs

    def latest(self, observation_type: str, subject_id: str, limit: int = 50) -> list[Observation]:
        conn = self._connect()
        try:
            if self._duckdb is not None:
                table = _table_name(observation_type)
                exists = conn.execute(
                    "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [table]
                ).fetchone()
                if exists is None:
                    rows = []
                else:
                    rows = conn.execute(
                        f"""
                        SELECT id, timestamp, subject_id, value, context, source_module
                        FROM {table}
                        WHERE subject_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        [subject_id, limit],
                    ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, timestamp, subject_id, value, context, source_module
                    FROM observations
                    WHERE observation_type = ? AND subject_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (observation_type, subject_id, limit),
                ).fetchall()
        finally:
            conn.close()
        out = []
        for row in rows:
            context = row[4]
            if isinstance(context, str):
                context = json.loads(context)
            out.append(Observation(id=row[0], timestamp=row[1], subject_id=row[2], value=float(row[3]), context=context, source_module=row[5], observation_type=observation_type))
        return out

