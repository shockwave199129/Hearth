"""Append-only observation store with DuckDB when available."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config import DATA_DIR

LEARNING_DB_PATH = DATA_DIR / "hearth_learning.duckdb"
LEARNING_SQLITE_FALLBACK_PATH = DATA_DIR / "hearth_learning.sqlite3"


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

    def _connect(self):
        if self._duckdb is not None:
            return self._duckdb.connect(str(self.path))
        conn = sqlite3.connect(LEARNING_SQLITE_FALLBACK_PATH)
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
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {observation_type}_observations (
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
                    f"INSERT INTO {observation_type}_observations VALUES (?, ?, ?, ?, ?, ?)",
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
                rows = conn.execute(
                    f"""
                    SELECT id, timestamp, subject_id, value, context, source_module
                    FROM {observation_type}_observations
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

