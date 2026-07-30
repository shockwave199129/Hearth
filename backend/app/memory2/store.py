"""Storage for Book Volume 4's memory system (Ch 14): Chroma holds
embeddings + similarity search for episodic/semantic entries; SQLite holds
fast, non-vector lookups (status, priority, decay-relevant timestamps) and
the consolidation log. Each store is used only for what it's genuinely best
at, mirroring Volume 1's general storage philosophy.

Human-readable content (summary/fact text) is encrypted before it touches
disk, same pattern as `app.memory.long_term`. Embeddings and the Chroma
client/collections are injectable so this module never requires a live
embedding server in tests."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Callable

from app.config import DATA_DIR, VECTOR_STORE_DIR
from app.memory2.models import EmotionalMetadata, EpisodicMemory, MemoryStatus, SemanticMemory
from app.security.crypto import decrypt, encrypt

MEMORY2_INDEX_PATH = DATA_DIR / "memory2_index.sqlite3"

EmbedFn = Callable[[str], list[float]]

_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_index (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,               -- episodic | semantic
    status TEXT NOT NULL,
    priority REAL NOT NULL DEFAULT 1.0,
    confidence REAL,                  -- semantic only
    reference_count INTEGER NOT NULL DEFAULT 0,
    last_reinforced TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS emotional_metadata (
    memory_id TEXT PRIMARY KEY,
    valence TEXT NOT NULL,
    intensity REAL NOT NULL,
    category TEXT NOT NULL,
    sensitivity_flag INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS consolidation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    merged_into TEXT NOT NULL,
    merged_from TEXT NOT NULL,        -- JSON list of ids
    occurred_at TEXT NOT NULL
);
"""


def _default_embed(text: str) -> list[float]:
    from app.memory.embedder import embed

    return embed(text)


def _make_chroma_collection(name: str):
    import chromadb

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    return client.get_or_create_collection(name)


class MemoryStore:
    def __init__(
        self,
        *,
        index_path: Path = MEMORY2_INDEX_PATH,
        embed_fn: EmbedFn | None = None,
        episodic_collection=None,
        semantic_collection=None,
    ):
        self.index_path = index_path
        self._embed = embed_fn or _default_embed
        self._episodic_collection = episodic_collection
        self._semantic_collection = semantic_collection

    # -- lazy collection access -------------------------------------------------
    def _episodic(self):
        if self._episodic_collection is None:
            self._episodic_collection = _make_chroma_collection("memory2_episodic")
        return self._episodic_collection

    def _semantic(self):
        if self._semantic_collection is None:
            self._semantic_collection = _make_chroma_collection("memory2_semantic")
        return self._semantic_collection

    def _connect(self) -> sqlite3.Connection:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.index_path)
        conn.executescript(_INDEX_SCHEMA)
        return conn

    # -- episodic ---------------------------------------------------------------
    def save_episodic(self, mem: EpisodicMemory, *, priority: float = 1.0) -> None:
        payload = encrypt(mem.model_dump_json()).decode("latin1")
        self._episodic().upsert(
            ids=[mem.id],
            documents=[payload],
            embeddings=[self._embed(mem.summary)],
            metadatas=[{"user_id": mem.user_id, "kind": "episodic"}],
        )
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO memory_index (id, user_id, kind, status, priority, confidence, reference_count, last_reinforced, created_at)
                   VALUES (?, ?, 'episodic', ?, ?, NULL, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET status=excluded.status, priority=excluded.priority,
                       reference_count=excluded.reference_count, last_reinforced=excluded.last_reinforced""",
                (
                    mem.id,
                    mem.user_id,
                    mem.status.value,
                    priority,
                    mem.reference_count,
                    mem.last_reinforced.isoformat(),
                    mem.timestamp.isoformat(),
                ),
            )
            self._upsert_emotional_metadata(conn, mem.id, mem.emotional_metadata)
            conn.commit()
        finally:
            conn.close()

    def _upsert_emotional_metadata(self, conn: sqlite3.Connection, mem_id: str, meta: EmotionalMetadata) -> None:
        conn.execute(
            """INSERT INTO emotional_metadata (memory_id, valence, intensity, category, sensitivity_flag)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(memory_id) DO UPDATE SET valence=excluded.valence, intensity=excluded.intensity,
                   category=excluded.category, sensitivity_flag=excluded.sensitivity_flag""",
            (mem_id, meta.valence, meta.intensity, meta.category, int(meta.sensitivity_flag)),
        )

    def get_episodic(self, mem_id: str, user_id: str) -> EpisodicMemory | None:
        result = self._episodic().get(ids=[mem_id])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        if not ids or not docs or not metas or metas[0].get("user_id") != user_id:
            return None
        return EpisodicMemory.model_validate_json(decrypt(docs[0].encode("latin1")))

    def list_episodic(self, user_id: str, status: MemoryStatus | None = None) -> list[EpisodicMemory]:
        result = self._episodic().get(where={"user_id": user_id})
        docs = result.get("documents") or []
        out = [EpisodicMemory.model_validate_json(decrypt(d.encode("latin1"))) for d in docs if d]
        if status is not None:
            out = [m for m in out if m.status == status]
        return out

    def update_episodic(self, mem: EpisodicMemory, *, re_embed: bool = False, priority: float | None = None) -> None:
        # Chroma's update() silently re-embeds `documents` with its own
        # default embedding function (dimension mismatch against ours) if
        # `embeddings` isn't explicitly supplied — always pass ours,
        # regardless of `re_embed` (which only signals "the summary text
        # itself changed", for callers who want to reason about that).
        payload = encrypt(mem.model_dump_json()).decode("latin1")
        self._episodic().update(
            ids=[mem.id],
            documents=[payload],
            embeddings=[self._embed(mem.summary)],
            metadatas=[{"user_id": mem.user_id, "kind": "episodic"}],
        )
        conn = self._connect()
        try:
            if priority is not None:
                conn.execute(
                    "UPDATE memory_index SET status=?, reference_count=?, last_reinforced=?, priority=? WHERE id=?",
                    (mem.status.value, mem.reference_count, mem.last_reinforced.isoformat(), priority, mem.id),
                )
            else:
                conn.execute(
                    "UPDATE memory_index SET status=?, reference_count=?, last_reinforced=? WHERE id=?",
                    (mem.status.value, mem.reference_count, mem.last_reinforced.isoformat(), mem.id),
                )
            self._upsert_emotional_metadata(conn, mem.id, mem.emotional_metadata)
            conn.commit()
        finally:
            conn.close()

    def delete_episodic(self, mem_id: str, user_id: str) -> None:
        if self.get_episodic(mem_id, user_id) is None:
            return
        self._episodic().delete(ids=[mem_id])
        conn = self._connect()
        try:
            conn.execute("DELETE FROM memory_index WHERE id = ?", (mem_id,))
            conn.execute("DELETE FROM emotional_metadata WHERE memory_id = ?", (mem_id,))
            conn.commit()
        finally:
            conn.close()

    def search_episodic(self, query: str, user_id: str, k: int = 8) -> list[tuple[EpisodicMemory, float]]:
        result = self._episodic().query(query_embeddings=[self._embed(query)], n_results=k, where={"user_id": user_id})
        docs = (result.get("documents") or [[]])[0] or []
        dists = (result.get("distances") or [[]])[0] or []
        out = []
        for doc, dist in zip(docs, dists):
            if not doc:
                continue
            out.append((EpisodicMemory.model_validate_json(decrypt(doc.encode("latin1"))), float(dist)))
        return out

    def neighbors_episodic(self, mem: EpisodicMemory, k: int = 6) -> list[tuple[EpisodicMemory, float]]:
        """Same-user candidates near `mem` in embedding space, excluding
        itself — the raw material for promotion (Ch 9) and consolidation
        (Ch 12) clustering."""
        return [
            (candidate, dist)
            for candidate, dist in self.search_episodic(mem.summary, mem.user_id, k=k + 1)
            if candidate.id != mem.id
        ]

    def priority(self, mem_id: str) -> float:
        conn = self._connect()
        try:
            row = conn.execute("SELECT priority FROM memory_index WHERE id = ?", (mem_id,)).fetchone()
        finally:
            conn.close()
        return float(row[0]) if row else 1.0

    # -- semantic -----------------------------------------------------------
    def save_semantic(self, mem: SemanticMemory) -> None:
        payload = encrypt(mem.model_dump_json()).decode("latin1")
        self._semantic().upsert(
            ids=[mem.id],
            documents=[payload],
            embeddings=[self._embed(mem.fact)],
            metadatas=[{"user_id": mem.user_id, "kind": "semantic"}],
        )
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO memory_index (id, user_id, kind, status, priority, confidence, reference_count, last_reinforced, created_at)
                   VALUES (?, ?, 'semantic', ?, 1.0, ?, 0, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET status=excluded.status, confidence=excluded.confidence,
                       last_reinforced=excluded.last_reinforced""",
                (mem.id, mem.user_id, mem.status.value, mem.confidence, mem.last_reinforced.isoformat(), mem.last_reinforced.isoformat()),
            )
            self._upsert_emotional_metadata(conn, mem.id, mem.emotional_metadata)
            conn.commit()
        finally:
            conn.close()

    def get_semantic(self, mem_id: str, user_id: str) -> SemanticMemory | None:
        result = self._semantic().get(ids=[mem_id])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        if not ids or not docs or not metas or metas[0].get("user_id") != user_id:
            return None
        return SemanticMemory.model_validate_json(decrypt(docs[0].encode("latin1")))

    def list_semantic(self, user_id: str, status: MemoryStatus | None = None) -> list[SemanticMemory]:
        result = self._semantic().get(where={"user_id": user_id})
        docs = result.get("documents") or []
        out = [SemanticMemory.model_validate_json(decrypt(d.encode("latin1"))) for d in docs if d]
        if status is not None:
            out = [m for m in out if m.status == status]
        return out

    def update_semantic(self, mem: SemanticMemory) -> None:
        # See update_episodic's comment — embeddings must always be passed
        # explicitly or Chroma silently substitutes its own default embedder.
        payload = encrypt(mem.model_dump_json()).decode("latin1")
        self._semantic().update(
            ids=[mem.id],
            documents=[payload],
            embeddings=[self._embed(mem.fact)],
            metadatas=[{"user_id": mem.user_id, "kind": "semantic"}],
        )
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE memory_index SET status=?, confidence=?, last_reinforced=? WHERE id=?",
                (mem.status.value, mem.confidence, mem.last_reinforced.isoformat(), mem.id),
            )
            self._upsert_emotional_metadata(conn, mem.id, mem.emotional_metadata)
            conn.commit()
        finally:
            conn.close()

    def delete_semantic(self, mem_id: str, user_id: str) -> None:
        if self.get_semantic(mem_id, user_id) is None:
            return
        self._semantic().delete(ids=[mem_id])
        conn = self._connect()
        try:
            conn.execute("DELETE FROM memory_index WHERE id = ?", (mem_id,))
            conn.execute("DELETE FROM emotional_metadata WHERE memory_id = ?", (mem_id,))
            conn.commit()
        finally:
            conn.close()

    def search_semantic(self, query: str, user_id: str, k: int = 8) -> list[tuple[SemanticMemory, float]]:
        result = self._semantic().query(query_embeddings=[self._embed(query)], n_results=k, where={"user_id": user_id})
        docs = (result.get("documents") or [[]])[0] or []
        dists = (result.get("distances") or [[]])[0] or []
        out = []
        for doc, dist in zip(docs, dists):
            if not doc:
                continue
            out.append((SemanticMemory.model_validate_json(decrypt(doc.encode("latin1"))), float(dist)))
        return out

    def semantic_sharing_entities(self, user_id: str, entities: list[str]) -> list[SemanticMemory]:
        """Semantic facts whose `fact` text mentions any of `entities` —
        used by contradiction detection (Ch 13), which only needs to check
        facts that plausibly overlap, not the whole store."""
        if not entities:
            return []
        lowered = [e.lower() for e in entities]
        return [m for m in self.list_semantic(user_id) if any(e in m.fact.lower() for e in lowered)]

    # -- consolidation log ----------------------------------------------------
    def log_consolidation(self, merged_into: str, merged_from: list[str], occurred_at: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO consolidation_log (merged_into, merged_from, occurred_at) VALUES (?, ?, ?)",
                (merged_into, json.dumps(merged_from), occurred_at),
            )
            conn.commit()
        finally:
            conn.close()

    # -- cascade helper ---------------------------------------------------------
    def delete_all_for_user(self, user_id: str) -> None:
        for mem in self.list_episodic(user_id):
            self.delete_episodic(mem.id, user_id)
        for mem in self.list_semantic(user_id):
            self.delete_semantic(mem.id, user_id)
