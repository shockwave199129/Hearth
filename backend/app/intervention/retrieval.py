"""Semantic skill candidate retrieval (Book Vol 1 Ch 8) — vector search over
skill-description embeddings in Chroma, replacing token-overlap. Retrieval
answers "which skills are relevant?"; ranking (app.intervention.ranking)
answers "which relevant skill is best right now?" — kept as two separate
stages, per the book's own Candidate Retrieval -> Dynamic Scoring pipeline.

Embeddings and the Chroma collection are injectable so this never requires
a live embedding server in tests (mirroring app.memory2.store's pattern)."""
from __future__ import annotations

import hashlib
from typing import Callable

from app.skills.loader import Skill, load_catalog

EmbedFn = Callable[[str], list[float]]


def _default_embed(text: str) -> list[float]:
    from app.memory.embedder import embed

    return embed(text)


def _make_collection():
    import chromadb

    from app.config import VECTOR_STORE_DIR

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    return client.get_or_create_collection("skills_catalog")


def _skill_text(skill: Skill) -> str:
    return " — ".join(filter(None, [skill.summary, " ".join(skill.tags), " ".join(skill.manifest.when_use)]))


class SkillRetriever:
    """Skills are files, not code (Vol 5 Ch 2, Design Goal 2) — the catalog
    is re-read from disk on every call via `load_catalog()`, and this class
    only re-embeds a skill whose description text actually changed since
    the last sync (cheap content-hash check), so editing a skill's
    manifest/content never requires a code change or a manual re-index step."""

    def __init__(self, *, embed_fn: EmbedFn | None = None, collection=None):
        self._embed = embed_fn or _default_embed
        self._collection = collection

    def _col(self):
        if self._collection is None:
            self._collection = _make_collection()
        return self._collection

    def _sync_catalog(self, skills: list[Skill]) -> None:
        col = self._col()
        existing = col.get()
        existing_ids = list(existing.get("ids") or [])
        existing_metas = dict(zip(existing_ids, existing.get("metadatas") or []))

        ids, docs, embeddings, metadatas = [], [], [], []
        current_ids = set()
        for skill in skills:
            current_ids.add(skill.id)
            text = _skill_text(skill)
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            if (existing_metas.get(skill.id) or {}).get("content_hash") == content_hash:
                continue
            ids.append(skill.id)
            docs.append(text)
            embeddings.append(self._embed(text))
            metadatas.append({"content_hash": content_hash})
        if ids:
            col.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)

        stale = [i for i in existing_ids if i not in current_ids]
        if stale:
            col.delete(ids=stale)

    def retrieve(self, transcript: str, limit: int = 8) -> list[Skill]:
        skills = load_catalog()
        if not skills:
            return []
        self._sync_catalog(skills)
        result = self._col().query(query_embeddings=[self._embed(transcript)], n_results=min(limit, len(skills)))
        ids = (result.get("ids") or [[]])[0] or []
        by_id = {s.id: s for s in skills}
        return [by_id[i] for i in ids if i in by_id]


_default_retriever: SkillRetriever | None = None


def retrieve_candidates(transcript: str, limit: int = 8, *, retriever: SkillRetriever | None = None) -> list[Skill]:
    global _default_retriever
    if retriever is not None:
        return retriever.retrieve(transcript, limit)
    if _default_retriever is None:
        _default_retriever = SkillRetriever()
    return _default_retriever.retrieve(transcript, limit)
