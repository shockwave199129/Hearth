"""Long-term, cross-session memory — tool-based, not auto-injected into the
system prompt (see project-plan.md §5). Multi-profile: every record carries
a user_id in its Chroma metadata, and every read/write is scoped to one —
get/update/delete additionally verify the stored user_id matches before
acting, so a leaked/guessed memory id can't reach across profiles. Only the
encrypted fact text ever touches disk — the embedding vector stays
unencrypted since Chroma needs it for similarity search, but the
human-readable text never sits on disk unencrypted."""
import uuid
from datetime import datetime, timezone

from app.db.chroma_client import get_collection
from app.memory.embedder import embed
from app.security.crypto import decrypt, encrypt


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create(text: str, category: str, user_id: str) -> str:
    mem_id = str(uuid.uuid4())
    get_collection().add(
        documents=[encrypt(text).decode("latin1")],
        embeddings=[embed(text)],
        metadatas=[{"user_id": user_id, "category": category, "updated_at": _now_iso()}],
        ids=[mem_id],
    )
    return mem_id


def _owned_by(mem_id: str, user_id: str) -> bool:
    results = get_collection().get(ids=[mem_id])
    ids = results.get("ids") or []
    metas = results.get("metadatas") or []
    if not ids or not metas:
        return False
    return metas[0].get("user_id") == user_id


def update(mem_id: str, new_text: str, user_id: str) -> None:
    if not _owned_by(mem_id, user_id):
        return
    get_collection().update(
        ids=[mem_id],
        documents=[encrypt(new_text).decode("latin1")],
        embeddings=[embed(new_text)],
        metadatas=[{"updated_at": _now_iso()}],
    )


def delete(mem_id: str, user_id: str) -> None:
    if not _owned_by(mem_id, user_id):
        return
    get_collection().delete(ids=[mem_id])


def list_memories(user_id: str, category: str | None = None) -> list[dict]:
    """id + category + a short label only — never the full text, so a
    listing call can't accidentally dump everything into context."""
    # chromadb 1.5+ requires a single top-level operator — a bare multi-key
    # dict raises "Expected where to have exactly one operator".
    if category:
        where = {"$and": [{"user_id": user_id}, {"category": category}]}
    else:
        where = {"user_id": user_id}
    results = get_collection().get(where=where)
    # Empty collections often return documents/metadatas as None (not []) —
    # zip(None, ...) would 500 the Settings → Memory panel.
    ids = results.get("ids") or []
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []
    out: list[dict] = []
    for mem_id, doc, meta in zip(ids, docs, metas):
        if doc is None or meta is None:
            continue
        try:
            label = decrypt(doc.encode("latin1"))[:40]
        except Exception:
            label = "(unreadable)"
        out.append({"id": mem_id, "category": meta.get("category", ""), "label": label})
    return out


def get(mem_id: str, user_id: str) -> dict | None:
    results = get_collection().get(ids=[mem_id])
    ids = results.get("ids") or []
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []
    if not ids or not docs or not metas or metas[0].get("user_id") != user_id:
        return None
    return {
        "id": ids[0],
        "category": metas[0]["category"],
        "text": decrypt(docs[0].encode("latin1")),
    }


def search(query: str, user_id: str, k: int = 5) -> list[dict]:
    results = get_collection().query(
        query_embeddings=[embed(query)], n_results=k, where={"user_id": user_id}
    )
    ids = (results.get("ids") or [[]])[0] or []
    docs = (results.get("documents") or [[]])[0] or []
    metas = (results.get("metadatas") or [[]])[0] or []
    return [
        {"id": i, "category": m["category"], "text": decrypt(d.encode("latin1"))}
        for i, d, m in zip(ids, docs, metas)
        if d is not None and m is not None
    ]


def delete_all_for_user(user_id: str) -> None:
    """Cascade helper for profile deletion — see main.py's
    DELETE /api/profiles/{user_id} handler."""
    results = get_collection().get(where={"user_id": user_id})
    ids = results.get("ids") or []
    if ids:
        get_collection().delete(ids=ids)
