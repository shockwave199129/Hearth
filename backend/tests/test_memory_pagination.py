"""Pagination for ``long_term.list_memories`` — mirrors chat_history's
``items``/``has_more`` contract without needing a live Chroma instance."""

from __future__ import annotations

from app.memory import long_term
from app.security.crypto import encrypt


class FakeCollection:
    def __init__(self, rows: list[tuple[str, str, dict]]):
        # (id, plaintext, metadata)
        self.rows = rows

    def get(self, where=None, limit=None, offset=None, **kwargs):
        user_id = None
        category = None
        if where and "user_id" in where:
            user_id = where["user_id"]
        elif where and "$and" in where:
            for clause in where["$and"]:
                if "user_id" in clause:
                    user_id = clause["user_id"]
                if "category" in clause:
                    category = clause["category"]

        filtered = [
            (mid, text, meta)
            for mid, text, meta in self.rows
            if meta.get("user_id") == user_id and (category is None or meta.get("category") == category)
        ]
        start = offset or 0
        end = start + limit if limit is not None else None
        page = filtered[start:end]
        return {
            "ids": [r[0] for r in page],
            "documents": [encrypt(r[1]).decode("latin1") for r in page],
            "metadatas": [r[2] for r in page],
        }


def test_list_memories_paginates(monkeypatch):
    rows = [
        (f"id-{i}", f"memory text number {i}", {"user_id": "u1", "category": "fact"})
        for i in range(5)
    ]
    monkeypatch.setattr(long_term, "get_collection", lambda: FakeCollection(rows))

    page1 = long_term.list_memories("u1", limit=2, offset=0)
    assert len(page1["items"]) == 2
    assert page1["has_more"] is True
    assert page1["items"][0]["id"] == "id-0"
    assert page1["items"][0]["label"].startswith("memory text")

    page2 = long_term.list_memories("u1", limit=2, offset=2)
    assert [m["id"] for m in page2["items"]] == ["id-2", "id-3"]
    assert page2["has_more"] is True

    page3 = long_term.list_memories("u1", limit=2, offset=4)
    assert [m["id"] for m in page3["items"]] == ["id-4"]
    assert page3["has_more"] is False


def test_list_memories_clamps_limit(monkeypatch):
    monkeypatch.setattr(long_term, "get_collection", lambda: FakeCollection([]))
    page = long_term.list_memories("u1", limit=9999, offset=-5)
    assert page["limit"] == 200
    assert page["offset"] == 0
    assert page["items"] == []
    assert page["has_more"] is False
