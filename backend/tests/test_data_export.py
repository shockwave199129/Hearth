"""User-owned data export (docs/roadmap-v1.md 0.4) — app/data_export.py.

The property under test throughout is *completeness you can trust*: an
export that claims to be all of someone's data has to either contain it or
say plainly which part is missing. A silently-partial export is the failure
mode worth spending tests on, so most of these exercise a store that
cannot be read rather than the happy path.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import pytest

from app import data_export
from app.db import sqlite_models
from app.memory import chat_history, long_term
from app.memory2.models import EmotionalMetadata, EpisodicMemory, SemanticMemory
from app.memory2.store import MemoryStore
from app.onboarding.profile_schema import UserProfile
from app.voice import store as voiceprint_store


def _fake_embed(text: str, dim: int = 32) -> list[float]:
    """Same stand-in as tests/test_phase2_memory.py — no embedding server
    exists in this environment, and export doesn't care about vectors."""
    vec = [0.0] * dim
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


@pytest.fixture
def profile() -> UserProfile:
    return UserProfile(
        user_id="user-1",
        name="Ada",
        companion_name="Hearth",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def transcript_db(tmp_path, monkeypatch) -> Path:
    db = tmp_path / "profile.db"
    sqlite_models.close_pooled_connections()
    monkeypatch.setattr(chat_history, "CHAT_HISTORY_DB_PATH", db)
    # The export reads voiceprint metadata from the same file; without this it
    # would reach the real user-data profile.db.
    monkeypatch.setattr(voiceprint_store, "VOICEPRINT_DB_PATH", db)
    yield db
    sqlite_models.close_pooled_connections()


@pytest.fixture
def flat_memories(monkeypatch):
    """A real long_term store on an ephemeral Chroma collection."""
    collection = chromadb.EphemeralClient().get_or_create_collection(
        f"long_term_{uuid.uuid4().hex}"
    )
    monkeypatch.setattr(long_term, "get_collection", lambda: collection)
    monkeypatch.setattr(long_term, "embed", _fake_embed)
    return collection


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    client = chromadb.EphemeralClient()
    return MemoryStore(
        index_path=tmp_path / "idx.sqlite3",
        embed_fn=_fake_embed,
        episodic_collection=client.get_or_create_collection(f"episodic_{uuid.uuid4().hex}"),
        semantic_collection=client.get_or_create_collection(f"semantic_{uuid.uuid4().hex}"),
    )


def _episodic(user_id: str, summary: str) -> EpisodicMemory:
    now = datetime.now(timezone.utc)
    return EpisodicMemory(
        id=str(uuid.uuid4()),
        user_id=user_id,
        timestamp=now,
        summary=summary,
        emotional_metadata=EmotionalMetadata(valence="negative", intensity=0.6, category="anxiety"),
        last_reinforced=now,
    )


def _read_json(folder: Path, name: str):
    return json.loads((folder / name).read_text(encoding="utf-8"))


# --- The readers the export depends on --------------------------------------


def test_transcript_export_is_oldest_first_and_decrypted(transcript_db, profile):
    chat_history.record_turn(profile.user_id, "s1", 1, "user", "first thing I said")
    chat_history.record_turn(profile.user_id, "s1", 1, "assistant", "first reply")
    chat_history.record_turn(profile.user_id, "s2", 2, "user", "later thing")

    turns = chat_history.export_all(profile.user_id)

    assert [t["content"] for t in turns] == ["first thing I said", "first reply", "later thing"]


def test_transcript_export_is_scoped_to_one_profile(transcript_db, profile):
    chat_history.record_turn(profile.user_id, "s1", 1, "user", "mine")
    chat_history.record_turn("someone-else", "s1", 1, "user", "not mine")

    contents = [t["content"] for t in chat_history.export_all(profile.user_id)]

    assert contents == ["mine"]


def test_transcript_export_keeps_an_undecryptable_row_as_a_placeholder(transcript_db, profile):
    """One corrupt row must not cost the user every other row, and must not
    vanish silently either."""
    chat_history.record_turn(profile.user_id, "s1", 1, "user", "readable")
    conn = sqlite_models.get_connection(transcript_db)
    try:
        conn.execute(
            """INSERT INTO chat_history (user_id, session_id, turn_id, role, content, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (profile.user_id, "s1", 2, "user", "not-a-fernet-token", "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()

    turns = chat_history.export_all(profile.user_id)

    assert len(turns) == 2
    assert turns[0]["content"] == "readable"
    assert "could not be decrypted" in turns[1]["content"]


def test_flat_memory_export_carries_full_text_not_the_listing_label(flat_memories, profile):
    """`list_memories` truncates to 40 characters on purpose; an export that
    inherited that would quietly corrupt what the user gets back."""
    long_text = "I always get anxious the night before a flight, " * 3
    long_term.create(long_text, "pattern", profile.user_id)

    exported = long_term.export_all(profile.user_id)

    assert len(exported) == 1
    assert exported[0]["text"] == long_text
    assert len(exported[0]["text"]) > 40


def test_flat_memory_export_is_scoped_to_one_profile(flat_memories, profile):
    long_term.create("mine", "pattern", profile.user_id)
    long_term.create("not mine", "pattern", "someone-else")

    assert [m["text"] for m in long_term.export_all(profile.user_id)] == ["mine"]


# --- The export itself ------------------------------------------------------


def test_export_writes_every_file_and_a_truthful_manifest(
    tmp_path, transcript_db, flat_memories, store, profile
):
    chat_history.record_turn(profile.user_id, "s1", 1, "user", "hello there")
    chat_history.record_turn(profile.user_id, "s1", 1, "assistant", "hello back")
    long_term.create("she has a sister called Mira", "person", profile.user_id)
    store.save_episodic(_episodic(profile.user_id, "a hard week at work"))
    store.save_semantic(
        SemanticMemory(
            id=str(uuid.uuid4()),
            user_id=profile.user_id,
            fact="dislikes open-plan offices",
            last_reinforced=datetime.now(timezone.utc),
        )
    )

    result = data_export.export_user_data(profile, store, destination=tmp_path)
    folder = Path(result["path"])

    assert {p.name for p in folder.iterdir()} == {
        "profile.json",
        "memories.json",
        "transcript.json",
        "transcript.txt",
        "voice.json",
        "manifest.json",
        "README.txt",
    }
    manifest = _read_json(folder, "manifest.json")
    assert manifest["counts"] == {
        "long_term_memories": 1,
        "episodic_memories": 1,
        "semantic_memories": 1,
        "transcript_messages": 2,
    }
    # Empty, not absent: a consumer distinguishes "nothing missing" from
    # "something could not be read" by this field.
    assert manifest["incomplete"] == {}
    assert manifest["encrypted"] is False
    assert result["incomplete"] == {}


def test_export_contains_both_memory_stores(tmp_path, transcript_db, flat_memories, store, profile):
    """`memories.json` claims to be what Hearth remembers. Hearth has two
    memory stores, so exporting one of them would be a false claim."""
    long_term.create("flat memory", "person", profile.user_id)
    store.save_episodic(_episodic(profile.user_id, "an episodic memory"))

    result = data_export.export_user_data(profile, store, destination=tmp_path)
    memories = _read_json(Path(result["path"]), "memories.json")

    assert [m["text"] for m in memories["long_term"]] == ["flat memory"]
    assert [m["summary"] for m in memories["episodic"]] == ["an episodic memory"]
    assert memories["semantic"] == []


def test_export_profile_round_trips_as_a_profile(tmp_path, transcript_db, flat_memories, store, profile):
    result = data_export.export_user_data(profile, store, destination=tmp_path)

    exported = _read_json(Path(result["path"]), "profile.json")

    assert UserProfile.model_validate(exported) == profile


def test_readable_transcript_uses_the_names_the_user_chose(
    tmp_path, transcript_db, flat_memories, store, profile
):
    chat_history.record_turn(profile.user_id, "s1", 1, "user", "I'm tired")
    chat_history.record_turn(profile.user_id, "s1", 1, "assistant", "that sounds heavy")

    result = data_export.export_user_data(profile, store, destination=tmp_path)
    text = (Path(result["path"]) / "transcript.txt").read_text(encoding="utf-8")

    assert "Ada: I'm tired" in text
    assert "Hearth: that sounds heavy" in text
    assert text.index("I'm tired") < text.index("that sounds heavy")


def test_export_warns_in_the_folder_that_the_files_are_not_encrypted(
    tmp_path, transcript_db, flat_memories, store, profile
):
    """The UI warning is easy to forget by the time the folder is opened, or
    copied somewhere else — so the warning travels with the files."""
    result = data_export.export_user_data(profile, store, destination=tmp_path)

    readme = (Path(result["path"]) / "README.txt").read_text(encoding="utf-8")

    assert "NOT encrypted" in readme


def test_a_second_export_in_the_same_second_does_not_overwrite_the_first(
    tmp_path, transcript_db, flat_memories, store, profile
):
    first = data_export.export_user_data(profile, store, destination=tmp_path)
    second = data_export.export_user_data(profile, store, destination=tmp_path)

    assert first["path"] != second["path"]
    assert Path(first["path"]).is_dir() and Path(second["path"]).is_dir()


# --- Partial failure --------------------------------------------------------


class BrokenStore:
    def list_episodic(self, *args, **kwargs):
        raise RuntimeError("index is corrupt")

    def list_semantic(self, *args, **kwargs):
        raise RuntimeError("index is corrupt")


def test_unreadable_memory_store_is_reported_and_the_rest_still_exports(
    tmp_path, transcript_db, flat_memories, profile
):
    chat_history.record_turn(profile.user_id, "s1", 1, "user", "still mine")
    long_term.create("still exported", "person", profile.user_id)

    result = data_export.export_user_data(profile, BrokenStore(), destination=tmp_path)
    folder = Path(result["path"])
    manifest = _read_json(folder, "manifest.json")

    assert "index is corrupt" in manifest["incomplete"]["memory2_error"]
    assert "index is corrupt" in result["incomplete"]["memory2_error"]
    # The point of isolating the failure: the readable data still arrives.
    assert manifest["counts"]["transcript_messages"] == 1
    assert manifest["counts"]["long_term_memories"] == 1
    assert _read_json(folder, "transcript.json")[0]["content"] == "still mine"


def test_missing_memory2_store_is_reported_rather_than_shown_as_empty(
    tmp_path, transcript_db, flat_memories, profile
):
    """A half-set-up install has no tiered store. Reporting zero episodic
    memories there would tell the user Hearth remembers nothing, which is a
    different claim from "this could not be read"."""
    result = data_export.export_user_data(profile, None, destination=tmp_path)

    manifest = _read_json(Path(result["path"]), "manifest.json")

    assert "memory2_error" in manifest["incomplete"]


def test_unreadable_flat_memory_store_is_reported(tmp_path, transcript_db, store, profile, monkeypatch):
    def boom():
        raise RuntimeError("chroma is unavailable")

    monkeypatch.setattr(long_term, "get_collection", boom)

    result = data_export.export_user_data(profile, store, destination=tmp_path)

    manifest = _read_json(Path(result["path"]), "manifest.json")
    assert "chroma is unavailable" in manifest["incomplete"]["long_term_error"]


def test_unreadable_transcript_is_reported(
    # transcript_db is still needed even though chat_history is stubbed below:
    # the export also reads voiceprint metadata from profile.db.
    tmp_path, transcript_db, flat_memories, store, profile, monkeypatch
):
    monkeypatch.setattr(
        chat_history, "export_all", lambda user_id: (_ for _ in ()).throw(RuntimeError("db is locked"))
    )

    result = data_export.export_user_data(profile, store, destination=tmp_path)

    manifest = _read_json(Path(result["path"]), "manifest.json")
    assert "db is locked" in manifest["incomplete"]["transcript_error"]


# --- Destination ------------------------------------------------------------


def test_exports_live_outside_the_directory_a_reset_wipes():
    """`data_reset` and the uninstallers delete USER_DATA_DIR. An export
    stored under it would be destroyed by the very operation people take an
    export before doing."""
    from app import config

    exports = data_export.exports_root().resolve()

    assert config.USER_DATA_DIR.resolve() not in exports.parents
    assert exports != config.USER_DATA_DIR.resolve()
