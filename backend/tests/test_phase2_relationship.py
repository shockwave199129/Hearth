"""Phase 2 relationship track (Book Volume 3) tests — RelationshipProfile
persistence/restart, Development level computation (including moving
backward), Attachment signals, and the Growth Engine as sole writer."""
import asyncio
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import chromadb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.memory.chat_history as chat_history_module
import app.relationship.profile_store as relationship_profile_store
from app.growth.engine import GrowthEngine
from app.learning.observation_store import ObservationStore
from app.memory.short_term import ShortTermMemory
from app.memory2.store import MemoryStore
from app.relationship.state import (
    RelationshipProfile,
    TrustModel,
    compute_development_level,
    evaluate_attachment_signals,
    update_boundaries,
    UserBoundaries,
)


def _fake_embed(text: str, dim: int = 32) -> list[float]:
    vec = [0.0] * dim
    for word in text.lower().replace("—", "").split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


class _FakeLlm:
    def complete(self, prompt: str, max_tokens: int = 120) -> str:
        return "summary"


@pytest.fixture
def isolated_profile_db(tmp_path, monkeypatch):
    db_path = tmp_path / "profile.db"
    monkeypatch.setattr(relationship_profile_store, "RELATIONSHIP_PROFILE_DB_PATH", db_path)
    # GrowthEngine's attachment scoring (Vol 7 Ch 8) reads session
    # timestamps from chat_history — isolate that too so tests never touch
    # the real, shared default data directory.
    monkeypatch.setattr(chat_history_module, "CHAT_HISTORY_DB_PATH", db_path)
    return db_path


@pytest.fixture
def observation_store(tmp_path) -> ObservationStore:
    return ObservationStore(path=tmp_path / "hearth_learning.duckdb")


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    client = chromadb.EphemeralClient()
    return MemoryStore(
        index_path=tmp_path / "idx.sqlite3",
        embed_fn=_fake_embed,
        episodic_collection=client.get_or_create_collection(f"episodic_{uuid.uuid4().hex}"),
        semantic_collection=client.get_or_create_collection(f"semantic_{uuid.uuid4().hex}"),
    )


def _memory_with(messages: list[str]) -> ShortTermMemory:
    memory = ShortTermMemory(_FakeLlm())
    for text in messages:
        memory.add_turn(text, "I hear you, that sounds hard.")
    return memory


# --- DoD: restart resumes full relationship context -------------------------


def test_restart_resumes_full_relationship_context(isolated_profile_db, store, observation_store):
    engine = GrowthEngine(store=store, observation_store=observation_store)
    memory = _memory_with(
        [
            "My manager Sarah criticized my work again today and I feel so anxious about it",
            "Sarah was harsh with me again in the meeting, I am really anxious around her now",
            "Sarah gave me tough feedback in review too, it makes me so anxious every time",
        ]
    )
    result = asyncio.run(engine.process_session("u1", memory))
    assert result.episodic_memories_formed == 3
    assert result.relationship_profile.conversation_count == 1

    # "Restart" — a completely fresh read from persistent storage, no
    # in-memory state carried over.
    reloaded = relationship_profile_store.get_relationship_profile("u1")
    assert reloaded is not None
    assert reloaded.trust.general_trust == result.relationship_profile.trust.general_trust
    assert reloaded.trust.vulnerability_trust > 0.0
    assert reloaded.conversation_count == 1
    assert any(p.name == "Sarah" for p in reloaded.life_model.important_people)
    assert reloaded.schema_version == 1


def test_growth_engine_is_the_only_thing_that_persists_relationship_state(isolated_profile_db, store):
    """No profile exists yet; merely instantiating the engine or reading
    state must not write anything (Vol 3 Ch 13, Invariant 6)."""
    assert relationship_profile_store.get_relationship_profile("nobody") is None
    profile = relationship_profile_store.get_or_create_relationship_profile("nobody")
    assert profile.trust.general_trust == 0.0
    # get_or_create does not itself persist a stub row.
    assert relationship_profile_store.get_relationship_profile("nobody") is None


# --- Development level (Vol 3 Ch 5) — derived, can move backward -----------


def test_development_level_requires_trust_and_conversation_floor():
    low_trust = TrustModel(general_trust=0.05)
    assert compute_development_level(low_trust, conversation_count=1) == "stranger"

    mid_trust = TrustModel(general_trust=0.4, vulnerability_trust=0.3, advice_trust=0.2, consistency_confidence=0.4)
    assert (
        compute_development_level(mid_trust, conversation_count=8, communication_traits={"likes_reflection": 0.8})
        == "familiar"
    )


def test_development_level_moves_backward_after_a_long_gap():
    high_trust = TrustModel(general_trust=0.8, vulnerability_trust=0.75, advice_trust=0.6, consistency_confidence=0.8)
    fresh_level = compute_development_level(
        high_trust, conversation_count=35, disclosure_depth=0.8, communication_traits={"likes_reflection": 0.9}
    )
    assert fresh_level == "deep_long_term_companion"

    stale_level = compute_development_level(
        high_trust,
        conversation_count=35,
        disclosure_depth=0.8,
        communication_traits={"likes_reflection": 0.9},
        days_since_last_contact=200,
    )
    # A long gap genuinely lowers the level — never a stale high-water mark.
    assert stale_level in {"stranger", "acquaintance"}
    assert DEVELOPMENT_INDEX(stale_level) < DEVELOPMENT_INDEX(fresh_level)


def DEVELOPMENT_INDEX(level: str) -> int:
    from app.relationship.state import DEVELOPMENT_LEVELS

    return DEVELOPMENT_LEVELS.index(level)


def test_development_level_never_driven_by_conversation_count_alone():
    """A person who has talked daily for a superficial purpose should not
    outrank someone with a handful of deeply vulnerable conversations,
    purely on count (Vol 3 Ch 5)."""
    shallow_but_frequent = TrustModel(general_trust=0.1, vulnerability_trust=0.05, advice_trust=0.05, consistency_confidence=0.1)
    level = compute_development_level(shallow_but_frequent, conversation_count=500)
    assert level == "stranger"


# --- Attachment signals (Vol 3 Ch 4) — computed and logged only ------------


def test_attachment_signals_detect_warning_language_but_stay_advisory():
    signals = evaluate_attachment_signals(["you're all I have, I don't need anyone else"])
    assert signals.replacement_language_detected is True
    assert signals.has_warning_signal is True
    # No escalation mechanism exists on this object — it's a flag, not an action.
    assert not hasattr(signals, "escalate")


def test_attachment_signals_healthy_case():
    signals = evaluate_attachment_signals(["I had a great time with my friends this weekend, feeling good"])
    assert signals.replacement_language_detected is False
    assert signals.has_warning_signal is False
    assert signals.healthy_engagement_with_others is True


# --- Boundaries (Vol 3 Ch 7) — conservative, never loosens with intimacy ---


def test_boundaries_flag_avoid_topic_from_explicit_signal():
    boundaries = UserBoundaries()
    updated = update_boundaries(boundaries, "please don't ask about my ex again", "relationship")
    assert "relationship" in updated.avoid_topics


def test_boundaries_do_not_flag_on_ambiguous_reaction():
    boundaries = UserBoundaries()
    updated = update_boundaries(boundaries, "that was a weird day I guess", "work")
    assert updated.avoid_topics == []


# --- Trust never fed by raw engagement volume (Vol 3 Ch 3) -----------------


def test_trust_signals_not_driven_by_message_count(isolated_profile_db, store):
    """Filler/upbeat small talk may still form episodic memories (a real
    emotion classifier can legitimately read joy in "lol"/"nice") but none
    of it carries a vulnerability/life-event/goal significance marker — so
    trust must not move meaningfully, no matter how many messages there
    were (Vol 3 Ch 3: never fed by raw engagement volume)."""
    engine = GrowthEngine(store=store)
    chatty_but_shallow = _memory_with(["lol", "haha yeah", "cool", "nice", "same", "yep", "totally", "for real"])
    result = asyncio.run(engine.process_session("u2", chatty_but_shallow))
    assert result.relationship_profile.trust.vulnerability_trust < 0.1
    assert result.relationship_profile.trust.general_trust < 0.1
