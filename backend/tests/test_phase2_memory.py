"""Phase 2 memory track (Book Volume 4) tests — formation, promotion,
decay, retrieval, contradiction, and privacy controls.

Embeddings never touch a live embedding server (none is available in this
environment): a deterministic bag-of-words hash embedding stands in for
`app.memory.embedder.embed`, sharing dimensions for texts that share words
so similarity-based clustering/retrieval is actually exercised, not just
smoke-tested against noise."""
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import chromadb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.memory2.consolidation import merge, find_merge_candidates
from app.memory2.contradiction import check_and_resolve
from app.memory2.decay import compute_priority, reinforce
from app.memory2.formation import build_emotional_metadata, build_summary, find_candidates
from app.memory2.models import EmotionalMetadata, EpisodicMemory, MemoryStatus
from app.memory2.privacy import correct_episodic, delete_episodic_with_cascade, plain_language_summary
from app.memory2.promotion import find_clusters, run_promotion
from app.memory2.retrieval import retrieve
from app.memory2.store import MemoryStore


def _fake_embed(text: str, dim: int = 32) -> list[float]:
    vec = [0.0] * dim
    for word in text.lower().replace("—", "").split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    client = chromadb.EphemeralClient()
    return MemoryStore(
        index_path=tmp_path / "idx.sqlite3",
        embed_fn=_fake_embed,
        episodic_collection=client.get_or_create_collection(f"episodic_{uuid.uuid4().hex}"),
        semantic_collection=client.get_or_create_collection(f"semantic_{uuid.uuid4().hex}"),
    )


def _make_episodic(user_id: str, summary: str, *, entities=None, intensity=0.7, category="anxiety", valence="negative", markers=None, timestamp=None) -> EpisodicMemory:
    now = timestamp or datetime.now(timezone.utc)
    return EpisodicMemory(
        id=str(uuid.uuid4()),
        user_id=user_id,
        timestamp=now,
        summary=summary,
        entities=entities or [],
        significance_markers=markers or [],
        emotional_metadata=EmotionalMetadata(valence=valence, intensity=intensity, category=category),
        last_reinforced=now,
    )


# --- Formation (Vol 4 Ch 8) -------------------------------------------------


def test_formation_triggers_on_real_signals_and_stays_silent_otherwise():
    messages = [
        "hey how's it going",  # no trigger -> silence
        "My sister Maya just told me she's getting divorced and I feel awful for her",  # new entity + significance + emotion
    ]
    candidates = find_candidates(messages)
    assert len(candidates) == 1
    assert candidates[0].text.startswith("My sister Maya")
    assert "new_entity" in candidates[0].triggers


def test_summary_is_templated_not_freeform():
    messages = ["My sister Maya just told me she's getting divorced and I feel awful for her"]
    candidate = find_candidates(messages)[0]
    summary = build_summary(candidate)
    assert summary.count(" — ") == 2
    assert "Maya" in summary or "sister" in summary


# --- DoD: memory formed in one conversation retrieved in a later one -------


def test_memory_formed_in_one_conversation_is_retrieved_in_a_later_one(store):
    user_id = "u1"
    # "Conversation 1"
    messages = [
        "My manager Sarah criticized my work again today and I feel so anxious about it",
    ]
    candidate = find_candidates(messages)[0]
    mem = _make_episodic(
        user_id, build_summary(candidate), entities=candidate.entities, markers=candidate.markers + candidate.triggers,
        intensity=candidate.emotion_intensity or 0.7, category=candidate.emotion_category, valence="negative",
    )
    store.save_episodic(mem)

    # "Conversation 2" — a fresh session, same store (same profile/user).
    results = retrieve(store, "How is work with Sarah going?", user_id)
    assert any("Sarah" in r.text for r in results)


# --- Decay (Vol 4 Ch 10) — reduces priority, never deletes ------------------


def test_decay_reduces_priority_but_never_deletes(store):
    now = datetime.now(timezone.utc)
    fresh = _make_episodic("u1", "Sarah — anxiety — new entity", entities=["Sarah"], timestamp=now)
    stale = fresh.model_copy(update={"id": str(uuid.uuid4()), "last_reinforced": now - timedelta(days=90)})
    store.save_episodic(fresh)
    store.save_episodic(stale)

    fresh_priority = compute_priority(fresh, now=now)
    stale_priority = compute_priority(stale, now=now)
    assert stale_priority < fresh_priority
    # Still present and active — decay never deletes.
    assert store.get_episodic(stale.id, "u1").status == MemoryStatus.active


def test_reinforcement_resets_decay():
    now = datetime.now(timezone.utc)
    old = _make_episodic("u1", "Sarah — anxiety — repetition", timestamp=now - timedelta(days=60))
    old = old.model_copy(update={"last_reinforced": now - timedelta(days=60)})
    reinforced = reinforce(old, now=now)
    assert reinforced.last_reinforced == now
    assert reinforced.reference_count == old.reference_count + 1
    assert compute_priority(reinforced, now=now) > compute_priority(old, now=now)


# --- Promotion (Vol 4 Ch 9) --------------------------------------------------


def test_promotion_clusters_and_builds_templated_semantic_fact(store):
    user_id = "u1"
    now = datetime.now(timezone.utc)
    for i in range(3):
        mem = _make_episodic(
            user_id, f"Sarah — anxiety — mention {i}", entities=["Sarah"], intensity=0.7,
            timestamp=now - timedelta(days=i),
        )
        store.save_episodic(mem)

    clusters = find_clusters(store, user_id)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3

    promoted = run_promotion(store, user_id)
    assert len(promoted) == 1
    assert promoted[0].fact == "User tends to experience anxiety in response to Sarah."
    assert 0.0 < promoted[0].confidence <= 1.0
    assert set(promoted[0].source_episodes) == {m.id for m in clusters[0]}


def test_promotion_leaves_irregular_cluster_unpromoted_without_fallback(store):
    # A cluster below MIN_CLUSTER_SIZE never gets promoted at all.
    user_id = "u1"
    store.save_episodic(_make_episodic(user_id, "Sarah — anxiety — a", entities=["Sarah"]))
    store.save_episodic(_make_episodic(user_id, "Sarah — anxiety — b", entities=["Sarah"]))
    assert run_promotion(store, user_id) == []


def test_promotion_uses_narrow_llm_fallback_only_when_provided(store):
    from app.memory2.promotion import promote_cluster

    # A cluster with no shared entity doesn't fit the template.
    cluster = [
        _make_episodic("u1", "conversation — anxiety — a", entities=[]),
        _make_episodic("u1", "conversation — anxiety — b", entities=[]),
        _make_episodic("u1", "conversation — anxiety — c", entities=[]),
    ]
    assert promote_cluster(cluster, "u1") is None  # no fallback configured -> stays unpromoted

    fallback_calls = []

    def fallback(summaries):
        fallback_calls.append(summaries)
        return "User feels anxious in unpredictable situations."

    result = promote_cluster(cluster, "u1", llm_fallback=fallback)
    assert result is not None
    assert result.fact == "User feels anxious in unpredictable situations."
    assert len(fallback_calls) == 1


# --- Contradiction (Vol 4 Ch 13) — reduces confidence, never overwrites -----


def test_contradiction_reduces_confidence_never_overwrites(store):
    user_id = "u1"
    now = datetime.now(timezone.utc)
    for i in range(3):
        store.save_episodic(_make_episodic(user_id, f"Sarah — anxiety — mention {i}", entities=["Sarah"], timestamp=now - timedelta(days=i)))
    promoted = run_promotion(store, user_id)
    fact_id = promoted[0].id
    original_confidence = promoted[0].confidence

    calm_episode = _make_episodic(user_id, "Sarah — joy — new entity", entities=["Sarah"], intensity=0.6, category="joy", valence="positive")
    store.save_episodic(calm_episode)
    contradictions = check_and_resolve(store, calm_episode)

    assert len(contradictions) == 1
    updated = store.get_semantic(fact_id, user_id)
    assert updated.confidence < original_confidence
    assert updated.status == MemoryStatus.active  # never overwritten/removed
    assert updated.fact == promoted[0].fact  # fact text itself untouched


# --- Consolidation (Vol 4 Ch 12) — merges without destroying source data ---


def test_consolidation_merges_near_duplicates_retaining_sources(store):
    user_id = "u1"
    now = datetime.now(timezone.utc)
    a = _make_episodic(user_id, "Sarah — anxiety — mention a", entities=["Sarah"], intensity=0.5, timestamp=now)
    b = _make_episodic(user_id, "Sarah — anxiety — mention b", entities=["Sarah"], intensity=0.9, timestamp=now - timedelta(days=1))
    store.save_episodic(a)
    store.save_episodic(b)

    groups = find_merge_candidates(store, user_id, now=now)
    assert len(groups) == 1
    merged = merge(groups[0], now=now)
    assert merged.status == MemoryStatus.consolidated
    assert merged.emotional_metadata.intensity == 0.9  # inherits the strongest, not an average
    assert set(merged.merged_from) == {a.id, b.id} - {merged.id}


# --- Privacy (Vol 4 Ch 15): view / correct / delete, end to end ------------


def test_privacy_view_correct_delete_end_to_end(store):
    user_id = "u1"
    now = datetime.now(timezone.utc)
    for i in range(3):
        store.save_episodic(_make_episodic(user_id, f"Sarah — anxiety — mention {i}", entities=["Sarah"], timestamp=now - timedelta(days=i)))
    promoted = run_promotion(store, user_id)
    assert promoted

    # View — plain-language, grouped by theme, not a raw dump.
    summary = plain_language_summary(store, user_id)
    assert "Sarah" in summary["remembered_moments_by_theme"]
    assert summary["total_moments_remembered"] == 3
    assert summary["total_general_facts"] == 1

    # Correct — applied immediately.
    target = store.list_episodic(user_id)[0]
    corrected = correct_episodic(store, target.id, user_id, corrected_summary="Sarah — corrected — mention")
    assert corrected.summary == "Sarah — corrected — mention"
    assert store.get_episodic(target.id, user_id).summary == "Sarah — corrected — mention"

    # Delete — hard delete, cascades a confidence reduction to the dependent fact.
    fact_before = store.get_semantic(promoted[0].id, user_id)
    affected = delete_episodic_with_cascade(store, target.id, user_id)
    assert store.get_episodic(target.id, user_id) is None  # gone, not decayed-to-zero
    assert len(affected) == 1
    assert affected[0].confidence < fact_before.confidence
    assert target.id not in affected[0].source_episodes
