"""Phase 5 (Book Volumes 7 + 8 — Learning & Growth, Evaluation) tests."""
import asyncio
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import chromadb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.evaluation.worker as evaluation_worker_module
import app.memory.chat_history as chat_history_module
import app.relationship.profile_store as relationship_profile_store
from app.benchmarks.cross_volume_runner import discover_cases, run_cross_volume_benchmarks
from app.benchmarks.runner import gate_release
from app.evaluation.invariants import run_invariant_checks
from app.evaluation.worker import EvaluationWorker
from app.growth.engine import GrowthEngine
from app.intervention.engine import InterventionEngine
from app.intervention.retrieval import SkillRetriever
from app.learning import attachment as attachment_module
from app.learning.coldstart import blend, confidence_curve
from app.learning.observation_store import InvalidTrustObservationError, ObservationStore
from app.learning.pipeline import fold_ewma, recompute_value
from app.learning.recompute import recompute_all
from app.memory.short_term import ShortTermMemory
from app.memory2.store import MemoryStore
from app.onboarding.profile_schema import UserProfile
from app.onboarding.profile_store import create_profile
from app.relationship.state import AttachmentSignals
from app.safety2.benchmark_runner import discover_benchmarks as discover_safety_benchmarks
from app.safety2.worker import SafetyWorker
from app.skills.benchmark_runner import discover_benchmarks as discover_skill_benchmarks


def _fake_embed(text: str, dim: int = 32) -> list[float]:
    vec = [0.0] * dim
    for word in text.lower().replace("—", "").split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def _fake_intervention_engine() -> InterventionEngine:
    client = chromadb.EphemeralClient()
    retriever = SkillRetriever(embed_fn=_fake_embed, collection=client.get_or_create_collection(f"skills_{uuid.uuid4().hex}"))
    return InterventionEngine(retriever=retriever)


class _FakeLlm:
    def complete(self, prompt: str, max_tokens: int = 120, temperature: float = 0.0) -> str:
        return "summary"


# --- Cold-start (Vol 7 Ch 11) ------------------------------------------------


def test_confidence_curve_increases_smoothly_with_sample_size():
    assert confidence_curve(0) == 0.0
    low = confidence_curve(2)
    mid = confidence_curve(10)
    high = confidence_curve(1000)
    assert 0.0 < low < mid < high < 1.0
    assert high > 0.99


def test_blend_uses_prior_when_sample_size_low_and_observed_when_high():
    low_sample = blend(0.5, 0.7, 0.9, sample_size=1)
    high_sample = blend(0.5, 0.7, 0.9, sample_size=10_000)
    assert abs(low_sample - 0.7) < abs(low_sample - 0.9)  # closer to prior when cold
    assert abs(high_sample - 0.9) < 0.01  # nearly all-observed at high sample size


def test_blend_falls_back_to_population_default_without_a_prior():
    result = blend(0.5, None, 0.9, sample_size=1)
    assert abs(result - 0.5) < abs(result - 0.9)


# --- Recomputation Pipeline (Vol 7 Ch 4) — the two-point-blend bug fix ------


def test_fold_ewma_uses_every_observation_not_just_the_latest(tmp_path):
    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    for _ in range(10):
        store.append("communication", "likes_reflection", 1.0, {}, "test")
    observations = store.latest("communication", "likes_reflection", 20)

    folded = fold_ewma(0.5, observations, alpha=0.05)
    single_observation_blend = 0.05 * 1.0 + 0.95 * 0.5  # the bug this replaces
    assert folded > single_observation_blend + 0.01  # materially different, not a coincidence


def test_recompute_value_reports_unchanged_when_within_tolerance(tmp_path):
    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    result = recompute_value(store, "communication", "likes_reflection", current=0.5, alpha=0.05)
    assert result.changed is False
    assert result.sample_size == 0


# --- Schema-level Trust guard (Vol 7 Ch 3/Invariant 4) ----------------------


def test_trust_observation_requires_valid_derivation(tmp_path):
    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    with pytest.raises(InvalidTrustObservationError):
        store.append("relationship", "general_trust", 1.0, {}, "test")
    with pytest.raises(InvalidTrustObservationError):
        store.append("relationship", "general_trust", 1.0, {"derivation": "message_frequency"}, "test")
    store.append("relationship", "general_trust", 1.0, {"derivation": "disclosure_depth"}, "test")  # ok


def test_non_trust_relationship_observations_unaffected(tmp_path):
    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    store.append("relationship", "attachment_signal", 1.0, {}, "test")  # should not raise


# --- Attachment scoring (Vol 7 Ch 8) ----------------------------------------


def test_contact_urgency_trend_detects_shrinking_gaps():
    now = datetime.now(timezone.utc)
    timestamps = []
    t = now - timedelta(days=40)
    for _ in range(6):
        timestamps.append(t)
        t += timedelta(days=3)
    for _ in range(5):
        timestamps.append(t)
        t += timedelta(days=1)
    trend = attachment_module.contact_urgency_trend(timestamps)
    assert trend > 0.3


def test_contact_urgency_trend_zero_with_insufficient_history():
    assert attachment_module.contact_urgency_trend([datetime.now(timezone.utc)]) == 0.0


def test_replacement_language_and_unavailability_scores():
    assert attachment_module.replacement_language_score(["you're all i have"]) > 0.0
    assert attachment_module.replacement_language_score(["just chatting"]) == 0.0
    assert attachment_module.unavailability_distress_score(["why weren't you here"]) > 0.0


def test_combined_attachment_score_never_exceeds_bounds():
    score = attachment_module.compute_combined_attachment_score(
        current_score=0.9, contact_urgency=1.0, replacement_language=1.0, unavailability_distress=1.0,
    )
    assert 0.0 <= score <= 1.0


# --- Growth Engine rewired onto the pipeline ---------------------------------


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    db_path = tmp_path / "profile.db"
    monkeypatch.setattr(relationship_profile_store, "RELATIONSHIP_PROFILE_DB_PATH", db_path)
    monkeypatch.setattr(chat_history_module, "CHAT_HISTORY_DB_PATH", db_path)
    return db_path


@pytest.fixture
def memory_store(tmp_path) -> MemoryStore:
    client = chromadb.EphemeralClient()
    return MemoryStore(
        index_path=tmp_path / "idx.sqlite3", embed_fn=_fake_embed,
        episodic_collection=client.get_or_create_collection(f"ep_{uuid.uuid4().hex}"),
        semantic_collection=client.get_or_create_collection(f"sem_{uuid.uuid4().hex}"),
    )


@pytest.fixture
def observation_store(tmp_path) -> ObservationStore:
    return ObservationStore(path=tmp_path / "hearth_learning.duckdb")


def _memory_with(messages: list[str]) -> ShortTermMemory:
    memory = ShortTermMemory(_FakeLlm())
    for text in messages:
        memory.add_turn(text, "I hear you, that sounds hard.")
    return memory


def test_growth_engine_writes_derivation_tagged_trust_observations(isolated_paths, memory_store, observation_store):
    engine = GrowthEngine(store=memory_store, observation_store=observation_store)
    memory = _memory_with(["My manager Sarah criticized my work again and I feel so anxious about it"])
    asyncio.run(engine.process_session("u1", memory))
    observations = observation_store.latest("relationship", "vulnerability_trust")
    assert observations
    assert observations[0].context["derivation"] in {"disclosure_depth", "consistency_observation"}


def test_growth_engine_trust_compounds_across_sessions(isolated_paths, memory_store, observation_store):
    engine = GrowthEngine(store=memory_store, observation_store=observation_store)
    messages = [
        "My manager Sarah criticized my work again today and I feel so anxious about it",
        "Sarah was harsh with me again in the meeting, I am really anxious around her now",
    ]
    result1 = asyncio.run(engine.process_session("u1", _memory_with(messages)))
    result2 = asyncio.run(engine.process_session("u1", _memory_with(messages)))
    # Real EWMA fold across sessions should move measurably, not just a
    # single-observation nudge each time.
    assert result2.relationship_profile.trust.vulnerability_trust > result1.relationship_profile.trust.vulnerability_trust


def test_growth_engine_computes_combined_attachment_score(isolated_paths, memory_store, observation_store):
    engine = GrowthEngine(store=memory_store, observation_store=observation_store)
    memory = _memory_with(["you're all i have, please don't leave me"])
    result = asyncio.run(engine.process_session("u1", memory))
    assert result.relationship_profile.attachment_signals.combined_score > 0.0


# --- learning/recompute.py rewired onto the pipeline + coldstart -----------


def test_recompute_all_covers_all_seven_communication_traits(tmp_path, monkeypatch):
    from app.cognitive.communication import TRAIT_KEYS

    profile_db = tmp_path / "profile.db"
    monkeypatch.setattr("app.onboarding.profile_store.PROFILE_DB_PATH", profile_db)
    profile = create_profile(__import__("app.onboarding.profile_schema", fromlist=["OnboardingRequest"]).OnboardingRequest(
        name="A", companion_name="Hearth"
    ))
    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    for trait in TRAIT_KEYS:
        store.append("communication", trait, 1.0, {}, "test")

    result = recompute_all(profile.user_id, store)
    assert set(TRAIT_KEYS).issubset(result.communication_traits.keys())
    for trait in TRAIT_KEYS:
        assert result.communication_traits[trait] > 0.5  # moved up from neutral


def test_recompute_all_skill_affinity_cold_start_uses_trait_prior(tmp_path, monkeypatch):
    profile_db = tmp_path / "profile.db"
    monkeypatch.setattr("app.onboarding.profile_store.PROFILE_DB_PATH", profile_db)
    from app.onboarding.profile_schema import OnboardingRequest

    profile = create_profile(OnboardingRequest(name="A", companion_name="Hearth", communication_traits={"likes_direct_advice": 0.9}))
    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    # Only ONE skill observation — deep cold-start territory.
    store.append("skill", "cognitive_reframing", 1.0, {}, "test")

    result = recompute_all(profile.user_id, store)
    # High likes_direct_advice prior should pull the cold-start estimate
    # above the bare neutral (0.5) default, even with just one observation.
    assert result.skill_affinity["cognitive_reframing"] > 0.5


# --- Evaluation Worker (Vol 8 Ch 3/4) ---------------------------------------


def test_invariant_checks_catch_validation_after_advice():
    results = run_invariant_checks({"reply_text": "You could try that. That sounds hard though.", "skill_id": None, "composed_with": None, "is_safety_response": False})
    validation_result = next(r for r in results if "Validation comes before" in r.rule)
    assert validation_result.result == "fail"


def test_invariant_checks_catch_crisis_composed():
    results = run_invariant_checks({"reply_text": "reply", "skill_id": "crisis_support", "composed_with": "validation", "is_safety_response": False})
    crisis_result = next(r for r in results if "Crisis Support" in r.rule)
    assert crisis_result.result == "fail"


def test_invariant_checks_log_based_invariant_reported_honestly():
    results = run_invariant_checks({"reply_text": "reply", "skill_id": None, "composed_with": None, "is_safety_response": False})
    log_based = next(r for r in results if r.check_type == "log_based")
    assert log_based.result == "not_automatically_checkable"


def test_evaluation_worker_flags_human_review_for_unmeasurable_metrics(tmp_path):
    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    worker = EvaluationWorker(store)
    result = worker.evaluate("u1", "hello", "That sounds hard. You could try journaling.")
    assert result.success_proxies["felt_understanding"] is None
    assert result.success_proxies["reduction_in_expressed_isolation"] is None
    assert result.human_review_flagged is True


def test_evaluation_worker_trust_consistency_proxy_uses_real_trust_snapshot(tmp_path):
    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    worker = EvaluationWorker(store)
    result = worker.evaluate(
        "u1", "hello", "That sounds hard.",
        trust_snapshot={"general_trust": 0.4, "vulnerability_trust": 0.6, "advice_trust": 0.2, "consistency_confidence": 0.5},
    )
    assert result.success_proxies["trust_consistency"] == pytest.approx(0.425, abs=0.001)


def test_evaluation_worker_dual_writes_safety_findings(tmp_path, monkeypatch):
    import app.safety2.audit as audit_module

    monkeypatch.setattr(audit_module, "SAFETY_AUDIT_DB_PATH", tmp_path / "profile.db")
    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    worker = EvaluationWorker(store)
    worker.evaluate("u1", "hello", "reply", safety_findings={"category": "acute_distress"}, is_safety_response=True)
    from app.db.sqlite_models import get_connection

    conn = get_connection(audit_module.SAFETY_AUDIT_DB_PATH)
    try:
        row = conn.execute("SELECT category FROM safety_audit WHERE user_id = ?", ("u1",)).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == "acute_distress"


def test_evaluation_writes_dedicated_evaluation_observation_type(tmp_path):
    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    worker = EvaluationWorker(store)
    worker.evaluate("u1", "hello", "reply")
    assert store.latest("evaluation", "invariant_adherence")
    assert store.latest("evaluation", "anti_pattern")


# --- Unified benchmarks + release gate (Vol 8 Ch 7/9) -----------------------


def test_benchmarks_unified_under_one_root():
    skill_cases = discover_skill_benchmarks()
    safety_cases = discover_safety_benchmarks()
    cross_volume_cases = discover_cases()
    assert len(skill_cases) == 8
    assert len(safety_cases) == 30
    assert len(cross_volume_cases) == 2


def test_cross_volume_benchmarks_pass():
    engine = _fake_intervention_engine()
    results = run_cross_volume_benchmarks(intervention_engine=engine)
    assert all(r.passed for r in results)


def test_release_gate_allows_when_everything_passes():
    engine = _fake_intervention_engine()
    decision = gate_release(intervention_engine=engine)
    assert decision.allowed is True
    assert decision.blocked_by_safety is False
    assert decision.blocked_by_skill is False


def test_release_gate_blocks_safety_regression_with_no_override():
    class AlwaysWrongSafetyWorker(SafetyWorker):
        def assess(self, *args, **kwargs):
            from app.safety2.worker import SafetyAssessment

            return SafetyAssessment(category="none", confidence=0.0, route="ordinary")

    engine = _fake_intervention_engine()
    decision = gate_release(intervention_engine=engine, safety_worker=AlwaysWrongSafetyWorker(), override_reason="ignore this, please ship anyway")
    assert decision.allowed is False
    assert decision.blocked_by_safety is True  # override reason must not affect this


def test_release_gate_skill_regression_overridable():
    from app.intervention import engine as engine_module

    class AlwaysListenEngine(InterventionEngine):
        def plan(self, transcript, profile, context, crisis=False):
            from app.intervention.engine import InterventionPlan

            return InterventionPlan(strategy="listen", primary_skill=None, secondary_skill=None, candidate_ids=[])

    broken_engine = AlwaysListenEngine()
    without_override = gate_release(intervention_engine=broken_engine)
    assert without_override.allowed is False
    assert without_override.blocked_by_skill is True

    with_override = gate_release(intervention_engine=broken_engine, override_reason="reviewed by QA 2026-07-30, known trade-off")
    assert with_override.allowed is True
    assert with_override.override_reason is not None
