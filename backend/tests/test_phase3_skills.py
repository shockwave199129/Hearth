"""Phase 3 (Book Volume 5 — Intervention & Skills) tests.

Retrieval embeddings never touch a live embedding server (none is
available in this environment): a deterministic bag-of-words hash
embedding stands in, sharing dimensions for texts that share words, so the
Chroma-backed semantic retrieval path is actually exercised."""
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sys

import chromadb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.intervention.engine import InterventionContext, InterventionEngine
from app.intervention.observation import mark_skill_used, resolve_pending_observation
from app.intervention.ranking import compose_skills, rank_candidates
from app.intervention.retrieval import SkillRetriever
from app.cognitive.mind_state import MindState
from app.learning.observation_store import ObservationStore
from app.onboarding.profile_schema import UserProfile
from app.skills.benchmark_runner import discover_benchmarks, run_benchmarks
from app.skills.loader import get_skill, load_catalog


def _fake_embed(text: str, dim: int = 32) -> list[float]:
    vec = [0.0] * dim
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


@pytest.fixture
def engine() -> InterventionEngine:
    client = chromadb.EphemeralClient()
    retriever = SkillRetriever(embed_fn=_fake_embed, collection=client.get_or_create_collection(f"skills_{uuid.uuid4().hex}"))
    return InterventionEngine(retriever=retriever)


def _profile(**overrides) -> UserProfile:
    defaults = dict(user_id="u1", name="A", companion_name="Companion", created_at=datetime.now(timezone.utc))
    defaults.update(overrides)
    return UserProfile(**defaults)


def _context(**overrides) -> InterventionContext:
    defaults = dict(stage="listening")
    defaults.update(overrides)
    return InterventionContext(**defaults)


# --- Catalog / manifest validation -----------------------------------------


def test_skill_catalog_loads_structured_skills():
    skills = load_catalog()
    ids = {skill.id for skill in skills}
    assert {
        "validation", "grounding", "journaling", "cognitive_reframing",
        "boundary_setting", "sleep_hygiene", "crisis_support",
    }.issubset(ids)


def test_manifests_validate():
    assert get_skill("grounding").manifest.skill_id == "grounding"


def test_legacy_citations_carried_into_content_and_manifest():
    """Book Vol 5 Ch 3's Safety notes / provenance shouldn't be lost when a
    skill is restructured — the legacy library/*.md source citations and
    NEEDS REVIEW warning must survive into the new content.md + manifest."""
    for skill_id in ("validation", "grounding", "journaling", "cognitive_reframing", "boundary_setting", "sleep_hygiene"):
        skill = get_skill(skill_id)
        assert skill.manifest.source, f"{skill_id} manifest missing source"
        assert "NEEDS REVIEW" in skill.manifest.source
        assert "Source" in skill.content
        assert "NEEDS REVIEW" in skill.content


# --- Intervention Engine: ordinary routing ----------------------------------


def test_intervention_engine_routes_validation(engine):
    plan = engine.plan("I feel so hurt and alone.", _profile(), _context(stage="listening"))
    assert plan.strategy == "validate"
    assert plan.primary_skill and plan.primary_skill.skill.id == "validation"


def test_intervention_engine_routes_grounding(engine):
    plan = engine.plan("I can't breathe and everything is spinning.", _profile(), _context(stage="supporting"))
    assert plan.strategy == "ground"
    assert plan.primary_skill and plan.primary_skill.skill.id == "grounding"


def test_intervention_engine_no_skill_when_nothing_relevant(engine):
    """Vol 1 Ch 8: an Intervention Strategy is not limited to picking a
    skill — sometimes the right call is simply to listen."""
    plan = engine.plan("I understand what happened and I just want to think about it.", _profile(), _context())
    assert plan.strategy == "listen"
    assert plan.primary_skill is None


def test_sleep_hygiene_not_confused_with_grounding_on_racing_thoughts(engine):
    """Regression: "racing" alone used to collide with sleep_hygiene's
    night-time "racing thoughts" trigger, since grounding's own trigger list
    also contained bare "racing"."""
    plan = engine.plan("My mind keeps racing when I try to sleep.", _profile(), _context())
    assert plan.primary_skill and plan.primary_skill.skill.id == "sleep_hygiene"


# --- Crisis routing (Vol 5 Ch 14, Invariant 8) ------------------------------


def test_crisis_bypasses_ordinary_scoring(engine):
    plan = engine.plan("I want to hurt myself.", _profile(), _context(stage="supporting"), crisis=True)
    assert plan.strategy == "crisis_support"
    assert plan.primary_skill and plan.primary_skill.skill.id == "crisis_support"
    assert plan.secondary_skill is None


def test_crisis_support_excluded_from_ordinary_candidates(engine):
    """crisis_support must never win or even appear via ordinary scoring —
    detection isn't this engine's job (Ch 14)."""
    plan = engine.plan("I want to hurt myself.", _profile(), _context(), crisis=False)
    assert "crisis_support" not in plan.candidate_ids


def test_crisis_support_never_composed():
    crisis_skill = get_skill("crisis_support")
    validation_skill = get_skill("validation")
    from app.intervention.ranking import RankedSkill

    primary = RankedSkill(skill=crisis_skill, score=1.0, reason="crisis path")
    other = RankedSkill(skill=validation_skill, score=0.9, reason="high score")
    composed = compose_skills(primary, [other])
    assert composed == [primary]


# --- Composition compatibility (Vol 5 Ch 13) --------------------------------


def test_compose_skills_respects_compatibility_table():
    grounding = get_skill("grounding")
    reframing = get_skill("cognitive_reframing")
    from app.intervention.ranking import RankedSkill

    primary = RankedSkill(skill=grounding, score=0.8, reason="x")
    incompatible_other = RankedSkill(skill=reframing, score=0.9, reason="x")
    composed = compose_skills(primary, [incompatible_other])
    assert len(composed) == 1  # incompatible pair never composed


def test_compose_skills_requires_minimum_secondary_relevance():
    """Regression for the unconditional `break`: a low-scoring, barely
    relevant compatible skill must not get composed just because it's the
    first compatible item in the list."""
    validation = get_skill("validation")
    sleep_hygiene = get_skill("sleep_hygiene")
    from app.intervention.ranking import RankedSkill

    primary = RankedSkill(skill=validation, score=0.8, reason="x")
    barely_relevant = RankedSkill(skill=sleep_hygiene, score=0.05, reason="x")
    composed = compose_skills(primary, [barely_relevant])
    assert len(composed) == 1


def test_compose_skills_attaches_genuinely_relevant_secondary():
    validation = get_skill("validation")
    reframing = get_skill("cognitive_reframing")
    from app.intervention.ranking import RankedSkill

    primary = RankedSkill(skill=validation, score=0.8, reason="x")
    relevant_other = RankedSkill(skill=reframing, score=0.5, reason="x")
    composed = compose_skills(primary, [relevant_other])
    assert len(composed) == 2
    assert composed[1].skill.id == "cognitive_reframing"


# --- Historical effectiveness / skill_affinity read (Vol 3 Ch 6) -----------


def test_ranking_reads_skill_affinity_from_profile():
    skills = load_catalog()
    profile = _profile()
    high_affinity_context = _context(skill_affinity={"journaling": 0.95})
    neutral_context = _context(skill_affinity={})
    ranked_high = {r.skill.id: r.score for r in rank_candidates("just thinking out loud", skills, profile, high_affinity_context)}
    ranked_neutral = {r.skill.id: r.score for r in rank_candidates("just thinking out loud", skills, profile, neutral_context)}
    assert ranked_high["journaling"] > ranked_neutral["journaling"]


def test_ranking_context_penalty_deprioritizes_recent_skill():
    skills = load_catalog()
    profile = _profile()
    fresh = rank_candidates("I feel so hurt and alone.", skills, profile, _context())
    penalized = rank_candidates(
        "I feel so hurt and alone.", skills, profile, _context(recent_skill_ids=("validation", "validation"))
    )
    fresh_score = {r.skill.id: r.score for r in fresh}["validation"]
    penalized_score = {r.skill.id: r.score for r in penalized}["validation"]
    assert penalized_score < fresh_score


# --- Skill Observation (Vol 5 Ch 16) ----------------------------------------


def test_skill_observation_resolves_after_next_turn(tmp_path):
    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    mind_state = MindState(stage="supporting", emotion="fear", emotion_confidence=0.8)
    skill = get_skill("grounding")

    mark_skill_used(mind_state, skill=skill, composed_with=None)
    assert mind_state.pending_skill_id == "grounding"

    # Next turn: emotion improved, and the user kept engaging substantively.
    mind_state.emotion = "neutral"
    mind_state.nlp_available = True
    resolve_pending_observation(mind_state, new_user_message="Okay that actually helped a lot, I feel steadier now", store=store)

    assert mind_state.pending_skill_id is None  # cleared after resolving
    observations = store.latest("skill", "grounding")
    assert observations
    assert observations[0].context["emotional_shift"] == "improved"
    assert observations[0].context["user_reaction_signal"] == "continued_engagement"
    assert observations[0].value == 1.0


def test_skill_observation_no_op_when_nothing_pending(tmp_path):
    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    mind_state = MindState()
    resolve_pending_observation(mind_state, new_user_message="hi", store=store)
    assert store.latest("skill", "grounding") == []


# --- Benchmark runner (Vol 5 Ch 6, Invariant 4) -----------------------------


def test_benchmarks_are_discovered():
    cases = discover_benchmarks()
    assert len(cases) == 8
    assert {c.skill_id for c in cases} == {
        "validation", "grounding", "journaling", "cognitive_reframing",
        "boundary_setting", "sleep_hygiene", "crisis_support",
    }


def test_all_skill_benchmarks_pass(engine):
    results = run_benchmarks(engine=engine)
    failed = [r for r in results if not r.passed]
    assert not failed, [(r.case.file.name, r.actual_strategy, r.actual_skill) for r in failed]
