from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.intervention.engine import InterventionEngine
from app.onboarding.profile_schema import UserProfile
from app.skills.loader import get_skill, load_catalog


def _profile() -> UserProfile:
    return UserProfile(
        user_id="u1",
        name="A",
        companion_name="Companion",
        created_at=datetime.now(timezone.utc),
    )


def test_skill_catalog_loads_structured_skills():
    skills = load_catalog()
    ids = {skill.id for skill in skills}
    assert {"validation", "grounding", "journaling", "cognitive_reframing", "boundary_setting", "sleep_hygiene", "crisis_support"}.issubset(ids)


def test_manifests_validate():
    assert get_skill("grounding").manifest.skill_id == "grounding"


def test_intervention_engine_routes_validation():
    engine = InterventionEngine()
    plan = engine.plan("I feel so hurt and alone.", _profile(), "listening")
    assert plan.strategy == "validate"
    assert plan.primary_skill and plan.primary_skill.skill.id == "validation"


def test_intervention_engine_routes_grounding():
    engine = InterventionEngine()
    plan = engine.plan("I can't breathe and everything is spinning.", _profile(), "supporting")
    assert plan.strategy == "ground"
    assert plan.primary_skill and plan.primary_skill.skill.id == "grounding"


def test_intervention_engine_routes_crisis():
    engine = InterventionEngine()
    plan = engine.plan("I want to hurt myself.", _profile(), "supporting", crisis=True)
    assert plan.strategy == "crisis_support"
    assert plan.primary_skill and plan.primary_skill.skill.id == "crisis_support"
