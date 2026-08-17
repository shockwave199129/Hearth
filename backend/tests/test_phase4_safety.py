"""Phase 4 (Book Volume 6 — Safety Framework) tests."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cognitive.mind_state import MindState
from app.cognitive.scheduler import CognitiveScheduler, SafetyNotAssessedError
from app.evaluation.worker import EvaluationWorker, SAFETY_FINDINGS_RETENTION_DAYS
from app.learning.observation_store import ObservationStore
from app.onboarding.profile_schema import UserProfile
from app.relationship.engine import RelationshipState
from app.relationship.state import AttachmentSignals
from app.safety.escalation import EmailNotifier, LoggedNotifier, get_notifier, maybe_escalate
from app.safety2.audit import SafetyAuditEntry, purge_expired, record, retention_policy_disclosure
from app.safety2.benchmark_runner import discover_benchmarks, run_benchmarks
from app.safety2.llm_signal import get_llm_risk_signal
from app.safety2.worker import SafetyWorker


def _profile(**overrides) -> UserProfile:
    defaults = dict(user_id="u1", name="A", companion_name="Companion", created_at=datetime.now(timezone.utc))
    defaults.update(overrides)
    return UserProfile(**defaults)


class _FixedLlm:
    def __init__(self, score: float):
        self.score = score

    def complete(self, prompt: str, max_tokens: int = 8, temperature: float = 0.0) -> str:
        return str(self.score)


# --- Non-skippability (Invariant 1) -----------------------------------------


def test_scheduler_requires_safety_assessment_argument():
    sched = CognitiveScheduler()
    with pytest.raises(TypeError):
        sched.schedule("hi", MindState())


def test_scheduler_refuses_ordinary_scheduling_for_non_ordinary_safety():
    from app.safety2.worker import SafetyAssessment

    sched = CognitiveScheduler()
    crisis = SafetyAssessment(category="acute_self_risk", confidence=1.0, route="crisis_support")
    with pytest.raises(SafetyNotAssessedError):
        sched.schedule("help", MindState(), crisis)


def test_scheduler_proceeds_for_ordinary_safety():
    from app.safety2.worker import SafetyAssessment

    sched = CognitiveScheduler()
    ordinary = SafetyAssessment(category="none", confidence=0.0, route="ordinary")
    task = sched.schedule("hi", MindState(), ordinary)
    assert "prompt_builder" in task.workers


# --- Four detection layers ----------------------------------------------------


def test_layer1_rule_based_crisis_detection():
    worker = SafetyWorker()
    assessment = worker.assess("u1", "I want to kill myself", _profile())
    assert assessment.category == "acute_self_risk"
    assert assessment.confidence == 1.0
    assert assessment.route == "crisis_support"


def test_layer2_emotion_classifier_proxy_raises_acute_distress():
    worker = SafetyWorker()
    assessment = worker.assess("u1", "everything is fine I guess", _profile(), emotion="fear", emotion_confidence=0.9)
    assert assessment.category == "acute_distress"
    assert assessment.signals["high_intensity_negative_emotion"] is True


def test_layer2_low_confidence_emotion_does_not_trigger():
    worker = SafetyWorker()
    assessment = worker.assess("u1", "everything is fine I guess", _profile(), emotion="fear", emotion_confidence=0.2)
    assert assessment.category == "none"


def test_layer3_contextual_attachment_signals_trigger_without_message_keywords():
    worker = SafetyWorker()
    attachment = AttachmentSignals(replacement_language_detected=True, healthy_engagement_with_others=False)
    assessment = worker.assess("u1", "what are we doing today", _profile(), attachment_signals=attachment)
    assert assessment.category == "dependency_attachment"
    assert assessment.signals["dependency_message"] is False
    assert assessment.signals["attachment_pattern"] is True


def test_layer4_llm_can_raise_but_never_asserted_without_score():
    worker = SafetyWorker()
    # No LLM configured at all -> no corroboration signal, no change.
    assessment = worker.assess("u1", "just an ordinary day", _profile(), llm=None)
    assert assessment.category == "none"


def test_layer4_llm_raises_to_distress_band():
    worker = SafetyWorker()
    assessment = worker.assess("u1", "just an ordinary sounding message", _profile(), llm=_FixedLlm(0.6))
    assert assessment.category == "acute_distress"
    assert "LLM" in assessment.notes


def test_layer4_llm_raises_to_crisis_band_alone():
    worker = SafetyWorker()
    assessment = worker.assess("u1", "an indirect but worrying message", _profile(), llm=_FixedLlm(0.9))
    assert assessment.category == "acute_self_risk"
    assert assessment.route == "crisis_support"


def test_layer4_low_llm_score_never_overrides_high_rule_signal():
    """A low LLM read must never override a high rule-based signal (Vol 6
    Ch4/Invariant 2)."""
    worker = SafetyWorker()
    assessment = worker.assess("u1", "I want to kill myself", _profile(), llm=_FixedLlm(0.0))
    assert assessment.category == "acute_self_risk"
    assert assessment.confidence == 1.0


def test_get_llm_risk_signal_fails_soft():
    class Broken:
        def complete(self, *a, **k):
            raise RuntimeError("down")

    assert get_llm_risk_signal("anything", Broken()) == 0.0
    assert get_llm_risk_signal("anything", None) == 0.0


# --- Dependency/attachment false-trigger fix --------------------------------


def test_dependency_no_longer_triggers_from_profile_state_alone():
    """Regression: previously escalated purely from
    `general_trust > 0.2 and boundaries == "firm"` with no signal in the
    current message at all."""
    worker = SafetyWorker()
    relationship = RelationshipState(
        general_trust=0.5, vulnerability_trust=0.1, advice_trust=0.1, consistency_confidence=0.1,
        boundaries="firm", life_model="unknown",
    )
    assessment = worker.assess("u1", "just talking about the weather today", _profile(), relationship)
    assert assessment.category == "none"
    assert assessment.route == "ordinary"


def test_dependency_still_triggers_from_message_content():
    worker = SafetyWorker()
    assessment = worker.assess("u1", "you are all i have, please don't leave me", _profile())
    assert assessment.category == "dependency_attachment"


# --- Resource store (Vol 6 Ch7) ----------------------------------------------


def test_resources_load_global_and_region():
    worker = SafetyWorker()
    global_only = worker.load_resources(None)
    assert global_only["resources"]
    assert all(r.get("verified") for r in global_only["resources"])

    us = worker.load_resources("us")
    ids = [r["id"] for r in us["resources"]]
    assert "us_988_lifeline" in ids
    # Region entries are layered ahead of global fallback.
    assert ids.index("us_988_lifeline") < ids.index(global_only["resources"][0]["id"])

    india = worker.load_resources("in")
    india_ids = [r["id"] for r in india["resources"]]
    assert "in_tele_manas" in india_ids
    assert india_ids.index("in_tele_manas") < india_ids.index(global_only["resources"][0]["id"])


def test_resources_unknown_region_falls_back_to_global():
    worker = SafetyWorker()
    result = worker.load_resources("nowhere")
    assert result["resources"] == worker.load_resources(None)["resources"]


def test_resources_not_stale_today():
    worker = SafetyWorker()
    assert worker.resources_are_stale() is False


# --- Escalation notifier -----------------------------------------------------


def test_get_notifier_email_unconfigured_falls_back_to_logged():
    notifier = get_notifier("email")
    assert isinstance(notifier, LoggedNotifier)


def test_get_notifier_sms_always_logged_stub():
    assert isinstance(get_notifier("sms"), LoggedNotifier)


def test_email_notifier_requires_configuration():
    notifier = EmailNotifier(host="", from_address="")
    assert notifier.is_configured() is False
    with pytest.raises(RuntimeError):
        notifier.send("hi", "email", "someone@example.com")


def test_maybe_escalate_falls_back_gracefully_on_notifier_failure(tmp_path, monkeypatch):
    import app.safety.escalation as escalation_module

    monkeypatch.setattr(escalation_module, "ESCALATION_DB_PATH", tmp_path / "profile.db")

    class AlwaysFailsNotifier:
        def send(self, message, method, value):
            raise RuntimeError("network down")

    class FakeProfile:
        emergency_contact_consent = True
        emergency_contact_value = "someone@example.com"
        emergency_contact_method = "email"
        companion_name = "Hearth"
        name = "Test"

    monkeypatch.setattr(escalation_module, "get_profile", lambda uid: FakeProfile())
    monkeypatch.setattr(escalation_module, "event_count", lambda uid, window: 99)
    # No monkeypatch of last_escalation itself — a fresh tmp_path db has no
    # prior escalations, so the real function naturally returns None here,
    # letting us check its real return value again after the call below.

    # Should not raise even though the notifier itself fails.
    maybe_escalate("u1", reason="test", notifier=AlwaysFailsNotifier())
    last = escalation_module.last_escalation("u1")
    assert last is not None


# --- Evaluation worker retention fix -----------------------------------------


def test_evaluation_worker_safety_findings_retention_is_not_immediate(tmp_path, monkeypatch):
    import app.safety2.audit as audit_module

    monkeypatch.setattr(audit_module, "SAFETY_AUDIT_DB_PATH", tmp_path / "profile.db")

    store = ObservationStore(path=tmp_path / "hearth.duckdb")
    worker = EvaluationWorker(store)
    before = datetime.now(timezone.utc)
    worker.evaluate("u1", "transcript", "reply", safety_findings={"category": "acute_distress"})
    # Reach into the audit table directly to check what was actually stored.
    from app.db.sqlite_models import get_connection

    conn = get_connection(audit_module.SAFETY_AUDIT_DB_PATH)
    try:
        row = conn.execute(
            "SELECT retention_expiry FROM safety_audit WHERE user_id = ? ORDER BY id DESC LIMIT 1", ("u1",)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    expiry = datetime.fromisoformat(row[0])
    # Must be materially in the future, not "now" (the original bug).
    assert expiry > before + timedelta(days=SAFETY_FINDINGS_RETENTION_DAYS - 1)


# --- Audit log: retention disclosure + purge ---------------------------------


def test_retention_policy_disclosure_is_plain_language():
    text = retention_policy_disclosure()
    assert "30 days" in text
    assert "delete" in text.lower()


def test_audit_purge_respects_retention_window(tmp_path, monkeypatch):
    import app.safety2.audit as audit_module

    monkeypatch.setattr(audit_module, "SAFETY_AUDIT_DB_PATH", tmp_path / "profile.db")
    now = datetime.now(timezone.utc)
    record(SafetyAuditEntry(
        user_id="u1", timestamp=now.isoformat(), category="acute_distress", confidence_signals={},
        response_taken="grounding", outcome_notes="test", retention_expiry=(now - timedelta(days=1)).isoformat(),
    ))
    record(SafetyAuditEntry(
        user_id="u1", timestamp=now.isoformat(), category="acute_distress", confidence_signals={},
        response_taken="grounding", outcome_notes="test", retention_expiry=(now + timedelta(days=29)).isoformat(),
    ))
    purged = purge_expired(now)
    assert purged == 1


# --- Safety benchmark suite (Vol 6 Ch13) ------------------------------------


def test_safety_benchmarks_discovered_per_category_and_non_crisis():
    cases = discover_benchmarks()
    categories = {c.category_dir for c in cases}
    assert {
        "acute_self_risk", "acute_distress", "dependency_attachment",
        "disclosed_harm_to_others", "out_of_scope_clinical", "non_crisis",
    } == categories
    crisis_cases = [c for c in cases if c.category_dir != "non_crisis"]
    non_crisis_cases = [c for c in cases if c.category_dir == "non_crisis"]
    # Over-escalation gets an equally-sized benchmark set (Vol 6 Ch13).
    assert len(non_crisis_cases) == len(crisis_cases)


def test_all_safety_benchmarks_pass():
    results = run_benchmarks()
    failed = [r for r in results if not r.passed]
    assert not failed, [(r.case.file.name, r.actual_category, r.actual_route) for r in failed]


def test_zero_over_escalation_on_non_crisis_set():
    results = run_benchmarks()
    non_crisis = [r for r in results if r.case.category_dir == "non_crisis"]
    over_escalated = [r for r in non_crisis if not r.passed]
    assert over_escalated == []
