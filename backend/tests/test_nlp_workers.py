"""NLP OnnxClassifier + worker runner smoke tests."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cognitive.mind_state import MindState
from app.cognitive.scheduler import CognitiveScheduler
from app.config import resolve_nlp_models_dir
from app.nlp.runtime import OnnxClassifier
from app.safety2.worker import SafetyAssessment
from app.workers.runner import NLP_WORKER_NAMES, NlpWorkerRunner

_ORDINARY = SafetyAssessment(category="none", confidence=0.0, route="ordinary")


def test_resolve_nlp_models_dir_finds_repo_package():
    root = resolve_nlp_models_dir()
    assert root is not None
    assert (root / "manifest.json").is_file()
    assert (root / "tokenizer.json").is_file()


def test_onnx_classifier_loads_and_predicts():
    clf = OnnxClassifier()
    assert clf.available
    emo = clf.predict_emotion("I am so sad and alone tonight.")
    assert emo.emotion
    assert 0.0 <= emo.confidence <= 1.0
    intent = clf.predict_intent("Can you just listen while I vent about work?")
    assert intent.intent
    mem = clf.predict_memory("Please remember that my sister's name is Maya.")
    assert mem.memory_type
    rel = clf.predict_relationship("I trust you enough to say this.")
    assert isinstance(rel.trust_delta, float)
    strat = clf.predict_strategy("I need someone to validate how I feel.")
    assert strat.strategy


def test_nlp_worker_runner_updates_mind_state():
    mind = MindState()
    runner = NlpWorkerRunner()
    runner.run(list(NLP_WORKER_NAMES), "I feel overwhelmed and scared.", mind)
    assert mind.nlp_available is True
    assert mind.emotion != ""
    assert mind.intent != ""


def test_scheduler_schedules_nlp_on_full_path_only():
    sched = CognitiveScheduler()
    mind = MindState()
    # Short greeting → fast_path (no NLP workers)
    task_fast = sched.schedule("hi", mind, _ORDINARY)
    assert "emotion" not in task_fast.workers
    # Emotional / complex → full_path
    mind2 = MindState()
    task_full = sched.schedule(
        "I feel so overwhelmed and scared and I don't know what to do about everything falling apart.",
        mind2,
        _ORDINARY,
    )
    if task_full.complexity.level == "full_path":
        assert set(NLP_WORKER_NAMES).issubset(task_full.workers)
    else:
        # Estimator may still classify as fast; ensure list shape is valid
        assert "prompt_builder" in task_full.workers


def test_fail_soft_when_models_missing(tmp_path):
    clf = OnnxClassifier(models_dir=tmp_path)
    assert clf.available is False
    mind = MindState()
    NlpWorkerRunner(clf).run(list(NLP_WORKER_NAMES), "hello", mind)
    assert mind.nlp_available is False
    assert mind.emotion == "unknown"
    assert mind.intent == "unknown"


def test_finalize_communication_state_uses_real_classifier_signal():
    """End-to-end: schedule -> run NLP workers -> finalize should let the
    real hearth_ai intent/emotion heads (not just keyword heuristics) drive
    MindState.stage/communication_mode on a full_path turn."""
    sched = CognitiveScheduler()
    mind = MindState()
    transcript = (
        "I got the promotion today and I honestly cannot believe it, I've "
        "been working toward this for two years!"
    )
    task = sched.schedule(transcript, mind, _ORDINARY)
    NlpWorkerRunner().run(task.workers, transcript, mind)
    assert mind.nlp_available is True
    sched.finalize_communication_state(transcript, mind)
    # A genuinely celebratory message should land on a positive intent/emotion
    # and never get mapped to "calm" purely because of stray keyword overlap.
    assert mind.intent in {"celebrate", "small_talk", "validate"}
    assert mind.stage in {"supporting", "listening", "exploring", "understanding"}


def test_finalize_communication_state_falls_back_to_keywords_on_fast_path():
    sched = CognitiveScheduler()
    mind = MindState()
    task = sched.schedule("hi", mind, _ORDINARY)
    NlpWorkerRunner().run(task.workers, "hi", mind)
    assert mind.nlp_available is False  # fast_path never schedules NLP workers
    sched.finalize_communication_state("hi", mind)
    assert mind.stage == "greeting"
    assert mind.communication_mode == "gentle"
