"""Phase 0 scheduler: choose fast-path vs full-path cognition."""

from dataclasses import dataclass, field

from app.cognitive.budget import FAST_PATH_BUDGET, FULL_PATH_BUDGET, ThinkingBudget
from app.cognitive.communication import infer_mode, infer_stage, update_question_streak
from app.cognitive.complexity import ComplexityDecision, ComplexityEstimator
from app.cognitive.mind_state import MindState
from app.safety2.worker import SafetyAssessment
from app.workers.runner import NLP_WORKER_NAMES


@dataclass(frozen=True)
class CognitiveTask:
    transcript: str
    complexity: ComplexityDecision
    budget: ThinkingBudget
    workers: list[str] = field(default_factory=list)
    route: str = "llm"


class SafetyNotAssessedError(RuntimeError):
    """Raised when the Scheduler is asked to run ordinary cognition for a
    turn the Safety Worker did not clear (Book Vol 6 Ch4/Invariant 1) — the
    Safety Worker is the one mandatory worker that must never be skipped or
    cached, and this makes bypassing it a structural (call-signature-level)
    impossibility rather than merely a documented convention: `schedule()`
    cannot even be called without a `SafetyAssessment` in hand, and refuses
    to proceed with ordinary scheduling unless that assessment's `route` is
    "ordinary"."""


class CognitiveScheduler:
    def __init__(self, estimator: ComplexityEstimator | None = None):
        self.estimator = estimator or ComplexityEstimator()

    def schedule(self, transcript: str, mind_state: MindState, safety: SafetyAssessment, session_summary: str = "") -> CognitiveTask:
        """Decides complexity/budget/workers only. Stage/mode/etc. are NOT
        set here anymore — call `finalize_communication_state` after the
        NLP workers this returns have actually run, so stage/mode inference
        can use this turn's real classifier signal instead of only keyword
        heuristics (see communication.py's `infer_stage`/`infer_mode`).

        `safety` must be a `SafetyAssessment` already produced by
        `SafetyWorker.assess()` for THIS transcript — see
        `SafetyNotAssessedError`."""
        if safety.route != "ordinary":
            raise SafetyNotAssessedError(
                f"refusing to run ordinary scheduling for a non-ordinary safety route ({safety.route!r}, "
                f"category={safety.category!r}) — route through the safety response path instead"
            )
        complexity = self.estimator.estimate(transcript, prior_context=session_summary)
        budget = FULL_PATH_BUDGET if complexity.level == "full_path" else FAST_PATH_BUDGET
        # Computed from the PRIOR turn's reply before it's overwritten below —
        # see app/cognitive/communication.py's question-frequency rule.
        mind_state.consecutive_question_turns = update_question_streak(
            mind_state.last_assistant_message, mind_state.consecutive_question_turns
        )
        mind_state.complexity_level = complexity.level
        mind_state.thinking_budget_mode = budget.mode
        mind_state.turn_count += 1
        mind_state.last_user_message = transcript

        # NLP heads run on full_path only (plan); fast_path stays heuristic-only.
        nlp = list(NLP_WORKER_NAMES) if complexity.level == "full_path" else []
        workers = [*nlp, "prompt_builder", "llm", "response_composer"]
        return CognitiveTask(
            transcript=transcript,
            complexity=complexity,
            budget=budget,
            workers=workers,
        )

    def finalize_communication_state(self, transcript: str, mind_state: MindState) -> None:
        """Call after NlpWorkerRunner has (maybe) populated mind_state for
        this turn. Stage/mode selection prefers the real hearth_ai
        classifier signal (mind_state.intent/emotion) when it's actually
        available this turn, falling back to keyword heuristics for
        fast_path turns or whenever the model is unavailable/unsure."""
        intent = mind_state.intent if mind_state.nlp_available else None
        emotion = mind_state.emotion if mind_state.nlp_available else None
        mind_state.stage = infer_stage(
            transcript,
            mind_state.complexity_level,
            intent=intent,
            intent_confidence=mind_state.intent_confidence,
        )
        mind_state.communication_mode = infer_mode(
            transcript,
            mind_state.stage,
            emotion=emotion,
            emotion_confidence=mind_state.emotion_confidence,
        )
        mind_state.question_frequency = "low" if mind_state.stage in {"greeting", "listening", "closing"} else "moderate"
        mind_state.verbosity = "short" if mind_state.complexity_level == "fast_path" else "balanced"
        mind_state.support_level = "high" if mind_state.stage in {"supporting", "planning"} else "medium"
