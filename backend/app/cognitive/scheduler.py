"""Phase 0 scheduler: choose fast-path vs full-path cognition."""

from dataclasses import dataclass, field

from app.cognitive.budget import FAST_PATH_BUDGET, FULL_PATH_BUDGET, ThinkingBudget
from app.cognitive.complexity import ComplexityDecision, ComplexityEstimator
from app.cognitive.mind_state import MindState
from app.workers.runner import NLP_WORKER_NAMES


@dataclass(frozen=True)
class CognitiveTask:
    transcript: str
    complexity: ComplexityDecision
    budget: ThinkingBudget
    workers: list[str] = field(default_factory=list)
    route: str = "llm"


class CognitiveScheduler:
    def __init__(self, estimator: ComplexityEstimator | None = None):
        self.estimator = estimator or ComplexityEstimator()

    def schedule(self, transcript: str, mind_state: MindState, session_summary: str = "") -> CognitiveTask:
        complexity = self.estimator.estimate(transcript, prior_context=session_summary)
        budget = FULL_PATH_BUDGET if complexity.level == "full_path" else FAST_PATH_BUDGET
        mind_state.complexity_level = complexity.level
        mind_state.thinking_budget_mode = budget.mode
        mind_state.stage = self._infer_stage(transcript, complexity.level)
        mind_state.communication_mode = self._infer_mode(transcript, mind_state.stage)
        mind_state.question_frequency = "low" if mind_state.stage in {"greeting", "listening", "closing"} else "moderate"
        mind_state.verbosity = "short" if complexity.level == "fast_path" else "balanced"
        mind_state.support_level = "high" if mind_state.stage in {"supporting", "planning"} else "medium"
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

    def _infer_stage(self, transcript: str, complexity_level: str) -> str:
        text = transcript.strip().lower()
        if not text:
            return "listening"
        if any(greet in text for greet in ("good morning", "good evening", "hi", "hello", "hey")) and len(text.split()) <= 5:
            return "greeting"
        if any(end in text for end in ("bye", "goodbye", "talk later", "see you")):
            return "closing"
        if complexity_level == "fast_path":
            return "listening"
        if "?" in text:
            return "understanding"
        if any(token in text for token in ("what should", "what do i do", "help me plan", "next step", "should i")):
            return "planning"
        if any(token in text for token in ("need", "want", "should", "help", "please")):
            return "supporting"
        return "exploring"

    def _infer_mode(self, transcript: str, stage: str) -> str:
        if stage in {"greeting", "closing"}:
            return "gentle"
        if any(token in transcript.lower() for token in ("panic", "overwhelmed", "can't", "cannot", "stuck", "lost")):
            return "calm"
        return "warm"
