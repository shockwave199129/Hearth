"""Run Emotion / Intent / Memory / Relationship / Strategy workers into MindState."""

from __future__ import annotations

import logging

from app.cognitive.mind_state import MindState
from app.nlp.labels import INTENT_TO_GOAL
from app.nlp.runtime import OnnxClassifier

logger = logging.getLogger(__name__)

NLP_WORKER_NAMES = ("emotion", "intent", "memory", "relationship", "strategy")


def _reset_nlp_fields(mind_state: MindState) -> None:
    mind_state.emotion = "unknown"
    mind_state.emotion_confidence = 0.0
    mind_state.intent = "unknown"
    mind_state.intent_confidence = 0.0
    mind_state.memory_store = False
    mind_state.memory_type = None
    mind_state.memory_importance = 0.0
    mind_state.relationship_trust_delta = 0.0
    mind_state.relationship_vulnerability = 0.0
    mind_state.relationship_openness = 0.0
    mind_state.relationship_comfort = 0.0
    mind_state.strategy_hint = None
    mind_state.strategy_confidence = 0.0
    mind_state.nlp_available = False


class NlpWorkerRunner:
    """Dispatch scheduled NLP workers; never raises into the turn path."""

    def __init__(self, classifier: OnnxClassifier | None = None):
        self.classifier = classifier if classifier is not None else OnnxClassifier()

    def run(self, workers: list[str], transcript: str, mind_state: MindState) -> None:
        nlp_workers = [w for w in workers if w in NLP_WORKER_NAMES]
        if not nlp_workers:
            return
        try:
            self._run_inner(nlp_workers, transcript, mind_state)
        except Exception:
            logger.exception("NLP workers failed — applying fail-soft defaults")
            _reset_nlp_fields(mind_state)

    def _run_inner(self, workers: list[str], transcript: str, mind_state: MindState) -> None:
        if not self.classifier.available:
            _reset_nlp_fields(mind_state)
            return

        mind_state.nlp_available = True

        # One call: runs the shared encoder once for every requested task
        # when the loaded models use graph_kind=="head_only" (nlp-track),
        # instead of one full encoder+head pass per task.
        preds = self.classifier.predict_all(transcript, workers)

        if "emotion" in preds:
            pred = preds["emotion"]
            mind_state.emotion = pred.emotion
            mind_state.emotion_confidence = round(pred.confidence, 4)

        if "intent" in preds:
            pred = preds["intent"]
            mind_state.intent = pred.intent
            mind_state.intent_confidence = round(pred.confidence, 4)
            if pred.intent != "unknown" and pred.confidence >= 0.25:
                mind_state.goal = INTENT_TO_GOAL.get(pred.intent, mind_state.goal)

        if "memory" in preds:
            pred = preds["memory"]
            mind_state.memory_store = pred.store
            mind_state.memory_type = pred.memory_type if pred.store else None
            mind_state.memory_importance = round(pred.importance, 4)

        if "relationship" in preds:
            pred = preds["relationship"]
            mind_state.relationship_trust_delta = round(pred.trust_delta, 4)
            mind_state.relationship_vulnerability = round(pred.vulnerability, 4)
            mind_state.relationship_openness = round(pred.openness, 4)
            mind_state.relationship_comfort = round(pred.comfort, 4)

        if "strategy" in preds:
            pred = preds["strategy"]
            mind_state.strategy_hint = pred.strategy
            mind_state.strategy_confidence = round(pred.confidence, 4)
