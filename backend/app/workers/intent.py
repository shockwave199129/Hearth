"""Intent / Goal worker — companion need → MindState.intent (+ goal)."""

from app.cognitive.mind_state import MindState
from app.nlp.labels import INTENT_TO_GOAL
from app.nlp.runtime import OnnxClassifier


def apply_intent(classifier: OnnxClassifier, transcript: str, mind_state: MindState) -> None:
    pred = classifier.predict_intent(transcript)
    mind_state.intent = pred.intent
    mind_state.intent_confidence = round(pred.confidence, 4)
    if pred.intent != "unknown" and pred.confidence >= 0.25:
        mind_state.goal = INTENT_TO_GOAL.get(pred.intent, mind_state.goal)
