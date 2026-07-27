"""Strategy worker — intervention hint for PromptBuilder context only."""

from app.cognitive.mind_state import MindState
from app.nlp.runtime import OnnxClassifier


def apply_strategy(classifier: OnnxClassifier, transcript: str, mind_state: MindState) -> None:
    pred = classifier.predict_strategy(transcript)
    mind_state.strategy_hint = pred.strategy
    mind_state.strategy_confidence = round(pred.confidence, 4)
