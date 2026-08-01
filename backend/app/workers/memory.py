"""Memory worker — soft store/type/importance signals (formation still durable writer)."""

from app.cognitive.mind_state import MindState
from app.nlp.runtime import OnnxClassifier


def apply_memory(classifier: OnnxClassifier, transcript: str, mind_state: MindState) -> None:
    pred = classifier.predict_memory(transcript)
    mind_state.memory_store = pred.store
    mind_state.memory_type = pred.memory_type if pred.store else None
    mind_state.memory_importance = round(pred.importance, 4)
