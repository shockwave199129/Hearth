"""Relationship worker — soft deltas only (Growth Engine owns durable state)."""

from app.cognitive.mind_state import MindState
from app.nlp.runtime import OnnxClassifier


def apply_relationship(classifier: OnnxClassifier, transcript: str, mind_state: MindState) -> None:
    pred = classifier.predict_relationship(transcript)
    mind_state.relationship_trust_delta = round(pred.trust_delta, 4)
    mind_state.relationship_vulnerability = round(pred.vulnerability, 4)
    mind_state.relationship_openness = round(pred.openness, 4)
    mind_state.relationship_comfort = round(pred.comfort, 4)
