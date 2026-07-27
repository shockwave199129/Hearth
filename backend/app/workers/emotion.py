"""Emotion worker — GoEmotions → Hearth display emotion on MindState."""

from app.cognitive.mind_state import MindState
from app.nlp.runtime import OnnxClassifier


def apply_emotion(classifier: OnnxClassifier, transcript: str, mind_state: MindState) -> None:
    pred = classifier.predict_emotion(transcript)
    mind_state.emotion = pred.emotion
    mind_state.emotion_confidence = round(pred.confidence, 4)
