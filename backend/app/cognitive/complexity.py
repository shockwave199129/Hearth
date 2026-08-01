"""Deterministic conversation complexity estimation."""

from dataclasses import dataclass
import re


_QUESTION_RE = re.compile(r"[?]")
_EMOTION_RE = re.compile(
    r"\b(anxious|panic|panicking|overwhelmed|stressed|sad|lonely|angry|hurt|upset|scared|afraid|hopeless|panic attack)\b",
    re.IGNORECASE,
)
_DEEP_RE = re.compile(
    r"\b(because|always|never|relationship|job|family|friend|work|future|meaning|worthless|can't cope|don't know what to do)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ComplexityDecision:
    level: str
    score: float
    reason: str


class ComplexityEstimator:
    def estimate(self, text: str, prior_context: str = "") -> ComplexityDecision:
        normalized = text.strip()
        score = 0.0
        reasons: list[str] = []

        word_count = len(normalized.split())
        if word_count >= 18:
            score += 0.35
            reasons.append("long message")
        if _QUESTION_RE.search(normalized):
            score += 0.12
            reasons.append("question")
        if _EMOTION_RE.search(normalized):
            score += 0.35
            reasons.append("emotion cue")
        if _DEEP_RE.search(normalized):
            score += 0.22
            reasons.append("deeper topic")
        if prior_context.strip():
            score += 0.08
            reasons.append("session continuation")

        level = "fast_path" if score < 0.45 and word_count <= 12 else "full_path"
        reason = ", ".join(reasons) if reasons else "simple check-in"
        return ComplexityDecision(level=level, score=round(min(score, 1.0), 2), reason=reason)

