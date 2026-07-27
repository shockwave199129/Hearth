"""Background evaluation worker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.learning.observation_store import ObservationStore
from app.safety2.audit import SafetyAuditEntry, record as record_safety


@dataclass(frozen=True)
class EvaluationResult:
    invariant_score: float
    anti_pattern_score: float
    success_proxy: float
    notes: str
    timestamp: datetime


class EvaluationWorker:
    def __init__(self, store: ObservationStore | None = None):
        self.store = store or ObservationStore()

    def evaluate(self, user_id: str, transcript: str, reply_text: str, safety_findings: dict | None = None) -> EvaluationResult:
        invariant_score = 1.0
        if not reply_text.strip():
            invariant_score -= 0.4
        if len(reply_text.split()) > 120:
            invariant_score -= 0.2
        anti_pattern_score = 1.0
        if "?" in reply_text and reply_text.count("?") > 1:
            anti_pattern_score -= 0.2
        success_proxy = 0.5
        if any(token in reply_text.lower() for token in ("i'm here", "that makes sense", "i hear you", "we can")):
            success_proxy += 0.3
        notes = "evaluation complete"
        now = datetime.now(timezone.utc)
        self.store.append("communication", "evaluation_invariant", invariant_score, {"transcript": transcript}, "evaluation_worker")
        self.store.append("communication", "evaluation_antipattern", anti_pattern_score, {"reply": reply_text}, "evaluation_worker")
        self.store.append("communication", "evaluation_success", success_proxy, {"reply": reply_text}, "evaluation_worker")
        if safety_findings:
            record_safety(
                SafetyAuditEntry(
                    user_id=user_id,
                    timestamp=now.isoformat(),
                    category=safety_findings.get("category", "evaluation"),
                    confidence_signals=safety_findings,
                    response_taken="evaluation",
                    outcome_notes=notes,
                    retention_expiry=now.isoformat(),
                )
            )
        return EvaluationResult(invariant_score, anti_pattern_score, success_proxy, notes, now)

