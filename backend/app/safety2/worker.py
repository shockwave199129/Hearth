"""Mandatory safety worker (Book Vol 6 Ch 4) — runs on every message,
never skipped or cached. Combines four independent signal layers rather
than resting on any single one (Design Goal 2/Invariant 2):

  1. Rule-based patterns — `app.safety.crisis_detector` (high-precision,
     phrase-based) plus this module's own regex signals.
  2. A dedicated-classifier stand-in — reuses Volume 1's existing, already-
     trained, non-LLM Emotion classifier as a corroborating signal. This is
     NOT the "dedicated safety classifier trained specifically for risk
     categories" Ch4 calls for — that requires real clinical training data
     and professional validation this codebase cannot fabricate. It is
     documented here as an interim proxy, and is scoped to only ever raise
     the `acute_distress` category, never assert `acute_self_risk` on its
     own.
  3. Contextual signals from Volume 3/4 — `AttachmentSignals` (Vol 3 Ch4,
     computed from actual recent message content by
     `app.relationship.state.evaluate_attachment_signals`), not raw profile
     state. This is the fix for the previous false trigger, which escalated
     purely from `general_trust > 0.2 and boundaries == "firm"` with no
     signal in the current message at all.
  4. The LLM's own read (`app.safety2.llm_signal`) — one corroborating
     signal that can only RAISE an assessment, never override a high
     rule/classifier signal downward (Design Goal 3/Invariant 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import re

import yaml

from app.onboarding.profile_schema import UserProfile
from app.relationship.engine import RelationshipState
from app.relationship.state import AttachmentSignals
from app.safety import crisis_detector
from app.safety2.llm_signal import get_llm_risk_signal

RESOURCES_ROOT = Path(__file__).resolve().parent / "resources"

# Layer 2 (emotion-classifier proxy) — deliberately conservative: only ever
# raises acute_distress, never asserts acute_self_risk.
_HIGH_RISK_EMOTIONS = {"fear", "grief", "sadness", "anxiety"}
EMOTION_SIGNAL_CONFIDENCE_FLOOR = 0.75
EMOTION_SIGNAL_WEIGHT = 0.5

# Vol 7 Ch 8's combined Attachment score, feeding this worker's dependency
# category alongside the message/regex-based signals above.
ATTACHMENT_SCORE_CONFIDENCE_FLOOR = 0.5

# Layer 4 (LLM corroboration) thresholds — graduated, per Ch10's
# confidence-banded response model.
LLM_RAISE_TO_DISTRESS_THRESHOLD = 0.5
LLM_RAISE_TO_CRISIS_THRESHOLD = 0.85
LLM_DISTRESS_WEIGHT = 0.7


@dataclass(frozen=True)
class SafetyAssessment:
    category: str
    confidence: float
    signals: dict[str, bool] = field(default_factory=dict)
    route: str = "ordinary"
    response_key: str = "normal"
    notes: str = ""


class SafetyWorker:
    """Runs on every message; cannot be skipped or cached — enforced
    structurally by `app.cognitive.scheduler.CognitiveScheduler.schedule`,
    which requires a `SafetyAssessment` argument and refuses to run
    ordinary scheduling for a non-"ordinary" one."""

    _DEPENDENCY_RE = re.compile(r"\b(you are all i have|don't leave me|need you all the time|can't cope without you)\b", re.I)
    _HARM_OTHERS_RE = re.compile(r"\b(hurt someone|hurt them|harm them|hurt my partner|hurt my friend|hurt my boss)\b", re.I)
    _DISTRESS_RE = re.compile(r"\b(panic|can't breathe|overwhelmed|spiraling|racing thoughts|numb|terrified)\b", re.I)
    _CLINICAL_REQUEST_RE = re.compile(r"\b(diagnose|therapy plan|treatment plan|medication advice|clinical)\b", re.I)

    def assess(
        self,
        user_id: str,
        transcript: str,
        profile: UserProfile,
        relationship: RelationshipState | None = None,
        *,
        attachment_signals: AttachmentSignals | None = None,
        emotion: str = "unknown",
        emotion_confidence: float = 0.0,
        llm: Any | None = None,
    ) -> SafetyAssessment:
        text = transcript.strip()

        # Layer 1: rule-based.
        acute = crisis_detector.detect(text)
        signals = {
            "acute_self_risk": acute is not None,
            "distress": bool(self._DISTRESS_RE.search(text)),
            "dependency_message": bool(self._DEPENDENCY_RE.search(text)),
            "harm_others": bool(self._HARM_OTHERS_RE.search(text)),
            "clinical_request": bool(self._CLINICAL_REQUEST_RE.search(text)),
            # Layer 3: contextual (Vol 3's Attachment Model), message-content
            # derived — never raw profile/trust state alone.
            "attachment_pattern": bool(attachment_signals is not None and attachment_signals.has_warning_signal),
            # Vol 7 Ch 8's three-stream combined score, finally feeding this
            # escalation the way Vol 3 Ch 4 always said it eventually would —
            # a graduated signal (Ch 10's confidence-banded response model),
            # distinct from the boolean has_warning_signal above.
            "attachment_score_elevated": bool(
                attachment_signals is not None and attachment_signals.combined_score >= ATTACHMENT_SCORE_CONFIDENCE_FLOOR
            ),
            # Layer 2: emotion-classifier proxy.
            "high_intensity_negative_emotion": (
                emotion in _HIGH_RISK_EMOTIONS and emotion_confidence >= EMOTION_SIGNAL_CONFIDENCE_FLOOR
            ),
        }

        # Layer 4: LLM corroboration — computed regardless of what the other
        # layers found, so it can independently catch what they missed.
        llm_score = get_llm_risk_signal(text, llm)
        signals["llm_corroboration"] = llm_score >= LLM_RAISE_TO_DISTRESS_THRESHOLD

        category = "none"
        confidence = 0.0
        route = "ordinary"
        response_key = "normal"
        notes = "no safety concerns"

        if signals["acute_self_risk"]:
            category, confidence, route, response_key = "acute_self_risk", 1.0, "crisis_support", "crisis_support"
            notes = "crisis detector matched (rule-based)"
        elif signals["harm_others"]:
            category, confidence, route, response_key = "disclosed_harm_to_others", 0.8, "deescalate_and_respond", "harm_others"
            notes = "possible disclosure of harm to others"
        elif signals["dependency_message"] or signals["attachment_pattern"] or signals["attachment_score_elevated"]:
            # All three require an actual signal derived from THIS message,
            # recent message content, or the Vol 7 Ch8 combined score
            # (itself computed from real conversation history) — never
            # profile trust/boundary state alone.
            if signals["dependency_message"]:
                confidence, notes = 0.7, "dependency language in this message"
            elif signals["attachment_pattern"]:
                confidence, notes = 0.45, "attachment warning pattern from recent messages"
            else:
                confidence = round(0.4 + 0.3 * attachment_signals.combined_score, 3)
                notes = f"attachment combined_score elevated ({attachment_signals.combined_score:.2f})"
            category, route, response_key = "dependency_attachment", "gentle_autonomy", "dependency"
        elif signals["clinical_request"]:
            category, confidence, route, response_key = "out_of_scope_clinical", 0.55, "decline_and_redirect", "clinical_boundary"
            notes = "clinical request detected"
        elif signals["distress"] or signals["high_intensity_negative_emotion"]:
            base = 0.5 if signals["distress"] else 0.0
            emotion_boost = EMOTION_SIGNAL_WEIGHT if signals["high_intensity_negative_emotion"] else 0.0
            confidence = min(1.0, base + emotion_boost)
            category, route, response_key = "acute_distress", "grounding", "grounding"
            notes = "distress signal (rule-based and/or emotion-classifier proxy)"

        # LLM corroboration can only RAISE an assessment, never lower one a
        # rule/classifier layer already made — and never asserts the
        # highest-stakes category entirely on its own unless its score
        # clears a materially higher bar than the "raise to distress" band.
        if category == "none":
            if llm_score >= LLM_RAISE_TO_CRISIS_THRESHOLD:
                category, confidence, route, response_key = "acute_self_risk", llm_score, "crisis_support", "crisis_support"
                notes = "elevated to crisis by LLM corroboration signal alone — no rule/classifier match"
            elif llm_score >= LLM_RAISE_TO_DISTRESS_THRESHOLD:
                category, confidence, route, response_key = "acute_distress", round(llm_score * LLM_DISTRESS_WEIGHT, 3), "grounding", "grounding"
                notes = "elevated by LLM corroboration signal alone"
        elif category == "acute_distress" and llm_score >= LLM_RAISE_TO_CRISIS_THRESHOLD:
            category, confidence, route, response_key = "acute_self_risk", max(confidence, llm_score), "crisis_support", "crisis_support"
            notes += "; escalated to crisis by strong LLM corroboration"

        return SafetyAssessment(category=category, confidence=confidence, signals=signals, route=route, response_key=response_key, notes=notes)

    def load_resources(self, region: str | None = None) -> dict:
        data = yaml.safe_load((RESOURCES_ROOT / "global.yaml").read_text(encoding="utf-8"))
        resources = list(data.get("resources", []))
        if region:
            region_file = RESOURCES_ROOT / "regions" / f"{region}.yaml"
            if region_file.exists():
                region_data = yaml.safe_load(region_file.read_text(encoding="utf-8")) or {}
                # Region-specific entries first — Vol 6 Ch7's region-aware
                # selection, falling back to global only when unavailable.
                resources = list(region_data.get("resources", [])) + resources
        return {"last_updated": data.get("last_updated"), "source_notes": data.get("source_notes"), "resources": resources}

    def resources_are_stale(self, region: str | None = None) -> bool:
        """Vol 6 Ch7: a resource store that hasn't been reviewed recently
        should be flagged internally, not silently keep serving."""
        data = yaml.safe_load((RESOURCES_ROOT / "global.yaml").read_text(encoding="utf-8"))
        last_updated = data.get("last_updated")
        stale_after_days = int(data.get("stale_after_days", 180))
        if not last_updated:
            return True
        age_days = (datetime.now(timezone.utc).date() - datetime.fromisoformat(last_updated).date()).days
        return age_days > stale_after_days

    def log(self, user_id: str, assessment: SafetyAssessment, response_taken: str, retention_days: int = 30) -> None:
        from app.safety2.audit import SafetyAuditEntry, record

        now = datetime.now(timezone.utc)
        record(
            SafetyAuditEntry(
                user_id=user_id,
                timestamp=now.isoformat(),
                category=assessment.category,
                confidence_signals=assessment.signals,
                response_taken=response_taken,
                outcome_notes=assessment.notes,
                retention_expiry=(now + timedelta(days=retention_days)).isoformat(),
            )
        )
