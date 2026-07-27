"""Mandatory safety worker for phase 4."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

import yaml

from app.memory import long_term
from app.onboarding.profile_schema import UserProfile
from app.relationship.engine import RelationshipState
from app.safety import crisis_detector


@dataclass(frozen=True)
class SafetyAssessment:
    category: str
    confidence: float
    signals: dict[str, bool] = field(default_factory=dict)
    route: str = "ordinary"
    response_key: str = "normal"
    notes: str = ""


class SafetyWorker:
    """Runs on every message; cannot be skipped or cached."""

    _DEPENDENCY_RE = re.compile(r"\b(you are all i have|don't leave me|need you all the time|can't cope without you)\b", re.I)
    _HARM_OTHERS_RE = re.compile(r"\b(hurt someone|hurt them|harm them|hurt my partner|hurt my friend|hurt my boss)\b", re.I)
    _DISTRESS_RE = re.compile(r"\b(panic|can't breathe|overwhelmed|spiraling|racing thoughts|numb|terrified)\b", re.I)
    _CLINICAL_REQUEST_RE = re.compile(r"\b(diagnose|therapy plan|treatment plan|medication advice|clinical)\b", re.I)

    def assess(self, user_id: str, transcript: str, profile: UserProfile, relationship: RelationshipState | None = None) -> SafetyAssessment:
        text = transcript.strip()
        acute = crisis_detector.detect(text)
        signals = {
            "acute_self_risk": acute is not None,
            "distress": bool(self._DISTRESS_RE.search(text)),
            "dependency": bool(self._DEPENDENCY_RE.search(text)),
            "harm_others": bool(self._HARM_OTHERS_RE.search(text)),
            "clinical_request": bool(self._CLINICAL_REQUEST_RE.search(text)),
        }
        confidence = 0.0
        category = "none"
        route = "ordinary"
        response_key = "normal"
        notes = "no safety concerns"

        if acute is not None:
            category = "acute_self_risk"
            confidence = 1.0
            route = "crisis_support"
            response_key = "crisis_support"
            notes = "crisis detector matched"
        elif signals["harm_others"]:
            category = "disclosed_harm_to_others"
            confidence = 0.8
            route = "deescalate_and_respond"
            response_key = "harm_others"
            notes = "possible disclosure of harm to others"
        elif signals["dependency"] or (
            relationship is not None and relationship.general_trust > 0.2 and relationship.boundaries == "firm"
        ):
            category = "dependency_attachment"
            confidence = 0.7 if signals["dependency"] else 0.45
            route = "gentle_autonomy"
            response_key = "dependency"
            notes = "dependency or attachment pattern"
        elif signals["clinical_request"]:
            category = "out_of_scope_clinical"
            confidence = 0.55
            route = "decline_and_redirect"
            response_key = "clinical_boundary"
            notes = "clinical request detected"
        elif signals["distress"]:
            category = "acute_distress"
            confidence = 0.5
            route = "grounding"
            response_key = "grounding"
            notes = "distress keywords detected"

        return SafetyAssessment(category=category, confidence=confidence, signals=signals, route=route, response_key=response_key, notes=notes)

    def load_resources(self, region: str | None = None) -> dict:
        root = Path(__file__).resolve().parent / "resources"
        data = yaml.safe_load((root / "global.yaml").read_text(encoding="utf-8"))
        resources = list(data.get("resources", []))
        if region:
            region_file = root / "regions" / f"{region}.yaml"
            if region_file.exists():
                region_data = yaml.safe_load(region_file.read_text(encoding="utf-8")) or {}
                resources.extend(region_data.get("resources", []))
        return {"last_updated": data.get("last_updated"), "source_notes": data.get("source_notes"), "resources": resources}

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

