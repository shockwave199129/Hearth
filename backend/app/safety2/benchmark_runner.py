"""Safety benchmark runner (Book Vol 6 Ch 13) — a stricter bar than
Volume 5's skill benchmarks (Ch 6): a safety regression is release-blocking,
not a quality issue to schedule for later. Two distinct, equally important
suites:

  - Per-category detection recall — `benchmarks/<category>/*.yaml` — does a
    known-risk scenario for each Vol 6 Ch 3 category actually get flagged
    and routed correctly? Recall is weighted more heavily than precision
    here (Ch 5's sensitivity-first posture).
  - Over-escalation — `benchmarks/non_crisis/*.yaml` — an equally-sized set
    of ordinary venting/dark-humor/distress conversations that must NOT
    escalate, measuring the other failure direction (Ch 10).

This exercises `SafetyWorker.assess()` directly — the real rule-based and
contextual layers. The emotion-classifier and LLM-corroboration layers are
exercised only when a case opts in (`emotion`/`emotion_confidence` or
`llm_risk_score` fields), so this runs fast and hermetically by default
without requiring a live model.

    python -m app.safety2.benchmark_runner
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from app.benchmarks import SAFETY_BENCHMARKS_ROOT
from app.onboarding.profile_schema import UserProfile
from app.relationship.state import AttachmentSignals
from app.safety2.worker import SafetyWorker

# Attachment signal fixture used by cases that set `attachment_signal: true`
# — exercises the contextual (Vol 3) layer independent of the message-level
# regex, since Vol 3's AttachmentSignals is itself derived from recent
# message content elsewhere (app.relationship.state.evaluate_attachment_signals).
_WARNING_ATTACHMENT_SIGNALS = AttachmentSignals(
    replacement_language_detected=True,
    escalating_contact_frequency=False,
    distress_about_unavailability=False,
    healthy_engagement_with_others=False,
)


@dataclass(frozen=True)
class SafetyBenchmarkCase:
    category_dir: str
    file: Path
    input: str
    expected_category: str
    expected_route: str
    attachment_signal: bool
    emotion: str
    emotion_confidence: float
    llm_risk_score: float | None


@dataclass(frozen=True)
class SafetyBenchmarkResult:
    case: SafetyBenchmarkCase
    passed: bool
    actual_category: str
    actual_route: str


def discover_benchmarks() -> list[SafetyBenchmarkCase]:
    cases: list[SafetyBenchmarkCase] = []
    if not SAFETY_BENCHMARKS_ROOT.exists():
        return cases
    for category_dir in sorted(p for p in SAFETY_BENCHMARKS_ROOT.iterdir() if p.is_dir()):
        for bench_file in sorted(category_dir.glob("*.yaml")):
            data = yaml.safe_load(bench_file.read_text(encoding="utf-8")) or {}
            cases.append(
                SafetyBenchmarkCase(
                    category_dir=category_dir.name,
                    file=bench_file,
                    input=data["input"],
                    expected_category=data["expected_category"],
                    expected_route=data["expected_route"],
                    attachment_signal=bool(data.get("attachment_signal", False)),
                    emotion=data.get("emotion", "unknown"),
                    emotion_confidence=float(data.get("emotion_confidence", 0.0)),
                    llm_risk_score=data.get("llm_risk_score"),
                )
            )
    return cases


def _default_profile() -> UserProfile:
    from datetime import datetime, timezone

    return UserProfile(user_id="safety-benchmark", name="Benchmark", companion_name="Hearth", created_at=datetime.now(timezone.utc))


class _FixedScoreLlm:
    """Test double for the LLM-corroboration layer — returns a fixed score
    string instead of calling a real model."""

    def __init__(self, score: float):
        self.score = score

    def complete(self, prompt: str, max_tokens: int = 8, temperature: float = 0.0) -> str:
        return str(self.score)


def run_benchmarks(worker: SafetyWorker | None = None, profile: UserProfile | None = None) -> list[SafetyBenchmarkResult]:
    worker = worker or SafetyWorker()
    profile = profile or _default_profile()
    results: list[SafetyBenchmarkResult] = []
    for case in discover_benchmarks():
        attachment_signals = _WARNING_ATTACHMENT_SIGNALS if case.attachment_signal else None
        llm = _FixedScoreLlm(case.llm_risk_score) if case.llm_risk_score is not None else None
        assessment = worker.assess(
            "safety-benchmark",
            case.input,
            profile,
            attachment_signals=attachment_signals,
            emotion=case.emotion,
            emotion_confidence=case.emotion_confidence,
            llm=llm,
        )
        passed = assessment.category == case.expected_category and assessment.route == case.expected_route
        results.append(
            SafetyBenchmarkResult(case=case, passed=passed, actual_category=assessment.category, actual_route=assessment.route)
        )
    return results


def summarize(results: list[SafetyBenchmarkResult]) -> str:
    lines = []
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"[{status}] {r.case.category_dir}/{r.case.file.name}: "
            f"expected category={r.case.expected_category!r} route={r.case.expected_route!r} -> "
            f"got category={r.actual_category!r} route={r.actual_route!r}"
        )
    passed = sum(1 for r in results if r.passed)
    non_crisis_results = [r for r in results if r.case.category_dir == "non_crisis"]
    over_escalations = [r for r in non_crisis_results if not r.passed]
    lines.append(f"\n{passed}/{len(results)} passed")
    lines.append(f"over-escalation rate: {len(over_escalations)}/{len(non_crisis_results)} non-crisis cases incorrectly escalated")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(run_benchmarks()))
