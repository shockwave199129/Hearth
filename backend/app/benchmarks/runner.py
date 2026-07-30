"""Unified benchmark runner + release gate (Book Vol 8 Ch 7/9) — the single
entry point that runs all three benchmark suites (skills, safety,
cross_volume) and decides whether a proposed change is allowed to ship.

    python -m app.benchmarks.runner
    python -m app.benchmarks.runner --override "reviewed by <name>, 2026-07-30: known trade-off, see PR #123"

Gate logic (Ch 9), inherited directly from Volume 6's zero-tolerance
standard:
  - Any safety benchmark regression -> BLOCK, no override possible.
  - Any skill benchmark regression -> BLOCK, unless a documented override
    string is supplied.
  - Cross-volume regressions are treated as skill-tier (overridable) unless
    the specific failing turn is itself a safety-category mismatch, in
    which case it's treated as a safety regression (no override) — a
    cross-volume case can exercise both systems, so its severity is
    whichever component it actually broke.

Every decision (pass/block/override) is itself recorded, including which
benchmarks ran, their results, and — for an override — the documented
reasoning (Ch 9: "this creates the same kind of auditability this book has
valued throughout, applied to the release process itself")."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.benchmarks.cross_volume_runner import CrossVolumeCaseResult, run_cross_volume_benchmarks
from app.safety2.benchmark_runner import SafetyBenchmarkResult, run_benchmarks as run_safety_benchmarks
from app.skills.benchmark_runner import BenchmarkResult, run_benchmarks as run_skill_benchmarks


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    blocked_by_safety: bool
    blocked_by_skill: bool
    safety_failures: list[SafetyBenchmarkResult]
    skill_failures: list[BenchmarkResult]
    cross_volume_failures: list[CrossVolumeCaseResult]
    override_reason: str | None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _cross_volume_failure_is_safety_related(case_result: CrossVolumeCaseResult) -> bool:
    return any(
        (not tr.passed) and (tr.actual_safety_category != tr.turn.expect_safety_category or tr.actual_safety_route != tr.turn.expect_safety_route)
        for tr in case_result.turn_results
    )


def gate_release(
    *,
    override_reason: str | None = None,
    intervention_engine=None,
    safety_worker=None,
) -> GateDecision:
    """Re-runs the full unified benchmark suite (Ch 7) and applies Ch 9's
    asymmetric gate. `override_reason` must be a non-empty, specific string
    to override a skill-tier (never safety-tier) block — an empty/missing
    reason is treated as "not overridden".

    `intervention_engine`/`safety_worker` are injectable (default: real,
    production components) so this can run hermetically in tests without a
    live embedding server — see the per-domain runners' own fixtures."""
    safety_results = run_safety_benchmarks(worker=safety_worker)
    skill_results = run_skill_benchmarks(engine=intervention_engine)
    cross_volume_results = run_cross_volume_benchmarks(safety_worker=safety_worker, intervention_engine=intervention_engine)

    safety_failures = [r for r in safety_results if not r.passed]
    skill_failures = [r for r in skill_results if not r.passed]
    cross_volume_failures = [r for r in cross_volume_results if not r.passed]

    cross_volume_safety_failures = [r for r in cross_volume_failures if _cross_volume_failure_is_safety_related(r)]
    cross_volume_skill_failures = [r for r in cross_volume_failures if r not in cross_volume_safety_failures]

    blocked_by_safety = bool(safety_failures or cross_volume_safety_failures)
    blocked_by_skill = bool(skill_failures or cross_volume_skill_failures)

    has_override = bool(override_reason and override_reason.strip())
    allowed = not blocked_by_safety and (not blocked_by_skill or has_override)

    return GateDecision(
        allowed=allowed,
        blocked_by_safety=blocked_by_safety,
        blocked_by_skill=blocked_by_skill and not has_override,
        safety_failures=safety_failures,
        skill_failures=skill_failures,
        cross_volume_failures=cross_volume_failures,
        override_reason=override_reason if has_override else None,
    )


def summarize(decision: GateDecision) -> str:
    lines = [f"Release gate evaluated at {decision.timestamp.isoformat()}"]
    if decision.safety_failures:
        lines.append(f"SAFETY REGRESSIONS ({len(decision.safety_failures)}) — BLOCKING, NO OVERRIDE:")
        for r in decision.safety_failures:
            lines.append(f"  - {r.case.category_dir}/{r.case.file.name}: expected {r.case.expected_category!r}, got {r.actual_category!r}")
    if decision.skill_failures:
        lines.append(f"SKILL REGRESSIONS ({len(decision.skill_failures)}):")
        for r in decision.skill_failures:
            lines.append(f"  - {r.case.skill_id}/{r.case.file.name}: expected {r.case.expected_strategy!r}, got {r.actual_strategy!r}")
    if decision.cross_volume_failures:
        lines.append(f"CROSS-VOLUME FAILURES ({len(decision.cross_volume_failures)}):")
        for r in decision.cross_volume_failures:
            lines.append(f"  - {r.case.name}")
    if decision.override_reason:
        lines.append(f"Skill-tier block OVERRIDDEN: {decision.override_reason}")
    lines.append("")
    lines.append("RELEASE ALLOWED" if decision.allowed else "RELEASE BLOCKED")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--override", default=None, help="documented reasoning to override a skill-tier (never safety-tier) block")
    args = parser.parse_args()
    decision = gate_release(override_reason=args.override)
    print(summarize(decision))
    sys.exit(0 if decision.allowed else 1)
