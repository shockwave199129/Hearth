"""Skill benchmark runner (Book Vol 5 Ch 6) — until this module existed,
every `benchmarks/*.yaml` file under a skill folder was read by nothing.
Volume 5 Invariant 4: no skill may reach production status without passing
its benchmark conversations.

This runs the deterministic, structural half of Ch 6's gate: does the real
Intervention Engine (retrieval + additive ranking + composition — no LLM
call) select the strategy/skill each benchmark expects, including the
`avoid` case where no skill should activate at all? It does not generate or
judge actual LLM response text against each skill's specific acceptance
criteria (grounding = concrete/present-moment, journaling = never a list,
etc.) — that half needs a live model and belongs to eval/llm_judge.py's
rubric-based harness, run separately.

    python -m app.skills.benchmark_runner
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.benchmarks import SKILLS_BENCHMARKS_ROOT
from app.cognitive.communication import infer_stage
from app.intervention.engine import InterventionContext, InterventionEngine
from app.onboarding.profile_schema import UserProfile


@dataclass(frozen=True)
class BenchmarkCase:
    skill_id: str
    file: Path
    input: str
    expected_strategy: str
    expect_skill: str | None


@dataclass(frozen=True)
class BenchmarkResult:
    case: BenchmarkCase
    passed: bool
    actual_strategy: str
    actual_skill: str | None


def discover_benchmarks() -> list[BenchmarkCase]:
    """Walks the unified `benchmarks/skills/<skill_id>/*.yaml` (Vol 8 Ch 7)
    — no longer nested under each skill's own category folder; skill
    category was always organizational-only (Vol 5 Ch 4), never affecting
    retrieval or benchmarking."""
    cases: list[BenchmarkCase] = []
    if not SKILLS_BENCHMARKS_ROOT.exists():
        return cases
    for skill_dir in sorted(p for p in SKILLS_BENCHMARKS_ROOT.iterdir() if p.is_dir()):
        for bench_file in sorted(skill_dir.glob("*.yaml")):
            data = yaml.safe_load(bench_file.read_text(encoding="utf-8")) or {}
            cases.append(
                BenchmarkCase(
                    skill_id=skill_dir.name,
                    file=bench_file,
                    input=data["input"],
                    expected_strategy=data["expected_strategy"],
                    expect_skill=data.get("expect_skill"),
                )
            )
    return cases


def _default_profile() -> UserProfile:
    return UserProfile(user_id="benchmark", name="Benchmark", companion_name="Hearth", created_at=datetime.now(timezone.utc))


def run_benchmarks(
    engine: InterventionEngine | None = None, profile: UserProfile | None = None
) -> list[BenchmarkResult]:
    """A case whose `expect_skill` is `crisis_support` is run through the
    dedicated crisis path (`crisis=True`) — matching Ch 14: crisis content
    is reached once a crisis is already known, never discovered by this
    engine's ordinary scoring. Every other case runs ordinary planning."""
    engine = engine or InterventionEngine()
    profile = profile or _default_profile()
    results: list[BenchmarkResult] = []
    for case in discover_benchmarks():
        is_crisis_case = case.expect_skill == "crisis_support"
        context = InterventionContext(stage=infer_stage(case.input, "full_path"))
        plan = engine.plan(case.input, profile, context, crisis=is_crisis_case)
        actual_skill = plan.primary_skill.skill.id if plan.primary_skill else None
        passed = plan.strategy == case.expected_strategy and actual_skill == case.expect_skill
        results.append(
            BenchmarkResult(case=case, passed=passed, actual_strategy=plan.strategy, actual_skill=actual_skill)
        )
    return results


def summarize(results: list[BenchmarkResult]) -> str:
    lines = []
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"[{status}] {r.case.skill_id}/{r.case.file.name}: "
            f"expected strategy={r.case.expected_strategy!r} skill={r.case.expect_skill!r} -> "
            f"got strategy={r.actual_strategy!r} skill={r.actual_skill!r}"
        )
    passed = sum(1 for r in results if r.passed)
    lines.append(f"\n{passed}/{len(results)} passed")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(run_benchmarks()))
