"""Cross-volume benchmark runner (Book Vol 8 Ch 7) — the genuinely new
category neither Volume 5's skill benchmarks nor Volume 6's safety
benchmarks cover: multi-turn scenarios exercising several systems together
(here: the Safety Worker's routing, Volume 6, and the Intervention
Engine's ordinary scoring, Volume 1/5, across a turn sequence), catching
integration issues per-component testing can't see by construction.

Scoped deliberately: this drives `SafetyWorker` + `InterventionEngine`
turn-by-turn, not a full simulated conversation through every subsystem
(memory formation, relationship growth, prompt building) — that would
need a live LLM. It is real integration testing of the two systems whose
hand-off (Volume 6 Ch 5/6: crisis bypasses ordinary scoring, then hands
back once the acute moment passes) is exactly what a single-component
benchmark can't exercise."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.benchmarks import CROSS_VOLUME_BENCHMARKS_ROOT
from app.cognitive.communication import infer_stage
from app.intervention.engine import InterventionContext, InterventionEngine
from app.onboarding.profile_schema import UserProfile
from app.safety2.worker import SafetyWorker


@dataclass(frozen=True)
class CrossVolumeTurn:
    input: str
    expect_safety_category: str
    expect_safety_route: str
    expect_skill: str | None


@dataclass(frozen=True)
class CrossVolumeCase:
    name: str
    file: Path
    turns: list[CrossVolumeTurn]


@dataclass(frozen=True)
class CrossVolumeTurnResult:
    turn: CrossVolumeTurn
    passed: bool
    actual_safety_category: str
    actual_safety_route: str
    actual_skill: str | None


@dataclass(frozen=True)
class CrossVolumeCaseResult:
    case: CrossVolumeCase
    turn_results: list[CrossVolumeTurnResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.turn_results)


def discover_cases() -> list[CrossVolumeCase]:
    cases: list[CrossVolumeCase] = []
    if not CROSS_VOLUME_BENCHMARKS_ROOT.exists():
        return cases
    for bench_file in sorted(CROSS_VOLUME_BENCHMARKS_ROOT.glob("*.yaml")):
        data = yaml.safe_load(bench_file.read_text(encoding="utf-8")) or {}
        turns = [
            CrossVolumeTurn(
                input=t["input"],
                expect_safety_category=t["expect_safety_category"],
                expect_safety_route=t["expect_safety_route"],
                expect_skill=t.get("expect_skill"),
            )
            for t in data.get("turns", [])
        ]
        cases.append(CrossVolumeCase(name=data.get("name", bench_file.stem), file=bench_file, turns=turns))
    return cases


def _default_profile() -> UserProfile:
    return UserProfile(user_id="cross-volume-benchmark", name="Benchmark", companion_name="Hearth", created_at=datetime.now(timezone.utc))


def run_cross_volume_benchmarks(
    safety_worker: SafetyWorker | None = None,
    intervention_engine: InterventionEngine | None = None,
    profile: UserProfile | None = None,
) -> list[CrossVolumeCaseResult]:
    safety_worker = safety_worker or SafetyWorker()
    intervention_engine = intervention_engine or InterventionEngine()
    profile = profile or _default_profile()

    results: list[CrossVolumeCaseResult] = []
    for case in discover_cases():
        turn_results: list[CrossVolumeTurnResult] = []
        for turn in case.turns:
            safety = safety_worker.assess(profile.user_id, turn.input, profile)
            if safety.route == "ordinary":
                context = InterventionContext(stage=infer_stage(turn.input, "full_path"))
                plan = intervention_engine.plan(turn.input, profile, context, crisis=False)
                actual_skill = plan.primary_skill.skill.id if plan.primary_skill else None
            else:
                actual_skill = None
            passed = (
                safety.category == turn.expect_safety_category
                and safety.route == turn.expect_safety_route
                and actual_skill == turn.expect_skill
            )
            turn_results.append(
                CrossVolumeTurnResult(
                    turn=turn, passed=passed, actual_safety_category=safety.category,
                    actual_safety_route=safety.route, actual_skill=actual_skill,
                )
            )
        results.append(CrossVolumeCaseResult(case=case, turn_results=turn_results))
    return results


def summarize(results: list[CrossVolumeCaseResult]) -> str:
    lines = []
    for case_result in results:
        status = "PASS" if case_result.passed else "FAIL"
        lines.append(f"[{status}] {case_result.case.name} ({case_result.case.file.name})")
        for i, tr in enumerate(case_result.turn_results):
            turn_status = "ok" if tr.passed else "MISMATCH"
            lines.append(
                f"    turn {i + 1} [{turn_status}]: expected category={tr.turn.expect_safety_category!r} "
                f"route={tr.turn.expect_safety_route!r} skill={tr.turn.expect_skill!r} -> "
                f"got category={tr.actual_safety_category!r} route={tr.actual_safety_route!r} skill={tr.actual_skill!r}"
            )
    passed = sum(1 for r in results if r.passed)
    lines.append(f"\n{passed}/{len(results)} cases passed")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(run_cross_volume_benchmarks()))
