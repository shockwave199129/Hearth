"""Golden transcript regression for hearth NLP classifiers.

Checks:
  1. Predictions stay in allowed label sets (constraints)
  2. Snapshot lock matches exported smoke/full ONNX package (exact)
  3. Per-head classify latency p95 ≤ budget (default 25ms)

Run::

    cd backend && python3 -m app.eval.nlp_golden
    cd backend && python3 -m app.eval.nlp_golden --update   # refresh snapshots

pytest: ``tests/test_nlp_golden.py`` (skips if ONNX package missing).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.nlp.runtime import OnnxClassifier

CASES_PATH = Path(__file__).resolve().parent / "nlp_golden_cases.json"
DEFAULT_LATENCY_MS = 25.0
LATENCY_WARMUP = 2
LATENCY_RUNS = 20


@dataclass
class CaseResult:
    case_id: str
    ok: bool
    errors: list[str]


def load_cases(path: Path = CASES_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _predict_bundle(clf: OnnxClassifier, text: str) -> dict[str, Any]:
    e = clf.predict_emotion(text)
    i = clf.predict_intent(text)
    m = clf.predict_memory(text)
    s = clf.predict_strategy(text)
    return {
        "emotion": e.emotion,
        "intent": i.intent,
        "strategy": s.strategy,
        "memory_store": m.store,
        "memory_type": m.memory_type if m.store else None,
    }


def check_case(clf: OnnxClassifier, case: dict[str, Any]) -> CaseResult:
    errors: list[str] = []
    pred = _predict_bundle(clf, case["text"])
    constraints = case.get("constraints") or {}

    for key, field in (
        ("emotion_in", "emotion"),
        ("intent_in", "intent"),
        ("strategy_in", "strategy"),
    ):
        allowed = constraints.get(key)
        if allowed and pred[field] not in allowed:
            errors.append(f"{field}={pred[field]!r} not in {key}")

    mem_allowed = constraints.get("memory_type_in")
    if mem_allowed and pred["memory_type"] is not None and pred["memory_type"] not in mem_allowed:
        errors.append(f"memory_type={pred['memory_type']!r} not in memory_type_in")

    snap = case.get("snapshot") or {}
    for field, expected in snap.items():
        actual = pred.get(field)
        if actual != expected:
            errors.append(f"snapshot {field}: expected {expected!r}, got {actual!r}")

    return CaseResult(case["id"], ok=not errors, errors=errors)


def measure_latency_ms(clf: OnnxClassifier, text: str, *, runs: int = LATENCY_RUNS) -> dict[str, float]:
    """Per-head wall times (each head ONNX includes its own encoder copy)."""
    heads = {
        "emotion": clf.predict_emotion,
        "intent": clf.predict_intent,
        "memory": clf.predict_memory,
        "relationship": clf.predict_relationship,
        "strategy": clf.predict_strategy,
    }
    out: dict[str, float] = {}
    for name, fn in heads.items():
        for _ in range(LATENCY_WARMUP):
            fn(text)
        samples: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn(text)
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        p95_idx = max(0, int(round(0.95 * (len(samples) - 1))))
        out[name] = samples[p95_idx]
        out[f"{name}_mean"] = statistics.fmean(samples)
    return out


def run_suite(
    clf: OnnxClassifier | None = None,
    *,
    cases_path: Path = CASES_PATH,
    check_latency: bool = True,
) -> tuple[list[CaseResult], dict[str, float], list[str]]:
    suite = load_cases(cases_path)
    clf = clf or OnnxClassifier()
    suite_errors: list[str] = []
    if not clf.available:
        return [], {}, ["NLP classifiers unavailable — install models/nlp or set NLP_MODELS_DIR"]

    results = [check_case(clf, case) for case in suite["cases"]]
    latency: dict[str, float] = {}
    if check_latency and suite["cases"]:
        budget = float(suite.get("latency_budget_ms_per_head_p95", DEFAULT_LATENCY_MS))
        probe = suite["cases"][0]["text"]
        latency = measure_latency_ms(clf, probe)
        for head in ("emotion", "intent", "memory", "relationship", "strategy"):
            p95 = latency.get(head, 0.0)
            if p95 > budget:
                suite_errors.append(f"latency p95 {head}={p95:.1f}ms exceeds {budget}ms")
    return results, latency, suite_errors


def update_snapshots(cases_path: Path = CASES_PATH, clf: OnnxClassifier | None = None) -> None:
    suite = load_cases(cases_path)
    clf = clf or OnnxClassifier()
    if not clf.available:
        raise SystemExit("NLP classifiers unavailable; cannot update snapshots")
    for case in suite["cases"]:
        case["snapshot"] = _predict_bundle(clf, case["text"])
    cases_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    print(f"updated snapshots in {cases_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="Rewrite golden snapshots from current models")
    parser.add_argument("--no-latency", action="store_true")
    args = parser.parse_args(argv)

    if args.update:
        update_snapshots()
        return 0

    results, latency, suite_errors = run_suite(check_latency=not args.no_latency)
    if suite_errors and not results:
        for err in suite_errors:
            print(f"ERROR: {err}")
        return 2

    failed = [r for r in results if not r.ok]
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"[{status}] {r.case_id}")
        for err in r.errors:
            print(f"       - {err}")
    if latency:
        print("latency p95 (ms):", {k: round(v, 2) for k, v in latency.items() if not k.endswith("_mean")})
    for err in suite_errors:
        print(f"ERROR: {err}")

    if failed or suite_errors:
        print(f"\n{len(failed)} case(s) failed, {len(suite_errors)} suite error(s)")
        return 1
    print(f"\nAll {len(results)} golden cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
