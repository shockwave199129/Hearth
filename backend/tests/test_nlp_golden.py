"""Golden NLP transcript regression + install-path smoke."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import resolve_nlp_models_dir
from app.eval.nlp_golden import run_suite
from app.nlp.runtime import OnnxClassifier, classify_available
from app.setup.nlp_models import (
    find_nlp_source,
    nlp_package_complete,
)


pytestmark = pytest.mark.skipif(
    not classify_available(),
    reason="NLP ONNX package not installed (models/nlp or NLP_MODELS_DIR)",
)


def test_ensure_nlp_models_installs_into_models_dir(tmp_path, monkeypatch):
    src = find_nlp_source()
    assert src is not None and nlp_package_complete(src)

    # Point MODELS_DIR at a temp tree via NLP install helper's destination.
    dest = tmp_path / "nlp"
    monkeypatch.setattr("app.setup.nlp_models.NLP_INSTALL_DIR", dest)
    monkeypatch.setattr("app.setup.nlp_models.MODELS_DIR", tmp_path)

    from app.setup import nlp_models as nm

    installed = nm.ensure_nlp_models(log=lambda _m: None)
    assert installed == dest
    assert nlp_package_complete(dest)

    # Idempotent
    again = nm.ensure_nlp_models(log=lambda _m: None)
    assert again == dest


def test_resolve_prefers_install_or_repo():
    root = resolve_nlp_models_dir()
    assert root is not None
    assert nlp_package_complete(root) or (root / "manifest.json").is_file()


def test_golden_transcript_suite_passes():
    """Correctness only — deterministic, so this one gates CI.

    check_latency=False deliberately: the p95 budget lives in its own
    perf-marked test below. Mixing them meant a slow shared runner could
    fail this on wall-clock alone and report it as a golden-case
    regression, which is the fastest way to teach people to ignore a
    red build."""
    results, _latency, suite_errors = run_suite(check_latency=False)
    assert not suite_errors, suite_errors
    assert results, "expected golden cases"
    failed = [r for r in results if not r.ok]
    assert not failed, [f"{r.case_id}: {r.errors}" for r in failed]


@pytest.mark.perf
def test_golden_transcript_suite_latency_budget():
    """Plan success criterion: p95 ≤ 25ms/head on CPU after warm-up.

    Marked `perf` and excluded from CI (`-m "not perf"`) — GitHub's shared
    runners are noisy-neighbour VMs, so this measures the runner, not the
    model. Run it on real target hardware: `pytest -m perf`."""
    _results, latency, _suite_errors = run_suite()
    for head in ("emotion", "intent", "memory", "relationship", "strategy"):
        assert latency[head] <= 25.0, f"{head} p95={latency[head]:.1f}ms"


def test_classifier_fail_soft_missing_dir(tmp_path):
    clf = OnnxClassifier(models_dir=tmp_path)
    assert clf.available is False
