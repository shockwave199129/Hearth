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
    results, latency, suite_errors = run_suite()
    assert not suite_errors, suite_errors
    assert results, "expected golden cases"
    failed = [r for r in results if not r.ok]
    assert not failed, [f"{r.case_id}: {r.errors}" for r in failed]
    # Plan success criterion: p95 ≤ 25ms/head on CPU after warm-up
    for head in ("emotion", "intent", "memory", "relationship", "strategy"):
        assert latency[head] <= 25.0, f"{head} p95={latency[head]:.1f}ms"


def test_classifier_fail_soft_missing_dir(tmp_path):
    clf = OnnxClassifier(models_dir=tmp_path)
    assert clf.available is False
