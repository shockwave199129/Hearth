"""Supply-chain integrity for downloaded model artifacts (CODE_REVIEW SEC-1).

`llama-server` and the setup Python were pinned by SHA-256 in an earlier
pass; the model weights were not. They are now, and these tests cover the
part that is easy to get wrong: not that a good download succeeds, but that
a bad one is *refused* — and refused in a way that leaves nothing usable
behind for the next run to pick up.

The signature assertions look pedantic. They are regression guards against
the specific mistake this change was fixing: a default `revision` value
silently resolving to a mutable `main`.
"""

import hashlib
import inspect
import io
from pathlib import Path

import pytest

from app.setup import models, nlp_models


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


# --- pinning completeness -------------------------------------------------


def test_every_tier_gguf_is_pinned():
    """A tier whose GGUF is unpinned fails only on that hardware tier, which
    is exactly the kind of gap that ships."""
    for local_name, remote_name in models._LFM2_REMOTE_FILENAMES.items():
        assert (models.LFM2_REPO, remote_name) in models._SHA256_BY_REPO_FILE, (
            f"{local_name} maps to {remote_name}, which has no pinned digest"
        )


def test_every_required_nlp_file_is_pinned():
    unpinned = [
        rel for rel in nlp_models._NLP_REQUIRED_RELATIVE
        if rel not in nlp_models._SHA256_BY_RELATIVE
    ]
    assert unpinned == []


@pytest.mark.parametrize(
    "revision",
    [
        models.LFM2_REVISION,
        models.EMBEDDING_REVISION,
        models.TTS_PARLER_REVISION,
        models.TTS_KOKORO_REVISION,
    ],
)
def test_revisions_are_commit_shas_not_branch_names(revision):
    """`main` is mutable; a 40-hex commit SHA is not."""
    assert len(revision) == 40
    assert all(c in "0123456789abcdef" for c in revision)


def test_download_helpers_require_an_explicit_revision():
    for fn in (models._download_hf, models._stream_http_to):
        param = inspect.signature(fn).parameters["revision"]
        assert param.default is inspect.Parameter.empty, (
            f"{fn.__name__} has a default revision — a caller that forgets to "
            "pass one would silently resolve against a mutable ref"
        )


# --- rejection paths ------------------------------------------------------


def test_verify_sha256_accepts_a_matching_file(tmp_path, monkeypatch):
    path = tmp_path / "model.gguf"
    digest = _write(path, b"weights")
    monkeypatch.setitem(models._SHA256_BY_REPO_FILE, ("repo", "model.gguf"), digest)

    models._verify_sha256(path, "repo", "model.gguf", lambda _msg: None)
    assert path.exists()


def test_verify_sha256_deletes_a_mismatched_file(tmp_path, monkeypatch):
    """Deletion is the load-bearing half: _usable_file() only checks that a
    file is readable, so a rejected artifact left on disk would be treated
    as 'already have it' and skipped on the next run."""
    path = tmp_path / "model.gguf"
    _write(path, b"tampered")
    monkeypatch.setitem(models._SHA256_BY_REPO_FILE, ("repo", "model.gguf"), "00" * 32)

    with pytest.raises(models.ModelIntegrityError, match="SHA-256 mismatch"):
        models._verify_sha256(path, "repo", "model.gguf", lambda _msg: None)
    assert not path.exists()


def test_verify_sha256_fails_closed_on_an_unpinned_file(tmp_path):
    """Adding a download without pinning it must stop the build, not opt
    that file out of verification."""
    path = tmp_path / "surprise.gguf"
    _write(path, b"unpinned")

    with pytest.raises(models.ModelIntegrityError, match="no pinned SHA-256"):
        models._verify_sha256(path, "repo", "surprise.gguf", lambda _msg: None)
    assert not path.exists()


def test_nlp_download_rejects_a_tampered_file(tmp_path, monkeypatch):
    """The digest is checked on the .partial temp file, so a failed artifact
    never exists at its final path at all."""
    monkeypatch.setattr(nlp_models, "NLP_INSTALL_DIR", tmp_path)
    monkeypatch.setitem(nlp_models._SHA256_BY_RELATIVE, "emotion/labels.json", "00" * 32)
    monkeypatch.setattr(
        nlp_models.urllib.request, "urlopen",
        lambda *a, **k: io.BytesIO(b"not the labels you pinned"),
    )

    dest = tmp_path / "emotion" / "labels.json"
    with pytest.raises(nlp_models.NlpIntegrityError, match="SHA-256 mismatch"):
        nlp_models._download_file("https://x/y", dest, "emotion/labels.json", lambda _m: None)

    assert not dest.exists()
    assert list(tmp_path.rglob("*.partial")) == []


def test_nlp_download_refuses_a_non_https_bucket(tmp_path, monkeypatch):
    """The base URL is env-overridable. The digest check would catch a
    swapped graph anyway, but there's no reason to accept the downgrade."""
    monkeypatch.setattr(nlp_models, "NLP_INSTALL_DIR", tmp_path)
    monkeypatch.setattr(nlp_models, "NLP_MODELS_BUCKET_URL", "http://insecure.example/nlp")
    logged: list[str] = []

    assert nlp_models.download_nlp_models(logged.append) is None
    assert any("non-HTTPS" in line for line in logged)


def test_nlp_download_discards_a_stale_file_that_no_longer_matches(tmp_path, monkeypatch):
    """A file present from an earlier run is re-verified, not assumed good —
    the one path where an artifact predating the digest map could otherwise
    be installed unchecked."""
    monkeypatch.setattr(nlp_models, "NLP_INSTALL_DIR", tmp_path)
    monkeypatch.setattr(nlp_models, "NLP_MODELS_BUCKET_URL", "https://bucket.example/nlp")
    stale = tmp_path / "manifest.json"
    _write(stale, b"stale contents")

    fetched: list[str] = []

    def _fake_download(url, dest, rel, log):
        fetched.append(rel)
        raise OSError("stop here — re-fetch was attempted, which is the point")

    monkeypatch.setattr(nlp_models, "_download_file", _fake_download)

    assert nlp_models.download_nlp_models(lambda _m: None) is None
    assert fetched == ["manifest.json"]


def test_download_models_passes_the_pinned_revisions(monkeypatch):
    """End-to-end through the names the setup flow actually calls, so a
    future caller that drops the revision argument is caught here."""
    calls: list[tuple[str, str, str]] = []

    def _fake_download_hf(repo_id, remote_filename, local_path, log, revision):
        calls.append((repo_id, remote_filename, revision))

    monkeypatch.setattr(models, "_download_hf", _fake_download_hf)
    monkeypatch.setattr(models, "ensure_kokoro_model", lambda log: None)
    monkeypatch.setattr(models, "ensure_nlp_models", lambda log: None)

    class _Tier:
        llm_gguf = "lfm2.5-1.2b-q4_k_m.gguf"
        tts_engine = "kokoro"

    models.download_models(_Tier(), log=lambda _m: None)

    revisions = {repo: rev for repo, _file, rev in calls}
    assert revisions[models.LFM2_REPO] == models.LFM2_REVISION
    assert revisions[models.EMBEDDING_REPO] == models.EMBEDDING_REVISION
