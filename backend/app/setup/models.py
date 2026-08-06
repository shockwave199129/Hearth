"""Model downloads shared by the in-app setup flow and scripts/setup.py's
standalone CLI (kept working for manual/headless use — see its own
docstring).

LLM + embedding GGUFs and Parler/Kokoro TTS weights are written as plain
files under MODELS_DIR / TTS_MODELS_DIR. We never open Hugging Face hub
`snapshots/` symlinks — those raise OSError Errno 22 on Windows (seen
under both `%USERPROFILE%\\.cache` and `{install}\\userdata\\hf-home`).
Moonshine STT still auto-downloads via moonshine-voice on first construct.

Every repo is pinned to an immutable commit SHA and every downloaded weight
file is verified against a pinned SHA-256 before it is accepted. A bare
repo id resolves to `main`, which is mutable: the owner (or anyone who
takes over the account) can force-push different weights under the same
filename, and every first-run setup after that would install them
silently. Same reasoning as scripts/fetch_llama_cpp.py — TLS attests who
served the bytes, not which bytes we expected.

This is trust-on-first-use pinning: it does not prove upstream was honest
on the day the digests were recorded, but it does mean every later install
gets byte-identical artifacts, and a silent re-upload fails the download
instead of shipping. Any change to the maps below must be a deliberate,
reviewed commit — regenerate them with
`python -m app.setup.models --print-checksums` from `backend/`.
"""
import hashlib
import os
import shutil
from pathlib import Path
from typing import Callable

from app.config import (
    EMBEDDING_MODEL_FILE,
    EMBEDDING_MODELS_DIR,
    LLM_MODELS_DIR,
    TTS_KOKORO_DIR,
    TTS_KOKORO_REPO,
    TTS_PARLER_DIR,
    TTS_PARLER_REPO,
)
from app.hardware.tier_manager import TierConfig
from app.setup.nlp_models import ensure_nlp_models

# LiquidAI/LFM2.5-1.2B-Instruct-GGUF — verified repo/filenames (July 2026).
# Remote filenames differ slightly from the local names tier_manager.py
# expects, so files are renamed on save rather than changing tier_manager.py.
LFM2_REPO = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
LFM2_REVISION = "76022b8bfa64af5862d6bce90a676c3cc9b17b52"
_LFM2_REMOTE_FILENAMES = {
    "lfm2.5-1.2b-bf16.gguf": "LFM2.5-1.2B-Instruct-BF16.gguf",
    "lfm2.5-1.2b-q8_0.gguf": "LFM2.5-1.2B-Instruct-Q8_0.gguf",
    "lfm2.5-1.2b-q6_k.gguf": "LFM2.5-1.2B-Instruct-Q6_K.gguf",
    "lfm2.5-1.2b-q4_k_m.gguf": "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
}

# unsloth/embeddinggemma-300m-GGUF — filename matches config.EMBEDDING_MODEL_FILE exactly.
EMBEDDING_REPO = "unsloth/embeddinggemma-300m-GGUF"
EMBEDDING_REVISION = "6661a6504c30d8304af13455cb4a5d4f5bc6011f"

TTS_PARLER_REVISION = "fe1bd939bb05464d39a76e20cf8a35d4e6885571"
TTS_KOKORO_REVISION = "e28a545879111ad76f8eb598764da71def783328"

# SHA-256 per (repo, remote filename), captured from the pinned revisions
# above. Hugging Face exposes these directly for LFS-backed files
# (HfApi.model_info(..., files_metadata=True)), which is exactly the set of
# files that carry model weights — see _print_checksums.
_SHA256_BY_REPO_FILE = {
    (LFM2_REPO, "LFM2.5-1.2B-Instruct-BF16.gguf"):
        "3d80914b903cd6f3cc041208cf20ec46a3224f840c732e5fd7698832b4743d1b",
    (LFM2_REPO, "LFM2.5-1.2B-Instruct-Q8_0.gguf"):
        "f6b981dcb86917fa463f78a362320bd5e2dc45445df147287eedb85e5a30d26a",
    (LFM2_REPO, "LFM2.5-1.2B-Instruct-Q6_K.gguf"):
        "c5e895c191a066f6b26a8f09f10e94cdb799e579216f87df61a7e27beacd9a2b",
    (LFM2_REPO, "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"):
        "b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5",
    (EMBEDDING_REPO, "embeddinggemma-300M-Q8_0.gguf"):
        "a0f7b4e13c397a6e1b32c2de75b1f65a14c92ec524d5f674d94a4290a1c4969b",
    (TTS_KOKORO_REPO, "model.onnx"):
        "65330db8adaedb57562c5e1cb7fc2e5afae2b27d67003564ff2170f1c48273ee",
    (TTS_KOKORO_REPO, "voices.json"):
        "dc24670e8333cb30990726c5d99e991afc14645139d1a9d2d1858d4fba08df05",
    # Parler arrives via snapshot_download (a whole tree, not named files),
    # so the revision pin is the primary control there. These two are the
    # LFS-backed members — the model weights and the sentencepiece model,
    # i.e. everything that is not small declarative JSON.
    (TTS_PARLER_REPO, "model.safetensors"):
        "2e549192e0ef60cc2627cbd9d2c54ef0985ded3576d8ae7bf8744267cd6d2427",
    (TTS_PARLER_REPO, "spiece.model"):
        "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
}

ProgressFn = Callable[[str], None]


class ModelIntegrityError(RuntimeError):
    """A downloaded artifact did not match its pinned digest."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, repo_id: str, remote_filename: str, log: ProgressFn) -> None:
    """Fail closed against _SHA256_BY_REPO_FILE.

    An unpinned entry is treated as an error rather than skipped: the whole
    point of the map is that adding a new download without pinning it should
    stop the build, not quietly opt that file out of verification.
    """
    expected = _SHA256_BY_REPO_FILE.get((repo_id, remote_filename))
    if expected is None:
        path.unlink(missing_ok=True)
        raise ModelIntegrityError(
            f"no pinned SHA-256 for {repo_id}/{remote_filename}. Run "
            "`python -m app.setup.models --print-checksums` and update "
            "_SHA256_BY_REPO_FILE before shipping."
        )
    actual = _sha256_file(path)
    if actual != expected:
        # Removed, not left in place: _usable_file() only checks readability,
        # so a rejected file left on disk would be treated as "already have
        # it" and skipped on the next run.
        path.unlink(missing_ok=True)
        raise ModelIntegrityError(
            f"SHA-256 mismatch for {repo_id}/{remote_filename}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "Refusing to install. If this artifact was legitimately re-cut "
            "upstream, verify it by hand and update the pinned revision and "
            "digest in a reviewed commit."
        )
    log(f"  sha256 ok ({actual[:16]}… {remote_filename})")


def _force_hf_no_symlinks() -> None:
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    try:
        from huggingface_hub import constants as hf_constants

        hf_constants.HF_HUB_DISABLE_SYMLINKS = True
    except ImportError:
        pass


def _usable_file(path: Path) -> bool:
    """True only if path is a real readable file (rejects broken Windows
    hub reparse points that raise Errno 22 on open)."""
    try:
        if path.is_symlink():
            return False
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        with path.open("rb") as f:
            return bool(f.read(1))
    except OSError:
        return False


def _stream_http_to(
    repo_id: str, remote_filename: str, dest: Path, log: ProgressFn, revision: str
) -> None:
    """Download via HTTPS into a brand-new regular file — never touches the
    hub snapshots/ symlink layout.

    `revision` is not optional: this is the Windows fallback path, so a
    default of `main` here would quietly reopen the mutable-ref hole that
    pinning the primary path closed."""
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    import urllib.request

    url = hf_hub_url(repo_id=repo_id, filename=remote_filename, revision=revision)
    headers = build_hf_headers()
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    log(f"streaming {remote_filename} from Hugging Face ...")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    tmp.replace(dest)


def _copy_regular_file(src: Path, dest: Path) -> None:
    """Write dest as a new regular file. Raises OSError if src is unreadable
    (e.g. Windows hub symlink / Errno 22)."""
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    with src.open("rb") as rf, tmp.open("wb") as wf:
        shutil.copyfileobj(rf, wf, length=1024 * 1024)
    tmp.replace(dest)


def _download_hf(
    repo_id: str, remote_filename: str, local_path: Path, log: ProgressFn, revision: str
) -> None:
    if _usable_file(local_path):
        # Re-verify rather than trusting presence: an artifact that landed
        # before this file was pinned, or one a mismatch left behind on an
        # older build, is exactly what the digest is here to catch.
        _verify_sha256(local_path, repo_id, remote_filename, log)
        log(f"already have {local_path.name}")
        return

    _force_hf_no_symlinks()
    local_path.parent.mkdir(parents=True, exist_ok=True)

    staging = local_path.parent / f".hf-staging-{local_path.stem}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    log(f"downloading {remote_filename} from {repo_id}@{revision[:12]} ...")
    try:
        from huggingface_hub import hf_hub_download

        # local_dir replicates the file as a real path under staging — do not
        # use the default hub cache (snapshots/ symlinks → Errno 22 on Windows).
        result = hf_hub_download(
            repo_id=repo_id,
            filename=remote_filename,
            revision=revision,
            local_dir=str(staging),
        )
        candidates = [staging / remote_filename, Path(result)]
        src = next((p for p in candidates if _usable_file(p)), None)
        if src is None:
            raise OSError(22, "downloaded path unreadable (likely a hub symlink)")
        if local_path.exists() or local_path.is_symlink():
            local_path.unlink(missing_ok=True)
        _copy_regular_file(src, local_path)
    except OSError as exc:
        log(f"local_dir download failed ({exc}); falling back to direct HTTP stream")
        if local_path.exists() or local_path.is_symlink():
            local_path.unlink(missing_ok=True)
        _stream_http_to(repo_id, remote_filename, local_path, log, revision)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    if not _usable_file(local_path):
        raise RuntimeError(f"failed to produce a readable file at {local_path}")
    # After both paths converge, so the Windows fallback cannot skip it.
    _verify_sha256(local_path, repo_id, remote_filename, log)


def ensure_parler_model(log: ProgressFn = print) -> Path:
    """Plain-file copy of parler-tts-tiny-v1 under TTS_PARLER_DIR."""
    config = TTS_PARLER_DIR / "config.json"
    if _usable_file(config):
        # Not re-hashed on this path — see ensure_kokoro_model for why.
        log(f"already have {TTS_PARLER_DIR.name}")
        return TTS_PARLER_DIR

    _force_hf_no_symlinks()
    if TTS_PARLER_DIR.exists():
        shutil.rmtree(TTS_PARLER_DIR, ignore_errors=True)

    from huggingface_hub import snapshot_download

    log(f"downloading {TTS_PARLER_REPO}@{TTS_PARLER_REVISION[:12]} ...")
    TTS_PARLER_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        TTS_PARLER_REPO, revision=TTS_PARLER_REVISION, local_dir=str(TTS_PARLER_DIR)
    )
    if not _usable_file(TTS_PARLER_DIR / "config.json"):
        raise RuntimeError(
            f"downloaded {TTS_PARLER_DIR} but config.json is missing or unreadable"
        )
    # The revision pin fixes the whole tree; these two carry the actual
    # weights, so they get a digest check as well. A mismatch takes the
    # directory with it — a half-rejected snapshot is worse than none.
    try:
        for name in ("model.safetensors", "spiece.model"):
            _verify_sha256(TTS_PARLER_DIR / name, TTS_PARLER_REPO, name, log)
    except ModelIntegrityError:
        shutil.rmtree(TTS_PARLER_DIR, ignore_errors=True)
        raise
    return TTS_PARLER_DIR


def ensure_kokoro_model(log: ProgressFn = print) -> Path:
    """Plain-file copy of kokoro ONNX assets under TTS_KOKORO_DIR."""
    model_onnx = TTS_KOKORO_DIR / "model.onnx"
    voices = TTS_KOKORO_DIR / "voices.json"
    if _usable_file(model_onnx) and _usable_file(voices):
        # Deliberately not re-hashed here. This runs on every TTS engine
        # construction, and hashing 230MB at each app start buys nothing:
        # verification happens where the bytes arrive (_download_hf below),
        # and a mismatch deletes the file rather than leaving it to be found
        # later. Tampering with an already-installed file is a local-attacker
        # scenario that pinning downloads was never the defense for.
        log(f"already have {TTS_KOKORO_DIR.name}")
        return TTS_KOKORO_DIR

    _force_hf_no_symlinks()
    if TTS_KOKORO_DIR.exists():
        shutil.rmtree(TTS_KOKORO_DIR, ignore_errors=True)

    log(f"downloading {TTS_KOKORO_REPO}@{TTS_KOKORO_REVISION[:12]} ...")
    TTS_KOKORO_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("model.onnx", "voices.json"):
        dest = TTS_KOKORO_DIR / name
        _download_hf(TTS_KOKORO_REPO, name, dest, log, TTS_KOKORO_REVISION)
    return TTS_KOKORO_DIR


def download_models(tier: TierConfig, log: ProgressFn = print) -> None:
    remote_name = _LFM2_REMOTE_FILENAMES[tier.llm_gguf]
    _download_hf(LFM2_REPO, remote_name, LLM_MODELS_DIR / tier.llm_gguf, log, LFM2_REVISION)
    _download_hf(
        EMBEDDING_REPO,
        EMBEDDING_MODEL_FILE,
        EMBEDDING_MODELS_DIR / EMBEDDING_MODEL_FILE,
        log,
        EMBEDDING_REVISION,
    )
    if tier.tts_engine in ("parler_gpu", "parler_cpu"):
        ensure_parler_model(log)
    else:
        ensure_kokoro_model(log)
    # Optional: hearth_ai emotion/intent/… ONNX package (fail-soft if absent).
    ensure_nlp_models(log)


def _print_checksums() -> None:
    """Re-pin every repo after a model bump: prints the current commit SHA
    for each repo plus a paste-ready _SHA256_BY_REPO_FILE block.

    Digests come from the hub's own file metadata rather than from
    downloading ~5GB — Hugging Face records the SHA-256 of every LFS-backed
    file, and LFS is exactly where the weights live. Files served from git
    proper (small JSON configs) have no such digest and are covered by the
    revision pin alone.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    repos = (
        ("LFM2_REPO", LFM2_REPO),
        ("EMBEDDING_REPO", EMBEDDING_REPO),
        ("TTS_KOKORO_REPO", TTS_KOKORO_REPO),
        ("TTS_PARLER_REPO", TTS_PARLER_REPO),
    )
    # Only the files this app actually downloads. These repos publish several
    # quantizations we never fetch, and emitting those too would grow the map
    # with entries nobody verifies and nobody notices going stale.
    wanted = {repo_file for repo_file in _SHA256_BY_REPO_FILE}
    lines: list[str] = []
    for const_name, repo_id in repos:
        info = api.model_info(repo_id, files_metadata=True)
        print(f"# {const_name} = {repo_id!r}  revision = {info.sha!r}")
        for sibling in info.siblings:
            if (repo_id, sibling.rfilename) not in wanted:
                continue
            lfs = getattr(sibling, "lfs", None)
            digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            if digest:
                lines.append(f'    ({const_name}, "{sibling.rfilename}"):\n        "{digest}",')
    print("\n_SHA256_BY_REPO_FILE = {")
    print("\n".join(lines))
    print("}")


if __name__ == "__main__":
    import sys

    if "--print-checksums" in sys.argv[1:]:
        _print_checksums()
    else:
        print("usage: python -m app.setup.models --print-checksums", file=sys.stderr)
        sys.exit(1)
