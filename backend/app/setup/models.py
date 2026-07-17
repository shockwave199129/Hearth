"""Model downloads shared by the in-app setup flow and scripts/setup.py's
standalone CLI (kept working for manual/headless use — see its own
docstring).

LLM + embedding GGUFs and Parler/Kokoro TTS weights are written as plain
files under MODELS_DIR / TTS_MODELS_DIR. We never open Hugging Face hub
`snapshots/` symlinks — those raise OSError Errno 22 on Windows (seen
under both `%USERPROFILE%\\.cache` and `{install}\\userdata\\hf-home`).
Moonshine STT still auto-downloads via moonshine-voice on first construct.
"""
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

# LiquidAI/LFM2.5-1.2B-Instruct-GGUF — verified repo/filenames (July 2026).
# Remote filenames differ slightly from the local names tier_manager.py
# expects, so files are renamed on save rather than changing tier_manager.py.
LFM2_REPO = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
_LFM2_REMOTE_FILENAMES = {
    "lfm2.5-1.2b-bf16.gguf": "LFM2.5-1.2B-Instruct-BF16.gguf",
    "lfm2.5-1.2b-q8_0.gguf": "LFM2.5-1.2B-Instruct-Q8_0.gguf",
    "lfm2.5-1.2b-q6_k.gguf": "LFM2.5-1.2B-Instruct-Q6_K.gguf",
    "lfm2.5-1.2b-q4_k_m.gguf": "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
}

# unsloth/embeddinggemma-300m-GGUF — filename matches config.EMBEDDING_MODEL_FILE exactly.
EMBEDDING_REPO = "unsloth/embeddinggemma-300m-GGUF"

ProgressFn = Callable[[str], None]


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


def _stream_http_to(repo_id: str, remote_filename: str, dest: Path, log: ProgressFn) -> None:
    """Download via HTTPS into a brand-new regular file — never touches the
    hub snapshots/ symlink layout."""
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    import urllib.request

    url = hf_hub_url(repo_id=repo_id, filename=remote_filename)
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


def _download_hf(repo_id: str, remote_filename: str, local_path: Path, log: ProgressFn) -> None:
    if _usable_file(local_path):
        log(f"already have {local_path.name}")
        return

    _force_hf_no_symlinks()
    local_path.parent.mkdir(parents=True, exist_ok=True)

    staging = local_path.parent / f".hf-staging-{local_path.stem}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    log(f"downloading {remote_filename} from {repo_id} ...")
    try:
        from huggingface_hub import hf_hub_download

        # local_dir replicates the file as a real path under staging — do not
        # use the default hub cache (snapshots/ symlinks → Errno 22 on Windows).
        result = hf_hub_download(
            repo_id=repo_id,
            filename=remote_filename,
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
        _stream_http_to(repo_id, remote_filename, local_path, log)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    if not _usable_file(local_path):
        raise RuntimeError(f"failed to produce a readable file at {local_path}")


def ensure_parler_model(log: ProgressFn = print) -> Path:
    """Plain-file copy of parler-tts-tiny-v1 under TTS_PARLER_DIR."""
    config = TTS_PARLER_DIR / "config.json"
    if _usable_file(config):
        log(f"already have {TTS_PARLER_DIR.name}")
        return TTS_PARLER_DIR

    _force_hf_no_symlinks()
    if TTS_PARLER_DIR.exists():
        shutil.rmtree(TTS_PARLER_DIR, ignore_errors=True)

    from huggingface_hub import snapshot_download

    log(f"downloading {TTS_PARLER_REPO} ...")
    TTS_PARLER_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(TTS_PARLER_REPO, local_dir=str(TTS_PARLER_DIR))
    if not _usable_file(TTS_PARLER_DIR / "config.json"):
        raise RuntimeError(
            f"downloaded {TTS_PARLER_DIR} but config.json is missing or unreadable"
        )
    return TTS_PARLER_DIR


def ensure_kokoro_model(log: ProgressFn = print) -> Path:
    """Plain-file copy of kokoro ONNX assets under TTS_KOKORO_DIR."""
    model_onnx = TTS_KOKORO_DIR / "model.onnx"
    voices = TTS_KOKORO_DIR / "voices.json"
    if _usable_file(model_onnx) and _usable_file(voices):
        log(f"already have {TTS_KOKORO_DIR.name}")
        return TTS_KOKORO_DIR

    _force_hf_no_symlinks()
    if TTS_KOKORO_DIR.exists():
        shutil.rmtree(TTS_KOKORO_DIR, ignore_errors=True)

    log(f"downloading {TTS_KOKORO_REPO} ...")
    TTS_KOKORO_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("model.onnx", "voices.json"):
        dest = TTS_KOKORO_DIR / name
        _download_hf(TTS_KOKORO_REPO, name, dest, log)
    return TTS_KOKORO_DIR


def download_models(tier: TierConfig, log: ProgressFn = print) -> None:
    remote_name = _LFM2_REMOTE_FILENAMES[tier.llm_gguf]
    _download_hf(LFM2_REPO, remote_name, LLM_MODELS_DIR / tier.llm_gguf, log)
    _download_hf(
        EMBEDDING_REPO,
        EMBEDDING_MODEL_FILE,
        EMBEDDING_MODELS_DIR / EMBEDDING_MODEL_FILE,
        log,
    )
    if tier.tts_engine in ("parler_gpu", "parler_cpu"):
        ensure_parler_model(log)
    else:
        ensure_kokoro_model(log)
