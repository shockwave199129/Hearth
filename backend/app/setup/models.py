"""Model downloads shared by the in-app setup flow and scripts/setup.py's
standalone CLI (kept working for manual/headless use — see its own
docstring).

LLM + embedding GGUFs are copied into MODELS_DIR. Parler/Kokoro TTS weights
are snapshot_download'd into a plain local_dir under TTS_MODELS_DIR so
Pipeline() does not open Hugging Face hub snapshot symlinks (those have
failed on Windows with Errno 22 when the user path contains spaces).
Moonshine STT still auto-downloads via moonshine-voice on first construct.
"""
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


def _usable_file(path: Path) -> bool:
    """True only if path is a real readable file (rejects broken Windows
    hub reparse points that raise Errno 22 on open)."""
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        with path.open("rb") as f:
            return bool(f.read(1))
    except OSError:
        return False


def _download_hf(repo_id: str, remote_filename: str, local_path: Path, log: ProgressFn) -> None:
    if local_path.exists():
        log(f"already have {local_path.name}")
        return
    from huggingface_hub import hf_hub_download

    log(f"downloading {remote_filename} from {repo_id} ...")
    cached_path = hf_hub_download(repo_id=repo_id, filename=remote_filename)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(Path(cached_path).read_bytes())


def ensure_parler_model(log: ProgressFn = print) -> Path:
    """Plain-file copy of parler-tts-tiny-v1 under TTS_PARLER_DIR."""
    config = TTS_PARLER_DIR / "config.json"
    if _usable_file(config):
        log(f"already have {TTS_PARLER_DIR.name}")
        return TTS_PARLER_DIR

    if TTS_PARLER_DIR.exists():
        # Partial or symlink-broken tree from a previous attempt.
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

    if TTS_KOKORO_DIR.exists():
        shutil.rmtree(TTS_KOKORO_DIR, ignore_errors=True)

    from huggingface_hub import hf_hub_download

    log(f"downloading {TTS_KOKORO_REPO} ...")
    TTS_KOKORO_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("model.onnx", "voices.json"):
        path = hf_hub_download(
            TTS_KOKORO_REPO, name, local_dir=str(TTS_KOKORO_DIR)
        )
        # hf_hub_download may still return a cache path; ensure a real file
        # exists under our local_dir name.
        dest = TTS_KOKORO_DIR / name
        if not _usable_file(dest):
            dest.write_bytes(Path(path).read_bytes())
    return TTS_KOKORO_DIR


def download_models(tier: TierConfig, log: ProgressFn = print) -> None:
    remote_name = _LFM2_REMOTE_FILENAMES[tier.llm_gguf]
    _download_hf(LFM2_REPO, remote_name, LLM_MODELS_DIR / tier.llm_gguf, log)
    _download_hf(
        EMBEDDING_REPO, EMBEDDING_MODEL_FILE, EMBEDDING_MODELS_DIR / EMBEDDING_MODEL_FILE, log
    )
    if tier.tts_engine in ("parler_gpu", "parler_cpu"):
        ensure_parler_model(log)
    else:
        ensure_kokoro_model(log)
