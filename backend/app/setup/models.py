"""Model downloads shared by the in-app setup flow and scripts/setup.py's
standalone CLI (kept working for manual/headless use — see its own
docstring). Only handles LLM + embedding GGUFs: TTS/STT model weights
auto-download on first engine construction already (ParlerEngine/
KokoroEngine/MoonshineEngine, see backend/app/tts/tts_engines.py and
backend/app/stt/moonshine_engine.py) — nothing extra needed for those
once Pipeline() is actually constructed post-setup.
"""
from pathlib import Path
from typing import Callable

from app.config import EMBEDDING_MODEL_FILE, EMBEDDING_MODELS_DIR, LLM_MODELS_DIR
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


def _download_hf(repo_id: str, remote_filename: str, local_path: Path, log: ProgressFn) -> None:
    if local_path.exists():
        log(f"already have {local_path.name}")
        return
    from huggingface_hub import hf_hub_download

    log(f"downloading {remote_filename} from {repo_id} ...")
    cached_path = hf_hub_download(repo_id=repo_id, filename=remote_filename)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(Path(cached_path).read_bytes())


def download_models(tier: TierConfig, log: ProgressFn = print) -> None:
    remote_name = _LFM2_REMOTE_FILENAMES[tier.llm_gguf]
    _download_hf(LFM2_REPO, remote_name, LLM_MODELS_DIR / tier.llm_gguf, log)
    _download_hf(EMBEDDING_REPO, EMBEDDING_MODEL_FILE, EMBEDDING_MODELS_DIR / EMBEDDING_MODEL_FILE, log)
