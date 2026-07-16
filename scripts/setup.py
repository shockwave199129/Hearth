#!/usr/bin/env python3
"""One-time setup: detects hardware/tier, then downloads only the model
files that tier actually needs into backend/models/. Safe to re-run —
skips anything already present. See project-plan.md §1/§10.

NOT downloaded here, by design:
- Moonshine (STT) — the `moonshine-voice` package auto-fetches/caches its
  own weights on first use (see backend/app/stt/moonshine_engine.py).
- Parler-TTS-Tiny-v1 (TTS, tiers S/A) — `ParlerTTSForConditionalGeneration
  .from_pretrained()` auto-downloads from HuggingFace on first call (see
  backend/app/tts/tts_engines.py). Needs internet + disk on first real
  run, same as any other first-use cache.
- Kokoro-82M (TTS, tiers B/C) — KokoroEngine's `hf_hub_download()` calls
  against NeuML/kokoro-fp16-onnx (model.onnx, voices.json) happen lazily on
  first engine construction, same as Parler above, not here.

Downloaded here:
- LFM2.5 GGUF (LLM) for the detected tier, from LiquidAI's official repo.
- EmbeddingGemma-300M Q8_0 GGUF (long-term memory embeddings) — needed on
  every tier.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import EMBEDDING_MODEL_FILE, EMBEDDING_MODELS_DIR, LLM_MODELS_DIR
from app.hardware.detect import detect_hardware
from app.hardware.tier_manager import pick_tier

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


def _download_hf(repo_id: str, remote_filename: str, local_path: Path) -> None:
    if local_path.exists():
        print(f"  already have {local_path.name}")
        return
    from huggingface_hub import hf_hub_download

    print(f"  downloading {remote_filename} from {repo_id} ...")
    cached_path = hf_hub_download(repo_id=repo_id, filename=remote_filename)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(Path(cached_path).read_bytes())


def main() -> None:
    hw = detect_hardware()
    tier = pick_tier(hw)
    print(f"Detected tier {tier.tier} (RAM: {hw['ram_gb']} GB, GPU: {hw['gpu_name'] or 'none detected'})\n")

    print("LLM:")
    remote_name = _LFM2_REMOTE_FILENAMES[tier.llm_gguf]
    _download_hf(LFM2_REPO, remote_name, LLM_MODELS_DIR / tier.llm_gguf)

    print("Embeddings (long-term memory, needed on every tier):")
    _download_hf(EMBEDDING_REPO, EMBEDDING_MODEL_FILE, EMBEDDING_MODELS_DIR / EMBEDDING_MODEL_FILE)

    if tier.tts_engine == "kokoro":
        print("TTS: Kokoro-82M — no manual download, auto-fetches from HuggingFace on first run.")
    else:
        print("TTS: Parler-TTS-Tiny-v1 — no manual download, auto-fetches from HuggingFace on first run.")

    print("STT: Moonshine — no manual download, auto-fetches/caches on first run.")
    print("\nDone. llama-server itself is a separate binary — see requirements-common.txt's LLM section.")


if __name__ == "__main__":
    main()
