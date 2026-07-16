#!/usr/bin/env python3
"""One-time setup: detects hardware/tier, then downloads only the model
files that tier actually needs into backend/models/. Safe to re-run —
skips anything already present. See project-plan.md §1/§10.

This is now largely superseded by the packaged app's own in-app setup
screen (see backend/app/setup/ and its own /api/setup/* endpoints), which
also installs the hardware-matched torch/onnxruntime build the packaged
app needs (this thin-build CI freeze no longer bundles either). Kept as a
manual/headless CLI alternative — e.g. for local dev running the backend
directly via a venv, where the torch/onnxruntime install already happened
via `pip install -r backend/requirements-gpu.txt` (or `-cpu.txt`) the
normal way and only the model download step below is still needed.

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

from app.hardware.detect import detect_hardware
from app.hardware.tier_manager import pick_tier
from app.setup.models import download_models


def main() -> None:
    hw = detect_hardware()
    tier = pick_tier(hw)
    print(f"Detected tier {tier.tier} (RAM: {hw['ram_gb']} GB, GPU: {hw['gpu_name'] or 'none detected'})\n")

    print("LLM + embeddings:")
    download_models(tier, log=lambda msg: print(f"  {msg}"))

    if tier.tts_engine == "kokoro":
        print("TTS: Kokoro-82M — no manual download, auto-fetches from HuggingFace on first run.")
    else:
        print("TTS: Parler-TTS-Tiny-v1 — no manual download, auto-fetches from HuggingFace on first run.")

    print("STT: Moonshine — no manual download, auto-fetches/caches on first run.")
    print("\nDone. llama-server itself is a separate binary — see requirements-common.txt's LLM section.")


if __name__ == "__main__":
    main()
