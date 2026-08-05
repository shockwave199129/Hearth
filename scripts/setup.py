#!/usr/bin/env python3
"""One-time setup: detects hardware/tier, then downloads only the model
files that tier actually needs into backend/models/. Safe to re-run —
skips anything already present. See docs/project-plan.md §1/§10.

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

Downloaded here:
- LFM2.5 GGUF (LLM) for the detected tier, from LiquidAI's official repo.
- EmbeddingGemma-300M Q8_0 GGUF (long-term memory embeddings) — needed on
  every tier.
- Parler or Kokoro TTS weights for the detected tier.
- Hearth NLP ONNX package when present under repo ``models/nlp`` or
  ``backend/bundled/nlp`` — copied to ``{MODELS_DIR}/nlp``. Optional;
  missing is OK (classifiers fail-soft).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.hardware.detect import detect_hardware
from app.hardware.tier_manager import pick_tier
from app.setup.models import download_models
from app.setup.nlp_models import ensure_nlp_models


def main() -> None:
    hw = detect_hardware()
    tier = pick_tier(hw)
    print(f"Detected tier {tier.tier} (RAM: {hw['ram_gb']} GB, GPU: {hw['gpu_name'] or 'none detected'})\n")

    print("LLM + embeddings + TTS + NLP:")
    download_models(tier, log=lambda msg: print(f"  {msg}"))
    ensure_nlp_models(log=lambda msg: print(f"  {msg}"))

    if tier.tts_engine == "kokoro":
        print("TTS: Kokoro-82M — weights fetched above (or already present).")
    else:
        print("TTS: Parler-TTS-Tiny-v1 — weights fetched above (or already present).")

    print("STT: Moonshine — no manual download, auto-fetches/caches on first run.")
    print("\nDone. llama-server itself is a separate binary — see requirements-common.txt's LLM section.")
    print("NLP: set NLP_MODELS_DIR to override; default install is {userdata}/models/nlp.")


if __name__ == "__main__":
    main()
