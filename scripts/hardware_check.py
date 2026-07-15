#!/usr/bin/env python3
"""Standalone hardware/tier probe — run before starting the app to see what
tier this machine will land on, without booting the full pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.hardware.detect import detect_hardware
from app.hardware.tier_manager import pick_tier

if __name__ == "__main__":
    hw = detect_hardware()
    tier = pick_tier(hw)
    print(f"RAM:  {hw['ram_gb']} GB")
    print(f"GPU:  {hw['gpu_name'] or 'none detected'}")
    print(f"VRAM: {hw['vram_gb']} GB")
    print(f"\n-> Tier {tier.tier}: LLM={tier.llm_gguf}, STT={tier.stt_model}, TTS={tier.tts_engine}")
