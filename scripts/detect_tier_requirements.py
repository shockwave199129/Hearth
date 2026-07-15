#!/usr/bin/env python3
"""Prints which of backend/requirements-{gpu,cpu}.txt matches this machine's
detected tier, so build_backend.sh/.ps1 can install the right one before
freezing. Not "gpu" as in "has a GPU" — it's "S/A" (chatterbox-tts, run on
GPU or CPU per tier.tts_engine) vs "B/C" (kokoro-onnx) — see
backend/requirements-gpu.txt's docstring-comment for why the split exists.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.hardware.detect import detect_hardware
from app.hardware.tier_manager import pick_tier

if __name__ == "__main__":
    tier = pick_tier(detect_hardware())
    print("requirements-cpu.txt" if tier.tts_engine == "kokoro" else "requirements-gpu.txt")
