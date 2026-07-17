"""Maps detected hardware to a model tier (S/A/B/C), see project-plan.md §2."""
import json
from dataclasses import dataclass, asdict

from app.config import LLM_MODELS_DIR, STT_MODELS_DIR, TTS_MODELS_DIR, TIER_CACHE_FILE
from app.hardware.detect import detect_hardware


@dataclass(frozen=True)
class TierConfig:
    tier: str
    llm_gguf: str
    n_gpu_layers: int
    stt_model: str
    tts_engine: str  # "parler_gpu" | "parler_cpu" | "kokoro"


_TIER_TABLE = {
    "S": TierConfig("S", "lfm2.5-1.2b-bf16.gguf", 99, "moonshine-base", "parler_gpu"),
    "A": TierConfig("A", "lfm2.5-1.2b-q8_0.gguf", 99, "moonshine-base", "parler_cpu"),
    "B": TierConfig("B", "lfm2.5-1.2b-q6_k.gguf", 0, "moonshine-tiny", "kokoro"),
    "C": TierConfig("C", "lfm2.5-1.2b-q4_k_m.gguf", 0, "moonshine-tiny", "kokoro"),
}


def pick_tier(hw: dict) -> TierConfig:
    """S/A (parler + GPU layers) only when NVIDIA is present. AMD/no-GPU
    machines fall through to B/C (kokoro) by RAM — otherwise Windows AMD
    with high VRAM would get parler_gpu while setup installs CPU torch."""
    has_nvidia = bool(hw.get("has_nvidia"))
    if has_nvidia and hw["vram_gb"] >= 8:
        key = "S"
    elif has_nvidia and (hw["vram_gb"] >= 4 or hw["ram_gb"] >= 16):
        key = "A"
    elif hw["ram_gb"] >= 8:
        key = "B"
    else:
        key = "C"
    return _TIER_TABLE[key]


def resolve_paths(tier: TierConfig) -> dict:
    return {
        "llm_path": str(LLM_MODELS_DIR / tier.llm_gguf),
        "stt_dir": str(STT_MODELS_DIR / tier.stt_model),
        "tts_dir": str(TTS_MODELS_DIR),
    }


def detect_and_cache_tier() -> TierConfig:
    """Re-runs detection every launch (cheap) rather than trusting a stale cache,
    but still writes the result so Settings can show it without re-probing."""
    hw = detect_hardware()
    tier = pick_tier(hw)
    TIER_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TIER_CACHE_FILE.write_text(json.dumps({"hardware": hw, "tier": asdict(tier)}, indent=2))
    return tier
