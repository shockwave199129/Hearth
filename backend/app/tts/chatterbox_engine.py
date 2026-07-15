"""TTS engines: Chatterbox-Turbo (tiers S/A) with a Kokoro-82M fallback for
tiers B/C, where Chatterbox would be usable but slow enough to hurt the
conversational feel. See project-plan.md §2.
"""
from pathlib import Path

import numpy as np

from app.config import TTS_MODELS_DIR
from app.hardware.tier_manager import TierConfig

VOICE_PROFILES_DIR = Path(__file__).resolve().parent / "voice_profiles"


class TtsEngine:
    sample_rate: int

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        raise NotImplementedError


class ChatterboxEngine(TtsEngine):
    """Uses Chatterbox-Turbo specifically (not the standard/non-Turbo model)
    — this is the only variant that interprets inline paralinguistic tags
    ([laugh], [chuckle], [cough]) in the input text; the standard model
    would just read them aloud as literal words. See
    config.CHATTERBOX_STYLE_SYSTEM_PROMPT_ADDITION for how the model is
    told to use those tags.

    `exaggeration` is still passed on every call despite Turbo's own log
    warning that it's "ignored" — verified against the installed package's
    actual source (chatterbox.tts_turbo): generate() forwards it straight
    into prepare_conditionals(), which bakes it into the emotion_adv
    conditioning tensor every time (this engine always passes
    audio_prompt_path, so conditionals are rebuilt every call, not cached
    once at startup) — the warning is misleading, the value measurably
    changes the conditioning. 0.5 is the value Chatterbox's own docs
    recommend as the neutral default."""

    def __init__(self, device: str):
        from chatterbox.tts import ChatterboxTTS  # deferred: heavy torch import

        self._model = ChatterboxTTS.from_pretrained(device=device)
        self.sample_rate = self._model.sr

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        audio_prompt = str(VOICE_PROFILES_DIR / f"{voice}.wav")
        wav = self._model.generate(text, audio_prompt_path=audio_prompt, exaggeration=0.5)
        return wav.squeeze().cpu().numpy()


class KokoroEngine(TtsEngine):
    _VOICE_MAP = {"male": "am_adam", "female": "af_sarah"}

    def __init__(self):
        from kokoro_onnx import Kokoro  # deferred: onnxruntime import

        # v1.0 filenames — the older kokoro-v0_19.onnx/voices.json pairing
        # this used to reference is no longer what's published at
        # thewh1teagle/kokoro-onnx's releases; see scripts/setup.py.
        self._kokoro = Kokoro(
            str(TTS_MODELS_DIR / "kokoro-v1.0.onnx"),
            str(TTS_MODELS_DIR / "voices-v1.0.bin"),
        )
        self.sample_rate = 24000

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        samples, sr = self._kokoro.create(
            text, voice=self._VOICE_MAP.get(voice, "af_sarah"), speed=1.0, lang="en-us"
        )
        self.sample_rate = sr
        return samples


def get_tts_engine(tier: TierConfig) -> TtsEngine:
    if tier.tts_engine == "chatterbox_gpu":
        return ChatterboxEngine(device="cuda")
    if tier.tts_engine == "chatterbox_cpu":
        return ChatterboxEngine(device="cpu")
    return KokoroEngine()
