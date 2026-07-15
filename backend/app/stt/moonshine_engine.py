"""Moonshine STT wrapper. Model size (base/tiny) is picked by tier_manager.

Uses the `moonshine-voice` PyPI package (github.com/moonshine-ai/moonshine).
The project this originally depended on — the plain `usefulsensors/moonshine`
git repo, installed via `useful-moonshine @ git+...` — has since been
rewritten into a much larger multi-platform "Moonshine Voice" project and
no longer installs via that git URL at all (confirmed: it has neither a
setup.py nor a pyproject.toml anymore). `moonshine-voice` is the actual
current Python package; `get_model_for_language()` still auto-downloads
and caches its own model files on first use, same as before — no separate
manual download step needed.
"""
import numpy as np

from app.config import SAMPLE_RATE

_MODEL_ARCH_BY_TIER_NAME = {
    "moonshine-base": "BASE",
    "moonshine-tiny": "TINY",
}


class MoonshineEngine:
    def __init__(self, model_name: str):
        """model_name: 'moonshine-base' or 'moonshine-tiny' (from TierConfig.stt_model)."""
        from moonshine_voice import ModelArch, get_model_for_language  # deferred: heavy, native lib
        from moonshine_voice.transcriber import Transcriber

        wanted_arch = getattr(ModelArch, _MODEL_ARCH_BY_TIER_NAME[model_name])
        model_path, resolved_arch = get_model_for_language("en", wanted_model_arch=wanted_arch)
        self._transcriber = Transcriber(model_path=model_path, model_arch=resolved_arch)

    def transcribe(self, audio: np.ndarray) -> str:
        """audio: mono float32 PCM at SAMPLE_RATE (16kHz), range [-1, 1]."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        transcript = self._transcriber.transcribe_without_streaming(audio.tolist(), sample_rate=SAMPLE_RATE)
        return " ".join(line.text for line in transcript.lines).strip()
