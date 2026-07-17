"""TTS engines: Parler-TTS-Tiny-v1 (tiers S/A) with an ONNX Kokoro-82M
fallback for tiers B/C, where Parler would be usable but slow enough to
hurt the conversational feel. See project-plan.md §2.

Weights are loaded from plain directories under TTS_MODELS_DIR (populated by
app.setup.models during first-run setup), not from the Hugging Face hub
cache's blobs/snapshots layout — see ensure_parler_model / ensure_kokoro_model.
"""
import json
from pathlib import Path

import numpy as np

from app.hardware.tier_manager import TierConfig


class TtsEngine:
    sample_rate: int

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        raise NotImplementedError


class ParlerEngine(TtsEngine):
    """Uses parler-tts/parler-tts-tiny-v1. Unlike Chatterbox, Parler has no
    audio-prompt voice cloning and no inline paralinguistic tags — voice is
    steered entirely by a natural-language description passed alongside the
    text, so _VOICE_DESCRIPTIONS below stands in for what chatterbox_engine's
    voice_profiles/*.wav files used to do."""

    _VOICE_DESCRIPTIONS = {
        "male": "A male speaker with a warm, calm voice delivers his words "
        "at a moderate pace in a quiet room.",
        "female": "A female speaker with a warm, calm voice delivers her "
        "words at a moderate pace in a quiet room.",
    }

    def __init__(self, device: str):
        # deferred: heavy torch/transformers import
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        from app.setup.models import ensure_parler_model

        # tier_manager.py's tier pick is based on detected hardware (VRAM),
        # not on whether the installed `torch` build actually has CUDA
        # support — plain `pip install torch` on Windows/Linux can resolve
        # to a CPU-only build depending on index/platform (see
        # requirements-gpu.txt's own comment on this exact gap for Linux
        # CI). Rather than hard-crash with "Torch not compiled with CUDA
        # enabled" partway through loading the model, fall back to CPU and
        # say why — the tier's hardware detection was right, the torch
        # install just doesn't match it.
        if device == "cuda" and not torch.cuda.is_available():
            import logging

            logging.getLogger("hearth").warning(
                "Tier detection picked device=cuda but torch.cuda.is_available() "
                "is False — falling back to CPU. Install a CUDA-enabled torch "
                "build (e.g. `pip install torch --index-url "
                "https://download.pytorch.org/whl/cu121`, matching your GPU "
                "driver) to actually use the GPU."
            )
            device = "cpu"

        model_dir = str(ensure_parler_model())
        self._device = device
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_dir
        ).to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.sample_rate = self._model.config.sampling_rate

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        import torch

        description = self._VOICE_DESCRIPTIONS.get(
            voice, self._VOICE_DESCRIPTIONS["female"]
        )
        input_ids = self._tokenizer(description, return_tensors="pt").input_ids.to(
            self._device
        )
        prompt_input_ids = self._tokenizer(text, return_tensors="pt").input_ids.to(
            self._device
        )
        with torch.no_grad():
            generation = self._model.generate(
                input_ids=input_ids, prompt_input_ids=prompt_input_ids
            )
        return generation.cpu().numpy().squeeze()


class KokoroEngine(TtsEngine):
    """Uses NeuML/kokoro-fp16-onnx via onnxruntime + ttstokenizer, per that
    model card's ONNX Runtime example. Files come from TTS_KOKORO_DIR
    (see app.setup.models.ensure_kokoro_model)."""

    _VOICE_MAP = {"male": "am_adam", "female": "af_sky"}

    def __init__(self):
        import onnxruntime

        from app.setup.models import ensure_kokoro_model

        model_dir = ensure_kokoro_model()
        model_path = model_dir / "model.onnx"
        voices_path = model_dir / "voices.json"
        self._session = onnxruntime.InferenceSession(str(model_path))
        self._voices = json.loads(Path(voices_path).read_text(encoding="utf-8"))
        self._tokenizer = None  # built lazily below: ttstokenizer import is deferred too
        self.sample_rate = 24000

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        if self._tokenizer is None:
            from ttstokenizer import IPATokenizer

            self._tokenizer = IPATokenizer()

        voice_id = self._VOICE_MAP.get(voice, self._VOICE_MAP["female"])
        tokens = self._tokenizer(text)
        speaker = np.array(self._voices[voice_id], dtype=np.float32)
        outputs = self._session.run(
            None,
            {
                "tokens": [[0, *tokens, 0]],
                "style": speaker[len(tokens)],
                "speed": np.ones(1, dtype=np.float32) * 1.0,
            },
        )
        return outputs[0]


def get_tts_engine(tier: TierConfig) -> TtsEngine:
    if tier.tts_engine == "parler_gpu":
        return ParlerEngine(device="cuda")
    if tier.tts_engine == "parler_cpu":
        return ParlerEngine(device="cpu")
    return KokoroEngine()
