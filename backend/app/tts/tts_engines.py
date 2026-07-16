"""TTS engines: Parler-TTS-Tiny-v1 (tiers S/A) with an ONNX Kokoro-82M
fallback for tiers B/C, where Parler would be usable but slow enough to
hurt the conversational feel. See project-plan.md §2.
"""
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
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self._device = device
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-tiny-v1"
        ).to(device)
        self._tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-tiny-v1")
        self.sample_rate = self._model.config.sampling_rate

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        import torch

        description = self._VOICE_DESCRIPTIONS.get(voice, self._VOICE_DESCRIPTIONS["female"])
        input_ids = self._tokenizer(description, return_tensors="pt").input_ids.to(self._device)
        prompt_input_ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self._device)
        with torch.no_grad():
            generation = self._model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        return generation.cpu().numpy().squeeze()


class KokoroEngine(TtsEngine):
    """Uses NeuML/kokoro-fp16-onnx (huggingface.co/NeuML/kokoro-fp16-onnx)
    directly via onnxruntime + ttstokenizer, per that model card's own ONNX
    Runtime example — not its alternative txtai pipeline, whose own base
    install pulls in torch/transformers/faiss-cpu regardless, which would
    defeat the point of this being the lightweight tier B/C fallback.
    Model files (model.onnx, voices.json) auto-download from HuggingFace on
    first use via hf_hub_download, same as ParlerEngine — no manual
    download step."""

    _VOICE_MAP = {"male": "am_adam", "female": "af_sky"}
    _REPO_ID = "NeuML/kokoro-fp16-onnx"

    def __init__(self):
        import json

        import onnxruntime
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(self._REPO_ID, "model.onnx")
        voices_path = hf_hub_download(self._REPO_ID, "voices.json")
        self._session = onnxruntime.InferenceSession(model_path)
        self._voices = json.loads(Path(voices_path).read_text())
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
