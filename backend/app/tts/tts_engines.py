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
        import logging

        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        from app.setup.models import ensure_parler_model

        log = logging.getLogger("hearth")

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
            log.warning(
                "Tier detection picked device=cuda but torch.cuda.is_available() "
                "is False — falling back to CPU. Install a CUDA-enabled torch "
                "build (e.g. `pip install torch --index-url "
                "https://download.pytorch.org/whl/cu121`, matching your GPU "
                "driver) to actually use the GPU."
            )
            device = "cpu"

        model_dir = str(ensure_parler_model())
        self._device = device
        # fp16 on CUDA: faster synthesize, leaves headroom for the LLM ctx.
        # Weights still live on the GPU — only the sampling Generator may
        # fall back to CPU if CUDA generators misbehave.
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=dtype
        ).to(device)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.sample_rate = self._model.config.sampling_rate
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                log.info(
                    "Parler TTS on cuda (%s), VRAM free/total after load: %.1f / %.1f GiB",
                    dtype,
                    free_b / (1024**3),
                    total_b / (1024**3),
                )
            except Exception:
                log.info("Parler TTS on cuda (%s)", dtype)
        else:
            log.info("Parler TTS on cpu (%s)", dtype)

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        import hashlib
        import logging

        import torch

        log = logging.getLogger("hearth")
        description = self._VOICE_DESCRIPTIONS.get(
            voice, self._VOICE_DESCRIPTIONS["female"]
        )
        input_ids = self._tokenizer(description, return_tensors="pt").input_ids.to(
            self._device
        )
        prompt_input_ids = self._tokenizer(text, return_tensors="pt").input_ids.to(
            self._device
        )
        # Stable seed from voice+text (sha256, not Python hash()) so live
        # chat and days-later replay use the same Parler sampling path.
        seed = int(hashlib.sha256(f"{voice}\0{text}".encode()).hexdigest()[:8], 16)

        def _greedy() -> torch.Tensor:
            return self._model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
                do_sample=False,
            )

        def _sample(generator: torch.Generator | None) -> torch.Tensor:
            kwargs = {
                "input_ids": input_ids,
                "prompt_input_ids": prompt_input_ids,
                "do_sample": True,
                "temperature": 0.7,
            }
            if generator is not None:
                kwargs["generator"] = generator
            return self._model.generate(**kwargs)

        with torch.inference_mode():
            generation = None
            if self._device.startswith("cuda"):
                # Prefer CUDA sampling — model weights stay on GPU either way.
                try:
                    generation = _sample(
                        torch.Generator(device=self._device).manual_seed(seed)
                    )
                except Exception:
                    log.exception(
                        "Parler CUDA generator sampling failed — retrying with cuda seed"
                    )
                    try:
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        generation = _sample(None)
                    except Exception:
                        log.exception(
                            "Parler CUDA sampling failed — retrying greedy on GPU"
                        )
                        generation = _greedy()
            else:
                try:
                    generation = _sample(
                        torch.Generator(device="cpu").manual_seed(seed)
                    )
                except Exception:
                    log.exception("Parler CPU sampling failed — retrying greedy")
                    generation = _greedy()

        audio_arr = (
            np.asarray(generation.detach().cpu().numpy()).reshape(-1).astype(np.float32)
        )
        if not np.isfinite(audio_arr).all():
            audio_arr = np.nan_to_num(audio_arr, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(audio_arr))) if audio_arr.size else 0.0
        if peak > 1.0:
            audio_arr = audio_arr / peak
        if audio_arr.size == 0:
            raise RuntimeError("Parler returned empty audio")
        return audio_arr


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
        audio_arr = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if not np.isfinite(audio_arr).all():
            audio_arr = np.nan_to_num(audio_arr, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(audio_arr))) if audio_arr.size else 0.0
        if peak > 1.0:
            audio_arr = audio_arr / peak
        if audio_arr.size == 0:
            raise RuntimeError("Kokoro returned empty audio")
        return audio_arr


def get_tts_engine(tier: TierConfig) -> TtsEngine:
    if tier.tts_engine == "parler_gpu":
        return ParlerEngine(device="cuda")
    if tier.tts_engine == "parler_cpu":
        return ParlerEngine(device="cpu")
    return KokoroEngine()
