"""Voice activity detection — is this buffer actually speech?

Replaces nothing: it *adds* a gate. The frontend still ends an utterance on
trailing RMS silence (`useAudioRecorder.ts`, mirrored for `--cli` in
`app/audio_io.py`), because that is an endpointing decision that has to
happen live in the browser. This runs once on the completed buffer, before
transcription, and answers a different question: was there speech in it at
all?

That matters because RMS energy cannot tell speech from a television, a
song, a fan, or a passing car — anything above the threshold keeps the
window open and lands in the transcript as if the user had said it. Silero
rejects all of those: measured probabilities on synthetic signals are
~0.0005 for silence, ~0.002 for white noise and ~0.0008 for a 440 Hz tone,
against a 0.5 speech threshold.

**Fails open.** With no model file, or on any inference error,
``has_speech`` returns True and the turn proceeds exactly as it does today.
A missing optional model must never be able to make Hearth ignore someone
who is talking to it — the cost of a wrongly-dropped turn is much higher
than the cost of transcribing a noisy one, and this whole module is an
improvement on the status quo rather than a dependency of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import SAMPLE_RATE, VAD_MODEL_PATH

logger = logging.getLogger("hearth.vad")

# Silero v5 consumes exactly 512 samples per step at 16 kHz and carries a
# (2, batch, 128) LSTM state between steps.
_WINDOW = 512
_STATE_SHAPE = (2, 1, 128)

SPEECH_PROBABILITY_THRESHOLD = 0.5
# Enough consecutive speech to be a word rather than a click. 3 windows is
# ~96 ms; below that, door latches and mouth noise start counting as speech.
MIN_SPEECH_WINDOWS = 3


@dataclass(frozen=True)
class VadResult:
    """``speech_seconds`` is what callers should gate duration on — it is the
    amount of *speech*, not the length of the recording, which is what
    speaker embedding quality actually depends on."""

    has_speech: bool
    speech_ratio: float
    speech_seconds: float
    available: bool


def vad_available(model_path: Path | None = None) -> bool:
    path = model_path or VAD_MODEL_PATH
    return path is not None and path.is_file()


class SileroVad:
    """Lazily-loaded Silero VAD ONNX session.

    Held as a single long-lived instance on the Pipeline: creating an
    onnxruntime session costs far more than running one, and the LSTM state
    is per-call rather than per-session, so one session is safe to reuse
    across utterances.
    """

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or VAD_MODEL_PATH
        self._session = None
        self.available = vad_available(self.model_path)
        if not self.available:
            logger.info("VAD model not present at %s — speech gating disabled", self.model_path)

    def _ensure_session(self):
        if self._session is None:
            import onnxruntime as ort  # deferred: native lib, only needed on a voice turn

            options = ort.SessionOptions()
            # One thread: this runs on the turn's critical path alongside STT
            # and the LLM, and a 512-sample graph gains nothing from a
            # thread pool while competing for the same cores.
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        return self._session

    def probabilities(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> list[float]:
        session = self._ensure_session()
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        rate = np.array(sample_rate, dtype=np.int64)
        out: list[float] = []
        for start in range(0, len(samples) - _WINDOW + 1, _WINDOW):
            chunk = samples[start : start + _WINDOW].reshape(1, -1)
            probability, state = session.run(None, {"input": chunk, "state": state, "sr": rate})
            out.append(float(probability[0][0]))
        return out

    def assess(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> VadResult:
        """Never raises. See the module docstring on failing open."""
        if not self.available:
            return VadResult(has_speech=True, speech_ratio=1.0, speech_seconds=0.0, available=False)
        try:
            probabilities = self.probabilities(audio, sample_rate)
        except Exception:
            logger.exception("VAD inference failed — treating the buffer as speech")
            return VadResult(has_speech=True, speech_ratio=1.0, speech_seconds=0.0, available=False)

        if not probabilities:
            # Shorter than one 512-sample window (32 ms). Too short to be a
            # word, and too short for the model to see at all.
            return VadResult(has_speech=False, speech_ratio=0.0, speech_seconds=0.0, available=True)

        flags = [p >= SPEECH_PROBABILITY_THRESHOLD for p in probabilities]
        speech_windows = sum(flags)
        longest_run = current_run = 0
        for flag in flags:
            current_run = current_run + 1 if flag else 0
            longest_run = max(longest_run, current_run)
        return VadResult(
            # A *run* rather than a total: three scattered windows across a
            # ten-second recording is a noise floor twitching over the
            # threshold, not somebody speaking.
            has_speech=longest_run >= MIN_SPEECH_WINDOWS,
            speech_ratio=speech_windows / len(flags),
            speech_seconds=speech_windows * _WINDOW / float(sample_rate),
            available=True,
        )
