"""End-to-end checks against the real voice models.

Skipped unless `models/voice/` has been populated by
`scripts/fetch_voice_models.py` — so CI and a fresh clone skip these, and
`test_voice_verification.py` remains the suite that always runs. This file
is the only thing that would catch a model swap or an ONNX-interface change
breaking the actual numbers, which unit tests with a fake embedder cannot.

These are the properties measured while choosing the models (see
`app/voice/embedder.py` on why ResNet34 and not CAM++). They are asserted
loosely — real thresholds belong in
`scripts/calibrate_speaker_threshold.py`, run against real recordings — but
tightly enough that a regression from "works" to "0.30 EER" fails here
rather than in someone's living room.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import SPEAKER_MODEL_PATH, VAD_MODEL_PATH
from app.voice.embedder import SpeakerEmbedder, cosine_similarity
from app.voice.vad import SileroVad

SR = 16000

pytestmark = pytest.mark.skipif(
    not (VAD_MODEL_PATH.is_file() and SPEAKER_MODEL_PATH.is_file()),
    reason="voice models absent — run scripts/fetch_voice_models.py",
)


def _tone(freq: float, seconds: float = 1.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(seconds: float = 1.0, scale: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(SR * seconds)) * scale).astype(np.float32)


# --- VAD --------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, signal",
    [
        ("silence", np.zeros(SR, dtype=np.float32)),
        ("white noise", _noise()),
        ("pure tone", _tone(440.0)),
        ("low rumble", _tone(80.0, amplitude=0.5)),
    ],
)
def test_vad_rejects_non_speech(label, signal):
    """The whole point of adding VAD: a fan, a rumble, or a tone clears the
    RMS endpointer and would otherwise be transcribed as if the user spoke."""
    result = SileroVad().assess(signal)

    assert result.available is True
    assert result.has_speech is False, f"{label} was classified as speech"


def test_vad_reports_a_speech_ratio_near_zero_for_noise():
    result = SileroVad().assess(_noise(seconds=2.0))

    assert result.speech_ratio < 0.1
    assert result.speech_seconds < 0.2


def test_vad_probabilities_are_well_formed():
    probabilities = SileroVad().probabilities(_noise(seconds=1.0))

    assert len(probabilities) == SR // 512
    assert all(0.0 <= p <= 1.0 for p in probabilities)


# --- Speaker embedder -------------------------------------------------------


def test_embedder_returns_a_normalised_vector_of_the_expected_width():
    embedding = SpeakerEmbedder().embed(_noise(seconds=3.0))

    assert embedding is not None
    assert embedding.shape == (256,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-4)


def test_embedder_is_deterministic():
    """Same audio must give the same vector: enrollment and verification
    happen at different times, and any nondeterminism here reads as the
    user's voice having changed."""
    embedder = SpeakerEmbedder()
    audio = _noise(seconds=3.0, seed=7)

    first = embedder.embed(audio)
    second = embedder.embed(audio)

    assert cosine_similarity(first, second) > 0.9999


def test_embedder_declines_audio_too_short_to_embed():
    assert SpeakerEmbedder().embed(np.zeros(200, dtype=np.float32)) is None


def test_distinct_signals_are_not_collapsed_to_one_embedding():
    """A sanity floor rather than a speaker test: synthetic tones are not
    speech, so this only asserts the model distinguishes inputs at all. Real
    same/different-speaker separation was measured on LibriSpeech while
    choosing the model (0.0% EER over 435 pairs) and belongs in the
    calibration script, not here."""
    embedder = SpeakerEmbedder()

    a = embedder.embed(_tone(180.0, seconds=3.0))
    b = embedder.embed(_noise(seconds=3.0, seed=11))

    assert a is not None and b is not None
    assert cosine_similarity(a, b) < 0.99
