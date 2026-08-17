"""Pins `app.voice.features.fbank` to the Kaldi reference it reimplements.

Skipped unless torchaudio is installed, which it deliberately is not in the
CPU tier or in CI: this is the one reason to install it, and it is a
correctness pin rather than a gate. Run it after touching features.py:

    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    pytest tests/test_voice_features.py

Why this pin exists at all. The speaker model is trained on
`torchaudio.compliance.kaldi.fbank` output, and a wrong feature extractor
does not fail loudly — it produces embeddings that no longer discriminate
speakers, which surfaces as a verifier that rejects its own user. During
development a log floor of 1e-38 instead of Kaldi's FLT_EPSILON produced
features that still correlated 0.987 with the reference and still looked
plausible, while degrading speaker EER from 0% to ~30%. Correlation is not
enough; this asserts absolute agreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.voice.features import fbank

kaldi = pytest.importorskip(
    "torchaudio.compliance.kaldi", reason="torchaudio is an optional dev-only reference"
)
torch = pytest.importorskip("torch")

SR = 16000
TOLERANCE = 1e-3


def _reference(audio: np.ndarray, *, cmn: bool = True) -> np.ndarray:
    feats = kaldi.fbank(
        torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0) * (1 << 15),
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=SR,
        window_type="hamming",
        use_energy=False,
    ).numpy()
    return feats - feats.mean(axis=0, keepdims=True) if cmn else feats


def _signals() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260817)
    t = np.arange(SR * 3) / SR
    speechlike = (
        0.3 * np.sin(2 * np.pi * 140 * t)
        + 0.2 * np.sin(2 * np.pi * 300 * t) * np.sin(2 * np.pi * 3 * t)
        + 0.05 * rng.standard_normal(len(t))
    ).astype(np.float32)
    # Leading silence is the case that exposed the log-floor bug: silent
    # frames pin every mel bin to the floor, and a wrong floor then drags the
    # CMN mean and corrupts every other frame in the utterance.
    with_silence = np.concatenate(
        [np.zeros(SR // 2, dtype=np.float32), speechlike[: SR * 2]]
    ).astype(np.float32)
    return {
        "speechlike": speechlike,
        "leading_silence": with_silence,
        "all_silence": np.zeros(SR, dtype=np.float32),
        "loud_noise": (rng.standard_normal(SR) * 0.4).astype(np.float32),
        "quiet": (rng.standard_normal(SR) * 1e-5).astype(np.float32),
    }


@pytest.mark.parametrize("name", sorted(_signals()))
def test_fbank_matches_kaldi_reference(name):
    signal = _signals()[name]
    mine = fbank(signal)
    reference = _reference(signal)

    assert mine.shape == reference.shape
    assert np.abs(mine - reference).max() < TOLERANCE


def test_fbank_matches_kaldi_reference_without_cmn():
    signal = _signals()["speechlike"]

    mine = fbank(signal, apply_cmn=False)

    assert np.abs(mine - _reference(signal, cmn=False)).max() < TOLERANCE


def test_silent_frames_hit_the_kaldi_log_floor_not_a_lower_one():
    """The specific regression: log(FLT_EPSILON) ~= -15.94, not -87.3."""
    mine = fbank(np.zeros(SR, dtype=np.float32), apply_cmn=False)

    assert np.allclose(mine, np.log(np.finfo(np.float32).eps), atol=1e-4)


def test_too_short_input_returns_no_frames_rather_than_raising():
    assert fbank(np.zeros(100, dtype=np.float32)).shape == (0, 80)
