"""Kaldi-compatible log-mel filterbank features, in numpy.

Needed because the speaker-embedding model (see `app.voice.embedder`) is a
WeSpeaker export, and WeSpeaker's models are trained on
``torchaudio.compliance.kaldi.fbank`` output. Getting these features even
slightly wrong does not fail loudly — it silently produces embeddings that
no longer discriminate speakers, which would show up as a verifier that
rejects the real user. So this is a reimplementation rather than a
dependency, and it is pinned against the reference by test.

Why not just call torchaudio: it would drag `torch` into the hardware tier
that can least afford it. torch is already an optional dependency for the
Parler TTS tiers, but tier C runs CPU-only with `onnxruntime` alone, and
voice input has to work there. This module is ~40 lines of numpy against a
~2 GB dependency.

Verified equivalent to
``kaldi.fbank(wav * (1 << 15), num_mel_bins=80, frame_length=25,
frame_shift=10, dither=0.0, window_type='hamming', use_energy=False)``
followed by per-utterance CMN, to within 5e-4 absolute on real speech —
see tests/test_voice_features.py, which skips unless torchaudio is
installed and is the only reason to install it.

Four details that are easy to get wrong, each of which cost real debugging:

1. **The log floor is FLT_EPSILON (~1.19e-07), not the denormal minimum.**
   Kaldi floors mel energies at `std::numeric_limits<float>::epsilon()`,
   giving log(-15.94) for a silent frame. Flooring lower (e.g. 1e-38 →
   -87.3) looks harmless, but LibriSpeech-style leading silence then drags
   the CMN mean down by tens of nats and corrupts *every* frame in the
   utterance, not just the silent ones.
2. **The window is a symmetric hamming**, matching WeSpeaker's explicit
   `window_type='hamming'` — not Kaldi's "povey" default, and not numpy's
   periodic `np.hamming(N+1)[:N]`.
3. **Mel triangles are interpolated in the mel domain, not in Hz.**
4. **Audio is scaled to int16 range** before framing.
"""

from __future__ import annotations

import numpy as np

NUM_MEL_BINS = 80
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10
PREEMPHASIS = 0.97
LOW_FREQ_HZ = 20.0
# Kaldi's log floor: std::numeric_limits<float>::epsilon(). See note 1 above.
_LOG_FLOOR = float(np.finfo(np.float32).eps)


def _mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 1127.0 * np.log(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_filterbank(sample_rate: int, n_fft: int, num_bins: int) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    mels = _mel(freqs)
    edges = np.linspace(_mel(LOW_FREQ_HZ), _mel(sample_rate / 2.0), num_bins + 2)
    bank = np.zeros((num_bins, len(freqs)))
    for i in range(num_bins):
        low, center, high = edges[i], edges[i + 1], edges[i + 2]
        rising = (mels - low) / (center - low)
        falling = (high - mels) / (high - center)
        bank[i] = np.clip(np.minimum(rising, falling), 0.0, None)
    return bank


def fbank(
    audio: np.ndarray,
    sample_rate: int = 16000,
    *,
    num_bins: int = NUM_MEL_BINS,
    apply_cmn: bool = True,
) -> np.ndarray:
    """80-dim log-mel filterbank for mono float32 audio in [-1, 1].

    Returns ``(frames, num_bins)`` float32. An input shorter than one
    analysis window returns an empty ``(0, num_bins)`` array rather than
    raising — callers gate on frame count anyway, and a too-short buffer is
    an ordinary condition on the voice path, not an error.
    """
    window_size = int(sample_rate * FRAME_LENGTH_MS / 1000)
    hop = int(sample_rate * FRAME_SHIFT_MS / 1000)
    n_fft = 1
    while n_fft < window_size:
        n_fft *= 2

    samples = np.asarray(audio, dtype=np.float64).reshape(-1) * 32768.0
    n_frames = 1 + (len(samples) - window_size) // hop
    if n_frames < 1:
        return np.zeros((0, num_bins), dtype=np.float32)

    offsets = np.arange(window_size)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = samples[offsets]
    frames = frames - frames.mean(axis=1, keepdims=True)  # remove DC offset

    emphasized = np.empty_like(frames)
    emphasized[:, 0] = frames[:, 0] - PREEMPHASIS * frames[:, 0]
    emphasized[:, 1:] = frames[:, 1:] - PREEMPHASIS * frames[:, :-1]

    spectrum = np.abs(np.fft.rfft(emphasized * np.hamming(window_size), n=n_fft)) ** 2
    energies = spectrum @ _mel_filterbank(sample_rate, n_fft, num_bins).T
    feats = np.log(np.maximum(energies, _LOG_FLOOR))
    if apply_cmn:
        feats = feats - feats.mean(axis=0, keepdims=True)
    return feats.astype(np.float32)
