"""Shared audio helpers for HTTP replay endpoints."""

from __future__ import annotations

import io
import wave

import numpy as np


def pcm_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Wraps float32 PCM as an in-memory WAV file (stdlib `wave`, no new
    dependency) — used for on-demand replay of a past reply."""
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())
    return buf.getvalue()
