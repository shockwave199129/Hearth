"""Mic capture and speaker playback. Phase 1 uses simple energy-based
silence detection to end an utterance — good enough to close the loop;
a proper VAD (e.g. webrtcvad/silero-vad) is a reasonable later upgrade,
not needed to validate the core pipeline.
"""
import numpy as np
import sounddevice as sd

from app.config import SAMPLE_RATE

_SILENCE_RMS_THRESHOLD = 0.02
_SILENCE_HOLD_S = 1.2
_MAX_UTTERANCE_S = 30.0
_BLOCK_S = 0.1


def record_utterance() -> np.ndarray:
    """Blocks until the user stops talking (or hits the max duration), then
    returns the captured mono float32 audio at SAMPLE_RATE."""
    block_frames = int(_BLOCK_S * SAMPLE_RATE)
    silence_blocks_needed = int(_SILENCE_HOLD_S / _BLOCK_S)
    max_blocks = int(_MAX_UTTERANCE_S / _BLOCK_S)

    chunks: list[np.ndarray] = []
    silence_run = 0
    heard_speech = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block_frames) as stream:
        for _ in range(max_blocks):
            block, _ = stream.read(block_frames)
            block = block[:, 0]
            chunks.append(block)
            rms = float(np.sqrt(np.mean(np.square(block))))
            if rms >= _SILENCE_RMS_THRESHOLD:
                heard_speech = True
                silence_run = 0
            else:
                silence_run += 1
            if heard_speech and silence_run >= silence_blocks_needed:
                break

    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    sd.play(audio, samplerate=sample_rate)
    sd.wait()
