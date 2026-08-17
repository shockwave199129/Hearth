"""Speaker embeddings — WeSpeaker ResNet34 (VoxCeleb, large-margin) via ONNX.

Takes mono float32 audio, returns a 256-dim L2-normalised embedding whose
cosine similarity against another embedding is a speaker-similarity score.

**Why ResNet34 and not CAM++.** CAM++ is smaller and faster and was the
first choice. Measured on 10 LibriSpeech speakers (3 utterances each, 435
pairs) with identical features, CAM++ scored ~30% EER — unusable — while
ResNet34 scored 0.0% EER with same-speaker similarity ≥0.553 against
different-speaker ≤0.420. The CAM++ ONNX export evidently expects some
different input convention; rather than reverse-engineer it, this uses the
model that demonstrably works. If someone revisits CAM++ for its smaller
size, reproduce that measurement first — the failure is silent, and a
speaker verifier that is quietly wrong is worse than none.

Both models declare the same interface (``feats`` [B, T, 80] → ``embs``),
so the input signature agreeing is not evidence that a swap is safe.

**Fails closed**, unlike the VAD. With no model file, verification reports
"unavailable" and the pipeline treats every turn as unverified rather than
as verified — see `app.voice.verification`. Guessing "it's the user" when
we cannot check would make the audit trail false.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.config import SAMPLE_RATE, SPEAKER_MODEL_PATH
from app.voice.features import fbank

logger = logging.getLogger("hearth.speaker")

EMBEDDING_DIM = 256
# Below this much *speech* (not recording length), embeddings stop being
# discriminative. Measured on the LibriSpeech probe set, the margin between
# same-speaker and different-speaker similarity collapses under ~1.5 s and
# inverts entirely at 0.5 s (same-min 0.017 vs different-max 0.298). 2.0 s
# is the first duration with a positive margin; treat anything shorter as
# unknown rather than scoring it.
MIN_SPEECH_SECONDS = 2.0


def speaker_model_available(model_path: Path | None = None) -> bool:
    path = model_path or SPEAKER_MODEL_PATH
    return path is not None and path.is_file()


class SpeakerEmbedder:
    """Lazily-loaded embedding session, held once on the Pipeline."""

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or SPEAKER_MODEL_PATH
        self._session = None
        self.available = speaker_model_available(self.model_path)
        if not self.available:
            logger.info(
                "speaker model not present at %s — voice verification disabled", self.model_path
            )

    def _ensure_session(self):
        if self._session is None:
            import onnxruntime as ort  # deferred: native lib

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        return self._session

    def embed(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray | None:
        """L2-normalised embedding, or None when the audio is unusable.

        Normalised here rather than at comparison time so that averaging
        several enrollment embeddings into a centroid is an average of unit
        vectors, and every stored voiceprint is on the same scale.
        """
        if not self.available:
            return None
        features = fbank(audio, sample_rate)
        if features.shape[0] < 10:  # <~120 ms — nothing to embed
            return None
        try:
            session = self._ensure_session()
            embedding = session.run(None, {"feats": features[None, :, :]})[0][0]
        except Exception:
            logger.exception("speaker embedding failed")
            return None
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm < 1e-8:
            return None
        return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two embeddings, robust to either being unnormalised."""
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
