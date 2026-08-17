"""Speaker verification: enrollment, scoring, and what a score may decide.

One-to-one only: an utterance is compared against the **active profile's**
enrolled voiceprint and nothing else. It deliberately does not try to pick
which of several profiles a voice belongs to. That would turn verification
into identification between people, multiply the biometric exposure, and
create a path for one profile's speech to be written into another's memory.

## The safety rule this module exists to encode

A score never silences anyone. `SpeakerVerdict.decision` is advisory:

- The safety/crisis check runs on **every** utterance regardless of verdict.
  A disclosure from an unrecognised voice is still a disclosure.
- The turn is still transcribed, answered, and stored. Nothing is dropped.
- The only thing a negative verdict does is stop the turn from forming
  durable *memories* about the user — see `memory.formation` and
  `growth.engine`, which skip messages flagged `speaker_verified=False`.

That asymmetry is deliberate. Emotional arousal moves a voice away from a
calm enrollment sample — crying, whispering, hoarseness, illness — so the
utterances most likely to be scored low are disproportionately the ones that
matter most. Anything that discards audio on a low score is a defect, not a
tuning choice.

## Threshold

`MATCH_THRESHOLD` is deliberately permissive. Measured against 10
LibriSpeech speakers (enroll on 2 utterances, probe with a third), 0.40 gave
a 0.0% false-rejection rate and a 2.2% false-acceptance rate; 0.45 gave 0/0.
0.40 is chosen over the apparently-better 0.45 because the two errors are
not symmetric here — see above — and because that corpus is clean read
speech from a studio microphone, which is the easy case. Real conditions
(a laptop mic, a room, a distressed voice) will move same-speaker scores
down, so the shipped default should sit below the measured knee, not on it.

**This is not calibrated for real users yet.** Clean read speech is not
crying into a laptop. `scripts/calibrate_speaker_threshold.py` re-derives
the numbers from real recordings; until that has been run on realistic
audio, treat the default as provisional.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.config import SAMPLE_RATE, SPEAKER_MATCH_THRESHOLD
from app.voice import consent, store
from app.voice.embedder import MIN_SPEECH_SECONDS, SpeakerEmbedder, cosine_similarity

logger = logging.getLogger("hearth.speaker")

MATCH_THRESHOLD = SPEAKER_MATCH_THRESHOLD
MODEL_ID = "wespeaker-voxceleb-resnet34-LM"

# Enrollment needs several samples: one is brittle to mic position, room and
# posture, and there is no way to tell a bad single sample from a good one.
MIN_ENROLLMENT_SAMPLES = 3
# Mean pairwise similarity among the enrollment samples. Genuinely
# same-speaker clean samples sat at ~0.76+ in the probe set, so 0.45 flags
# an enrollment that captured different speakers, mostly silence, or a
# broken microphone — better to ask the user to redo it than to build a
# centroid that will misjudge every later turn.
MIN_ENROLLMENT_COHESION = 0.45


@dataclass(frozen=True)
class SpeakerVerdict:
    """``decision`` is one of:

    ``unavailable``   model weights absent — nothing was checked
    ``not_enrolled``  no voiceprint for this profile
    ``too_short``     not enough speech to score honestly
    ``match``         similarity at or above threshold
    ``unrecognized``  similarity below threshold

    ``verified`` is the tri-state the rest of the app consumes: True on
    ``match``, False on ``unrecognized``, and **None** for every
    can't-tell case. None must not be read as False — "we did not check"
    and "this is not the user" have different consequences, and only the
    latter should suppress memory formation.
    """

    decision: str
    score: float | None
    verified: bool | None
    threshold: float = MATCH_THRESHOLD

    @property
    def checked(self) -> bool:
        return self.decision in ("match", "unrecognized")


@dataclass(frozen=True)
class EnrollmentResult:
    ok: bool
    sample_count: int = 0
    cohesion: float | None = None
    error: str | None = None
    # Distinguishes "you have not agreed yet" from "those recordings were no
    # good", so the UI can show the consent step instead of a retry hint.
    needs_consent: bool = False


def _centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(embeddings)
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 1e-8 else mean


def _cohesion(embeddings: list[np.ndarray]) -> float:
    pairs = [
        cosine_similarity(embeddings[i], embeddings[j])
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
    ]
    return float(np.mean(pairs)) if pairs else 1.0


def enroll(
    user_id: str,
    samples: list[np.ndarray],
    embedder: SpeakerEmbedder,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> EnrollmentResult:
    """Build and persist a voiceprint from several recordings.

    Rejects rather than half-succeeds: too few usable samples, or samples
    that disagree with each other, produce an error the UI can act on. A
    weak enrollment is not a smaller version of a good one — it is a
    verifier that will be wrong on every subsequent turn.

    Consent is checked **here**, not only in the route, because BIPA-style
    statutes bind *collection*: any code path that could produce a stored
    template has to pass through this gate. `app.voice.consent` holds the
    record; see docs/compliance.md §6.
    """
    if not embedder.available:
        return EnrollmentResult(ok=False, error="The voice model isn't installed.")

    if not consent.has_current_consent(user_id):
        return EnrollmentResult(
            ok=False,
            error="Voice recognition needs your agreement first.",
            needs_consent=True,
        )

    embeddings: list[np.ndarray] = []
    for sample in samples:
        seconds = len(np.asarray(sample).reshape(-1)) / float(sample_rate)
        if seconds < MIN_SPEECH_SECONDS:
            continue
        embedding = embedder.embed(sample, sample_rate)
        if embedding is not None:
            embeddings.append(embedding)

    if len(embeddings) < MIN_ENROLLMENT_SAMPLES:
        return EnrollmentResult(
            ok=False,
            sample_count=len(embeddings),
            error=(
                f"Need at least {MIN_ENROLLMENT_SAMPLES} recordings of "
                f"{MIN_SPEECH_SECONDS:.0f} seconds or more."
            ),
        )

    cohesion = _cohesion(embeddings)
    if cohesion < MIN_ENROLLMENT_COHESION:
        return EnrollmentResult(
            ok=False,
            sample_count=len(embeddings),
            cohesion=cohesion,
            error=(
                "Those recordings didn't sound like the same voice in the same place. "
                "Try again somewhere quiet, with only one person speaking."
            ),
        )

    store.save(user_id, _centroid(embeddings), sample_count=len(embeddings), model_id=MODEL_ID)
    return EnrollmentResult(ok=True, sample_count=len(embeddings), cohesion=cohesion)


def verify(
    user_id: str,
    audio: np.ndarray,
    embedder: SpeakerEmbedder,
    *,
    speech_seconds: float | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> SpeakerVerdict:
    """Score one utterance against the profile's voiceprint. Never raises.

    ``speech_seconds`` should come from the VAD, not from the buffer length:
    a 10-second recording holding 0.4 seconds of speech is a too-short
    sample wearing a long one's clothes.
    """
    if not embedder.available:
        return SpeakerVerdict("unavailable", None, None)

    voiceprint = store.get(user_id)
    if voiceprint is None:
        return SpeakerVerdict("not_enrolled", None, None)
    if voiceprint.model_id != MODEL_ID:
        # Enrolled under a different model: the template is uncomparable, so
        # this is "not enrolled" rather than a score nobody can interpret.
        logger.warning(
            "voiceprint for %s was enrolled with %s, current model is %s — re-enrollment needed",
            user_id, voiceprint.model_id, MODEL_ID,
        )
        return SpeakerVerdict("not_enrolled", None, None)

    duration = (
        speech_seconds
        if speech_seconds is not None
        else len(np.asarray(audio).reshape(-1)) / float(sample_rate)
    )
    if duration < MIN_SPEECH_SECONDS:
        return SpeakerVerdict("too_short", None, None)

    embedding = embedder.embed(audio, sample_rate)
    if embedding is None:
        return SpeakerVerdict("too_short", None, None)

    score = cosine_similarity(embedding, voiceprint.embedding)
    if score >= MATCH_THRESHOLD:
        return SpeakerVerdict("match", score, True)
    return SpeakerVerdict("unrecognized", score, False)
