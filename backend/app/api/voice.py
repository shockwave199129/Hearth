"""Speaker-enrollment routes.

Enrollment stores biometric data (docs/compliance.md §6), so it is opt-in,
inspectable, and independently deletable. Four routes and nothing else: check
status, record consent, enroll, forget.

Consent is its own route rather than a flag riding alongside the audio. That
keeps it a separate, auditable act performed *before* any biometric data is
collected — which is what BIPA-style statutes require — and means the audio
body stays a plain PCM bundle.

There is deliberately **no route that returns the stored template**, and no
route that scores arbitrary audio. The first would hand a biometric
identifier to anything that can reach the loopback API; the second would
turn Hearth into a general-purpose voice-matching oracle. Verification
happens only as part of a real conversational turn, in `Pipeline.respond`.

Audio arrives as a raw request body of little-endian float32 mono PCM at
`SAMPLE_RATE`, several samples concatenated with an explicit sample count —
the same wire format the conversation websocket already uses for a turn, so
the frontend reuses its existing recorder without a second encoder.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request

from app.config import (
    SAMPLE_RATE,
    VOICE_BIOMETRIC_CONSENT_TEXT,
    VOICE_BIOMETRIC_CONSENT_VERSION,
    VOICEPRINT_RETENTION_DAYS,
)
from app.deps import Pipeline, get_pipeline
from app.voice import consent as voice_consent
from app.voice import retention
from app.voice import store as voiceprint_store
from app.voice import verification
from app.voice.embedder import MIN_SPEECH_SECONDS

logger = logging.getLogger("hearth")

router = APIRouter()

# One 6-second sample at 16 kHz float32 is ~384 KB; five is under 2 MB. This
# bounds a malformed or hostile request before it becomes a large allocation.
_MAX_BODY_BYTES = 8 * 1024 * 1024


def _consent_text(companion_name: str) -> str:
    """The single reviewable copy of the consent wording (app.config), with the
    companion's name filled in. Served rather than duplicated in the frontend
    so the text a user agrees to is exactly the text recorded against their
    `consent_version`."""
    return VOICE_BIOMETRIC_CONSENT_TEXT.format(
        companion=companion_name or "Hearth",
        retention_years=max(VOICEPRINT_RETENTION_DAYS // 365, 1),
    )


@router.get("/api/voice/enrollment")
def api_enrollment_status(pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    """Whether this profile has a voiceprint, whether consent is on record,
    and whether the feature can run at all. Never includes the template.

    Also the point at which the retention schedule is enforced for a profile
    that is open but idle — see app/voice/retention.py.
    """
    user_id = pipeline.profile.user_id
    retention.enforce(user_id)
    expiry = retention.expiry_for(user_id)
    return {
        "model_available": pipeline.speaker_embedder.available,
        "vad_available": pipeline.vad.available,
        "required_samples": verification.MIN_ENROLLMENT_SAMPLES,
        "min_seconds_per_sample": MIN_SPEECH_SECONDS,
        "consent_text": _consent_text(pipeline.profile.companion_name),
        "retention_days": VOICEPRINT_RETENTION_DAYS,
        "expires_at": expiry.isoformat() if expiry else None,
        **voice_consent.status(user_id),
        **voiceprint_store.metadata(user_id),
    }


@router.post("/api/voice/consent")
def api_record_consent(pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    """Record written consent to the current biometric-consent wording.

    Takes no body, exactly like POST /api/profile/attestation: there is
    nothing for the client to assert beyond "the user agreed", the timestamp
    is stamped server-side so it cannot be backdated, and the wording is
    identified by `VOICE_BIOMETRIC_CONSENT_VERSION` rather than sent up and
    trusted. Recording consent collects nothing on its own — enrollment is
    still a separate, explicit act.
    """
    record = voice_consent.record(pipeline.profile.user_id)
    return {
        "ok": True,
        "consented_at": record.consented_at,
        "consent_version": record.consent_version,
        "current_consent_version": VOICE_BIOMETRIC_CONSENT_VERSION,
    }


@router.post("/api/voice/enrollment")
async def api_enroll(request: Request, pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    """Enroll (or re-enroll) this profile's voiceprint.

    Samples are sent as one body: a little-endian uint32 count, then that
    many little-endian uint32 sample-lengths, then the float32 PCM runs back
    to back. Re-enrolling replaces the previous template rather than
    averaging into it — a user redoing enrollment is correcting it, and
    blending a bad attempt into a good one would defeat the point.
    """
    body = await request.body()
    if len(body) > _MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Too much audio in one enrollment request.")
    try:
        samples = _decode_samples(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Malformed enrollment audio: {exc}") from exc

    result = verification.enroll(pipeline.profile.user_id, samples, pipeline.speaker_embedder)
    if not result.ok:
        # 403 when consent is missing, because that is a permission state
        # rather than bad input, and the UI shows the consent step instead of
        # a "try recording again" hint. Otherwise 422: the usual causes are
        # recordings that were too short or captured more than one voice,
        # which the user can fix by redoing them.
        raise HTTPException(
            status_code=403 if result.needs_consent else 422,
            detail=result.error or "Enrollment failed.",
        )
    logger.info(
        "voiceprint enrolled for %s from %s samples (cohesion %.3f)",
        pipeline.profile.user_id, result.sample_count, result.cohesion or float("nan"),
    )
    return {
        "ok": True,
        "sample_count": result.sample_count,
        **voiceprint_store.metadata(pipeline.profile.user_id),
    }


@router.delete("/api/voice/enrollment")
def api_forget_voice(pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    """Delete the voiceprint without touching anything else in the profile.

    Idempotent, and returns ok even when nothing was enrolled: "my voice is
    not stored" is the state the caller asked for either way.
    """
    voiceprint_store.delete(pipeline.profile.user_id)
    # Consent is withdrawn with the template: permission to hold a biometric
    # identifier does not outlive the identifier, so re-enrolling asks again.
    voice_consent.revoke(pipeline.profile.user_id)
    return {"ok": True, "enrolled": False, "consent_recorded": False}


def _decode_samples(body: bytes) -> list[np.ndarray]:
    """Parse the count-prefixed float32 sample bundle described above."""
    if len(body) < 4:
        raise ValueError("body too short to contain a sample count")
    count = int(np.frombuffer(body[:4], dtype="<u4")[0])
    if not 1 <= count <= 16:
        raise ValueError(f"sample count {count} out of range")
    header_end = 4 + 4 * count
    if len(body) < header_end:
        raise ValueError("truncated sample length table")
    lengths = np.frombuffer(body[4:header_end], dtype="<u4").astype(np.int64)
    expected = header_end + int(lengths.sum()) * 4
    if len(body) != expected:
        raise ValueError(f"expected {expected} bytes for the declared lengths, got {len(body)}")

    samples: list[np.ndarray] = []
    offset = header_end
    for length in lengths:
        end = offset + int(length) * 4
        samples.append(np.frombuffer(body[offset:end], dtype="<f4").astype(np.float32))
        offset = end
    if any(len(s) / SAMPLE_RATE > 60 for s in samples):
        raise ValueError("a sample was longer than 60 seconds")
    return samples
