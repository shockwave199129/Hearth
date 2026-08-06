"""Status / check-in / safety transparency routes."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.deps import Pipeline, get_pipeline
from app.hardware.detect import detect_hardware
from app.checkin.state import get_last_checkin
from app.safety import crisis_detector, escalation
from app.safety2.audit import pending_entry_count, retention_policy_disclosure

router = APIRouter()


@router.get("/api/status")
def get_status(pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    tier = pipeline.tier
    return {
        "tier": tier.tier,
        "llm_gguf": tier.llm_gguf,
        "stt_model": tier.stt_model,
        "tts_engine": tier.tts_engine,
        "n_gpu_layers": tier.n_gpu_layers,
        "ctx_size": tier.ctx_size,
        "hardware": detect_hardware(),
    }


@router.get("/api/checkin")
def api_get_checkin(pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    last = get_last_checkin(pipeline.profile.user_id)
    days_since = (datetime.now(timezone.utc).date() - last.date()).days if last else None
    return {
        "last_checkin_at": last.isoformat() if last else None,
        "days_since_last_checkin": days_since,
    }


@router.get("/api/safety/status")
def api_get_safety_status(pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    """Read-only transparency surface — same 'never actually hidden'
    principle as /api/memories, /api/skills, /api/checkin. See
    docs/project-plan.md §9."""
    user_id = pipeline.profile.user_id
    last = escalation.last_escalation(user_id)
    return {
        "recent_crisis_events": crisis_detector.event_count(user_id, within_days=7),
        "last_escalation_at": last.isoformat() if last else None,
        "safety_log_retention_policy": retention_policy_disclosure(),
        "safety_log_entries_retained": pending_entry_count(user_id),
    }
