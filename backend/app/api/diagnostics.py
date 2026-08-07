"""Crash-report API — list / send / dismiss pending local buffers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.diagnostics import crash_log

router = APIRouter()


class FrontendCrashBody(BaseModel):
    message: str = Field(..., max_length=2000)
    stack: str = Field("", max_length=32000)
    component: str | None = Field(None, max_length=200)


@router.get("/api/crash-reports/pending")
def api_list_pending() -> dict:
    return {"reports": crash_log.list_pending()}


@router.post("/api/crash-reports")
def api_record_frontend_crash(body: FrontendCrashBody) -> dict:
    """Frontend unhandled errors — written to the same pending queue the
    next-launch prompt reads. Never auto-uploaded."""
    summary = crash_log.record_crash(
        source="frontend",
        message=body.message,
        stack=body.stack,
        extra={"component": body.component},
    )
    return summary


@router.post("/api/crash-reports/{report_id}/send")
def api_send_report(report_id: str) -> dict:
    """User opted in — needs internet; uploads to the crash-logs S3 prefix."""
    try:
        return crash_log.send_report(report_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="crash report not found") from None
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None


@router.post("/api/crash-reports/{report_id}/dismiss")
def api_dismiss_report(report_id: str) -> dict:
    if not crash_log.dismiss_report(report_id):
        raise HTTPException(status_code=404, detail="crash report not found")
    return {"ok": True}
