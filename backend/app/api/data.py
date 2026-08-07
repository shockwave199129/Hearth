"""Local-data reset endpoints.

The API is loopback-only and protected by the per-launch token middleware.
The destructive action is intentionally a POST with no silent background
trigger; the Settings UI always presents an explicit confirmation first.
"""

from fastapi import APIRouter

from app.deps import get_pipeline_optional, set_pipeline
from app.setup.data_reset import reset_local_data

router = APIRouter()


@router.post("/api/data/reset")
def reset_data() -> dict[str, bool]:
    """Erase local runtime data while preserving profile identity."""
    pipeline = get_pipeline_optional()
    if pipeline is not None:
        pipeline.shutdown()
        set_pipeline(None)
    profile_retained = reset_local_data()
    return {"ok": True, "profile_retained": profile_retained}
