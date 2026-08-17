"""Local-data reset and export endpoints.

The API is loopback-only and protected by the per-launch token middleware.
The destructive action is intentionally a POST with no silent background
trigger; the Settings UI always presents an explicit confirmation first.

Export is a POST for the same reason — it writes files to disk, so it is not
a safely-repeatable GET a prefetch or a refresh could trigger.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.data_export import export_user_data
from app.deps import Pipeline, get_pipeline, get_pipeline_optional, set_pipeline
from app.setup.data_reset import reset_local_data

logger = logging.getLogger("hearth")

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


@router.post("/api/data/export")
def export_data(pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    """Write the active profile, its memories, and its transcript to plain
    files under ``~/Hearth-exports`` and return the folder path.

    The caller cannot choose the destination — see `app.data_export`. The
    response carries `incomplete` rather than a bare success flag: a store
    that could not be read must be visible to the user, since a silently
    partial export of "all your data" is worse than a failed one.
    """
    try:
        return export_user_data(pipeline.profile, pipeline.growth_engine.store)
    except OSError as exc:
        # Disk full, a read-only home, or a permissions problem — the
        # plausible failures here are all environmental, and the user can
        # act on them once they are told which one it was.
        logger.exception("data export failed")
        raise HTTPException(status_code=500, detail=f"Could not write the export: {exc}") from exc
