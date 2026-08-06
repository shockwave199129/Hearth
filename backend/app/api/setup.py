"""First-run setup flow — packages, models, Pipeline warm-up."""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter

from app.deps import get_pipeline_optional, set_pipeline
from app.pipeline import Pipeline
from app.setup import orchestrator
from app.setup.installer import InstallProgress

logger = logging.getLogger("hearth")

router = APIRouter()

# One process-wide progress tracker for the setup flow — see
# app/setup/orchestrator.py. A thread rather than an async task since
# run_setup() does blocking subprocess/network calls throughout.
_setup_progress = InstallProgress()
_setup_thread: threading.Thread | None = None


@router.get("/api/setup/status")
def get_setup_status() -> dict:
    return orchestrator.detect_status()


@router.post("/api/setup/start")
def start_setup() -> dict:
    """Idempotent — if a setup run is already in flight, just returns its
    current progress instead of starting a second overlapping one."""
    global _setup_thread
    if _setup_thread is not None and _setup_thread.is_alive():
        return _setup_progress.snapshot()

    # Drop stale error/log from a previous failed attempt before the new run
    # starts — otherwise GET /api/setup/progress (and the Setup UI) keep
    # showing the old failure while detecting/installing again.
    _setup_progress.reset()

    def _run() -> None:
        orchestrator.run_setup(_setup_progress)
        # run_setup leaves step at downloading_models on success (never
        # "done") — "done" is reserved for after Pipeline() + mark_complete.
        if _setup_progress.snapshot()["step"] == "error":
            return

        _setup_progress.set_step("starting_engines")
        _setup_progress.append_log("Starting speech and language engines…")
        try:
            # Re-running setup against an already-running app (a Retry, or a
            # manual POST from /docs) must not build a second Pipeline: its
            # llama-server would fail to bind the port the live one holds,
            # leaving the new Pipeline wired to a dead process.
            if get_pipeline_optional() is None:
                set_pipeline(Pipeline())
            else:
                _setup_progress.append_log("Engines already running — reusing them.")
        except Exception as exc:
            # Packages/models installed fine, but constructing the actual
            # pipeline still failed (e.g. llama-server missing/broken) —
            # caught for real during this feature's own local verification,
            # not a hypothetical: without this, the UI would show "done"
            # forever while /api/status silently 503s with no explanation.
            logger.exception("Pipeline() construction failed after setup packages/models")
            _setup_progress.set_error(f"setup finished but the app failed to start: {exc}")
            return
        # Persist so the next launch skips Setup entirely (setup_state in
        # profile.db) — only after Pipeline actually starts, not merely
        # after pip/downloads finish. mark_complete before "done" so a
        # client that re-fetches /api/setup/status on done sees complete.
        orchestrator.mark_complete()
        _setup_progress.set_step("done")

    _setup_thread = threading.Thread(target=_run, daemon=True)
    _setup_thread.start()
    return _setup_progress.snapshot()


@router.get("/api/setup/progress")
def get_setup_progress() -> dict:
    return _setup_progress.snapshot()
