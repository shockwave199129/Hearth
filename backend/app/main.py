"""FastAPI app + websocket entrypoint, and a `--cli` mode that runs the loop
directly against the local mic/speaker (no frontend needed to validate
Phase 1: mic -> Moonshine -> LFM2.5 -> Parler-TTS/Kokoro -> speaker).

Composition root only — route handlers live under ``app.api.*``, and the
conversation engines live in ``app.pipeline``. See CODE_REVIEW ARCH-1.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, memory, profile, setup, skills, status
from app.api.auth import require_api_token, token_matches
from app.config import APP_HOST, APP_PORT
from app.deps import get_pipeline_optional, set_pipeline
from app.pipeline import Pipeline
from app.setup import orchestrator

logger = logging.getLogger("hearth")

# Kept for tests that historically patched / called through app.main
# (test_api_token.py). Prefer app.api.auth.token_matches and app.config.
_token_matches = token_matches

app = FastAPI(title="Hearth")

app.middleware("http")(require_api_token)

# The UI is never served from this process in the packaged app — Tauri loads
# frontend/dist from its own origin (https://tauri.localhost / tauri://localhost)
# and the frontend calls http://127.0.0.1:48173 (see frontend/src/lib/backendUrl.ts).
# Dev uses the Vite proxy on :48176. Backend is loopback-only (APP_HOST), so
# opening these origins is the right CORS surface, not "same origin".
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:48176",
        "http://127.0.0.1:48176",
        "http://tauri.localhost",
        "https://tauri.localhost",
        "tauri://localhost",
    ],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(status.router)
app.include_router(setup.router)
app.include_router(profile.router)
app.include_router(memory.router)
app.include_router(skills.router)
app.include_router(chat.router)


@app.on_event("startup")
def _startup() -> None:
    if orchestrator.detect_status()["complete"]:
        try:
            set_pipeline(Pipeline())
        except Exception:
            # Flag/assets said "done" but the app still can't start (e.g.
            # backend-deps wiped). Clear the flag so the Setup UI can recover
            # instead of leaving FastAPI dead on boot.
            logger.exception(
                "Pipeline() failed on startup despite setup marked complete — clearing setup_state"
            )
            from app.setup.state import clear_setup_complete

            clear_setup_complete()
            return
        # Backfill the DB flag for installs that already had packages/models
        # (e.g. scripts/setup.py) before setup_state existed.
        orchestrator.mark_complete()
    else:
        logger.info("Setup not complete yet — waiting for /api/setup/start before building the pipeline.")


@app.on_event("shutdown")
def _shutdown() -> None:
    pipeline = get_pipeline_optional()
    if pipeline is not None:
        pipeline.shutdown()
        set_pipeline(None)


def run_cli_loop() -> None:
    """Runs the pipeline directly against the local mic/speaker — the
    quickest way to validate Phase 1 end-to-end without the frontend."""
    from app.audio_io import play_audio, record_utterance

    pipeline = Pipeline()
    memory = pipeline.new_session_memory()
    print(f"Ready (tier {pipeline.tier.tier}). Speak after each prompt; Ctrl+C to quit.")
    try:
        while True:
            input("\n[press Enter, then speak]")
            audio = record_utterance()
            if audio.size == 0:
                print("(heard nothing)")
                continue
            transcript, reply_text, reply_audio, sample_rate, _turn_db_id = pipeline.respond(audio, memory)
            print(f"You: {transcript}")
            print(f"{pipeline.profile.companion_name}: {reply_text}")
            if reply_audio is not None:
                play_audio(reply_audio, sample_rate)
    except KeyboardInterrupt:
        pass
    finally:
        asyncio.run(pipeline.run_maintenance(memory))
        pipeline.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="run mic/speaker loop directly, no server")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.cli:
        run_cli_loop()
    else:
        import uvicorn

        uvicorn.run(app, host=APP_HOST, port=APP_PORT)
