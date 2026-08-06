"""Chat history + WebSocket conversation routes."""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Response, WebSocket, WebSocketDisconnect

from app import config
from app.api.audio import pcm_to_wav_bytes
from app.api.auth import token_matches
from app.deps import Pipeline, get_pipeline
from app.memory import chat_history

logger = logging.getLogger("hearth")

router = APIRouter()


@router.get("/api/chat_history")
def api_list_chat_history(
    limit: int = 40,
    before_id: int | None = None,
    pipeline: Pipeline = Depends(get_pipeline),
) -> dict:
    """Paginated chat rows for the Talk transcript. Newest page by default;
    pass ``before_id`` (smallest id already shown) to load an older page
    when the user scrolls up. Response: ``{items, has_more}``."""
    return chat_history.list_turns(pipeline.profile.user_id, limit, before_id)


@router.get("/api/chat_history/{row_id}/audio")
def api_replay_chat_history(row_id: int, pipeline: Pipeline = Depends(get_pipeline)) -> Response:
    """Re-synthesize the stored reply text with the profile's preferred
    voice (same TTS path as live chat). No audio files are kept on disk."""
    turn = chat_history.get_turn(pipeline.profile.user_id, row_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="turn not found")
    if turn["role"] != "assistant":
        raise HTTPException(status_code=400, detail="only assistant replies can be replayed")
    try:
        audio = pipeline.tts.synthesize(
            turn["content"],
            voice=pipeline.profile.preferred_voice,
            style=pipeline.profile.voice_style,
        )
        if audio is None or len(np.asarray(audio).reshape(-1)) == 0:
            raise RuntimeError("TTS returned empty audio")
        wav_bytes = pcm_to_wav_bytes(audio, pipeline.tts.sample_rate)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("TTS replay failed for turn %s", row_id)
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc
    return Response(content=wav_bytes, media_type="audio/wav")


@router.delete("/api/chat_history/{row_id}")
def api_delete_chat_history(row_id: int, pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    chat_history.delete_turn(pipeline.profile.user_id, row_id)
    return {"ok": True}


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Protocol: client sends either a binary frame per utterance (mono
    float32 PCM @ SAMPLE_RATE) or a text frame (JSON `{"type": "text",
    "text": "..."}`) for typed input — both share the same session/memory,
    so a conversation can freely mix voice and text turns. Server replies
    with one text frame (JSON metadata: transcript, reply_text,
    sample_rate, turn_db_id, has_audio) followed by a binary frame (mono
    float32 PCM reply audio) only when has_audio is true — skipped
    entirely when the profile has speak_replies off. When speak_replies is
    on, audio is synthesized before any reply frame is sent (voice is the
    product). Short-term memory is scoped to this one connection; long-term
    memory maintenance runs once, silently, when it ends — see
    docs/project-plan.md §5.

    The local API token travels as a `?token=` query parameter here rather
    than the header the HTTP routes use: the browser WebSocket constructor
    cannot set request headers, and Starlette's HTTP middleware never runs
    for a websocket scope regardless. Rejected before accept(), so the
    handshake fails outright instead of opening and immediately closing."""
    if config.API_TOKEN and not token_matches(ws.query_params.get("token")):
        await ws.close(code=1008)
        return
    await ws.accept()
    pipeline = get_pipeline()
    memory = pipeline.new_session_memory()
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect
            try:
                if "bytes" in message and message["bytes"] is not None:
                    audio = np.frombuffer(message["bytes"], dtype=np.float32)
                    transcript, reply_text, reply_audio, sample_rate, turn_db_id = await asyncio.to_thread(
                        pipeline.respond, audio, memory
                    )
                else:
                    payload = json.loads(message["text"])
                    transcript, reply_text, reply_audio, sample_rate, turn_db_id = await asyncio.to_thread(
                        pipeline.respond_to_text, payload["text"], memory
                    )
            except Exception:
                logger.exception("turn failed — notifying client without dropping the socket")
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "I couldn't speak that reply — please try again.",
                        }
                    )
                )
                continue

            has_audio = reply_audio is not None
            if pipeline.profile.speak_replies and not has_audio:
                # speak_replies on must never deliver text-only companion turns.
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "I couldn't speak that reply — please try again.",
                        }
                    )
                )
                continue

            await ws.send_text(
                json.dumps(
                    {
                        "transcript": transcript,
                        "reply_text": reply_text,
                        "sample_rate": sample_rate,
                        "turn_db_id": turn_db_id,
                        "has_audio": has_audio,
                    }
                )
            )
            if has_audio:
                try:
                    pcm = np.asarray(reply_audio, dtype=np.float32).reshape(-1)
                    await ws.send_bytes(pcm.tobytes())
                except Exception:
                    logger.exception(
                        "failed to send reply audio for turn %s",
                        turn_db_id,
                    )
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "I couldn't speak that reply — please try again.",
                            }
                        )
                    )
    except WebSocketDisconnect:
        logger.info("client disconnected")
    finally:
        await pipeline.run_maintenance(memory)
