"""FastAPI app + websocket entrypoint, and a `--cli` mode that runs the loop
directly against the local mic/speaker (no frontend needed to validate
Phase 1: mic -> Moonshine -> LFM2.5 -> Parler-TTS/Kokoro -> speaker).
"""

import argparse
import io
import json
import logging
import wave
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.checkin.state import delete_checkin, get_last_checkin
from app.checkin.tools import make_tools as make_checkin_tools
from app.config import (
    APP_HOST,
    APP_PORT,
    CHECKIN_PROMPT_TEMPLATE,
    MEMORY_SYSTEM_PROMPT_ADDITION,
    SAFETY_AUDIO_PATH,
    SAFETY_RESPONSE_TEXT,
    SKILLS_SYSTEM_PROMPT_ADDITION,
    SYSTEM_PROMPT_TEMPLATE,
)
from app.eval.self_check import flag_reply
from app.hardware.detect import detect_hardware
from app.hardware.tier_manager import detect_and_cache_tier
from app.llm.server_manager import LlmServer
from app.memory import chat_history, long_term
from app.memory.short_term import ShortTermMemory
from app.memory.tools import make_tools as make_memory_tools
from app.onboarding.active_profile import clear_active_user_id, get_active_user_id, set_active_user_id
from app.onboarding.profile_schema import OnboardingRequest, UserProfile
from app.onboarding.profile_store import (
    create_profile,
    delete_profile,
    get_profile,
    list_profiles,
    update_speak_replies,
)
from app.safety import crisis_detector, escalation
from app.skills.loader import get_skill, load_catalog
from app.skills.tools import SKILL_TOOLS
from app.stt.moonshine_engine import MoonshineEngine
from app.tts.tts_engines import get_tts_engine

logger = logging.getLogger("hearth")

_ROLE_TO_LC_MESSAGE = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}

# Appended to the system prompt for exactly one regeneration attempt when
# eval/self_check.py flags a reply — see project-plan.md §7.
_SELF_CHECK_NUDGE = "\n\nKeep it to 2-3 short spoken sentences, no lists, no clinical or diagnostic language."


def _to_lc_message(message: dict):
    return _ROLE_TO_LC_MESSAGE[message["role"]](content=message["content"])


def _load_safety_audio() -> tuple[np.ndarray, int]:
    """Loads the pre-synthesized crisis response (see config.SAFETY_AUDIO_PATH)
    via stdlib `wave` rather than adding a new audio dependency — this is
    read once per trigger, not a hot path. See app/safety/safety_audio/README.md."""
    with wave.open(str(SAFETY_AUDIO_PATH), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sample_rate


def _pcm_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Wraps float32 PCM as an in-memory WAV file (stdlib `wave`, no new
    dependency) — used for on-demand replay of a past reply."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wav_file.writeframes(pcm16.tobytes())
    return buf.getvalue()


app = FastAPI(title="Hearth")

# Dev-only: the Vite dev server (localhost:5173) calls /api/* directly.
# In production the frontend build is served from this same origin, so this
# has no effect once the app is packaged.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Used for LLM/STT/TTS before any real profile has ever been created, so
# --cli and /ws still work out of the box. Never persisted to `profiles` —
# a fixed sentinel user_id, distinct from any real (uuid4) profile id.
# get_active_user_id() is the source of truth for "has anyone onboarded yet"
# (see /api/profile below).
DEFAULT_PROFILE = UserProfile(
    user_id="default", name="friend", companion_name="Companion", created_at=datetime.now(timezone.utc)
)


class Pipeline:
    """Owns the tier decision, the three model engines, and the active
    profile; one instance per process, wired up in the FastAPI startup
    event and reused by --cli. Multiple profiles can exist per install
    (Settings → Profiles) but only one is ever active in a running
    Pipeline — switching is a deliberate action, not per-request routing."""

    def __init__(self):
        self.tier = detect_and_cache_tier()
        logger.info("Selected hardware tier: %s", self.tier.tier)

        self.llm = LlmServer(self.tier)
        self.llm.start()

        # ChatOpenAI just talks to the same llama-server /v1/chat/completions
        # endpoint the hand-rolled client used before — see project-plan.md §5
        # and the create_agent migration notes.
        self.chat_model = ChatOpenAI(model="local", base_url=f"{self.llm.base_url}/v1", api_key="not-needed")

        self.stt = MoonshineEngine(self.tier.stt_model)
        self.tts = get_tts_engine(self.tier)

        active_user_id = get_active_user_id()
        initial_profile = get_profile(active_user_id) if active_user_id else None
        self.set_profile(initial_profile or DEFAULT_PROFILE)

    def _build_tools(self, user_id: str) -> list:
        """Tools are bound to one profile via closure (make_tools) rather
        than taking user_id as an LLM-fillable argument — the model must
        never be able to name an arbitrary profile. Rebuilt on every
        set_profile call, i.e. whenever the active profile changes."""
        return make_memory_tools(user_id) + SKILL_TOOLS + make_checkin_tools(user_id)

    def set_profile(self, profile: UserProfile) -> None:
        self.profile = profile
        self.system_prompt = (
            SYSTEM_PROMPT_TEMPLATE.format(companion_name=profile.companion_name, name=profile.name)
            + MEMORY_SYSTEM_PROMPT_ADDITION
            + SKILLS_SYSTEM_PROMPT_ADDITION
        )
        self.agent = create_agent(
            model=self.chat_model,
            tools=self._build_tools(profile.user_id),
            middleware=[
                # Replaces the old hand-rolled max_iterations bound (was 4 for
                # live turns, 6 for maintenance) with one shared, slightly
                # more generous cap.
                ModelCallLimitMiddleware(run_limit=15),
                # Resilience against transient local llama-server hiccups.
                ModelRetryMiddleware(max_retries=2),
                # Defensive redaction in case a reply echoes something like
                # an emergency contact's email/URL.
                PIIMiddleware("email", strategy="redact", apply_to_output=True),
                PIIMiddleware("url", strategy="redact", apply_to_output=True),
                # Safety net, not the primary mechanism — ShortTermMemory's own
                # rolling window (project-plan.md §4) already keeps normal
                # per-turn message lists small. This guards the one case that
                # can still balloon a single invoke: run_maintenance's one-shot
                # full-session dump as a single user message.
                SummarizationMiddleware(model=self.chat_model, trigger=("messages", 40), keep=("messages", 20)),
            ],
        )

    def new_session_memory(self) -> ShortTermMemory:
        return ShortTermMemory(self.llm)

    def respond(self, audio: np.ndarray, memory: ShortTermMemory) -> tuple[str, str, np.ndarray | None, int, int]:
        """Voice input: transcribe, then everything else is shared with
        typed input via _handle_turn."""
        transcript = self.stt.transcribe(audio)
        return self._handle_turn(transcript, memory)

    def respond_to_text(self, text: str, memory: ShortTermMemory) -> tuple[str, str, np.ndarray | None, int, int]:
        """Typed input: no STT involved, otherwise identical turn handling
        (crisis check, agent run, self-check, chat history, optional TTS) —
        see project-plan.md's text-input support notes."""
        return self._handle_turn(text, memory)

    def _handle_turn(self, transcript: str, memory: ShortTermMemory) -> tuple[str, str, np.ndarray | None, int, int]:
        crisis_match = crisis_detector.detect(transcript)
        if crisis_match is not None:
            # Crisis audio always plays, regardless of speak_replies — see
            # UserProfile.speak_replies's docstring.
            return self._respond_to_crisis(transcript, crisis_match, memory)

        messages = self._build_messages(memory, transcript)
        reply_text = self._run_agent_with_self_check(messages)
        turn_id = memory.add_turn(transcript, reply_text)
        chat_history.record_turn(self.profile.user_id, memory.session_id, turn_id, "user", transcript)
        turn_db_id = chat_history.record_turn(
            self.profile.user_id, memory.session_id, turn_id, "assistant", reply_text
        )
        if not self.profile.speak_replies:
            return transcript, reply_text, None, 0, turn_db_id
        reply_audio = self.tts.synthesize(reply_text, voice=self.profile.preferred_voice)
        return transcript, reply_text, reply_audio, self.tts.sample_rate, turn_db_id

    def _respond_to_crisis(
        self, transcript: str, crisis_match: crisis_detector.CrisisMatch, memory: ShortTermMemory
    ) -> tuple[str, str, np.ndarray, int, int]:
        """On a crisis-detector match: skip the LLM and live TTS entirely in
        favor of a pre-synthesized response, per project-plan.md §9. This
        keeps the safety path independent of the model's judgment (and its
        latency) at the moment it matters most."""
        crisis_detector.record_event(crisis_match, self.profile.user_id)
        try:
            escalation.maybe_escalate(self.profile.user_id, reason=crisis_match.severity)
        except Exception:
            logger.exception("escalation check failed — safety response still proceeds")
        turn_id = memory.add_turn(transcript, SAFETY_RESPONSE_TEXT)
        chat_history.record_turn(self.profile.user_id, memory.session_id, turn_id, "user", transcript)
        turn_db_id = chat_history.record_turn(
            self.profile.user_id, memory.session_id, turn_id, "assistant", SAFETY_RESPONSE_TEXT
        )
        reply_audio, sample_rate = _load_safety_audio()
        return transcript, SAFETY_RESPONSE_TEXT, reply_audio, sample_rate, turn_db_id

    def _checkin_prompt_line(self) -> str:
        """Computed fresh each turn (single cheap row read) rather than
        cached per-session, so it self-corrects immediately after
        mark_checkin fires mid-session. See project-plan.md §8."""
        now = datetime.now(timezone.utc)
        last = get_last_checkin(self.profile.user_id)
        if last is None:
            checkin_status = "You have never asked how they're feeling."
        else:
            days = (now.date() - last.date()).days
            checkin_status = f"It has been {days} day{'s' if days != 1 else ''} since you last asked how they're feeling."
        return CHECKIN_PROMPT_TEMPLATE.format(date=now.date().isoformat(), checkin_status=checkin_status)

    def _build_messages(self, memory: ShortTermMemory, transcript: str) -> list[dict]:
        system_content = self.system_prompt + self._checkin_prompt_line()
        if memory.session_summary:
            system_content += f"\n\nEarlier in this session: {memory.session_summary}"
        messages = [{"role": "system", "content": system_content}]
        messages.extend(memory.as_api_messages())
        messages.append({"role": "user", "content": transcript})
        return messages

    def _run_agent(self, messages: list[dict]) -> str:
        """Lets the model call memory/skill/check-in tools before answering —
        see project-plan.md §5, §6, §8. The tool-calling loop, iteration
        bound (ModelCallLimitMiddleware), and retry/summarization/PII
        handling all live in the create_agent graph built in set_profile."""
        lc_messages = [_to_lc_message(m) for m in messages]
        try:
            result = self.agent.invoke({"messages": lc_messages})
        except Exception:
            logger.exception("agent invocation did not converge or failed")
            return "I'm here with you — say that again?"
        return (result["messages"][-1].content or "").strip()

    def _run_agent_with_self_check(self, messages: list[dict]) -> str:
        """Runtime pre-TTS self-check (project-plan.md §7) — a fast
        heuristic, not a second LLM call. Regenerates exactly once if
        flagged, then uses whatever comes back regardless, so this never
        blocks the conversation."""
        reply_text = self._run_agent(messages)
        reason = flag_reply(reply_text)
        if reason is None:
            return reply_text
        logger.info("self-check flagged reply (%s) — regenerating once", reason)
        nudged = [dict(m) for m in messages]
        nudged[0] = {**nudged[0], "content": nudged[0]["content"] + _SELF_CHECK_NUDGE}
        return self._run_agent(nudged)

    def run_maintenance(self, memory: ShortTermMemory) -> None:
        """End-of-session silent pass: lets the model create/update/delete
        memories based on the whole session, without it being a
        conversational event. Best-effort — a failure here should never
        take down session teardown."""
        if not memory.messages and not memory.session_summary:
            return
        transcript_lines = [f"{m['role']}: {m['content']}" for m in memory.messages]
        convo = "\n".join(filter(None, [memory.session_summary, *transcript_lines]))
        messages = [
            {"role": "system", "content": MEMORY_SYSTEM_PROMPT_ADDITION},
            {
                "role": "user",
                "content": (
                    "Review what you know about this person against this session — "
                    "create, update, or delete memories as needed. This is a silent "
                    f"background pass, not a reply to the user.\n\nSession:\n{convo}"
                ),
            },
        ]
        try:
            self._run_agent(messages)
        except Exception:
            logger.exception("end-of-session memory maintenance failed")

    def shutdown(self) -> None:
        self.llm.stop()


_pipeline: Pipeline | None = None


@app.on_event("startup")
def _startup() -> None:
    global _pipeline
    _pipeline = Pipeline()


@app.on_event("shutdown")
def _shutdown() -> None:
    if _pipeline is not None:
        _pipeline.shutdown()


@app.get("/api/status")
def get_status() -> dict:
    assert _pipeline is not None
    tier = _pipeline.tier
    return {
        "tier": tier.tier,
        "llm_gguf": tier.llm_gguf,
        "stt_model": tier.stt_model,
        "tts_engine": tier.tts_engine,
        "hardware": detect_hardware(),
    }


@app.get("/api/profile")
def api_get_profile() -> UserProfile:
    """404 (no active profile) is how the frontend tells 'never onboarded'
    apart from 'onboarded with default-ish answers'."""
    user_id = get_active_user_id()
    profile = get_profile(user_id) if user_id else None
    if profile is None:
        raise HTTPException(status_code=404, detail="no profile saved yet")
    return profile


class ProfileSettingsUpdate(BaseModel):
    speak_replies: bool


@app.put("/api/profile")
def api_update_profile(payload: ProfileSettingsUpdate) -> UserProfile:
    """Lets Settings flip lightweight preferences (currently just
    speak_replies) without redoing the whole onboarding flow."""
    user_id = get_active_user_id()
    profile = get_profile(user_id) if user_id else None
    if profile is None:
        raise HTTPException(status_code=404, detail="no profile saved yet")
    update_speak_replies(user_id, payload.speak_replies)
    updated = get_profile(user_id)
    assert _pipeline is not None
    # A plain attribute swap, not set_profile() — speak_replies doesn't
    # affect the system prompt or the agent's tools, so there's no need to
    # rebuild the create_agent graph just to flip this.
    _pipeline.profile = updated
    return updated


@app.post("/api/onboarding")
def api_complete_onboarding(payload: OnboardingRequest) -> UserProfile:
    """Creates a new profile and activates it — used for first-run
    onboarding AND for adding another profile later (Settings → Profiles →
    Add another profile reuses this same form/endpoint)."""
    profile = create_profile(payload)
    set_active_user_id(profile.user_id)
    assert _pipeline is not None
    _pipeline.set_profile(profile)
    return profile


@app.get("/api/profiles")
def api_list_profiles() -> list[UserProfile]:
    return list_profiles()


@app.post("/api/profiles/{user_id}/activate")
def api_activate_profile(user_id: str) -> UserProfile:
    profile = get_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="profile not found")
    set_active_user_id(user_id)
    assert _pipeline is not None
    _pipeline.set_profile(profile)
    return profile


@app.delete("/api/profiles/{user_id}")
def api_delete_profile(user_id: str) -> dict:
    """Cascades across every user_id-scoped table — memories, checkin,
    crisis/escalation history, and chat history — never a partial delete."""
    if get_profile(user_id) is None:
        raise HTTPException(status_code=404, detail="profile not found")
    was_active = get_active_user_id() == user_id

    delete_profile(user_id)
    long_term.delete_all_for_user(user_id)
    delete_checkin(user_id)
    crisis_detector.delete_events(user_id)
    escalation.delete_escalations(user_id)
    chat_history.delete_all_for_user(user_id)

    assert _pipeline is not None
    if was_active:
        remaining = list_profiles()
        if remaining:
            set_active_user_id(remaining[0].user_id)
            _pipeline.set_profile(remaining[0])
        else:
            clear_active_user_id()
            _pipeline.set_profile(DEFAULT_PROFILE)
    return {"ok": True}


class MemoryUpdateRequest(BaseModel):
    text: str


@app.get("/api/memories")
def api_list_memories(category: str | None = None) -> list[dict]:
    assert _pipeline is not None
    return long_term.list_memories(_pipeline.profile.user_id, category)


@app.get("/api/memories/{mem_id}")
def api_get_memory(mem_id: str) -> dict:
    assert _pipeline is not None
    result = long_term.get(mem_id, _pipeline.profile.user_id)
    if result is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return result


@app.put("/api/memories/{mem_id}")
def api_update_memory(mem_id: str, payload: MemoryUpdateRequest) -> dict:
    assert _pipeline is not None
    user_id = _pipeline.profile.user_id
    if long_term.get(mem_id, user_id) is None:
        raise HTTPException(status_code=404, detail="memory not found")
    long_term.update(mem_id, payload.text, user_id)
    return long_term.get(mem_id, user_id)


@app.delete("/api/memories/{mem_id}")
def api_delete_memory(mem_id: str) -> dict:
    assert _pipeline is not None
    long_term.delete(mem_id, _pipeline.profile.user_id)
    return {"ok": True}


@app.get("/api/skills")
def api_list_skills() -> list[dict]:
    """Read-only — the skills library is static reference content, not
    user data, so there's no edit/delete surface (unlike /api/memories)."""
    return [
        {"id": s.id, "title": s.title, "tags": s.tags, "summary": s.summary}
        for s in load_catalog()
    ]


@app.get("/api/skills/{skill_id}")
def api_get_skill(skill_id: str) -> dict:
    skill = get_skill(skill_id)
    if skill is None:
        raise HTTPException(status_code=404, detail="skill not found")
    return {"id": skill.id, "title": skill.title, "content": skill.content, "source": skill.source}


@app.get("/api/checkin")
def api_get_checkin() -> dict:
    assert _pipeline is not None
    last = get_last_checkin(_pipeline.profile.user_id)
    days_since = (datetime.now(timezone.utc).date() - last.date()).days if last else None
    return {
        "last_checkin_at": last.isoformat() if last else None,
        "days_since_last_checkin": days_since,
    }


@app.get("/api/safety/status")
def api_get_safety_status() -> dict:
    """Read-only transparency surface — same 'never actually hidden'
    principle as /api/memories, /api/skills, /api/checkin. See
    project-plan.md §9."""
    assert _pipeline is not None
    user_id = _pipeline.profile.user_id
    last = escalation.last_escalation(user_id)
    return {
        "recent_crisis_events": crisis_detector.event_count(user_id, within_days=7),
        "last_escalation_at": last.isoformat() if last else None,
    }


@app.get("/api/chat_history")
def api_list_chat_history(limit: int = 50) -> list[dict]:
    assert _pipeline is not None
    return chat_history.list_turns(_pipeline.profile.user_id, limit)


@app.get("/api/chat_history/{row_id}/audio")
def api_replay_chat_history(row_id: int) -> Response:
    """Re-synthesizes a stored past reply on demand via the normal TTS
    engine — no audio files are cached anywhere (see project-plan.md's
    audio_cache discussion; this replaces that with plain re-synthesis)."""
    assert _pipeline is not None
    turn = chat_history.get_turn(_pipeline.profile.user_id, row_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="turn not found")
    if turn["role"] != "assistant":
        raise HTTPException(status_code=400, detail="only assistant replies can be replayed")
    audio = _pipeline.tts.synthesize(turn["content"], voice=_pipeline.profile.preferred_voice)
    wav_bytes = _pcm_to_wav_bytes(audio, _pipeline.tts.sample_rate)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.delete("/api/chat_history/{row_id}")
def api_delete_chat_history(row_id: int) -> dict:
    assert _pipeline is not None
    chat_history.delete_turn(_pipeline.profile.user_id, row_id)
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Protocol: client sends either a binary frame per utterance (mono
    float32 PCM @ SAMPLE_RATE) or a text frame (JSON `{"type": "text",
    "text": "..."}`) for typed input — both share the same session/memory,
    so a conversation can freely mix voice and text turns. Server replies
    with one text frame (JSON metadata: transcript, reply_text,
    sample_rate, turn_db_id, has_audio) followed by a binary frame (mono
    float32 PCM reply audio) only when has_audio is true — skipped
    entirely when the profile has speak_replies off. Short-term memory is
    scoped to this one connection; long-term memory maintenance runs once,
    silently, when it ends — see project-plan.md §5."""
    await ws.accept()
    assert _pipeline is not None
    memory = _pipeline.new_session_memory()
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect
            if "bytes" in message and message["bytes"] is not None:
                audio = np.frombuffer(message["bytes"], dtype=np.float32)
                transcript, reply_text, reply_audio, sample_rate, turn_db_id = _pipeline.respond(audio, memory)
            else:
                payload = json.loads(message["text"])
                transcript, reply_text, reply_audio, sample_rate, turn_db_id = _pipeline.respond_to_text(
                    payload["text"], memory
                )
            await ws.send_text(json.dumps({
                "transcript": transcript,
                "reply_text": reply_text,
                "sample_rate": sample_rate,
                "turn_db_id": turn_db_id,
                "has_audio": reply_audio is not None,
            }))
            if reply_audio is not None:
                await ws.send_bytes(reply_audio.astype(np.float32).tobytes())
    except WebSocketDisconnect:
        logger.info("client disconnected")
    finally:
        _pipeline.run_maintenance(memory)


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
            play_audio(reply_audio, sample_rate)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.run_maintenance(memory)
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
