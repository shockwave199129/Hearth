"""FastAPI app + websocket entrypoint, and a `--cli` mode that runs the loop
directly against the local mic/speaker (no frontend needed to validate
Phase 1: mic -> Moonshine -> LFM2.5 -> Parler-TTS/Kokoro -> speaker).
"""

import argparse
import asyncio
import io
import json
import logging
import threading
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
from app.setup import orchestrator
from app.setup.installer import InstallProgress
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


def _profile_context_addition(profile: UserProfile) -> str:
    """Short spoken-context block from onboarding fields so the agent actually
    uses age/profession/stressors/gender — previously only name + companion
    name were injected into the system prompt."""
    parts: list[str] = []
    if profile.age_range:
        parts.append(f"They are in the {profile.age_range} age range.")
    if profile.gender:
        parts.append(f"They identify as {profile.gender}.")
    if profile.profession:
        parts.append(f"Their work or role is {profile.profession}.")
    if profile.stressors:
        joined = ", ".join(profile.stressors)
        parts.append(f"Things that have been weighing on them include: {joined}.")
    if not parts:
        return ""
    return "\n\nWhat you already know about them from setup:\n" + " ".join(parts)


def _pcm_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Wraps float32 PCM as an in-memory WAV file (stdlib `wave`, no new
    dependency) — used for on-demand replay of a past reply."""
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())
    return buf.getvalue()


app = FastAPI(title="Hearth")

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
            + _profile_context_addition(profile)
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

    def _synthesize_required(self, reply_text: str) -> tuple[np.ndarray, int]:
        """Voice is the product — synthesize with one retry, then raise.
        Never return a text-only companion turn when speak_replies is on."""
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                reply_audio = self.tts.synthesize(reply_text, voice=self.profile.preferred_voice)
                pcm = np.asarray(reply_audio, dtype=np.float32).reshape(-1)
                if pcm.size == 0:
                    raise RuntimeError("TTS returned empty audio")
                return pcm, self.tts.sample_rate
            except Exception as exc:
                last_exc = exc
                logger.exception("TTS attempt %s failed", attempt + 1)
        raise RuntimeError("TTS failed after retries") from last_exc

    def _commit_turn(
        self, memory: ShortTermMemory, transcript: str, reply_text: str
    ) -> int:
        turn_id = memory.add_turn(transcript, reply_text)
        chat_history.record_turn(self.profile.user_id, memory.session_id, turn_id, "user", transcript)
        return chat_history.record_turn(
            self.profile.user_id, memory.session_id, turn_id, "assistant", reply_text
        )

    def _handle_turn(self, transcript: str, memory: ShortTermMemory) -> tuple[str, str, np.ndarray | None, int, int]:
        crisis_match = crisis_detector.detect(transcript)
        if crisis_match is not None:
            # Crisis audio always plays, regardless of speak_replies — see
            # UserProfile.speak_replies's docstring.
            return self._respond_to_crisis(transcript, crisis_match, memory)

        messages = self._build_messages(memory, transcript)
        reply_text = self._run_agent_with_self_check(messages)
        if not self.profile.speak_replies:
            turn_db_id = self._commit_turn(memory, transcript, reply_text)
            return transcript, reply_text, None, 0, turn_db_id
        # Synthesize before committing history so a failed voice turn does not
        # leave a text-only reply that only shows up after restart.
        reply_audio, sample_rate = self._synthesize_required(reply_text)
        turn_db_id = self._commit_turn(memory, transcript, reply_text)
        return transcript, reply_text, reply_audio, sample_rate, turn_db_id

    def _respond_to_crisis(
        self, transcript: str, crisis_match: crisis_detector.CrisisMatch, memory: ShortTermMemory
    ) -> tuple[str, str, np.ndarray | None, int, int]:
        """On a crisis-detector match: skip the LLM in favor of the fixed
        SAFETY_RESPONSE_TEXT (project-plan.md §9). Speak it via the live TTS
        engine so the words match the text."""
        crisis_detector.record_event(crisis_match, self.profile.user_id)
        try:
            escalation.maybe_escalate(self.profile.user_id, reason=crisis_match.severity)
        except Exception:
            logger.exception("escalation check failed — safety response still proceeds")
        reply_audio, sample_rate = self._synthesize_required(SAFETY_RESPONSE_TEXT)
        turn_db_id = self._commit_turn(memory, transcript, SAFETY_RESPONSE_TEXT)
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

# One process-wide progress tracker for the setup flow — see
# app/setup/orchestrator.py. A thread rather than an async task since
# run_setup() does blocking subprocess/network calls throughout.
_setup_progress = InstallProgress()
_setup_thread: threading.Thread | None = None


def _require_pipeline() -> None:
    """Every endpoint below Pipeline() used to bare `assert _pipeline is
    not None` — a thin build (CI no longer bundles torch/onnxruntime, see
    the project setup plan, so _startup() can't construct Pipeline() until
    /api/setup/start finishes) means that's now an expected, recoverable
    state, not a should-never-happen bug — a 503 lets the frontend show
    "finish setup first" instead of crashing."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Setup not complete — see /api/setup/status")


@app.on_event("startup")
def _startup() -> None:
    global _pipeline
    if orchestrator.detect_status()["complete"]:
        try:
            _pipeline = Pipeline()
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
    if _pipeline is not None:
        _pipeline.shutdown()


@app.get("/api/status")
def get_status() -> dict:
    _require_pipeline()
    tier = _pipeline.tier
    return {
        "tier": tier.tier,
        "llm_gguf": tier.llm_gguf,
        "stt_model": tier.stt_model,
        "tts_engine": tier.tts_engine,
        "n_gpu_layers": tier.n_gpu_layers,
        "ctx_size": tier.ctx_size,
        "hardware": detect_hardware(),
    }


@app.get("/api/setup/status")
def get_setup_status() -> dict:
    return orchestrator.detect_status()


@app.post("/api/setup/start")
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
        global _pipeline
        orchestrator.run_setup(_setup_progress)
        # run_setup leaves step at downloading_models on success (never
        # "done") — "done" is reserved for after Pipeline() + mark_complete.
        if _setup_progress.snapshot()["step"] == "error":
            return

        _setup_progress.set_step("starting_engines")
        _setup_progress.append_log("Starting speech and language engines…")
        try:
            _pipeline = Pipeline()
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


@app.get("/api/setup/progress")
def get_setup_progress() -> dict:
    return _setup_progress.snapshot()


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
    _require_pipeline()
    # A plain attribute swap, not set_profile() — speak_replies doesn't
    # affect the system prompt or the agent's tools, so there's no need to
    # rebuild the create_agent graph just to flip this.
    _pipeline.profile = updated
    return updated


@app.post("/api/onboarding")
def api_complete_onboarding(payload: OnboardingRequest) -> UserProfile:
    """Creates a new profile and activates it — used for first-run
    onboarding AND for adding another profile later (Settings → Profiles →
    Add another profile reuses this same form/endpoint).

    Profile + active_user_id are persisted first so a later launch still
    skips onboarding even if wiring the live Pipeline fails mid-request.
    """
    profile = create_profile(payload)
    set_active_user_id(profile.user_id)
    if _pipeline is not None:
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
    _require_pipeline()
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

    _require_pipeline()
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
    _require_pipeline()
    return long_term.list_memories(_pipeline.profile.user_id, category)


@app.get("/api/memories/{mem_id}")
def api_get_memory(mem_id: str) -> dict:
    _require_pipeline()
    result = long_term.get(mem_id, _pipeline.profile.user_id)
    if result is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return result


@app.put("/api/memories/{mem_id}")
def api_update_memory(mem_id: str, payload: MemoryUpdateRequest) -> dict:
    _require_pipeline()
    user_id = _pipeline.profile.user_id
    if long_term.get(mem_id, user_id) is None:
        raise HTTPException(status_code=404, detail="memory not found")
    long_term.update(mem_id, payload.text, user_id)
    return long_term.get(mem_id, user_id)


@app.delete("/api/memories/{mem_id}")
def api_delete_memory(mem_id: str) -> dict:
    _require_pipeline()
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
    _require_pipeline()
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
    _require_pipeline()
    user_id = _pipeline.profile.user_id
    last = escalation.last_escalation(user_id)
    return {
        "recent_crisis_events": crisis_detector.event_count(user_id, within_days=7),
        "last_escalation_at": last.isoformat() if last else None,
    }


@app.get("/api/chat_history")
def api_list_chat_history(limit: int = 40, before_id: int | None = None) -> dict:
    """Paginated chat rows for the Talk transcript. Newest page by default;
    pass ``before_id`` (smallest id already shown) to load an older page
    when the user scrolls up. Response: ``{items, has_more}``."""
    _require_pipeline()
    return chat_history.list_turns(_pipeline.profile.user_id, limit, before_id)


@app.get("/api/chat_history/{row_id}/audio")
def api_replay_chat_history(row_id: int) -> Response:
    """Re-synthesize the stored reply text with the profile's preferred
    voice (same TTS path as live chat). No audio files are kept on disk."""
    _require_pipeline()
    turn = chat_history.get_turn(_pipeline.profile.user_id, row_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="turn not found")
    if turn["role"] != "assistant":
        raise HTTPException(status_code=400, detail="only assistant replies can be replayed")
    try:
        audio = _pipeline.tts.synthesize(turn["content"], voice=_pipeline.profile.preferred_voice)
        if audio is None or len(np.asarray(audio).reshape(-1)) == 0:
            raise RuntimeError("TTS returned empty audio")
        wav_bytes = _pcm_to_wav_bytes(audio, _pipeline.tts.sample_rate)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("TTS replay failed for turn %s", row_id)
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc
    return Response(content=wav_bytes, media_type="audio/wav")


@app.delete("/api/chat_history/{row_id}")
def api_delete_chat_history(row_id: int) -> dict:
    _require_pipeline()
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
    entirely when the profile has speak_replies off. When speak_replies is
    on, audio is synthesized before any reply frame is sent (voice is the
    product). Short-term memory is scoped to this one connection; long-term
    memory maintenance runs once, silently, when it ends — see
    project-plan.md §5."""
    await ws.accept()
    _require_pipeline()
    memory = _pipeline.new_session_memory()
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect
            try:
                if "bytes" in message and message["bytes"] is not None:
                    audio = np.frombuffer(message["bytes"], dtype=np.float32)
                    transcript, reply_text, reply_audio, sample_rate, turn_db_id = await asyncio.to_thread(
                        _pipeline.respond, audio, memory
                    )
                else:
                    payload = json.loads(message["text"])
                    transcript, reply_text, reply_audio, sample_rate, turn_db_id = await asyncio.to_thread(
                        _pipeline.respond_to_text, payload["text"], memory
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
            if _pipeline.profile.speak_replies and not has_audio:
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
            if reply_audio is not None:
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
