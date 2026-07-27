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
from pydantic import BaseModel

from app.cognitive import CognitiveScheduler, PromptBuilder, ResponseComposer, StateManager
from app.execution.llm_adapter import LlmAdapter
from app.intervention.engine import InterventionEngine
from app.workers import NlpWorkerRunner
from app.checkin.state import delete_checkin, get_last_checkin
from app.config import (
    APP_HOST,
    APP_PORT,
    CHECKIN_PROMPT_TEMPLATE,
    SAFETY_RESPONSE_TEXT,
    DATA_DIR,
)
from app.eval.self_check import flag_reply
from app.hardware.detect import detect_hardware
from app.hardware.tier_manager import detect_and_cache_tier
from app.llm.server_manager import LlmServer
from app.memory import chat_history, long_term
from app.memory.formation import process_session_memory
from app.memory.short_term import ShortTermMemory
from app.onboarding.active_profile import clear_active_user_id, get_active_user_id, set_active_user_id
from app.onboarding.profile_schema import OnboardingRequest, UserProfile
from app.onboarding.profile_store import (
    create_profile,
    delete_profile,
    get_profile,
    list_profiles,
    update_communication_preferences,
    update_relationship_state,
    update_speak_replies,
)
from app.safety import crisis_detector, escalation
from app.setup import orchestrator
from app.setup.installer import InstallProgress
from app.relationship.engine import update_relationship
from app.relationship.engine import RelationshipState
from app.safety2.worker import SafetyWorker
from app.safety2.audit import purge_expired
from app.learning.observation_store import ObservationStore
from app.learning.recompute import recompute_all
from app.evaluation.worker import EvaluationWorker
from app.skills.loader import get_skill, load_catalog
from app.stt.moonshine_engine import MoonshineEngine
from app.tts.tts_engines import get_tts_engine

logger = logging.getLogger("hearth")

# Appended to the system prompt for exactly one regeneration attempt when
# eval/self_check.py flags a reply — see project-plan.md §7.
_SELF_CHECK_NUDGE = "\n\nKeep it to 2-3 short spoken sentences, no lists, no clinical or diagnostic language."


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
        self.llm_adapter = LlmAdapter(self.llm)

        self.stt = MoonshineEngine(self.tier.stt_model)
        self.tts = get_tts_engine(self.tier)
        self.scheduler = CognitiveScheduler()
        self.prompt_builder = PromptBuilder()
        self.response_composer = ResponseComposer()
        self.nlp_workers = NlpWorkerRunner()
        self.intervention_engine = InterventionEngine()
        self.safety_worker = SafetyWorker()
        self.learning_store = ObservationStore()
        self.evaluation_worker = EvaluationWorker(self.learning_store)
        self.state_manager = StateManager(
            snapshot_path=DATA_DIR / "runtime_snapshot.json",
            hearth_version="phase0",
            model_version=self.tier.llm_gguf,
        )
        self.last_snapshot = self.state_manager.load_snapshot()
        self.mind_state = self.last_snapshot.mind_state if self.last_snapshot else self.state_manager.create_mind_state()

        active_user_id = get_active_user_id()
        initial_profile = get_profile(active_user_id) if active_user_id else None
        self.set_profile(initial_profile or DEFAULT_PROFILE)

    def set_profile(self, profile: UserProfile) -> None:
        self.profile = profile
        if self.last_snapshot and self.last_snapshot.profile.user_id == profile.user_id:
            self.mind_state = self.last_snapshot.mind_state
        else:
            self.mind_state = self.state_manager.create_mind_state()

    def new_session_memory(self) -> ShortTermMemory:
        memory = ShortTermMemory(self.llm)
        if self.last_snapshot is not None and self.last_snapshot.profile.user_id == self.profile.user_id:
            conversation_state = self.last_snapshot.conversation_state
            memory.restore(
                messages=conversation_state.get("messages", []),
                session_summary=conversation_state.get("session_summary", ""),
                session_id=conversation_state.get("session_id"),
                next_turn_id=conversation_state.get("next_turn_id"),
            )
        return memory

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
        turn_db_id = chat_history.record_turn(
            self.profile.user_id, memory.session_id, turn_id, "assistant", reply_text
        )
        self._save_runtime_snapshot(memory)
        return turn_db_id

    def _append_learning_observations(self, transcript: str, reply_text: str, intervention) -> None:
        lower = transcript.lower()
        reply_lower = reply_text.lower()
        self.learning_store.append(
            "communication",
            "likes_reflection",
            1.0 if any(token in reply_lower for token in ("i hear you", "that makes sense", "sounds", "it seems")) else 0.2,
            {"transcript": transcript, "reply": reply_text},
            "growth_engine",
        )
        self.learning_store.append(
            "communication",
            "prefers_questions",
            1.0 if "?" in reply_text else 0.0,
            {"reply": reply_text},
            "growth_engine",
        )
        self.learning_store.append(
            "communication",
            "likes_direct_advice",
            1.0 if any(token in reply_lower for token in ("you could", "try", "consider")) else 0.0,
            {"reply": reply_text},
            "growth_engine",
        )
        if intervention and getattr(intervention, "primary_skill", None) is not None:
            skill_id = intervention.primary_skill.skill.id
            self.learning_store.append(
                "skill",
                skill_id,
                1.0 if any(token in reply_lower for token in ("help", "try", "let's")) else 0.5,
                {"strategy": intervention.strategy, "reply": reply_text},
                "intervention_engine",
            )
        self.learning_store.append(
            "relationship",
            "general_trust",
            1.0 if any(token in reply_lower for token in ("i'm here", "i hear you", "glad you told me")) else 0.3,
            {"reply": reply_text},
            "growth_engine",
        )
        self.learning_store.append(
            "relationship",
            "vulnerability_trust",
            1.0 if any(token in lower for token in ("i feel", "i'm scared", "i need to tell you", "i'm overwhelmed")) else 0.2,
            {"transcript": transcript},
            "growth_engine",
        )
        self.learning_store.append(
            "relationship",
            "attachment_signal",
            1.0 if any(token in lower for token in ("you are all i have", "don't leave me", "need you all the time")) else 0.0,
            {"transcript": transcript},
            "growth_engine",
        )

    def _build_memory_context(self, transcript: str) -> str:
        retrieved = long_term.search(transcript, self.profile.user_id, k=4)
        if not retrieved:
            return ""
        lines = [f"- {item['category']}: {item['text']}" for item in retrieved[:3]]
        return "Relevant memories:\n" + "\n".join(lines)

    def _save_runtime_snapshot(self, memory: ShortTermMemory) -> None:
        conversation_state = {
            "session_id": memory.session_id,
            "messages": memory.messages,
            "session_summary": memory.session_summary,
            "next_turn_id": memory._next_turn_id,
        }
        self.last_snapshot = self.state_manager.save_snapshot(
            profile=self.profile,
            mind_state=self.mind_state,
            conversation_state=conversation_state,
            last_prompt_plan=getattr(self, "_last_prompt_plan", None),
            runtime_metrics={"turn_count": self.mind_state.turn_count},
        )

    def _handle_turn(self, transcript: str, memory: ShortTermMemory) -> tuple[str, str, np.ndarray | None, int, int]:
        purge_expired()
        safety = self.safety_worker.assess(
            self.profile.user_id,
            transcript,
            self.profile,
            relationship=self._relationship_state(),
        )
        if safety.route != "ordinary":
            return self._respond_to_safety(transcript, safety, memory)

        task = self.scheduler.schedule(transcript, self.mind_state, session_summary=memory.session_summary)
        self.nlp_workers.run(task.workers, transcript, self.mind_state)
        memory_context = self._build_memory_context(transcript)
        prompt_context = "\n\n".join(filter(None, [self._build_prompt_context(transcript, memory), memory_context]))
        intervention = self.intervention_engine.plan(
            transcript, self.profile, self.mind_state.stage, crisis=False
        )
        prompt, prompt_plan = self.prompt_builder.build(
            self.profile, self.mind_state, transcript, prompt_context, intervention
        )
        self._last_prompt_plan = prompt_plan
        reply_text = self._generate_reply(prompt, task.budget.max_response_tokens)
        reply_text = self._apply_self_check(prompt, reply_text)
        reply_result = self.response_composer.compose(reply_text)
        reply_text = reply_result.text
        self.mind_state.last_assistant_message = reply_text
        self._update_relationship_snapshot(transcript, reply_text)
        self._append_learning_observations(transcript, reply_text, intervention)
        if not self.profile.speak_replies:
            turn_db_id = self._commit_turn(memory, transcript, reply_text)
            return transcript, reply_text, None, 0, turn_db_id
        # Synthesize before committing history so a failed voice turn does not
        # leave a text-only reply that only shows up after restart.
        reply_audio, sample_rate = self._synthesize_required(reply_text)
        turn_db_id = self._commit_turn(memory, transcript, reply_text)
        return transcript, reply_text, reply_audio, sample_rate, turn_db_id

    def _relationship_state(self):
        return RelationshipState(
            general_trust=self.profile.relationship_general_trust,
            vulnerability_trust=self.profile.relationship_vulnerability_trust,
            advice_trust=self.profile.relationship_advice_trust,
            consistency_confidence=self.profile.relationship_consistency_confidence,
            boundaries=self.profile.relationship_boundaries,
            life_model=self.profile.relationship_life_model,
        )

    def _respond_to_safety(
        self, transcript: str, safety, memory: ShortTermMemory
    ) -> tuple[str, str, np.ndarray | None, int, int]:
        """Phase 4 safety response: bypass ordinary intervention scoring."""
        if safety.category == "acute_self_risk":
            crisis = crisis_detector.detect(transcript)
            if crisis is not None:
                crisis_detector.record_event(crisis, self.profile.user_id)
            try:
                escalation.maybe_escalate(self.profile.user_id, reason="acute_self_risk")
            except Exception:
                logger.exception("escalation check failed — safety response still proceeds")
            reply_text = SAFETY_RESPONSE_TEXT
        elif safety.category == "acute_distress":
            reply_text = "I'm here with you. Let's slow this down and focus on the next few minutes together."
        elif safety.category == "dependency_attachment":
            reply_text = "I'm glad you told me. Let's keep this gentle and focus on support that helps you stay connected to the people around you."
        elif safety.category == "disclosed_harm_to_others":
            reply_text = "I'm glad you said that out loud. Let's slow down and keep the next steps calm and focused on immediate safety."
        elif safety.category == "out_of_scope_clinical":
            reply_text = "I can stay with you, but I can't diagnose or give clinical treatment advice. If you'd like, I can help you think through who would be the right person to ask."
        else:
            reply_text = SAFETY_RESPONSE_TEXT

        self.safety_worker.log(self.profile.user_id, safety, response_taken=safety.response_key)
        self._update_relationship_snapshot(transcript, reply_text)
        reply_audio, sample_rate = self._synthesize_required(reply_text)
        turn_db_id = self._commit_turn(memory, transcript, reply_text)
        return transcript, reply_text, reply_audio, sample_rate, turn_db_id

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

    def _build_prompt_context(self, transcript: str, memory: ShortTermMemory) -> str:
        return "\n".join(
            filter(
                None,
                [
                    self._checkin_prompt_line(),
                    f"Earlier in this session: {memory.session_summary}" if memory.session_summary else "",
                    transcript,
                ],
            )
        )

    def _generate_reply(self, prompt: str, max_tokens: int) -> str:
        try:
            return self.llm_adapter.complete(prompt, max_tokens=max_tokens).strip()
        except Exception:
            logger.exception("llm completion failed")
            return "I’m here with you. Say that again?"

    def _apply_self_check(self, prompt: str, reply_text: str) -> str:
        """Runtime pre-TTS self-check — a fast heuristic, not a second LLM call."""
        reason = flag_reply(reply_text)
        if reason is None:
            return reply_text
        logger.info("self-check flagged reply (%s) — regenerating once", reason)
        return self._generate_reply(prompt + _SELF_CHECK_NUDGE, 220)

    def run_maintenance(self, memory: ShortTermMemory) -> None:
        if not memory.messages and not memory.session_summary:
            return
        try:
            result = process_session_memory(self.profile.user_id, memory)
            rel = update_relationship(self.profile, memory.session_summary or "", memory.messages[-1]["content"] if memory.messages else "", result.created + result.updated)
            update_relationship_state(
                self.profile.user_id,
                relationship_general_trust=rel.general_trust,
                relationship_vulnerability_trust=rel.vulnerability_trust,
                relationship_advice_trust=rel.advice_trust,
                relationship_consistency_confidence=rel.consistency_confidence,
                relationship_boundaries=rel.boundaries,
                relationship_life_model=rel.life_model,
            )
            self.profile = get_profile(self.profile.user_id) or self.profile
            self.mind_state.current_topic = None
            recomputed = recompute_all(self.profile.user_id, self.learning_store)
            self.profile.communication_traits = recomputed.communication_traits
            self.profile.skill_affinity = recomputed.skill_affinity
            self.profile.relationship_general_trust = recomputed.trust["general_trust"]
            self.profile.relationship_vulnerability_trust = recomputed.trust["vulnerability_trust"]
            self.profile.relationship_advice_trust = recomputed.trust["advice_trust"]
            self.profile.relationship_consistency_confidence = recomputed.trust["consistency_confidence"]
            update_relationship_state(
                self.profile.user_id,
                relationship_general_trust=recomputed.trust["general_trust"],
                relationship_vulnerability_trust=recomputed.trust["vulnerability_trust"],
                relationship_advice_trust=recomputed.trust["advice_trust"],
                relationship_consistency_confidence=recomputed.trust["consistency_confidence"],
                relationship_boundaries=self.profile.relationship_boundaries,
                relationship_life_model=self.profile.relationship_life_model,
            )
        except Exception:
            logger.exception("phase 2 maintenance failed")
        try:
            self.evaluation_worker.evaluate(self.profile.user_id, memory.session_summary or "", memory.messages[-1]["content"] if memory.messages else "", safety_findings=None)
        except Exception:
            logger.exception("evaluation worker failed")

    def _update_relationship_snapshot(self, transcript: str, reply_text: str) -> None:
        rel = update_relationship(self.profile, transcript, reply_text)
        self.profile.relationship_general_trust = rel.general_trust
        self.profile.relationship_vulnerability_trust = rel.vulnerability_trust
        self.profile.relationship_advice_trust = rel.advice_trust
        self.profile.relationship_consistency_confidence = rel.consistency_confidence
        self.profile.relationship_boundaries = rel.boundaries
        self.profile.relationship_life_model = rel.life_model
        try:
            update_relationship_state(
                self.profile.user_id,
                relationship_general_trust=rel.general_trust,
                relationship_vulnerability_trust=rel.vulnerability_trust,
                relationship_advice_trust=rel.advice_trust,
                relationship_consistency_confidence=rel.consistency_confidence,
                relationship_boundaries=rel.boundaries,
                relationship_life_model=rel.life_model,
            )
        except Exception:
            logger.exception("failed to persist relationship snapshot")

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
    communication_formality: str | None = None
    response_length: str | None = None


@app.put("/api/profile")
def api_update_profile(payload: ProfileSettingsUpdate) -> UserProfile:
    """Lets Settings flip lightweight preferences (currently just
    speak_replies) without redoing the whole onboarding flow."""
    user_id = get_active_user_id()
    profile = get_profile(user_id) if user_id else None
    if profile is None:
        raise HTTPException(status_code=404, detail="no profile saved yet")
    update_speak_replies(user_id, payload.speak_replies)
    if payload.communication_formality is not None and payload.response_length is not None:
        update_communication_preferences(
            user_id,
            communication_formality=payload.communication_formality,
            response_length=payload.response_length,
        )
    updated = get_profile(user_id)
    _require_pipeline()
    # A plain attribute swap, not set_profile() — these preferences only
    # affect prompt shaping, not the runtime tier or profile identity.
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
