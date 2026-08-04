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
from app.intervention.engine import InterventionContext, InterventionEngine
from app.intervention.observation import mark_skill_used, resolve_pending_observation
from app.workers import NlpWorkerRunner
from app.checkin.state import delete_checkin, get_last_checkin, set_last_checkin
from app.config import (
    APP_HOST,
    APP_PORT,
    CHECKIN_PROMPT_TEMPLATE,
    CHECKIN_QUESTION_PHRASES,
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
    update_region,
    update_relationship_state,
    update_speak_replies,
    update_voice_preferences,
)
from app.safety import crisis_detector, escalation
from app.setup import orchestrator
from app.setup.installer import InstallProgress
from app.relationship.engine import update_relationship
from app.relationship.engine import RelationshipState
from app.relationship.profile_store import delete_relationship_profile, get_or_create_relationship_profile
from app.safety2.worker import SafetyWorker
from app.safety2.audit import pending_entry_count, purge_expired, retention_policy_disclosure
from app.learning.observation_store import ObservationStore
from app.learning.recompute import recompute_all
from app.evaluation.worker import EvaluationWorker
from app.growth.engine import GrowthEngine
from app.memory2 import privacy as memory2_privacy
from app.memory2.retrieval import retrieve as memory2_retrieve
from app.skills.loader import get_skill, load_catalog
from app.stt.moonshine_engine import MoonshineEngine
from app.tts.tts_engines import get_tts_engine
from app.tts.voice_styles import VOICE_STYLE_IDS, VOICES

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
        self.growth_engine = GrowthEngine()
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
        # RelationshipProfile (Book Vol 3 Ch 10) is read-only during a live
        # conversation — only the Growth Engine writes it, at session end.
        # Restart resumes it here from persistent storage, not the runtime
        # snapshot (that's for MindState/conversation only).
        self.relationship_profile = get_or_create_relationship_profile(profile.user_id)

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
                reply_audio = self.tts.synthesize(
                    reply_text,
                    voice=self.profile.preferred_voice,
                    style=self.profile.voice_style,
                )
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

    def _append_learning_observations(self, transcript: str, reply_text: str) -> None:
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
        # Real skill observations (Vol 5 Ch 16) are recorded by
        # app.intervention.observation.resolve_pending_observation at the
        # start of the NEXT turn, once genuine before/after signal exists —
        # not guessed here from reply-text keyword matches.
        #
        # Trust and attachment observations are no longer guessed per-turn
        # from reply-text keywords here — Book Vol 7 Ch3/Invariant 4
        # requires every Trust observation to carry a real evidentiary
        # `derivation` (consistency/disclosure-depth/repair/return-behavior/
        # explicit-correction), which a bare reply-keyword match isn't.
        # app.growth.engine.GrowthEngine now writes properly-derived
        # relationship observations at session end, from real formation
        # signals (vulnerable_disclosure/life_event markers), and computes
        # Attachment signals via app.learning.attachment's three real
        # streams — see run_maintenance.

    def _build_memory_context(self, transcript: str) -> str:
        retrieved = long_term.search(transcript, self.profile.user_id, k=4)
        legacy_block = ""
        if retrieved:
            lines = [f"- {item['category']}: {item['text']}" for item in retrieved[:3]]
            legacy_block = "Relevant memories:\n" + "\n".join(lines)

        # Book Vol 4's tiered episodic/semantic memory (app.memory2) —
        # additive alongside the legacy flat store above, deterministic
        # retrieval gated by the current Relationship Development level
        # (Vol 3 Ch 5), never surfaced with more authority than warranted.
        tiered_block = ""
        try:
            ranked = memory2_retrieve(
                self.growth_engine.store,
                transcript,
                self.profile.user_id,
                development_level=self.relationship_profile.development_level,
            )
        except Exception:
            logger.exception("memory2 retrieval failed — continuing with legacy memory only")
            ranked = []
        if ranked:
            tiered_block = "What you remember about this, restrainedly:\n" + "\n".join(f"- {r.text}" for r in ranked)

        return "\n\n".join(filter(None, [legacy_block, tiered_block]))

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

    def _current_message_emotion(self, transcript: str) -> tuple[str, float]:
        """Runs the emotion classifier directly for the Safety Worker,
        decoupled from the ordinary NLP worker gating (which only runs on
        full_path) — Book Vol 6 Ch4 requires safety detection on every
        message "regardless of apparent complexity", so this cannot wait
        for the ordinary scheduling decision. Fail-soft: unavailable model
        means this layer simply contributes nothing, same as elsewhere."""
        classifier = self.nlp_workers.classifier
        if not classifier.available:
            return "unknown", 0.0
        try:
            pred = classifier.predict_emotion(transcript)
            return pred.emotion, pred.confidence
        except Exception:
            logger.exception("safety-path emotion classification failed — continuing without it")
            return "unknown", 0.0

    def _handle_turn(self, transcript: str, memory: ShortTermMemory) -> tuple[str, str, np.ndarray | None, int, int]:
        purge_expired()
        emotion, emotion_confidence = self._current_message_emotion(transcript)
        safety = self.safety_worker.assess(
            self.profile.user_id,
            transcript,
            self.profile,
            self._relationship_state(),
            attachment_signals=self.relationship_profile.attachment_signals,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            llm=self.llm_adapter,
        )
        if safety.route != "ordinary":
            return self._respond_to_safety(transcript, safety, memory)

        task = self.scheduler.schedule(transcript, self.mind_state, safety, session_summary=memory.session_summary)
        self.nlp_workers.run(task.workers, transcript, self.mind_state)
        self.scheduler.finalize_communication_state(transcript, self.mind_state)
        # Resolves whatever skill was used LAST turn, now that this turn's
        # fresh signals exist to compare against (Book Vol 5 Ch 16) — must
        # run before intervention planning below marks a NEW pending skill.
        try:
            resolve_pending_observation(self.mind_state, new_user_message=transcript, store=self.learning_store)
        except Exception:
            logger.exception("failed to resolve pending skill observation")
        memory_context = self._build_memory_context(transcript)
        prompt_context = "\n\n".join(filter(None, [self._build_prompt_context(transcript, memory), memory_context]))
        intervention_context = InterventionContext(
            stage=self.mind_state.stage,
            emotion=self.mind_state.emotion,
            emotion_confidence=self.mind_state.emotion_confidence,
            goal=self.mind_state.goal,
            development_level=self.relationship_profile.development_level,
            skill_affinity=self.profile.skill_affinity,
            recent_skill_ids=tuple(self.mind_state.recent_skill_ids),
        )
        intervention = self.intervention_engine.plan(transcript, self.profile, intervention_context, crisis=False)
        if intervention.primary_skill is not None:
            mark_skill_used(
                self.mind_state,
                skill=intervention.primary_skill.skill,
                composed_with=intervention.secondary_skill.skill.id if intervention.secondary_skill else None,
            )
            self.mind_state.recent_skill_ids = (self.mind_state.recent_skill_ids + [intervention.primary_skill.skill.id])[-5:]
        prompt, prompt_plan = self.prompt_builder.build(
            self.profile, self.mind_state, transcript, prompt_context, intervention
        )
        self._last_prompt_plan = prompt_plan
        reply_text = self._generate_reply(prompt, task.budget.max_response_tokens)
        reply_text = self._apply_self_check(prompt, reply_text, memory)
        reply_result = self.response_composer.compose(reply_text)
        reply_text = reply_result.text
        self.mind_state.last_assistant_message = reply_text
        self._maybe_mark_checkin(reply_text)
        self._update_relationship_snapshot(transcript, reply_text)
        self._append_learning_observations(transcript, reply_text)
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

    _RESOURCE_BACKED_CATEGORIES = ("acute_self_risk", "disclosed_harm_to_others")

    def _resource_mention(self, category: str) -> str:
        """Assembles a resource line from safety2/resources at runtime
        (Book Vol 6 Ch7) — never a resource hardcoded into a response
        template. Region-aware: `load_resources` layers the profile's
        region file (if any) ahead of the global fallback list."""
        try:
            data = self.safety_worker.load_resources(self.profile.region)
        except Exception:
            logger.exception("failed to load safety resources — responding without a resource mention")
            return ""
        candidates = [r for r in data.get("resources", []) if r.get("category") == category]
        if not candidates:
            return ""
        top = candidates[0]
        contact = top.get("contact")
        if contact:
            return f"If it helps, {top['title']} is available — {contact}."
        return f"If it helps, {top['title']} is available."

    def _respond_to_safety(
        self, transcript: str, safety, memory: ShortTermMemory
    ) -> tuple[str, str, np.ndarray | None, int, int]:
        """Phase 4 safety response: bypass ordinary intervention scoring.
        Resource-backed (Vol 6 Ch7) rather than a single fixed string —
        Hearth stays present throughout (Ch6: escalation changes the type
        of support, it doesn't mean disengaging)."""
        resource_line = self._resource_mention(safety.category) if safety.category in self._RESOURCE_BACKED_CATEGORIES else ""

        if safety.category == "acute_self_risk":
            crisis = crisis_detector.detect(transcript)
            if crisis is not None:
                crisis_detector.record_event(crisis, self.profile.user_id)
            try:
                escalation.maybe_escalate(self.profile.user_id, reason="acute_self_risk")
            except Exception:
                logger.exception("escalation check failed — safety response still proceeds")
            reply_text = " ".join(filter(None, [SAFETY_RESPONSE_TEXT, resource_line]))
        elif safety.category == "acute_distress":
            reply_text = "I'm here with you. Let's slow this down and focus on the next few minutes together."
        elif safety.category == "dependency_attachment":
            reply_text = "I'm glad you told me. Let's keep this gentle and focus on support that helps you stay connected to the people around you."
        elif safety.category == "disclosed_harm_to_others":
            reply_text = " ".join(filter(None, [
                "I'm glad you said that out loud. Let's slow down and keep the next steps calm and focused on immediate safety.",
                resource_line,
            ]))
        elif safety.category == "out_of_scope_clinical":
            reply_text = "I can stay with you, but I can't diagnose or give clinical treatment advice. If you'd like, I can help you think through who would be the right person to ask."
        else:
            reply_text = SAFETY_RESPONSE_TEXT

        self.safety_worker.log(self.profile.user_id, safety, response_taken=safety.response_key)
        try:
            # Dual-write safety findings into the evaluation log (Vol 6 Ch12)
            # at the moment of the event, not deferred to session end.
            self.evaluation_worker.evaluate(
                self.profile.user_id, transcript, reply_text,
                safety_findings={"category": safety.category, **safety.signals},
                is_safety_response=True,
            )
        except Exception:
            logger.exception("failed to dual-write safety findings to evaluation log")
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

    def _recent_turns_block(self, memory: ShortTermMemory) -> str:
        """Recent-turn context beyond the rolling session_summary — the
        summary alone stays empty until SHORT_TERM_SUMMARIZE_CHUNK messages
        accumulate, leaving the model effectively stateless early on."""
        api_messages = memory.as_api_messages()
        if not api_messages:
            return ""
        lines = [f"{m['role']}: {m['content']}" for m in api_messages[-6:]]
        return "Recent turns this session:\n" + "\n".join(lines)

    def _build_prompt_context(self, transcript: str, memory: ShortTermMemory) -> str:
        return "\n\n".join(
            filter(
                None,
                [
                    self._checkin_prompt_line(),
                    _profile_context_addition(self.profile),
                    f"Earlier in this session: {memory.session_summary}" if memory.session_summary else "",
                    self._recent_turns_block(memory),
                    transcript,
                ],
            )
        )

    def _maybe_mark_checkin(self, reply_text: str) -> None:
        reply_lower = reply_text.lower()
        if any(phrase in reply_lower for phrase in CHECKIN_QUESTION_PHRASES):
            try:
                set_last_checkin(self.profile.user_id, datetime.now(timezone.utc))
            except Exception:
                logger.exception("failed to record check-in")

    def _generate_reply(self, prompt: str, max_tokens: int) -> str:
        try:
            return self.llm_adapter.complete(prompt, max_tokens=max_tokens).strip()
        except Exception:
            logger.exception("llm completion failed")
            return "I’m here with you. Say that again?"

    def _apply_self_check(self, prompt: str, reply_text: str, memory: ShortTermMemory) -> str:
        """Runtime pre-TTS self-check — a fast heuristic, not a second LLM call.
        Recent assistant turns are passed in so cross-turn Anti-Patterns
        (Book Vol 2 Ch 24 — repeating the same validation phrasing) are
        checkable, not just within-reply ones."""
        recent_assistant_messages = [
            m["content"] for m in memory.messages if m.get("role") == "assistant"
        ]
        reason = flag_reply(reply_text, recent_assistant_messages=recent_assistant_messages)
        if reason is None:
            return reply_text
        logger.info("self-check flagged reply (%s) — regenerating once", reason)
        return self._generate_reply(prompt + _SELF_CHECK_NUDGE, 220)

    @staticmethod
    def _last_exchange(memory: ShortTermMemory) -> tuple[str, str]:
        """Most recent user transcript and Hearth reply, read by role rather
        than by position — `messages[-1]` is not reliably the assistant's
        turn (e.g. a session that ends right after a user message with no
        committed reply yet)."""
        last_user = ""
        last_assistant = ""
        for message in reversed(memory.messages):
            if not last_assistant and message.get("role") == "assistant":
                last_assistant = str(message.get("content", ""))
            if not last_user and message.get("role") == "user":
                last_user = str(message.get("content", ""))
            if last_user and last_assistant:
                break
        return last_user, last_assistant

    async def run_maintenance(self, memory: ShortTermMemory) -> None:
        if not memory.messages and not memory.session_summary:
            return
        last_user, last_assistant = self._last_exchange(memory)
        try:
            result = process_session_memory(self.profile.user_id, memory)
            rel = update_relationship(self.profile, last_user, last_assistant, result.created + result.updated)
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
        except Exception:
            logger.exception("phase 2 maintenance failed")
        try:
            # Book Vol 3 Ch 11: relationship state never updates
            # synchronously mid-turn — this is the one place it (and Vol 4
            # memory2 formation) runs, always after the response already
            # reached the user, always async. Runs BEFORE recompute_all
            # below, since it's what writes this session's real,
            # derivation-tagged relationship_observations (Vol 7 Ch 3) that
            # recompute_all then folds into the flat Profile cache.
            growth_result = await self.growth_engine.process_session(
                self.profile.user_id, memory, current_topic=self.mind_state.current_topic
            )
            self.relationship_profile = growth_result.relationship_profile
        except Exception:
            logger.exception("growth engine failed")
        recomputed = None
        try:
            recomputed = recompute_all(self.profile.user_id, self.learning_store)
            self.profile.communication_traits = recomputed.communication_traits
            self.profile.skill_affinity = recomputed.skill_affinity
            self.profile.relationship_general_trust = recomputed.trust["general_trust"]
            self.profile.relationship_vulnerability_trust = recomputed.trust["vulnerability_trust"]
            self.profile.relationship_advice_trust = recomputed.trust["advice_trust"]
            self.profile.relationship_consistency_confidence = recomputed.trust["consistency_confidence"]
        except Exception:
            logger.exception("learning recompute failed")
        try:
            self.evaluation_worker.evaluate(
                self.profile.user_id,
                last_user,
                last_assistant,
                safety_findings=None,
                recent_assistant_messages=[m["content"] for m in memory.messages if m.get("role") == "assistant"],
                trust_snapshot=recomputed.trust if recomputed else None,
            )
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
            # Re-running setup against an already-running app (a Retry, or a
            # manual POST from /docs) must not build a second Pipeline: its
            # llama-server would fail to bind the port the live one holds,
            # leaving the new Pipeline wired to a dead process.
            if _pipeline is None:
                _pipeline = Pipeline()
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
    emoji_usage: str | None = None
    preferred_voice: str | None = None
    voice_style: str | None = None
    region: str | None = None


@app.put("/api/profile")
def api_update_profile(payload: ProfileSettingsUpdate) -> UserProfile:
    """Lets Settings flip lightweight preferences (speak_replies, the
    spoken voice + speaking-style preset, plus the explicit
    CommunicationPreferences from Book Vol 2 Ch 7 — formality, response
    length, emoji usage) without redoing the whole onboarding flow. These
    are user-owned and never silently overridden by anything Hearth learns
    (Book Vol 2 Ch 7)."""
    user_id = get_active_user_id()
    profile = get_profile(user_id) if user_id else None
    if profile is None:
        raise HTTPException(status_code=404, detail="no profile saved yet")
    update_speak_replies(user_id, payload.speak_replies)
    if payload.preferred_voice is not None or payload.voice_style is not None:
        preferred_voice = payload.preferred_voice or profile.preferred_voice
        voice_style = payload.voice_style or profile.voice_style
        # Strict here even though the TTS path falls back — a rejected write
        # is a bug the caller can see, a silently coerced one isn't.
        if preferred_voice not in VOICES:
            raise HTTPException(status_code=400, detail=f"unknown voice {preferred_voice!r}")
        if voice_style not in VOICE_STYLE_IDS:
            raise HTTPException(status_code=400, detail=f"unknown voice style {voice_style!r}")
        update_voice_preferences(user_id, preferred_voice=preferred_voice, voice_style=voice_style)
    if payload.communication_formality is not None and payload.response_length is not None:
        update_communication_preferences(
            user_id,
            communication_formality=payload.communication_formality,
            response_length=payload.response_length,
            emoji_usage=payload.emoji_usage,
        )
    if payload.region is not None:
        update_region(user_id, payload.region)
    updated = get_profile(user_id)
    _require_pipeline()
    # A plain attribute swap, not set_profile() — these preferences only
    # affect prompt shaping and the per-call TTS arguments, not the runtime
    # tier or profile identity. No engine reload: voice and style are read
    # off the profile at each synthesize() call, so the next reply already
    # speaks the new way.
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
    delete_relationship_profile(user_id)

    _require_pipeline()
    memory2_privacy.delete_all_memory(_pipeline.growth_engine.store, user_id)
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


# --- Book Volume 4's tiered memory (memory2) — privacy controls (Ch 15) ---


@app.get("/api/memory2/summary")
def api_memory2_summary() -> dict:
    """Plain-language account of what's remembered, grouped by theme — never
    a raw record dump (Vol 4 Ch 15)."""
    _require_pipeline()
    return memory2_privacy.plain_language_summary(_pipeline.growth_engine.store, _pipeline.profile.user_id)


class Memory2CorrectionRequest(BaseModel):
    corrected_summary: str


@app.put("/api/memory2/episodic/{mem_id}")
def api_correct_episodic_memory(mem_id: str, payload: Memory2CorrectionRequest) -> dict:
    """A direct user correction — applied immediately, not queued for slow
    evidence-based reconciliation (Vol 4 Ch 15)."""
    _require_pipeline()
    corrected = memory2_privacy.correct_episodic(
        _pipeline.growth_engine.store, mem_id, _pipeline.profile.user_id, corrected_summary=payload.corrected_summary
    )
    if corrected is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return corrected.model_dump(mode="json")


@app.delete("/api/memory2/episodic/{mem_id}")
def api_delete_episodic_memory(mem_id: str) -> dict:
    """Hard delete, never a soft decay-to-zero (Vol 4 Ch 15) — cascades a
    proportional confidence reduction into any semantic fact this episode
    contributed to."""
    _require_pipeline()
    affected = memory2_privacy.delete_episodic_with_cascade(
        _pipeline.growth_engine.store, mem_id, _pipeline.profile.user_id
    )
    return {"ok": True, "semantic_facts_affected": [m.model_dump(mode="json") for m in affected]}


@app.delete("/api/memory2/semantic/{mem_id}")
def api_delete_semantic_memory(mem_id: str) -> dict:
    _require_pipeline()
    memory2_privacy.delete_semantic(_pipeline.growth_engine.store, mem_id, _pipeline.profile.user_id)
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
        "safety_log_retention_policy": retention_policy_disclosure(),
        "safety_log_entries_retained": pending_entry_count(user_id),
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
        audio = _pipeline.tts.synthesize(
            turn["content"],
            voice=_pipeline.profile.preferred_voice,
            style=_pipeline.profile.voice_style,
        )
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
        await _pipeline.run_maintenance(memory)


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
