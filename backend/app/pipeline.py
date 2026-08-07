"""Conversation pipeline: STT → LLM → TTS + cognitive/safety/memory wiring.

Extracted from main.py so HTTP routers can depend on a Pipeline (or a test
double) without importing the FastAPI composition root.
"""

import logging
from datetime import datetime, timezone

import numpy as np

from app.cognitive import CognitiveScheduler, PromptBuilder, ResponseComposer, StateManager
from app.execution.llm_adapter import LlmAdapter
from app.intervention.engine import InterventionContext, InterventionEngine
from app.intervention.observation import mark_skill_used, resolve_pending_observation
from app.workers import NlpWorkerRunner
from app.checkin.state import get_last_checkin, set_last_checkin
from app.config import (
    CHECKIN_PROMPT_TEMPLATE,
    CHECKIN_QUESTION_PHRASES,
    SAFETY_RESPONSE_TEXT,
    DATA_DIR,
)
from app.eval.self_check import flag_reply
from app.hardware.tier_manager import detect_and_cache_tier
from app.llm.server_manager import LlmServer
from app.memory import chat_history, long_term
from app.memory.formation import process_session_memory
from app.memory.short_term import ShortTermMemory
from app.onboarding.active_profile import get_active_user_id
from app.onboarding.profile_schema import UserProfile
from app.onboarding.profile_store import get_profile, update_relationship_state
from app.safety import crisis_detector, escalation
from app.relationship.engine import update_relationship
from app.relationship.engine import RelationshipState
from app.relationship.profile_store import get_or_create_relationship_profile
from app.safety2.worker import SafetyWorker
from app.learning.observation_store import ObservationStore
from app.learning.recompute import recompute_all
from app.evaluation.worker import EvaluationWorker
from app.growth.engine import GrowthEngine
from app.memory2.retrieval import retrieve as memory2_retrieve
from app.stt.moonshine_engine import MoonshineEngine
from app.tts.tts_engines import get_tts_engine
from app.safety2.audit import purge_expired

logger = logging.getLogger("hearth")

# Appended to the system prompt for exactly one regeneration attempt when
# eval/self_check.py flags a reply — see docs/project-plan.md §7.
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
        see docs/project-plan.md's text-input support notes."""
        return self._handle_turn(text, memory)

    def _synthesize_reply(self, reply_text: str) -> tuple[np.ndarray | None, int]:
        """Synthesize with one retry, then give up and let the turn continue
        as text — callers pass the None straight through to the client.

        Voice is the product, so reaching the None here is a real defect and
        it logs a traceback saying so. What it must not do is discard the
        turn: this used to raise, the websocket turned that into an error
        frame, and the client — which only renders a turn once one arrives —
        showed neither the reply nor the user's own message. A TTS failure
        read as the app ignoring input entirely. Degrading to text keeps the
        conversation legible and keeps the failure visible instead of
        swallowing both.
        """
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
            except Exception:
                logger.exception("TTS attempt %s failed", attempt + 1)
        logger.error("TTS failed after retries — delivering %r as text only", reply_text[:80])
        return None, 0

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
        reply_audio, sample_rate = (
            self._synthesize_reply(reply_text) if self.profile.speak_replies else (None, 0)
        )
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
        reply_audio, sample_rate = self._synthesize_reply(reply_text)
        turn_db_id = self._commit_turn(memory, transcript, reply_text)
        return transcript, reply_text, reply_audio, sample_rate, turn_db_id

    def _checkin_prompt_line(self) -> str:
        """Computed fresh each turn (single cheap row read) rather than
        cached per-session, so it self-corrects immediately after
        mark_checkin fires mid-session. See docs/project-plan.md §8."""
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

