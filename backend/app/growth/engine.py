"""Growth Engine (Book Vol 1 Ch 12, the sole writer referenced throughout
Volumes 3 and 4) — analyzes a completed conversation and is the only
component that writes to `app.memory2` (episodic/semantic memory) and
`app.relationship.state.RelationshipProfile`. Always invoked as an async
entrypoint, and always after a response has already reached the user —
never synchronously mid-turn (Vol 3 Ch 11)."""
from __future__ import annotations

import logging
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from app.learning import attachment as attachment_pipeline
from app.learning.observation_store import ObservationStore
from app.learning.pipeline import recompute_value
from app.memory import chat_history
from app.memory.short_term import ShortTermMemory
from app.memory2.contradiction import check_and_resolve
from app.memory2.decay import reinforce
from app.memory2.formation import FormationCandidate, build_emotional_metadata, build_summary, find_candidates
from app.memory2.models import EpisodicMemory
from app.memory2.promotion import run_promotion
from app.memory2.store import MemoryStore
from app.nlp.runtime import OnnxClassifier
from app.relationship.profile_store import get_or_create_relationship_profile, save_relationship_profile
from app.relationship.state import (
    AttachmentSignals,
    ImportantPerson,
    LifeModel,
    OngoingSituation,
    RelationshipProfile,
    TrustModel,
    compute_development_level,
    derive_shared_history_candidates,
    evaluate_attachment_signals,
    update_boundaries,
)

logger = logging.getLogger(__name__)

MAX_SHARED_HISTORY_ENTRIES = 20
RECURRING_THEME_MIN_MENTIONS = 3

# Book Vol 7 Ch 7: Trust EWMA weight, applied via the shared Recomputation
# Pipeline (app.learning.pipeline) — same weight `learning/recompute.py`
# uses for the flat Profile cache, since both fold the same underlying
# relationship_observations.
TRUST_ALPHA = 0.06


@dataclass(frozen=True)
class GrowthEngineResult:
    episodic_memories_formed: int
    semantic_facts_promoted: int
    contradictions_resolved: int
    relationship_profile: RelationshipProfile


def _session_trust_signals(candidates: list[FormationCandidate]) -> dict[str, float]:
    """Rule-based per-session signals feeding TrustModel's moving average
    (Vol 3 Ch 3) — never a sharp jump, never derived from raw message
    volume or session length. Appended as real, derivation-tagged
    observations (Vol 7 Ch 3/Invariant 4) rather than blended inline."""
    total = max(1, len(candidates))
    vulnerability_hits = sum(1 for c in candidates if "vulnerable_disclosure" in c.markers)
    life_event_hits = sum(1 for c in candidates if "life_event" in c.markers)
    goal_hits = sum(1 for c in candidates if "stated_goal_or_plan" in c.markers)
    return {
        "vulnerability_trust": min(1.0, vulnerability_hits / total * 2),
        "general_trust": min(1.0, (vulnerability_hits + life_event_hits) / total),
        "advice_trust": min(1.0, goal_hits / total * 2),
        # A conversation that reached a normal end (not analyzed here as
        # abrupt/broken) is itself a small positive consistency signal —
        # Volume 2's Conversation Repair outcomes as a richer signal are a
        # future refinement; this is a legitimate, if modest, consistency
        # observation on its own.
        "consistency_confidence": 0.6,
    }


class GrowthEngine:
    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        classifier: OnnxClassifier | None = None,
        observation_store: ObservationStore | None = None,
    ):
        self.store = store or MemoryStore()
        self.classifier = classifier or OnnxClassifier()
        # Book Vol 7 Ch 3: the same unified DuckDB observation store other
        # components (skill observations, Phase 3; communication
        # observations, main.py) write to — pass the same instance the rest
        # of the Pipeline uses so Trust/Communication/Skill computations
        # all fold from one consistent history.
        self.observation_store = observation_store or ObservationStore()

    async def process_session(
        self,
        user_id: str,
        memory: ShortTermMemory,
        *,
        current_topic: str | None = None,
        now: datetime | None = None,
    ) -> GrowthEngineResult:
        now = now or datetime.now(timezone.utc)
        user_messages = [str(m.get("content", "")) for m in memory.messages if m.get("role") == "user"]
        if not user_messages:
            return GrowthEngineResult(0, 0, 0, get_or_create_relationship_profile(user_id))

        candidates = find_candidates(user_messages, self.classifier)
        formed: list[EpisodicMemory] = []
        contradiction_count = 0
        for candidate in candidates:
            mem = EpisodicMemory(
                id=str(uuid.uuid4()),
                user_id=user_id,
                timestamp=now,
                summary=build_summary(candidate),
                entities=candidate.entities,
                significance_markers=sorted(set(candidate.markers + candidate.triggers)),
                emotional_metadata=build_emotional_metadata(candidate),
                last_reinforced=now,
            )
            self._reinforce_related(user_id, mem, now=now)
            self.store.save_episodic(mem)
            formed.append(mem)
            contradiction_count += len(check_and_resolve(self.store, mem, now=now))

        promoted = run_promotion(self.store, user_id)

        profile = get_or_create_relationship_profile(user_id)
        profile = self._update_relationship(profile, candidates, formed, user_messages, current_topic, now)
        save_relationship_profile(profile)

        return GrowthEngineResult(
            episodic_memories_formed=len(formed),
            semantic_facts_promoted=len(promoted),
            contradictions_resolved=contradiction_count,
            relationship_profile=profile,
        )

    def _reinforce_related(self, user_id: str, new_mem: EpisodicMemory, *, now: datetime) -> None:
        """A new episodic touching the same entity as an existing one
        resets that existing memory's decay (Vol 4 Ch 10)."""
        if not new_mem.entities:
            return
        for existing in self.store.list_episodic(user_id):
            if existing.id == new_mem.id:
                continue
            if set(existing.entities) & set(new_mem.entities):
                self.store.update_episodic(reinforce(existing, now=now))

    _TRUST_DERIVATIONS = {
        "general_trust": "disclosure_depth",
        "vulnerability_trust": "disclosure_depth",
        "advice_trust": "disclosure_depth",
        "consistency_confidence": "consistency_observation",
    }

    def _write_trust_observations(self, user_id: str, signals: dict[str, float]) -> None:
        """Book Vol 7 Ch 3/Invariant 4: every Trust observation must carry a
        real evidentiary `derivation` — enforced at append time by
        `ObservationStore`, not just documented here."""
        for subscore, value in signals.items():
            self.observation_store.append(
                "relationship",
                subscore,
                value,
                {"derivation": self._TRUST_DERIVATIONS[subscore]},
                "growth_engine",
            )

    def _recompute_trust(self, user_id: str, current: TrustModel) -> TrustModel:
        """Book Vol 7 Ch 4/Ch 7 — folds ALL accumulated relationship
        observations (not just the latest session's), via the same shared
        pipeline `app.learning.recompute` uses for the flat Profile cache,
        so both stay consistent with one underlying computation."""
        updated = {}
        for subscore in ("general_trust", "vulnerability_trust", "advice_trust", "consistency_confidence"):
            result = recompute_value(
                self.observation_store, "relationship", subscore, current=getattr(current, subscore), alpha=TRUST_ALPHA
            )
            updated[subscore] = result.value if result.changed else getattr(current, subscore)
        return TrustModel(**updated)

    def _recompute_attachment(
        self, user_id: str, profile: RelationshipProfile, user_messages: list[str]
    ) -> AttachmentSignals:
        """Book Vol 7 Ch 8 — three independent streams combined into
        `combined_score`, finally feeding Phase 4's escalation
        (app.safety2.worker reads RelationshipProfile.attachment_signals)."""
        qualitative = evaluate_attachment_signals(user_messages)
        session_timestamps = chat_history.session_start_timestamps(user_id)
        contact_urgency = attachment_pipeline.contact_urgency_trend(session_timestamps)
        replacement_language = attachment_pipeline.replacement_language_score(user_messages)
        unavailability_distress = attachment_pipeline.unavailability_distress_score(user_messages)
        combined_score = attachment_pipeline.compute_combined_attachment_score(
            current_score=profile.attachment_signals.combined_score,
            contact_urgency=contact_urgency,
            replacement_language=replacement_language,
            unavailability_distress=unavailability_distress,
        )
        return qualitative.model_copy(update={"combined_score": combined_score})

    def _update_relationship(
        self,
        profile: RelationshipProfile,
        candidates: list[FormationCandidate],
        formed: list[EpisodicMemory],
        user_messages: list[str],
        current_topic: str | None,
        now: datetime,
    ) -> RelationshipProfile:
        signals = _session_trust_signals(candidates)
        self._write_trust_observations(profile.user_id, signals)
        trust = self._recompute_trust(profile.user_id, profile.trust)
        conversation_count = profile.conversation_count + 1
        days_since_last_contact = max(0.0, (now - profile.last_updated).total_seconds() / 86400.0)
        development_level = compute_development_level(
            trust,
            conversation_count=conversation_count,
            disclosure_depth=signals["vulnerability_trust"],
            communication_traits=profile.communication_traits,
            days_since_last_contact=days_since_last_contact,
        )
        attachment = self._recompute_attachment(profile.user_id, profile, user_messages)

        boundaries = profile.boundaries
        for text in user_messages:
            boundaries = update_boundaries(boundaries, text, current_topic)

        new_shared_history = derive_shared_history_candidates(
            [(m.summary, m.emotional_metadata.intensity, m.significance_markers) for m in formed]
        )
        existing_summaries = {entry.summary for entry in profile.shared_history}
        shared_history = list(profile.shared_history) + [
            entry for entry in new_shared_history if entry.summary not in existing_summaries
        ]
        shared_history = shared_history[-MAX_SHARED_HISTORY_ENTRIES:]

        life_model = self._update_life_model(profile.user_id, profile.life_model, formed, now)

        return profile.model_copy(
            update={
                "trust": trust,
                "development_level": development_level,
                "attachment_signals": attachment,
                "boundaries": boundaries,
                "life_model": life_model,
                "shared_history": shared_history,
                "conversation_count": conversation_count,
                "last_updated": now,
            }
        )

    def _update_life_model(
        self, user_id: str, life_model: LifeModel, formed: list[EpisodicMemory], now: datetime
    ) -> LifeModel:
        important_people = {p.name: p for p in life_model.important_people}
        ongoing_situations = {s.topic: s for s in life_model.ongoing_situations}
        for mem in formed:
            proper_nouns = [e for e in mem.entities if e[:1].isupper()]
            relation_words = [e for e in mem.entities if e[:1].islower()]
            relation_type = relation_words[0] if relation_words else "unknown"
            for name in proper_nouns:
                prior = important_people.get(name)
                important_people[name] = ImportantPerson(
                    name=name,
                    relationship_type=relation_type if relation_type != "unknown" else (prior.relationship_type if prior else "unknown"),
                    sentiment_context=mem.emotional_metadata.valence,
                    last_mentioned=now,
                )
            if "life_event" in mem.significance_markers:
                topic = proper_nouns[0] if proper_nouns else (relation_words[0] if relation_words else mem.summary)
                prior_situation = ongoing_situations.get(topic)
                ongoing_situations[topic] = OngoingSituation(
                    topic=topic,
                    status="ongoing",
                    first_mentioned=prior_situation.first_mentioned if prior_situation else now,
                    last_mentioned=now,
                )

        entity_counts = Counter(e for mem in self.store.list_episodic(user_id) for e in mem.entities)
        recurring_themes = sorted(e for e, count in entity_counts.items() if count >= RECURRING_THEME_MIN_MENTIONS)

        return LifeModel(
            important_people=list(important_people.values()),
            ongoing_situations=list(ongoing_situations.values()),
            recurring_themes=recurring_themes,
        )
