"""Live runtime cognition state."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class MindState(BaseModel):
    schema_version: int = 1
    stage: str = "idle"
    goal: str = "listen"
    communication_mode: str = "gentle"
    question_frequency: str = "low"
    verbosity: str = "balanced"
    support_level: str = "high"
    # Vol 2 Ch 6's mechanical question rule: count of consecutive assistant
    # turns that ended in a question, with no substantive statement between.
    consecutive_question_turns: int = 0
    current_topic: str | None = None
    complexity_level: str = "fast_path"
    last_user_message: str | None = None
    last_assistant_message: str | None = None
    turn_count: int = 0
    thinking_budget_mode: str = "fast_path"
    # NLP classifier soft signals (fail-soft defaults when ONNX missing)
    emotion: str = "unknown"
    emotion_confidence: float = 0.0
    intent: str = "unknown"
    intent_confidence: float = 0.0
    memory_store: bool = False
    memory_type: str | None = None
    memory_importance: float = 0.0
    relationship_trust_delta: float = 0.0
    relationship_vulnerability: float = 0.0
    relationship_openness: float = 0.0
    relationship_comfort: float = 0.0
    strategy_hint: str | None = None
    strategy_confidence: float = 0.0
    nlp_available: bool = False
    # Vol 1 Ch 8 Context Penalty — skill ids used in recent turns, so
    # ranking (app.intervention.ranking) can deprioritize repeating the
    # same suggestion. Most-recent last, capped by main.py.
    recent_skill_ids: list[str] = Field(default_factory=list)
    # Vol 5 Ch 16's SkillObservation — set when a skill is used, resolved
    # (and cleared) at the start of the next turn once fresh signals for
    # that next turn exist. See app.intervention.observation.
    pending_skill_id: str | None = None
    pending_skill_version: str | None = None
    pending_skill_composed_with: str | None = None
    pending_skill_emotion_before: str = "unknown"
    pending_skill_stage: str = "listening"
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
