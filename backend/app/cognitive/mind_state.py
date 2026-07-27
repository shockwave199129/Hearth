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
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
