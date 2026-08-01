"""Book Vol 1 Ch 8 contract objects — typed data models for inter-module
communication. Every major handoff between cognitive subsystems flows
through one of these rather than an unstructured string or dict."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class WorkerResult(BaseModel):
    """Output contract from any Worker (NLP, Safety, Evaluation, etc.)."""

    worker: str
    status: str = "ok"
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    elapsed_ms: float | None = None
    error: str | None = None


class ResponsePlan(BaseModel):
    """Plan produced by the cognitive layer before prompt construction —
    captures the *what* and *why* of the upcoming response."""

    strategy: str
    communication_mode: str = "supportive"
    communication_stage: str = "listening"
    skill_id: str | None = None
    skill_secondary_id: str | None = None
    memory_context_ids: list[str] = Field(default_factory=list)
    safety_cleared: bool = True
    max_response_tokens: int = 220
    notes: str = ""
