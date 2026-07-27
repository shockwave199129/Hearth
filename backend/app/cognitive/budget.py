"""Thinking budget presets for the phase 0 scheduler."""

from pydantic import BaseModel


class ThinkingBudget(BaseModel):
    mode: str
    max_prompt_tokens: int
    max_response_tokens: int
    max_latency_ms: int


FAST_PATH_BUDGET = ThinkingBudget(mode="fast_path", max_prompt_tokens=300, max_response_tokens=120, max_latency_ms=800)
FULL_PATH_BUDGET = ThinkingBudget(mode="full_path", max_prompt_tokens=900, max_response_tokens=220, max_latency_ms=1800)

