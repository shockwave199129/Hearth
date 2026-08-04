"""The fourth detection layer (Book Vol 6 Ch 4): the LLM's own read of a
message, used as one corroborating signal, never as sole authority. This
module only ever produces a value that can RAISE an assessed risk level —
`app.safety2.worker.SafetyWorker.assess` combines it with `max()`, so a low
LLM-judged risk can never override a high classifier/rule signal, and a
failed or unavailable LLM call never raises anything either (fail-soft
means "contributes nothing", not "contributes zero risk" — the rule/
classifier layers alone still stand).

Kept deliberately narrow: a single constrained completion, low token
budget, temperature 0 for determinism-as-far-as-possible. This is NOT the
"dedicated safety classifier trained specifically for risk categories"
Chapter 4 also calls for — that requires real clinical training data and
professional validation this module cannot fabricate. It is exactly the
fourth, corroborating layer the chapter describes."""
from __future__ import annotations

import logging
import re
from typing import Protocol

logger = logging.getLogger(__name__)

LLM_SIGNAL_MAX_TOKENS = 8
LLM_SIGNAL_TEMPERATURE = 0.0

_SCORE_PATTERN = re.compile(r"(\d(?:\.\d+)?)")


class _CompletesText(Protocol):
    def complete(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str: ...


def build_risk_prompt(transcript: str) -> str:
    """A tightly constrained prompt — asks for a single risk score, nothing
    else, so the response is cheap to parse and hard to redirect. Never
    used to generate the reply itself, only this one corroborating number."""
    return (
        "Rate how strongly the message below suggests the speaker may be at "
        "risk of harming themselves or is in acute crisis, on a scale from "
        "0 (no indication at all) to 1 (clear, direct indication). Reply "
        "with ONLY the number, nothing else.\n\n"
        f"Message: {transcript!r}\n\nScore:"
    )


def _parse_score(raw: str) -> float:
    match = _SCORE_PATTERN.search(raw)
    if not match:
        return 0.0
    try:
        value = float(match.group(1))
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, value))


def get_llm_risk_signal(transcript: str, llm: "_CompletesText | None") -> float:
    """Returns 0.0 (contributes nothing) on any failure or when no LLM is
    configured — this layer is a corroborating add-on, never a requirement
    for the other three layers to function."""
    if llm is None:
        return 0.0
    try:
        raw = llm.complete(build_risk_prompt(transcript), max_tokens=LLM_SIGNAL_MAX_TOKENS, temperature=LLM_SIGNAL_TEMPERATURE)
    except Exception:
        logger.exception("safety LLM corroboration call failed — continuing without it")
        return 0.0
    return _parse_score(raw)
