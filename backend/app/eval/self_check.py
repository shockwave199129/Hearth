"""Runtime pre-TTS self-check — a fast heuristic, NOT a second LLM call
(voice latency matters). The real quality assurance is the offline
rubric-based harness (eval/llm_judge.py, run during development); this is
just a light safety net catching the most obvious violations before a
reply is spoken. See project-plan.md §7.

If flag_reply() returns a reason, main.py's Pipeline regenerates the reply
exactly once with a short nudge appended, then uses whatever comes back
regardless — this never blocks the conversation."""
import re

_SENTENCE_BOUNDARY = re.compile(r"[.!?]+(?:\s|$)")
_LIST_MARKER = re.compile(r"(^|\n)\s*(\d+[.)]|[-*•])\s", re.MULTILINE)
_CLINICAL_TERMS = re.compile(
    r"\b(diagnos\w+|disorder|clinical(ly)?|pathology|syndrome|prescri\w+|symptomatology)\b", re.IGNORECASE
)

MAX_SENTENCES = 4


def flag_reply(text: str) -> str | None:
    """Returns a short reason if the reply looks like it violates the
    rubric (length/format, register — see eval/rubric.md), else None."""
    if not text.strip():
        return None
    sentence_count = len(_SENTENCE_BOUNDARY.findall(text))
    if sentence_count > MAX_SENTENCES:
        return "too long"
    if _LIST_MARKER.search(text):
        return "looks like a list"
    if _CLINICAL_TERMS.search(text):
        return "clinical/diagnostic language"
    return None
