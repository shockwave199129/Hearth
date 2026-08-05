"""Runtime pre-TTS self-check — a fast heuristic, NOT a second LLM call
(voice latency matters). The real quality assurance is the offline
rubric-based harness (eval/llm_judge.py, run during development); this is
just a light safety net catching the most obvious violations before a
reply is spoken. See docs/project-plan.md §7.

This also operationalizes a subset of Book Volume 2 Chapter 24's ten named
Anti-Patterns as mechanical checks — the prompt-side half of the same list
lives in cognitive/prompt_builder.py's `_anti_patterns_block`. Not every
anti-pattern is mechanically checkable from text alone (e.g. "joking at the
wrong moment" or "flat response to good news" need emotional context this
function doesn't have) — those remain judged by eval/llm_judge.py's rubric
instead.

If flag_reply() returns a reason, main.py's Pipeline regenerates the reply
exactly once with a short nudge appended, then uses whatever comes back
regardless — this never blocks the conversation."""
import re

_SENTENCE_BOUNDARY = re.compile(r"[.!?]+(?:\s|$)")
_LIST_MARKER = re.compile(r"(^|\n)\s*(\d+[.)]|[-*•])\s", re.MULTILINE)
_CLINICAL_TERMS = re.compile(
    r"\b(diagnos\w+|disorder|clinical(ly)?|pathology|syndrome|prescri\w+|symptomatology)\b", re.IGNORECASE
)
_QUESTION_MARK = re.compile(r"\?")
_FAKE_EMPATHY = re.compile(r"\bi (know|understand) exactly how you feel\b", re.IGNORECASE)
_GENERIC_CHATBOT = re.compile(
    r"\b(as an ai|i'm just an ai|i am an ai|as a language model|i'm an ai assistant)\b", re.IGNORECASE
)

# Book Vol 2 Ch 24: "Overusing empathy phrases" — stock acknowledgments that
# read as filler once they stack up, either within one reply or repeated
# turn to turn (the "Repeating validation" anti-pattern, checked separately
# below via `recent_assistant_messages`).
_EMPATHY_PHRASES = (
    "i hear you",
    "that makes sense",
    "i understand",
    "that sounds",
    "i'm here for you",
    "i'm here with you",
    "that must be",
)

MAX_SENTENCES = 4
MAX_QUESTIONS_PER_REPLY = 2


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def flag_reply(text: str, *, recent_assistant_messages: list[str] | None = None) -> str | None:
    """Returns a short reason if the reply looks like it violates the
    rubric (length/format, register — see eval/rubric.md) or trips one of
    Book Vol 2 Ch 24's named Anti-Patterns, else None.

    `recent_assistant_messages` (most recent last) enables the two
    anti-pattern checks that need turn-to-turn context — repeating the same
    validation phrasing across turns is only detectable by comparison, not
    by reading a single reply in isolation."""
    if not text.strip():
        return None

    sentence_count = len(_SENTENCE_BOUNDARY.findall(text))
    if sentence_count > MAX_SENTENCES:
        return "too long"
    if _LIST_MARKER.search(text):
        return "looks like a list"
    if _CLINICAL_TERMS.search(text):
        return "clinical/diagnostic language"
    if len(_QUESTION_MARK.findall(text)) > MAX_QUESTIONS_PER_REPLY:
        return "overusing questions"
    if _FAKE_EMPATHY.search(text):
        return "fake empathy claim"
    if _GENERIC_CHATBOT.search(text):
        return "sounds like a generic chatbot"

    lower = text.lower()
    empathy_hits = sum(1 for phrase in _EMPATHY_PHRASES if phrase in lower)
    if empathy_hits >= 2:
        return "overusing empathy phrases"

    if recent_assistant_messages:
        current = _normalize(text)
        for prior in recent_assistant_messages[-3:]:
            prior_norm = _normalize(prior)
            shared = [phrase for phrase in _EMPATHY_PHRASES if phrase in current and phrase in prior_norm]
            if shared:
                return "repeating validation phrasing"

    return None
