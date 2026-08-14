"""Identity disclosure — deterministic answers to "are you real?".

US companion-AI law (e.g. California SB 243) requires a clear disclosure
that the system is AI. The *primary* disclosure is structural: the
onboarding consent step and the persistent marker in the app surface — see
docs/compliance.md. This module is the conversational backstop for when
someone asks directly, mid-conversation, where the UI isn't the answer.

It is deliberately **not** an LLM call, for two reasons:

1. Book Vol 1 Invariant #5 — the LLM never decides strategy. What Hearth
   says about its own nature is not a generation decision.
2. eval/self_check.py's `_GENERIC_CHATBOT` anti-pattern matches "i am an
   ai" / "as a language model", and pipeline._apply_self_check *regenerates*
   a flagged reply rather than merely logging it. A truthful disclosure
   produced by the model would therefore be discarded and re-rolled.
   Intercepting before the model runs resolves that cleanly: the
   anti-pattern keeps its original meaning (unprompted chatbot filler is
   still a defect, Book Vol 2 Ch 24) and a direct question still gets a
   true answer.

Scope is deliberately narrow — questions about *what Hearth is*, not
questions about what it feels. "Do you actually care about me?" is an
emotional question, and answering it with a flat capability disclaimer
would be both unkind and unnecessary; that belongs to the ordinary
conversational path, which is already constrained from claiming humanity by
config.SYSTEM_PROMPT_TEMPLATE.
"""

import re


# Direct questions about Hearth's nature. Anchored on the question forms
# people actually use out loud — this runs on STT output, so it must
# tolerate missing punctuation and contractions.
_PATTERNS: tuple[str, ...] = (
    r"\bare you (a |an )?(real|human|person|people|bot|robot|ai|a\.?i\.?|machine|computer|program|software)\b",
    r"\bare you (actually|really|even) (a |an )?(real|human|person|bot|robot|ai|a\.?i\.?)\b",
    r"\byou'?re not (a |an )?(real|human|person|actual person)\b",
    r"\bis (this|that) (a |an )?(real|actual) (person|human)\b",
    r"\bam i (talking|speaking|chatting) (to|with) (a |an )?(real|actual|human|person|bot|robot|ai|a\.?i\.?)\b",
    r"\bwhat are you\b",
    r"\bare you (alive|conscious|sentient|self.?aware)\b",
    r"\bdo you exist\b",
)

_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in _PATTERNS]

# Phrases that look like identity questions but are about capability or
# availability, and are better served by the ordinary conversational path.
# Checked first so "are you real" inside "are you really listening" doesn't
# trigger a disclosure the user didn't ask for.
_EXCLUSIONS = (
    re.compile(r"\bare you (really|actually) (listening|there|sure|serious|okay|ok)\b", re.IGNORECASE),
    re.compile(r"\bare you (a |an )?(real|human)\w* (help|support)", re.IGNORECASE),
)


def is_identity_question(text: str) -> bool:
    """True when the user is directly asking what Hearth is.

    Kept conservative on purpose: a false negative falls through to the
    ordinary path, which is already forbidden from claiming humanity, while
    a false positive interrupts a real conversation with a disclosure
    nobody asked for."""
    if not text or not text.strip():
        return False
    if any(pattern.search(text) for pattern in _EXCLUSIONS):
        return False
    return any(pattern.search(text) for pattern in _COMPILED)
