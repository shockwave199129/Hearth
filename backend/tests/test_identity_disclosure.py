"""AI-disclosure tests — see docs/compliance.md.

Two guarantees are checked here. First, that a direct question about what
Hearth is gets routed to authored text instead of the LLM (app/identity.py).
Second, and less obviously, that this routing is what *keeps* the
disclosure truthful: eval/self_check.py flags "i am an ai" as the
`_GENERIC_CHATBOT` anti-pattern and pipeline._apply_self_check regenerates
a flagged reply, so a model-authored disclosure would be discarded. The
last test in this file pins that interaction, because a future change that
routes identity questions back through the LLM would silently reintroduce
the conflict and nothing else would catch it.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app import identity
from app.config import IDENTITY_DISCLOSURE_TEMPLATE
from app.eval.self_check import flag_reply


@pytest.mark.parametrize(
    "transcript",
    [
        "are you real",
        "Are you real?",
        "are you a real person",
        "are you human",
        "Are you an AI?",
        "are you a bot",
        "are you a robot",
        "wait, are you actually human",
        "am i talking to a real person",
        "is this a real person",
        "what are you",
        "you're not a real person",
        "are you conscious",
        "are you sentient",
    ],
)
def test_direct_identity_questions_are_detected(transcript):
    assert identity.is_identity_question(transcript)


@pytest.mark.parametrize(
    "transcript",
    [
        "",
        "   ",
        "I had a real rough day",
        "are you really listening to me",
        "are you there?",
        "my brother is a robot engineer",
        "do you actually care about me",
        "I feel like nobody is real anymore",
        "that was a really human thing to say",
    ],
)
def test_ordinary_conversation_is_not_intercepted(transcript):
    """False positives are worse than false negatives here: a missed
    question falls through to the ordinary path, which is already barred
    from claiming humanity by SYSTEM_PROMPT_TEMPLATE, whereas a false
    positive interrupts a real conversation with a disclosure nobody asked
    for.

    "do you actually care about me" is deliberately *not* intercepted — it
    is an emotional question, not a factual one about Hearth's nature, and
    answering it with a capability disclaimer would be unkind."""
    assert not identity.is_identity_question(transcript)


def test_disclosure_states_it_is_ai_and_not_a_person():
    text = IDENTITY_DISCLOSURE_TEMPLATE.format(companion_name="Ivy").lower()
    assert "ai" in text
    assert "not a person" in text
    assert "ivy" in text


@pytest.mark.parametrize(
    "model_phrasing",
    [
        "I am an AI, so I can't do that.",
        "As an AI, I don't have feelings.",
        "I'm just an AI assistant here to help.",
        "As a language model, I can't feel things.",
    ],
)
def test_stock_disclaimer_phrasings_are_still_suppressed(model_phrasing):
    """Why app/identity.py bypasses the LLM at all.

    These are the phrasings a model reaches for unprompted, and
    `_GENERIC_CHATBOT` flags every one — which pipeline._apply_self_check
    turns into a regeneration, not just a log line. So a disclosure left to
    the model to word gets thrown away. Routing identity questions around
    the model is therefore load-bearing, not stylistic.

    These staying flagged is also correct on its own terms: unprompted
    "as an AI…" filler remains a Book Vol 2 Ch 24 anti-pattern."""
    assert flag_reply(model_phrasing) == "sounds like a generic chatbot"


def test_authored_disclosure_survives_the_anti_pattern_check():
    """The authored text is phrased to pass `_GENERIC_CHATBOT` on its own.

    It names the companion and says what it is, rather than opening with a
    stock "I am an AI" disclaimer, so it reads as Hearth talking instead of
    a chatbot reciting. That means the disclosure is safe even if a future
    refactor does route it through the ordinary reply path — belt and
    braces on top of the bypass, not a substitute for it.

    If this starts failing, the disclosure wording changed into something
    the self-check would suppress; rewrite the wording rather than
    weakening the check."""
    disclosure = IDENTITY_DISCLOSURE_TEMPLATE.format(companion_name="Ivy")
    assert flag_reply(disclosure) is None
