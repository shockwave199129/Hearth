"""Phase 1 (Book Volume 2 — Communication) tests.

These check the deterministic, code-level guarantees that back Volume 2
Chapter 26's ten Communication Invariants. An LLM's actual word choice is
judged by the offline rubric harness (eval/llm_judge.py + eval/rubric.md),
which needs a running model and is not part of this suite — what's testable
here, without an LLM in the loop, is: section ordering, mechanical rule
enforcement, explicit-preference propagation, and the anti-pattern checker.
Each test is annotated with the invariant number(s) it operationalizes.
"""
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cognitive.communication import (
    CommunicationPreferences,
    CommunicationTraits,
    MAX_CONSECUTIVE_QUESTIONS,
    infer_mode,
    infer_stage,
    must_suppress_question,
    update_question_streak,
)
from app.cognitive.mind_state import MindState
from app.cognitive.prompt_builder import PromptBuilder
from app.eval.self_check import flag_reply
from app.onboarding.profile_schema import UserProfile


def _profile(**overrides) -> UserProfile:
    defaults = dict(
        user_id="u1",
        name="Subho",
        companion_name="Hearth",
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return UserProfile(**defaults)


def _section_index(prompt: str, marker: str) -> int:
    idx = prompt.find(marker)
    assert idx != -1, f"expected to find {marker!r} in prompt"
    return idx


# --- Invariants 1 & 2: user feels heard before Hearth tries to help; -------
# validation comes before problem solving.
def test_invariant_1_2_validation_precedes_help():
    profile = _profile()
    mind_state = MindState(stage="supporting")
    prompt, _ = PromptBuilder().build(profile, mind_state, "I don't know what to do about work")
    validation_idx = _section_index(prompt, "Validate the feeling itself")
    output_idx = _section_index(prompt, "User said:")
    assert validation_idx < output_idx


# --- Invariant 3: advice is offered, never imposed. ------------------------
def test_invariant_3_advice_offered_not_imposed():
    prompt, _ = PromptBuilder().build(_profile(), MindState(stage="supporting"), "hey")
    assert "before it's welcome" in prompt
    assert "advice" not in prompt.lower().split("output_block")  # sanity: no crash path


# --- Invariant 4: conversations build autonomy, not dependency. -----------
def test_invariant_4_no_validating_harmful_beliefs_and_closing_has_no_pressure():
    listening_prompt, _ = PromptBuilder().build(_profile(), MindState(stage="listening"), "nobody will ever hire me again")
    assert "harmful or distorted belief" in listening_prompt
    closing_prompt, _ = PromptBuilder().build(_profile(), MindState(stage="closing"), "I should go")
    assert "without any pressure to keep talking" in closing_prompt


# --- Invariant 5: warmth never at the cost of honesty. ---------------------
def test_invariant_5_fake_empathy_forbidden_in_prompt_and_caught_after_the_fact():
    prompt, _ = PromptBuilder().build(_profile(), MindState(stage="listening"), "this is so hard")
    assert "fake empathy" in prompt.lower()
    assert flag_reply("I know exactly how you feel, that must be awful.") == "fake empathy claim"


# --- Invariant 6: trust is earned gradually, never assumed. ---------------
def test_invariant_6_unlearned_traits_stay_neutral():
    traits = CommunicationTraits.from_profile(_profile())
    assert traits.likes_reflection == 0.5
    assert traits.likes_direct_advice == 0.5
    assert traits.prefers_questions == 0.5
    # A partially learned profile only moves the traits it actually has signal for.
    learned = CommunicationTraits.from_profile(_profile(communication_traits={"likes_direct_advice": 0.9}))
    assert learned.likes_direct_advice == 0.9
    assert learned.likes_reflection == 0.5  # untouched trait stays neutral


# --- Invariant 7: communication adapts to the person and the moment. ------
def test_invariant_7_mode_and_traits_change_the_built_prompt():
    calm_prompt, _ = PromptBuilder().build(
        _profile(communication_traits={"humor_receptiveness": 0.9}),
        MindState(stage="supporting", communication_mode="calm"),
        "I'm panicking about tomorrow",
    )
    assert "enjoy humor, but not right now" in calm_prompt

    direct_prompt, _ = PromptBuilder().build(
        _profile(communication_traits={"likes_direct_advice": 0.8}),
        MindState(stage="supporting", communication_mode="warm"),
        "what should I do",
    )
    assert "prefer direct suggestions" in direct_prompt


# --- Invariant 8: Hearth remembers to support, not to surprise. -----------
def test_invariant_8_anti_pattern_list_warns_against_surprising_memory_use():
    prompt, _ = PromptBuilder().build(_profile(), MindState(stage="exploring"), "hi")
    assert "surprises or unsettles them" in prompt


# --- Invariant 9: celebrate progress as sincerely as struggles. -----------
def test_invariant_9_anti_pattern_list_warns_against_flat_response_to_good_news():
    prompt, _ = PromptBuilder().build(_profile(), MindState(stage="exploring"), "hi")
    assert "flatly to good news" in prompt


# --- Invariant 10: every conversation leaves the person feeling at least --
# as respected/understood — mechanically enforced via the question rule and
# session-wide respect for explicit preferences.
def test_invariant_10_question_streak_forces_reflection_instead_of_interrogation():
    mind_state = MindState(stage="exploring", consecutive_question_turns=MAX_CONSECUTIVE_QUESTIONS)
    prompt, _ = PromptBuilder().build(_profile(), mind_state, "I guess so")
    assert "do NOT ask another question this turn" in prompt


def test_invariant_10_explicit_preferences_hold_across_a_whole_session():
    profile = _profile(emoji_usage="none", response_length="short", communication_formality="formal")
    builder = PromptBuilder()
    transcripts_and_stages = [
        ("hello there", "greeting"),
        ("I've been struggling with something", "listening"),
        ("thanks, talk soon", "closing"),
    ]
    # The per-turn assertions below are what prove the invariant: the same
    # three explicit-preference lines appear in every stage's prompt.
    for transcript, stage in transcripts_and_stages:
        prompt, _ = builder.build(profile, MindState(stage=stage), transcript)
        assert "Never use emoji." in prompt
        assert "Keep replies concise, usually 1-3 short sentences." in prompt
        assert "Use respectful, polished language without sounding stiff." in prompt


# --- Supporting mechanics (not invariants themselves, but what the above -
# invariant tests rely on being correct) -----------------------------------
def test_question_streak_increments_and_resets():
    assert update_question_streak(None, 0) == 0
    assert update_question_streak("What happened next?", 0) == 1
    assert update_question_streak("Did that help?", 1) == 2
    assert update_question_streak("That sounds like a lot.", 2) == 0


def test_must_suppress_question_threshold():
    assert must_suppress_question(MAX_CONSECUTIVE_QUESTIONS - 1) is False
    assert must_suppress_question(MAX_CONSECUTIVE_QUESTIONS) is True


def test_infer_stage_and_mode_basic_cases():
    assert infer_stage("hey", "fast_path") == "greeting"
    assert infer_stage("talk later", "fast_path") == "closing"
    assert infer_stage("what should I do about this", "full_path") == "planning"
    assert infer_mode("I'm so overwhelmed and stuck", "supporting") == "calm"
    assert infer_mode("just chatting", "exploring") == "warm"


def test_communication_preferences_from_profile_never_overridden():
    profile = _profile(emoji_usage="frequent", response_length="long", communication_formality="casual", preferred_voice="male")
    prefs = CommunicationPreferences.from_profile(profile)
    assert prefs == CommunicationPreferences(
        preferred_name="Subho", voice="male", emoji_usage="frequent", response_length="long", formality="casual"
    )


def test_self_check_overusing_questions():
    assert flag_reply("Are you okay? What happened? Do you want to talk about it?") == "overusing questions"


def test_self_check_repeating_validation_across_turns():
    prior = ["That sounds really frustrating to deal with."]
    current = "That sounds like a lot to carry today."
    assert flag_reply(current, recent_assistant_messages=prior) == "repeating validation phrasing"


def test_self_check_generic_chatbot_phrase():
    assert flag_reply("As an AI, I can't really feel that, but I'm here.") == "sounds like a generic chatbot"


def test_self_check_still_catches_original_three_checks():
    assert flag_reply("") is None
    assert flag_reply("Here is what to do:\n1. Breathe\n2. Relax\n3. Sleep") == "looks like a list"
    assert flag_reply("This sounds like a clinical anxiety disorder diagnosis.") == "clinical/diagnostic language"
    long_reply = "One. " * 6
    assert flag_reply(long_reply) == "too long"
