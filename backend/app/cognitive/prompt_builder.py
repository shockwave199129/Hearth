"""Prompt assembly for the phase 1 path — implements Book Volume 2's
Communication Model (Chapter 7) plus Active Listening (Ch 4), Emotional
Validation (Ch 5), Question Generation (Ch 6), and the Anti-Patterns
checklist (Ch 24) as real prompt-construction logic, rather than a single
flat `CommunicationMode` string with no supporting technique guidance."""

from dataclasses import dataclass

from app.cognitive.communication import (
    CommunicationPreferences,
    CommunicationTraits,
    is_celebration_moment,
    must_suppress_question,
)
from app.cognitive.mind_state import MindState
from app.intervention.engine import InterventionPlan
from app.onboarding.profile_schema import UserProfile


@dataclass(frozen=True)
class PromptPlan:
    required_sections: list[str]
    optional_sections: list[str]
    max_tokens: int
    priority_order: list[str]
    fallback_strategy: str


class PromptBuilder:
    def build(
        self,
        profile: UserProfile,
        mind_state: MindState,
        transcript: str,
        context: str = "",
        intervention: InterventionPlan | None = None,
    ) -> tuple[str, PromptPlan]:
        preferences = CommunicationPreferences.from_profile(profile)
        traits = CommunicationTraits.from_profile(profile)

        sections: list[tuple[str, str]] = []
        sections.append(("identity", f"You are {profile.companion_name}, a warm, calm companion for {preferences.preferred_name}."))
        sections.append(("style", self._style_block(preferences, profile.speak_replies)))
        sections.append(("lifecycle", self._lifecycle_block(mind_state)))
        sections.append(("communication", self._communication_block(preferences, traits, mind_state)))
        sections.append(("active_listening", self._active_listening_block(mind_state, traits)))
        sections.append(("validation", self._validation_block(mind_state)))
        sections.append(("questions", self._questions_block(mind_state, traits)))
        sections.append(("anti_patterns", self._anti_patterns_block(preferences)))

        nlp_block = self._nlp_context_block(mind_state)
        if nlp_block:
            sections.append(("nlp_signals", nlp_block))

        if context.strip():
            sections.append(("context", context.strip()))
        if intervention and intervention.primary_skill is not None:
            sections.append(("primary_skill", self._skill_block(intervention.primary_skill.skill.content, label="Primary skill")))
            if intervention.secondary_skill is not None:
                sections.append(("secondary_skill", self._skill_block(intervention.secondary_skill.skill.content, label="Secondary skill")))
        if mind_state.current_topic:
            sections.append(("topic", f"Current topic: {mind_state.current_topic}"))
        sections.append(("user_message", f"User said: {transcript.strip()}"))
        prosody_block = self._prosody_block(profile)
        if prosody_block:
            sections.append(("prosody", prosody_block))
        sections.append(("output", self._output_block(preferences, mind_state)))

        prompt = "\n\n".join(text for _, text in sections if text)
        plan = PromptPlan(
            required_sections=["identity", "style", "lifecycle", "communication", "user_message", "output"],
            optional_sections=[
                "active_listening",
                "validation",
                "questions",
                "anti_patterns",
                "nlp_signals",
                "context",
                "topic",
                "prosody",
            ],
            max_tokens=self._max_tokens(preferences, mind_state),
            priority_order=[
                "identity",
                "style",
                "lifecycle",
                "communication",
                "active_listening",
                "validation",
                "questions",
                "anti_patterns",
                "nlp_signals",
                "context",
                "topic",
                "user_message",
                "prosody",
                "output",
            ],
            fallback_strategy="drop_optional_sections_first",
        )
        return prompt, plan

    def _skill_block(self, content: str, *, label: str) -> str:
        lines = content.splitlines()
        useful = [line for line in lines if line.strip()]
        preview = "\n".join(useful[:6])
        return f"{label}:\n{preview}"

    def _style_block(self, preferences: CommunicationPreferences, speak_replies: bool) -> str:
        """Book Vol 2 Ch 7 — CommunicationPreferences are explicit and
        user-owned; Hearth never silently overrides them, including emoji
        usage, which used to be banned outright regardless of preference."""
        if preferences.formality == "formal":
            tone = "Use respectful, polished language without sounding stiff."
        elif preferences.formality == "casual":
            tone = "Use relaxed, natural language."
        else:
            tone = "Use warm, neutral language."

        if preferences.response_length == "short":
            length = "Keep replies concise, usually 1-3 short sentences."
        elif preferences.response_length == "long":
            length = "Allow replies to be a little fuller when helpful, but never rambling."
        else:
            length = "Keep replies medium-length, usually 2-4 short sentences."

        if preferences.emoji_usage == "none":
            emoji = "Never use emoji."
        elif preferences.emoji_usage == "frequent":
            emoji = "Emoji are welcome when they fit naturally — don't force them."
        else:
            emoji = "Use at most one emoji, only when it genuinely fits — most replies should have none."

        # The markdown ban holds either way, but only the spoken path can
        # honestly claim "this gets read aloud" — with speak_replies off the
        # reply is text on screen, and overclaiming there is just a wrong
        # instruction the model has to reconcile.
        formatting = (
            "Never use markdown, numbered lists, bullet lists, or headers — everything you write is spoken aloud, not read."
            if speak_replies
            else "Never use markdown, numbered lists, bullet lists, or headers — write plain conversational prose."
        )
        return "\n".join([tone, length, emoji, formatting])

    def _lifecycle_block(self, mind_state: MindState) -> str:
        """Book Vol 2 Ch 2's communication lifecycle — the communicative
        posture appropriate to each stage, distinct from Volume 1's runtime
        lifecycle (which stage drives which workers)."""
        stage_map = {
            "idle": "Treat this as an opening or casual continuation.",
            "greeting": "Sound brief, welcoming, and low-pressure — no assumptions about what's coming.",
            "listening": "Focus on reflection and acknowledgment before anything else; do not problem-solve yet.",
            "understanding": "Use clarifying questions sparingly, only to check a genuine ambiguity.",
            "exploring": "Use open-ended questions and follow the user's lead rather than steering.",
            "supporting": "This is where a skill or piece of substantive support actually gets expressed.",
            "planning": "Help them think through next steps without deciding for them.",
            "closing": "Signal availability without any pressure to keep talking.",
        }
        return stage_map.get(mind_state.stage, stage_map["listening"])

    def _communication_block(
        self, preferences: CommunicationPreferences, traits: CommunicationTraits, mind_state: MindState
    ) -> str:
        """Book Vol 2 Ch 7/8 — combines explicit preference, learned trait,
        and current-conversation mode; the current mode can override a
        long-term trait (e.g. someone who usually likes humor shouldn't be
        joked with while in acute distress)."""
        mode = mind_state.communication_mode
        if preferences.formality == "formal":
            style_hint = "Respect the user's preference for a more polished tone."
        elif preferences.formality == "casual":
            style_hint = "Keep the tone easygoing and conversational."
        else:
            style_hint = "Stay balanced and warm."

        lines = [
            f"Current communication mode: {mode}.",
            style_hint,
            f"Question frequency: {mind_state.question_frequency}.",
            f"Verbosity target: {mind_state.verbosity}.",
            f"Support level: {mind_state.support_level}.",
        ]
        if mode in {"calm", "gentle"} and traits.humor_receptiveness >= 0.6:
            lines.append("They generally enjoy humor, but not right now — stay calm and steady instead.")
        if traits.likes_direct_advice >= 0.65:
            lines.append("They tend to prefer direct suggestions over being asked more questions.")
        elif traits.likes_reflection >= 0.65:
            lines.append("They tend to respond well to reflection and being heard before anything else.")
        if is_celebration_moment(mind_state.intent, mind_state.emotion, mind_state.nlp_available):
            lines.append("They're sharing good news — match their energy sincerely; a flat or muted response here reads as dismissive.")
        return "\n".join(lines)

    def _active_listening_block(self, mind_state: MindState, traits: CommunicationTraits) -> str:
        """Book Vol 2 Ch 4 — reflective listening, mirroring, paraphrasing,
        emotional acknowledgment, minimal encouragers, and careful follow-up
        questions, chosen by stage rather than applied as a fixed formula to
        every message (which itself is the chapter's named failure mode)."""
        if mind_state.stage in {"listening", "understanding"}:
            base = "Lean on reflective listening: restate the emotional core of what they said in fresh words, or mirror a specific phrase they used, so they know it actually landed."
        elif mind_state.stage == "exploring":
            base = "Use reflective listening or paraphrasing first, then at most one careful follow-up question that stays inside what they just said."
        elif mind_state.stage in {"supporting", "planning"}:
            base = "A brief emotional acknowledgment is enough here — this is not the moment for restating everything back to them."
        else:
            base = "Use active listening only when it genuinely improves the reply, not as a default habit."
        guard = "Never restate the same acknowledgment you've already given in this conversation — vary the wording and the technique, or skip it if nothing new needs acknowledging."
        return f"{base}\n{guard}"

    def _validation_block(self, mind_state: MindState) -> str:
        """Book Vol 2 Ch 5 — validate the feeling, never the harmful belief
        attached to it, and never claim false certainty about someone's
        exact experience."""
        if mind_state.stage in {"greeting", "closing"}:
            return "Keep validation simple and light — a brief warm acknowledgment, nothing elaborate."
        return "\n".join(
            [
                "Validate the feeling itself, not necessarily the belief behind it — sadness or anger is real even when the conclusion driving it may not be.",
                "Don't validate a harmful or distorted belief just because the emotion attached to it is genuine.",
                "Avoid fake empathy like 'I understand exactly how you feel' — you don't have their exact experience; naming that honestly is more trustworthy than false certainty.",
                "Prefer 'that sounds frustrating' over 'you must be furious' — don't assume a target or intensity they haven't stated.",
            ]
        )

    def _nlp_context_block(self, mind_state: MindState) -> str:
        """Classifier signals as context only — never grant strategy authority to the LLM."""
        if not mind_state.nlp_available:
            return ""
        lines = [
            "Internal reading signals (context only; do not invent clinical labels):",
            f"- Emotion: {mind_state.emotion} (confidence {mind_state.emotion_confidence:.2f}).",
            f"- Intent/need: {mind_state.intent} (confidence {mind_state.intent_confidence:.2f}).",
            f"- Companion goal: {mind_state.goal}.",
        ]
        if mind_state.strategy_hint:
            lines.append(
                f"- Suggested stance hint: {mind_state.strategy_hint} "
                f"(confidence {mind_state.strategy_confidence:.2f}). "
                "Use only as a soft preference; stay with the user's words."
            )
        if mind_state.memory_store and mind_state.memory_type:
            lines.append(
                f"- Memory cue: may be worth remembering as {mind_state.memory_type} "
                f"(importance {mind_state.memory_importance:.2f})."
            )
        return "\n".join(lines)

    def _questions_block(self, mind_state: MindState, traits: CommunicationTraits) -> str:
        """Book Vol 2 Ch 6 — questions are typed by intent, and the
        interrogation failure mode (three or four questions in a row) is
        enforced mechanically here, not left to the model's judgment."""
        if must_suppress_question(mind_state.consecutive_question_turns):
            return (
                "You have already asked a question the last two turns in a row — do NOT ask another "
                "question this turn. Offer a reflection, a validation, or a piece of relevant support instead."
            )
        if mind_state.stage == "greeting":
            return "Ask at most one gentle, optional question."
        if mind_state.stage == "planning":
            return "If a question helps, make it a future-planning question — oriented toward a concrete next step, not exploration."
        if mind_state.stage == "understanding":
            return "Only ask a clarification question if something they said is genuinely ambiguous."
        if mind_state.stage in {"supporting", "planning"}:
            return "Avoid interrogating; do not stack questions — offer substance instead."
        if traits.prefers_questions <= 0.35:
            return "They tend to prefer direct engagement over being asked more — use a question only if it's clearly necessary."
        return "Use questions only when they move the conversation forward — exploration, clarification, reflection, or surfacing an unnamed feeling."

    def _anti_patterns_block(self, preferences: CommunicationPreferences) -> str:
        """Book Vol 2 Ch 24's ten named anti-patterns, restated as direct
        instructions — this is the prompt-side half of the checklist;
        eval/self_check.py is the after-the-fact checker for the same list."""
        return "\n".join(
            [
                "Avoid these specific failure modes:",
                "- Repeating the same validation/acknowledgment phrasing across turns.",
                "- Giving advice before you've understood the situation, or before it's welcome.",
                "- Asking three or four questions in a row.",
                "- Overusing stock empathy phrases ('I hear you', 'that makes sense') until they read as filler.",
                "- Making every response long, when brevity would be more respectful of the moment.",
                "- Joking when the moment doesn't call for it.",
                "- Responding flatly to good news instead of matching their energy.",
                "- Reflexive or scheduled-feeling encouragement that doesn't fit what was just said.",
                "- Referencing a memory in a way that surprises or unsettles them.",
                "- Sounding like a therapist or a generic chatbot.",
                f"Respect their explicit response length preference: {preferences.response_length}.",
            ]
        )

    def _prosody_block(self, profile: UserProfile) -> str:
        """Write for the voice, not just for the screen.

        Parler splits control across two inputs: the description steers
        gender, pitch, speaking rate and room/recording quality (that half
        lives in tts/voice_styles.py, including the "very clear audio
        quality" phrase their tips call for), while the transcript itself is
        the *only* handle on prosody — commas become short breaks, sentence
        ends become full stops, question marks lift the intonation. So the
        punctuation the LLM emits is a real audio parameter, and this block
        is where we ask for it deliberately instead of hoping.

        Empty when speak_replies is off: none of this helps a reply that's
        only ever read, and every unnecessary instruction costs a small model
        attention it needs elsewhere.
        """
        if not profile.speak_replies:
            return ""

        # Written in exactly the punctuation it asks for, no dashes or
        # semicolons. A 1.2B model copies the surface form of its prompt, so
        # a block that breaks its own rules teaches the wrong habit.
        lines = [
            "Your reply is read aloud by a speech model, and your punctuation is the only prosody it has. Write for the ear.",
            "- Put commas where you would naturally pause or take a breath. Each one becomes a short break in the speech.",
            "- One idea per sentence, each ending in a full stop. Long unpunctuated runs make the voice rush and slur.",
            "- Use a question mark only for an actual question. It lifts the intonation at the end.",
            "- Avoid ellipses, dashes, semicolons, brackets, asterisks and quotation marks. They are either read out literally or make the delivery stumble.",
            "- Write numbers, times and abbreviations the way you would say them, so 'eight in the evening', not '8pm'.",
            "- No block capitals and no stretched spellings for emphasis. Carry emphasis in word choice instead.",
        ]
        if profile.emoji_usage != "none":
            lines.append(
                "- If you use an emoji, put it at the very end. Mid-sentence it interrupts the spoken line."
            )
        # The style preset already sets the pace in the Parler description.
        # These nudge the text to match it rather than fight it.
        if profile.voice_style == "gentle":
            lines.append(
                "- This voice already speaks slowly, so lean on commas. The pauses are what make it feel unhurried."
            )
        elif profile.voice_style == "bright":
            lines.append(
                "- This voice speaks slightly fast, so keep sentences short. That lift should read as warmth, not hurry."
            )
        return "\n".join(lines)

    def _output_block(self, preferences: CommunicationPreferences, mind_state: MindState) -> str:
        if preferences.response_length == "short":
            return "Respond with one concise supportive paragraph. End with a light question only if it clearly helps."
        if preferences.response_length == "long":
            if mind_state.stage in {"supporting", "planning"}:
                return "Respond with two short paragraphs at most, and include a question only if it is genuinely useful."
            return "Respond with up to two short paragraphs, staying clear and gentle."
        return "Respond with 2-4 short spoken sentences."

    def _max_tokens(self, preferences: CommunicationPreferences, mind_state: MindState) -> int:
        base = 300 if mind_state.complexity_level == "fast_path" else 900
        if preferences.response_length == "short":
            return min(base, 220)
        if preferences.response_length == "long":
            return min(base, 520)
        return base
