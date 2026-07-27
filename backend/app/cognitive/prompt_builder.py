"""Prompt assembly for the phase 1 path."""

from dataclasses import dataclass

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
        sections: list[tuple[str, str]] = []
        sections.append(("identity", f"You are {profile.companion_name}, a warm, calm companion for {profile.name}."))
        sections.append(("style", self._style_block(profile)))
        sections.append(("lifecycle", self._lifecycle_block(mind_state)))
        sections.append(("communication", self._communication_block(profile, mind_state)))
        sections.append(("active_listening", self._active_listening_block(mind_state)))
        sections.append(("validation", self._validation_block(mind_state)))
        sections.append(("questions", self._questions_block(mind_state)))
        sections.append(("anti_patterns", self._anti_patterns_block(profile)))

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
        sections.append(("output", self._output_block(profile, mind_state)))

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
            ],
            max_tokens=self._max_tokens(profile, mind_state),
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

    def _style_block(self, profile: UserProfile) -> str:
        formality = profile.communication_formality
        response_length = profile.response_length
        if formality == "formal":
            tone = "Use respectful, polished language without sounding stiff."
        elif formality == "casual":
            tone = "Use relaxed, natural language."
        else:
            tone = "Use warm, neutral language."

        if response_length == "short":
            length = "Keep replies concise, usually 1-3 short sentences."
        elif response_length == "long":
            length = "Allow replies to be a little fuller when helpful, but never rambling."
        else:
            length = "Keep replies medium-length, usually 2-4 short sentences."

        return "\n".join(
            [
                tone,
                length,
                "Never use markdown, numbered lists, bullet lists, headers, or emoji.",
            ]
        )

    def _lifecycle_block(self, mind_state: MindState) -> str:
        stage_map = {
            "idle": "Treat this as an opening or casual continuation.",
            "greeting": "Sound brief, welcoming, and low-pressure.",
            "listening": "Focus on reflection and acknowledgment before advice.",
            "understanding": "Use clarifying questions sparingly and only when needed.",
            "exploring": "Use open-ended questions and follow the user's lead.",
            "supporting": "Offer substantive support only when it fits the conversation.",
            "planning": "Help think through next steps without taking over.",
            "closing": "Signal availability without pressure to keep talking.",
        }
        return stage_map.get(mind_state.stage, stage_map["listening"])

    def _communication_block(self, profile: UserProfile, mind_state: MindState) -> str:
        mode = mind_state.communication_mode
        if profile.communication_formality == "formal":
            style_hint = "Respect the user's preference for a more polished tone."
        elif profile.communication_formality == "casual":
            style_hint = "Keep the tone easygoing and conversational."
        else:
            style_hint = "Stay balanced and warm."

        return "\n".join(
            [
                f"Current communication mode: {mode}.",
                style_hint,
                f"Question frequency: {mind_state.question_frequency}.",
                f"Verbosity target: {mind_state.verbosity}.",
                f"Support level: {mind_state.support_level}.",
            ]
        )

    def _active_listening_block(self, mind_state: MindState) -> str:
        if mind_state.stage in {"listening", "understanding"}:
            return "Lean on reflective listening, paraphrasing, and short acknowledgments."
        if mind_state.stage == "exploring":
            return "Use reflective listening, then one careful follow-up question if needed."
        return "Use active listening only when it genuinely improves the reply."

    def _validation_block(self, mind_state: MindState) -> str:
        if mind_state.stage in {"greeting", "closing"}:
            return "Keep validation simple and light."
        return "Validate the feeling first, without validating harmful beliefs or overclaiming certainty."

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

    def _questions_block(self, mind_state: MindState) -> str:
        if mind_state.stage == "greeting":
            return "Ask at most one gentle, optional question."
        if mind_state.stage == "supporting":
            return "Avoid interrogating; do not stack questions."
        return "Use questions only when they move the conversation forward."

    def _anti_patterns_block(self, profile: UserProfile) -> str:
        return "\n".join(
            [
                "Avoid therapy-bot phrasing, walls of text, and mechanical empathy.",
                "Don't overuse 'I understand exactly how you feel.'",
                "Don't ask three or four questions in a row.",
                f"Respect the user's explicit response length preference: {profile.response_length}.",
            ]
        )

    def _output_block(self, profile: UserProfile, mind_state: MindState) -> str:
        if profile.response_length == "short":
            return "Respond with one concise supportive paragraph. End with a light question only if it clearly helps."
        if profile.response_length == "long":
            if mind_state.stage in {"supporting", "planning"}:
                return "Respond with two short paragraphs at most, and include a question only if it is genuinely useful."
            return "Respond with up to two short paragraphs, staying clear and gentle."
        return "Respond with 2-4 short spoken sentences."

    def _max_tokens(self, profile: UserProfile, mind_state: MindState) -> int:
        base = 300 if mind_state.complexity_level == "fast_path" else 900
        if profile.response_length == "short":
            return min(base, 220)
        if profile.response_length == "long":
            return min(base, 520)
        return base
