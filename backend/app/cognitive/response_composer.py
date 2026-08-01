"""Final response shaping for Hearth voice."""

from dataclasses import dataclass
import re


_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_FORMATTING_RE = re.compile(r"^[\s>*#`-]+", re.MULTILINE)


@dataclass(frozen=True)
class ResponseResult:
    text: str
    emotion: str = "steady"
    confidence: float = 0.7
    reflection_required: bool = False


class ResponseComposer:
    def compose(self, draft: str) -> ResponseResult:
        text = draft.strip()
        text = _LEADING_FORMATTING_RE.sub("", text)
        text = _WHITESPACE_RE.sub(" ", text)
        text = text.replace(" - ", ", ")
        sentences = re.split(r"(?<=[.!?])\s+", text)
        trimmed = " ".join(sentences[:4]).strip()
        if not trimmed:
            trimmed = "I’m here with you."
        return ResponseResult(text=trimmed)

