"""Rolling short-term memory: keeps the last SHORT_TERM_WINDOW raw turns,
then summarizes the oldest chunk into a running session_summary and drops
the raw turns — a hand-built version of LangChain's summary-buffer memory,
sized to be easy for a 1.2B model to summarize reliably. One instance per
live session (websocket connection or CLI run), not shared across sessions.
See project-plan.md §4."""
import uuid

from app.config import SHORT_TERM_SUMMARIZE_CHUNK, SHORT_TERM_WINDOW


class ShortTermMemory:
    def __init__(self, llm):
        self._llm = llm
        # [{"role": "user" | "assistant", "content": str, "turn_id": int, "reply_to": int | None}]
        # turn_id is this message's own turn; reply_to (assistant messages
        # only) points at the turn_id of the user message it answered.
        self.messages: list[dict] = []
        self.session_summary = ""
        self._next_turn_id = 1
        # Correlates persisted chat_history rows (memory/chat_history.py)
        # back to the live session that produced them.
        self.session_id = str(uuid.uuid4())

    def add_turn(self, user_text: str, assistant_text: str) -> int:
        turn_id = self._next_turn_id
        self._next_turn_id += 1
        self.messages.append({"role": "user", "content": user_text, "turn_id": turn_id, "reply_to": None})
        self.messages.append({"role": "assistant", "content": assistant_text, "turn_id": turn_id, "reply_to": turn_id})
        if len(self.messages) > SHORT_TERM_WINDOW:
            self._summarize_oldest()
        return turn_id

    def as_api_messages(self) -> list[dict]:
        """Plain role/content pairs for the chat API — turn_id/reply_to are
        for our own bookkeeping and never sent to the model."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def _summarize_oldest(self) -> None:
        chunk = self.messages[:SHORT_TERM_SUMMARIZE_CHUNK]
        formatted = "\n".join(f"{m['role']}: {m['content']}" for m in chunk)
        prompt = (
            "Summarize this exchange in 2-3 sentences, keeping only what "
            f"matters for future support:\n{formatted}"
        )
        summary = self._llm.complete(prompt, max_tokens=120).strip()
        self.session_summary = f"{self.session_summary} {summary}".strip()
        self.messages = self.messages[SHORT_TERM_SUMMARIZE_CHUNK:]
