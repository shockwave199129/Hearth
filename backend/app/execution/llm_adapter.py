"""Thin wrapper around the local llama-server completion endpoint."""

from app.llm.server_manager import LlmServer


class LlmAdapter:
    def __init__(self, llm_server: LlmServer):
        self._llm_server = llm_server

    def complete(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        return self._llm_server.complete(prompt, max_tokens=max_tokens, temperature=temperature)

