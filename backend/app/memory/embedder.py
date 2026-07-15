"""Separate, lightweight llama.cpp embedding server for EmbeddingGemma-300M.
Only started when long-term memory is actually used (first call to embed()),
so tiers B/C don't pay for it until a memory tool call happens.
See project-plan.md §5 — embedding calls are cheap and only fire at
create/update/search time, not on every message."""
import requests

from app.config import EMBEDDING_MODEL_FILE, EMBEDDING_MODELS_DIR, EMBEDDING_SERVER_HOST, EMBEDDING_SERVER_PORT
from app.llm.server_manager import LlamaCppProcess


class EmbeddingServer(LlamaCppProcess):
    def __init__(self):
        model_path = str(EMBEDDING_MODELS_DIR / EMBEDDING_MODEL_FILE)
        super().__init__(
            host=EMBEDDING_SERVER_HOST,
            port=EMBEDDING_SERVER_PORT,
            extra_args=["--model", model_path, "--embedding", "--n-gpu-layers", "0"],
        )

    def embed(self, text: str) -> list[float]:
        if not self.is_running():
            self.start()
        resp = requests.post(f"{self.base_url}/embedding", json={"content": text}, timeout=30)
        resp.raise_for_status()
        return resp.json()["embedding"]


_server: EmbeddingServer | None = None


def embed(text: str) -> list[float]:
    global _server
    if _server is None:
        _server = EmbeddingServer()
    return _server.embed(text)
