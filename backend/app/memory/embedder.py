"""Separate, lightweight llama.cpp embedding server for EmbeddingGemma-300M.
Only started when long-term memory is actually used (first call to embed()),
so tiers B/C don't pay for it until a memory tool call fires.
See docs/project-plan.md §5 — embedding calls are cheap and only fire at
create/update/search time, not on every message."""
import requests

from app.config import EMBEDDING_MODEL_FILE, EMBEDDING_MODELS_DIR, EMBEDDING_SERVER_HOST, EMBEDDING_SERVER_PORT
from app.hardware.detect import detect_hardware
from app.llm.server_manager import LlamaCppProcess


def _extract_embedding(payload: object) -> list[float]:
    """Pull the vector out of llama-server's /embedding response.

    The shape changed across llama.cpp releases: older builds answered
    ``{"embedding": [floats]}``, the pinned b10016 build answers
    ``[{"index": 0, "embedding": [[floats]]}]`` — one entry per input, each
    holding one row per pooled sequence. The OpenAI-compatible endpoint
    wraps the same thing in ``{"data": [...]}``. Accept all three so an
    upgrade of the bundled binary can't silently break long-term memory.
    """
    if isinstance(payload, dict):
        payload = payload.get("data", payload)

    if isinstance(payload, list):
        if not payload:
            raise ValueError("llama-server returned an empty embedding response")
        entry = payload[0]
        vector = entry.get("embedding") if isinstance(entry, dict) else entry
    elif isinstance(payload, dict):
        vector = payload.get("embedding")
    else:
        vector = None

    # Default pooling yields exactly one row; take it rather than the raw
    # per-token matrix.
    while isinstance(vector, list) and vector and isinstance(vector[0], list):
        vector = vector[0]

    if not isinstance(vector, list) or not vector or not isinstance(vector[0], (int, float)):
        raise ValueError(f"unrecognized /embedding response shape: {type(payload).__name__}")
    return [float(value) for value in vector]


def _embedding_gpu_layers() -> int:
    """Put EmbeddingGemma on the NVIDIA GPU when one is present so spare
    VRAM is used instead of leaving the embedder on CPU."""
    try:
        hw = detect_hardware()
    except Exception:
        return 0
    return -1 if hw.get("has_nvidia") else 0


class EmbeddingServer(LlamaCppProcess):
    def __init__(self):
        model_path = str(EMBEDDING_MODELS_DIR / EMBEDDING_MODEL_FILE)
        super().__init__(
            host=EMBEDDING_SERVER_HOST,
            port=EMBEDDING_SERVER_PORT,
            extra_args=[
                "--model",
                model_path,
                "--embedding",
                "--n-gpu-layers",
                str(_embedding_gpu_layers()),
            ],
        )

    def embed(self, text: str) -> list[float]:
        if not self.is_running():
            self.start()
        resp = requests.post(f"{self.base_url}/embedding", json={"content": text}, timeout=30)
        resp.raise_for_status()
        return _extract_embedding(resp.json())


_server: EmbeddingServer | None = None


def embed(text: str) -> list[float]:
    global _server
    if _server is None:
        _server = EmbeddingServer()
    return _server.embed(text)
