"""Spawns/monitors llama.cpp `llama-server` subprocesses. `LlmServer` runs
LFM2.5 — `.complete()` for plain-text completion (short-term summarizer,
eval harness); tool-calling chat goes through `langchain_openai.ChatOpenAI`
pointed at this same process's `/v1/chat/completions` (see main.py's
create_agent wiring), not through this class. `EmbeddingServer` in
memory/embedder.py reuses the shared `LlamaCppProcess` base for
EmbeddingGemma. See project-plan.md §1/§2/§5."""
import atexit
import platform
import subprocess
import threading
import time

import requests

from app.config import LLAMA_SERVER_BIN, LLM_SERVER_HOST, LLM_SERVER_PORT
from app.hardware.tier_manager import TierConfig, resolve_paths


class LlmServerError(RuntimeError):
    pass


def _windows_no_window_kwargs() -> dict:
    """Hide the console window for llama-server on Windows (Tauri already
    hides the parent hearth-backend console — grandchildren still flash
    without this)."""
    if platform.system() != "Windows":
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}


class LlamaCppProcess:
    """Generic spawn/health-check/stop for a `llama-server` subprocess.
    Subclasses add whatever request shape their use case needs."""

    def __init__(self, host: str, port: int, extra_args: list[str]):
        self.base_url = f"http://{host}:{port}"
        self._host = host
        self._port = port
        self._extra_args = extra_args
        self._proc: subprocess.Popen | None = None
        self._stdout_tail: list[str] = []
        self._drain_thread: threading.Thread | None = None

    def start(self, timeout_s: float = 60.0) -> None:
        if self.is_running():
            return
        cmd = [
            LLAMA_SERVER_BIN,
            "--host", self._host,
            "--port", str(self._port),
            *self._extra_args,
        ]
        self._stdout_tail = []
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            **_windows_no_window_kwargs(),
        )
        # Drain stdout so a chatty llama-server can't fill the pipe and
        # deadlock after health-check succeeds. Tail kept for early-exit errors.
        self._drain_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._drain_thread.start()
        atexit.register(self.stop)
        self._wait_until_ready(timeout_s)

    def _drain_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self._stdout_tail.append(line)
            self._stdout_tail = self._stdout_tail[-80:]

    def _wait_until_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                # Give the drain thread a moment to finish reading.
                if self._drain_thread is not None:
                    self._drain_thread.join(timeout=1.0)
                output = "".join(self._stdout_tail)
                raise LlmServerError(
                    f"llama-server exited early with code {self._proc.returncode}\n{output}"
                )
            try:
                if requests.get(f"{self.base_url}/health", timeout=1).ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise LlmServerError(f"llama-server did not become ready within {timeout_s}s")

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()


class LlmServer(LlamaCppProcess):
    def __init__(self, tier: TierConfig):
        self.tier = tier
        llm_path = resolve_paths(tier)["llm_path"]
        super().__init__(
            host=LLM_SERVER_HOST,
            port=LLM_SERVER_PORT,
            extra_args=[
                "--model", llm_path,
                "--n-gpu-layers", str(tier.n_gpu_layers),
                "--ctx-size", str(tier.ctx_size),
                "--jinja",  # needed for tool-call parsing, see memory/tools.py
            ],
        )

    def complete(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Plain-text completion — used for the short-term summarizer, which
        doesn't need tool calling."""
        if not self.is_running():
            raise LlmServerError("llama-server is not running — call start() first")
        resp = requests.post(
            f"{self.base_url}/completion",
            json={"prompt": prompt, "n_predict": max_tokens, "temperature": temperature},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["content"]
