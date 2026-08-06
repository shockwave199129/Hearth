"""FastAPI dependencies for the conversation pipeline.

Replaces the module-level ``_pipeline`` global that main.py used to expose
to every route handler. Routers take ``Depends(get_pipeline)`` (or the
optional variant) so tests can inject a fake without constructing real
STT/LLM/TTS engines.
"""

from __future__ import annotations

from fastapi import HTTPException

from app.pipeline import Pipeline

__all__ = [
    "Pipeline",
    "get_pipeline",
    "get_pipeline_optional",
    "set_pipeline",
]

_pipeline: Pipeline | None = None


def get_pipeline_optional() -> Pipeline | None:
    return _pipeline


def get_pipeline() -> Pipeline:
    """Raise 503 when setup has not finished building the pipeline yet.

    A thin build (CI no longer bundles torch/onnxruntime) means this is an
    expected, recoverable state until /api/setup/start finishes — a 503 lets
    the frontend show "finish setup first" instead of crashing.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Setup not complete — see /api/setup/status")
    return _pipeline


def set_pipeline(pipeline: Pipeline | None) -> None:
    global _pipeline
    _pipeline = pipeline
