"""Cognitive NLP workers — write soft signals into MindState (fail-soft)."""

from .runner import NLP_WORKER_NAMES, NlpWorkerRunner

__all__ = ["NLP_WORKER_NAMES", "NlpWorkerRunner"]
