"""Pydantic models for Book Volume 4's memory taxonomy — Episodic memory
(Ch 5), Semantic memory (Ch 6), and Emotional metadata (Ch 7, attached to
either as metadata, never a separate store). Every persisted object carries
a `schema_version` (Ch 14) so future template revisions never orphan
existing data."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

MEMORY_SCHEMA_VERSION = 1


class MemoryStatus(str, Enum):
    active = "active"
    decayed = "decayed"
    consolidated = "consolidated"
    contradicted = "contradicted"


class EmotionalMetadata(BaseModel):
    """Not a fifth store — metadata attached to an episodic/semantic entry
    (Vol 4 Ch 7), populated by the same non-LLM emotion classifier Volume 1
    already runs for live conversation."""

    schema_version: int = MEMORY_SCHEMA_VERSION
    valence: str = "neutral"  # positive | negative | mixed | neutral
    intensity: float = 0.0  # 0.0-1.0
    category: str = "neutral"  # e.g. grief, joy, anxiety, pride, shame
    sensitivity_flag: bool = False


class EpisodicMemory(BaseModel):
    """A specific, timestamped, remembered event (Vol 4 Ch 5). `summary` is
    always assembled from a template, never free-form LLM prose."""

    schema_version: int = MEMORY_SCHEMA_VERSION
    id: str
    user_id: str
    timestamp: datetime
    summary: str
    entities: list[str] = Field(default_factory=list)
    significance_markers: list[str] = Field(default_factory=list)
    emotional_metadata: EmotionalMetadata = Field(default_factory=EmotionalMetadata)
    reference_count: int = 0
    status: MemoryStatus = MemoryStatus.active
    last_reinforced: datetime
    # Ch 12 consolidation — ids of source entries a merged entry absorbed;
    # nothing is destroyed by a merge, only compacted.
    merged_from: list[str] = Field(default_factory=list)


class SemanticMemory(BaseModel):
    """A durable, general fact distilled from a cluster of episodic entries
    (Vol 4 Ch 6) — never created directly. `fact` is always templated."""

    schema_version: int = MEMORY_SCHEMA_VERSION
    id: str
    user_id: str
    fact: str
    source_episodes: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    emotional_metadata: EmotionalMetadata = Field(default_factory=EmotionalMetadata)
    last_reinforced: datetime
    status: MemoryStatus = MemoryStatus.active
