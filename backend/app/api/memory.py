"""Legacy flat memory + Book Vol 4 tiered memory (memory2) privacy routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.deps import Pipeline, get_pipeline
from app.memory import long_term
from app.memory2 import privacy as memory2_privacy

router = APIRouter()


class MemoryUpdateRequest(BaseModel):
    text: str


@router.get("/api/memories")
def api_list_memories(
    category: str | None = None,
    pipeline: Pipeline = Depends(get_pipeline),
) -> list[dict]:
    return long_term.list_memories(pipeline.profile.user_id, category)


@router.get("/api/memories/{mem_id}")
def api_get_memory(mem_id: str, pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    result = long_term.get(mem_id, pipeline.profile.user_id)
    if result is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return result


@router.put("/api/memories/{mem_id}")
def api_update_memory(
    mem_id: str,
    payload: MemoryUpdateRequest,
    pipeline: Pipeline = Depends(get_pipeline),
) -> dict:
    user_id = pipeline.profile.user_id
    if long_term.get(mem_id, user_id) is None:
        raise HTTPException(status_code=404, detail="memory not found")
    long_term.update(mem_id, payload.text, user_id)
    return long_term.get(mem_id, user_id)


@router.delete("/api/memories/{mem_id}")
def api_delete_memory(mem_id: str, pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    long_term.delete(mem_id, pipeline.profile.user_id)
    return {"ok": True}


# --- Book Volume 4's tiered memory (memory2) — privacy controls (Ch 15) ---


@router.get("/api/memory2/summary")
def api_memory2_summary(pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    """Plain-language account of what's remembered, grouped by theme — never
    a raw record dump (Vol 4 Ch 15)."""
    return memory2_privacy.plain_language_summary(pipeline.growth_engine.store, pipeline.profile.user_id)


class Memory2CorrectionRequest(BaseModel):
    corrected_summary: str


@router.put("/api/memory2/episodic/{mem_id}")
def api_correct_episodic_memory(
    mem_id: str,
    payload: Memory2CorrectionRequest,
    pipeline: Pipeline = Depends(get_pipeline),
) -> dict:
    """A direct user correction — applied immediately, not queued for slow
    evidence-based reconciliation (Vol 4 Ch 15)."""
    corrected = memory2_privacy.correct_episodic(
        pipeline.growth_engine.store, mem_id, pipeline.profile.user_id, corrected_summary=payload.corrected_summary
    )
    if corrected is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return corrected.model_dump(mode="json")


@router.delete("/api/memory2/episodic/{mem_id}")
def api_delete_episodic_memory(mem_id: str, pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    """Hard delete, never a soft decay-to-zero (Vol 4 Ch 15) — cascades a
    proportional confidence reduction into any semantic fact this episode
    contributed to."""
    affected = memory2_privacy.delete_episodic_with_cascade(
        pipeline.growth_engine.store, mem_id, pipeline.profile.user_id
    )
    return {"ok": True, "semantic_facts_affected": [m.model_dump(mode="json") for m in affected]}


@router.delete("/api/memory2/semantic/{mem_id}")
def api_delete_semantic_memory(mem_id: str, pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    memory2_privacy.delete_semantic(pipeline.growth_engine.store, mem_id, pipeline.profile.user_id)
    return {"ok": True}
