"""Read-only skills library routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.skills.loader import get_skill, load_catalog

router = APIRouter()


@router.get("/api/skills")
def api_list_skills() -> list[dict]:
    """Read-only — the skills library is static reference content, not
    user data, so there's no edit/delete surface (unlike /api/memories)."""
    return [
        {"id": s.id, "title": s.title, "tags": s.tags, "summary": s.summary}
        for s in load_catalog()
    ]


@router.get("/api/skills/{skill_id}")
def api_get_skill(skill_id: str) -> dict:
    skill = get_skill(skill_id)
    if skill is None:
        raise HTTPException(status_code=404, detail="skill not found")
    return {"id": skill.id, "title": skill.title, "content": skill.content, "source": skill.source}
