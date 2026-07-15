"""LangChain tools for the skills library — catalog metadata only ever
reaches the system prompt as a pointer (see config.SKILLS_SYSTEM_PROMPT_ADDITION);
full content is only returned on an explicit get_skill call. See
project-plan.md §6."""
from langchain_core.tools import tool

from app.skills.loader import get_skill as _get_skill, load_catalog


@tool
def list_skills(tag: str | None = None) -> list[dict]:
    """List available support techniques — id, title, tags, and a one-line
    summary only, never the full technique. Call this before reaching for a
    technique from general knowledge, so the reply is grounded in this
    library's vetted material instead of improvised.

    Args:
        tag: optional filter, e.g. anxiety, sleep, boundaries
    """
    catalog = load_catalog()
    if tag:
        catalog = [s for s in catalog if tag in s.tags]
    return [{"id": s.id, "title": s.title, "tags": s.tags, "summary": s.summary} for s in catalog]


@tool
def get_skill(id: str) -> dict:
    """Get the full content of one specific support technique, by an id from
    list_skills. Never quote or read this content back verbatim — restate it
    in your own words as part of a short, spoken reply."""
    skill = _get_skill(id)
    if skill is None:
        return {"error": "not found"}
    return {"id": skill.id, "title": skill.title, "content": skill.content, "source": skill.source}


SKILL_TOOLS = [list_skills, get_skill]
