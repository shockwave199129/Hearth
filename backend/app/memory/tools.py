"""LangChain tools for long-term memory — the only part of memory the model
actually sees; raw memory content stays out of the system prompt entirely.
See project-plan.md §5.

`make_tools(user_id)` builds tools bound to one profile via closure —
user_id is deliberately NOT an LLM-fillable tool argument (that would let
the model name an arbitrary profile), it's fixed by the backend at
construction time. main.py's Pipeline rebuilds the agent (and therefore
these tools) whenever the active profile changes — see Pipeline.set_profile."""
from langchain_core.tools import tool

from app.memory import long_term


def make_tools(user_id: str) -> list:
    @tool
    def list_memories(category: str | None = None) -> list[dict]:
        """List what you remember about this user — id, category, and a short
        label only. Call this before assuming you don't know something.

        Args:
            category: optional filter: preference | stressor | life_event | relationship | other
        """
        return long_term.list_memories(user_id, category)

    @tool
    def get_memory(id: str) -> dict:
        """Get the full text of a specific memory by id."""
        return long_term.get(id, user_id) or {"error": "not found"}

    @tool
    def search_memories(query: str) -> list[dict]:
        """Semantic search over memories for a topic, when you don't know the exact id."""
        return long_term.search(query, user_id)

    @tool
    def create_memory(text: str, category: str) -> dict:
        """Save a new fact worth remembering long-term. Check list_memories or
        search_memories first — if a similar memory already exists, call
        update_memory on it instead of creating a duplicate.

        Args:
            text: one self-contained fact, written in the third person (e.g.
                "Prefers walking over the gym" not "I like walking") — it may
                be recalled in a very different conversation later.
            category: exactly one of: preference | stressor | life_event | relationship | other
        """
        return {"id": long_term.create(text, category, user_id)}

    @tool
    def update_memory(id: str, text: str) -> dict:
        """Correct or refresh an existing memory — e.g. a stressor that's
        resolved, a changed circumstance.

        Args:
            text: the full replacement text — this overwrites the old memory
                entirely, it does not append to it.
        """
        long_term.update(id, text, user_id)
        return {"ok": True}

    @tool
    def delete_memory(id: str) -> dict:
        """Remove a memory that's no longer accurate or relevant."""
        long_term.delete(id, user_id)
        return {"ok": True}

    return [list_memories, get_memory, search_memories, create_memory, update_memory, delete_memory]
