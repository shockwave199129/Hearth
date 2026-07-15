"""mark_checkin — lets the model close the loop on dynamic check-ins itself,
so the backend never has to guess whether a check-in "counted". See
project-plan.md §8.

`make_tools(user_id)` binds the tool to one profile via closure, same
reasoning as memory/tools.py — user_id is never an LLM-fillable argument."""
from datetime import datetime, timezone

from langchain_core.tools import tool

from app.checkin.state import set_last_checkin


def make_tools(user_id: str) -> list:
    @tool
    def mark_checkin() -> dict:
        """Call this once, right after you've asked how the user is
        feeling/doing in your reply — not for every message, only when
        you've actually asked."""
        set_last_checkin(user_id, datetime.now(timezone.utc))
        return {"ok": True}

    return [mark_checkin]
