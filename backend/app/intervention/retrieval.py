"""Deterministic skill retrieval."""

from __future__ import annotations

import re

from app.skills.loader import Skill, load_catalog

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def retrieve_candidates(transcript: str, limit: int = 8) -> list[Skill]:
    query = _tokens(transcript)
    scored: list[tuple[int, Skill]] = []
    for skill in load_catalog():
        haystack = _tokens(
            " ".join([skill.summary, " ".join(skill.tags), " ".join(skill.manifest.when_use), skill.content])
        )
        score = len(query.intersection(haystack))
        lower = transcript.lower()
        if any(trigger.lower() in lower for trigger in skill.manifest.when_use):
            score += 2
        if skill.id == "crisis_support" and any(token in lower for token in ("hurt myself", "kill myself", "suicide", "self harm")):
            score += 5
        if score > 0:
            scored.append((score, skill))
    scored.sort(key=lambda item: (-item[0], item[1].id))
    return [skill for _, skill in scored[:limit]]
