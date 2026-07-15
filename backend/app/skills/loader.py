"""Parses front-matter from skills/library/*.md into a lightweight catalog
without needing to load full body content for a listing — see
project-plan.md §6. The library content is a starting draft distilled from
established frameworks (CBT/DBT, motivational interviewing, published
burnout/sleep research); it still needs review by a licensed mental health
professional before being treated as production-ready (see each file's
`source` front-matter field)."""
from dataclasses import dataclass
from pathlib import Path

import yaml

LIBRARY_DIR = Path(__file__).resolve().parent / "library"


@dataclass
class Skill:
    id: str
    title: str
    tags: list[str]
    summary: str
    source: str
    content: str


def _parse_file(path: Path) -> Skill:
    _, front_matter, body = path.read_text().split("---", 2)
    meta = yaml.safe_load(front_matter)
    return Skill(
        id=meta["id"],
        title=meta["title"],
        tags=meta.get("tags", []),
        summary=meta["summary"],
        source=meta.get("source", ""),
        content=body.strip(),
    )


def load_catalog() -> list[Skill]:
    return sorted((_parse_file(p) for p in LIBRARY_DIR.glob("*.md")), key=lambda s: s.id)


def get_skill(skill_id: str) -> Skill | None:
    for skill in load_catalog():
        if skill.id == skill_id:
            return skill
    return None
