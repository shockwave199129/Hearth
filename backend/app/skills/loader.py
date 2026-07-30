"""Validated skills library loader for phase 3.

Loads the THF folder structure:
skills/<category>/<skill_id>/{manifest.yaml,content.md}
and falls back to the legacy flat markdown catalog if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

SKILLS_ROOT = Path(__file__).resolve().parent
LIBRARY_DIR = SKILLS_ROOT / "library"
SCHEMA_VERSION_FILE = Path(__file__).resolve().parent / "_schema_version.txt"

_V0_VALUES = {
    "understanding_before_advice",
    "curiosity_over_assumption",
    "compassion_over_judgment",
    "presence_over_perfection",
    "honesty_over_illusion",
    "growth_over_dependency",
    "respect_autonomy",
    "hope_over_false_reassurance",
}


@dataclass(frozen=True)
class SkillManifest:
    name: str
    skill_id: str
    version: str
    description: str
    tags: list[str]
    when_use: list[str]
    values_implemented: list[str]
    avoid: list[str]
    conversation_depth: list[str]
    estimated_tokens: int
    estimated_latency: int
    deprecated: bool
    category: str
    source: str = ""


@dataclass(frozen=True)
class Skill:
    id: str
    title: str
    tags: list[str]
    summary: str
    source: str
    content: str
    manifest: SkillManifest
    path: Path


def _validate_manifest(meta: dict, *, skill_id: str, category: str) -> SkillManifest:
    required = ["name", "skill_id", "version", "description", "tags", "when_use", "values_implemented"]
    for key in required:
        if key not in meta:
            raise ValueError(f"missing required manifest field: {key}")
    if meta["skill_id"] != skill_id:
        raise ValueError(f"skill_id mismatch: manifest has {meta['skill_id']}, folder has {skill_id}")
    if meta.get("category", category) != category:
        raise ValueError(f"category mismatch for {skill_id}: {meta.get('category')} vs {category}")
    values = meta["values_implemented"]
    unknown = [v for v in values if v not in _V0_VALUES]
    if unknown:
        raise ValueError(f"unknown values_implemented for {skill_id}: {unknown}")
    avoid = meta.get("avoid", [])
    overlap = set(meta["when_use"]).intersection(avoid)
    if overlap:
        raise ValueError(f"when_use/avoid overlap for {skill_id}: {sorted(overlap)}")
    return SkillManifest(
        name=meta["name"],
        skill_id=meta["skill_id"],
        version=str(meta["version"]),
        description=meta["description"],
        tags=list(meta.get("tags", [])),
        when_use=list(meta.get("when_use", [])),
        values_implemented=list(values),
        avoid=list(avoid),
        conversation_depth=list(meta.get("conversation_depth", ["medium", "deep"])),
        estimated_tokens=int(meta.get("estimated_tokens", 150)),
        estimated_latency=int(meta.get("estimated_latency", 0)),
        deprecated=bool(meta.get("deprecated", False)),
        category=meta.get("category", category),
        source=str(meta.get("source", "")),
    )


def _parse_structured_skill(manifest_path: Path, content_path: Path, *, category: str, skill_id: str) -> Skill:
    meta = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    manifest = _validate_manifest(meta, skill_id=skill_id, category=category)
    content = content_path.read_text(encoding="utf-8").strip()
    return Skill(
        id=manifest.skill_id,
        title=manifest.name,
        tags=manifest.tags,
        summary=manifest.description,
        source=manifest.source or manifest.description,
        content=content,
        manifest=manifest,
        path=manifest_path.parent,
    )


def _parse_legacy_file(path: Path) -> Skill:
    _, front_matter, body = path.read_text(encoding="utf-8").split("---", 2)
    meta = yaml.safe_load(front_matter)
    skill_id = meta["id"]
    content = body.strip()
    manifest = SkillManifest(
        name=meta["title"],
        skill_id=skill_id,
        version="1.0.0",
        description=meta["summary"],
        tags=list(meta.get("tags", [])),
        when_use=list(meta.get("tags", [])),
        values_implemented=["understanding_before_advice"],
        avoid=[],
        conversation_depth=["medium", "deep"],
        estimated_tokens=150,
        estimated_latency=0,
        deprecated=False,
        category="legacy",
        source=str(meta.get("source", "")),
    )
    return Skill(id=skill_id, title=meta["title"], tags=manifest.tags, summary=meta["summary"], source=meta.get("source", ""), content=content, manifest=manifest, path=path)


def _structured_skill_paths() -> list[tuple[Path, Path, str, str]]:
    out: list[tuple[Path, Path, str, str]] = []
    if not SKILLS_ROOT.exists():
        return out
    for category_dir in sorted(p for p in SKILLS_ROOT.iterdir() if p.is_dir() and not p.name.startswith(("_", "library"))):
        for skill_dir in sorted(p for p in category_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            manifest = skill_dir / "manifest.yaml"
            content = skill_dir / "content.md"
            if manifest.exists() and content.exists():
                out.append((manifest, content, category_dir.name, skill_dir.name))
    return out


def load_catalog() -> list[Skill]:
    structured = _structured_skill_paths()
    if structured:
        return sorted((_parse_structured_skill(m, c, category=cat, skill_id=skill_id) for m, c, cat, skill_id in structured), key=lambda s: s.id)
    legacy = sorted(LIBRARY_DIR.glob("*.md"))
    return sorted((_parse_legacy_file(p) for p in legacy), key=lambda s: s.id)


def get_skill(skill_id: str) -> Skill | None:
    for skill in load_catalog():
        if skill.id == skill_id:
            return skill
    return None
