"""Single source for the shipping app version.

Resolution order (first hit wins):

1. ``HEARTH_APP_VERSION`` — set by the Tauri shell from its package version
   (itself stamped from the ``v*`` git tag at build time), or by CI /
   ``scripts/build_backend.*`` while freezing.
2. ``app/VERSION`` — baked next to this module during a freeze so a
   packaged backend still knows its version if launched without Tauri
   (``--cli``).
3. Nearest ``v*`` git tag in a source checkout (``v0.3.4`` → ``0.3.4``).
4. ``0.0.0`` — untagged / workflow_dispatch MSI-safe placeholder.

Always returns a numeric ``major.minor.patch`` string with no ``v`` prefix
(Windows MSI ``ProductVersion`` rejects pre-release suffixes).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

_VERSION_FILE = Path(__file__).resolve().parent / "VERSION"
_SEMVER = re.compile(r"^\d+\.\d+\.\d+")


def strip_v_prefix(value: str) -> str:
    value = value.strip()
    if value[:1] in ("v", "V"):
        value = value[1:]
    return value.strip()


def _from_git() -> str | None:
    # Repo root is two parents up from app/version.py in a checkout
    # (app/ → backend/ → repo). Frozen bundles have no .git.
    repo = Path(__file__).resolve().parents[2]
    if not (repo / ".git").exists():
        return None
    try:
        raw = subprocess.check_output(
            ["git", "describe", "--tags", "--match", "v*", "--abbrev=0"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not raw:
        return None
    return strip_v_prefix(raw)


def resolve_app_version() -> str:
    env = os.environ.get("HEARTH_APP_VERSION", "").strip()
    if env:
        return strip_v_prefix(env)

    if _VERSION_FILE.is_file():
        baked = strip_v_prefix(_VERSION_FILE.read_text(encoding="utf-8"))
        if baked:
            return baked

    git_ver = _from_git()
    if git_ver:
        return git_ver

    return "0.0.0"


def is_numeric_semver(value: str) -> bool:
    """True when value is MSI-safe ``major.minor.patch`` (optional further dots)."""
    return bool(_SEMVER.match(value))
