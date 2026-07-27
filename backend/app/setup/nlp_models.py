"""Install hearth_ai NLP ONNX package into ``MODELS_DIR/nlp``.

Source order (first complete package wins):
  1. ``backend/bundled/nlp`` (packaged / optional ship)
  2. Repo ``models/nlp`` (dev checkout)
  3. Existing ``NLP_MODELS_DIR`` env if it already points at a complete tree

Destination is always ``{MODELS_DIR}/nlp`` so runtime resolution prefers the
installed copy. Missing source → log and skip (fail-soft; app still runs).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

from app.config import BACKEND_DIR, MODELS_DIR

ProgressFn = Callable[[str], None]

NLP_INSTALL_DIR = MODELS_DIR / "nlp"

_NLP_REQUIRED_RELATIVE = (
    "manifest.json",
    "tokenizer.json",
    "encoder/model.onnx",
    "emotion/model.onnx",
    "emotion/labels.json",
    "intent/model.onnx",
    "intent/labels.json",
    "memory/model.onnx",
    "memory/labels.json",
    "relationship/model.onnx",
    "relationship/labels.json",
    "strategy/model.onnx",
    "strategy/labels.json",
)


def nlp_package_complete(root: Path | None) -> bool:
    if root is None or not root.is_dir():
        return False
    return all((root / rel).is_file() for rel in _NLP_REQUIRED_RELATIVE)


def find_nlp_source() -> Path | None:
    """Locate a complete ONNX package to install from."""
    candidates: list[Path] = []
    env = os.environ.get("NLP_MODELS_DIR", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(BACKEND_DIR / "bundled" / "nlp")
    if not getattr(__import__("sys"), "frozen", False):
        candidates.append(BACKEND_DIR.parent / "models" / "nlp")
    for path in candidates:
        if nlp_package_complete(path):
            return path
    return None


def ensure_nlp_models(log: ProgressFn = print) -> Path | None:
    """Copy NLP classifiers into ``MODELS_DIR/nlp`` if not already present.

    Returns the install directory when usable, else None.
    """
    if nlp_package_complete(NLP_INSTALL_DIR):
        log(f"already have NLP classifiers at {NLP_INSTALL_DIR}")
        return NLP_INSTALL_DIR

    src = find_nlp_source()
    if src is None:
        log("NLP package not found (bundled/ or models/nlp) — classifiers will fail-soft")
        return None

    if src.resolve() == NLP_INSTALL_DIR.resolve():
        log(f"NLP classifiers already at install path {NLP_INSTALL_DIR}")
        return NLP_INSTALL_DIR

    log(f"installing NLP classifiers from {src} → {NLP_INSTALL_DIR}")
    if NLP_INSTALL_DIR.exists():
        shutil.rmtree(NLP_INSTALL_DIR)
    shutil.copytree(src, NLP_INSTALL_DIR)

    if not nlp_package_complete(NLP_INSTALL_DIR):
        log(f"NLP install incomplete at {NLP_INSTALL_DIR}")
        return None
    log("NLP classifiers installed")
    return NLP_INSTALL_DIR
