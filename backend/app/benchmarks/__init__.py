"""The unified benchmark library (Book Vol 8 Ch 7) — Volume 5's per-skill
benchmarks and Volume 6's per-category safety benchmarks now live under one
root, plus a new `cross_volume/` category neither of them covers:

    benchmarks/
      skills/<skill_id>/*.yaml           # Volume 5 Ch 6
      safety/<risk_category>/*.yaml       # Volume 6 Ch 13
      cross_volume/*.yaml                 # new: multi-turn, multi-system

This module doesn't change either system's own acceptance standard — see
`app.skills.benchmark_runner` and `app.safety2.benchmark_runner` for the
per-domain runners, and `app.benchmarks.runner` for the release gate
(Ch 9) that unifies all three."""
from __future__ import annotations

from pathlib import Path

BENCHMARKS_ROOT = Path(__file__).resolve().parent
SKILLS_BENCHMARKS_ROOT = BENCHMARKS_ROOT / "skills"
SAFETY_BENCHMARKS_ROOT = BENCHMARKS_ROOT / "safety"
CROSS_VOLUME_BENCHMARKS_ROOT = BENCHMARKS_ROOT / "cross_volume"
