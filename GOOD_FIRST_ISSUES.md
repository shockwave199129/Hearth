# Seed list: good first issues for Hearth

A working list to copy into GitHub Issues (with `good first issue` /
`help wanted` labels) once the project is opened up — not itself a
contributor-facing doc. Grouped by effort and area, against the current
state of the tree.

Re-check the paths and counts below before opening any of these; they
describe a moving codebase and go stale as the work gets done.

## Low effort, non-technical or light-technical (best true "first" issues)

- **Add a new skill to the library** — `backend/app/skills/library/` has
  7 skills (grounding, journaling, boundary-setting, etc.) as plain
  Markdown. Adding one more (e.g. "gratitude practice" or
  "time-blocking for overwhelm") following the existing file pattern is
  approachable for a non-ML contributor and immediately useful.
- **Add a manifest-based skill with a benchmark** — the newer skill format
  is more structured than the plain library files: a
  `backend/app/skills/<category>/<skill>/` directory holding `manifest.yaml`
  + `content.md`, with its benchmark cases living separately under
  `backend/app/benchmarks/skills/<skill>/`. Seven skills use this format
  already; `burnout-recovery-research.md` is the only plain library file
  not yet ported, so that's the one remaining "learn the newer system"
  issue. After that, this becomes "add a genuinely new skill in the
  manifest format" instead.
- **Add a new eval golden case** — `backend/app/eval/nlp_golden_cases.json`
  and `backend/app/eval/test_transcripts/` (currently 5 transcripts:
  ambiguous distress, boundary setting, crisis trigger, low-energy
  venting, ordinary chat) could use more coverage — e.g. "grief",
  "burnout", "conflict de-escalation" transcripts.
- **Docs pass on `docs/architecture.md` / `docs/privacy.md`** — good for a
  contributor who wants to understand the whole system before writing
  code; documentation clarity issues are genuinely valuable here given
  privacy claims need to be exactly right.
- **Cross-platform build reports** — running
  `scripts/build_windows_installer.ps1` or the Tauri Linux build (see
  `desktop/src-tauri/README.md`) on a machine you have, and filing a
  detailed issue on what broke. No code needed, very high value.

## Medium effort, frontend

- **New Settings panel option** — `frontend/src/pages/Settings.tsx` +
  `frontend/src/components/*Panel.tsx` follow a consistent pattern; a
  self-contained addition (e.g. a data-export button wired to an existing
  backend route) is a reasonable first PR into the React codebase.
- **Accessibility pass on `VoiceOrb`/`AlertStack` components** — keyboard
  nav, ARIA labels, reduced-motion support. Concrete, scoped, and
  genuinely important for a voice-first app.
- **`TranscriptLog` improvements** — search/filter within a session
  transcript, or export to text/markdown.

## Medium effort, backend

- **Extend `backend/tests/`** — `test_phase3_skills.py` and
  `test_phase5_learning.py` exist; earlier phases (`checkin/`, `memory/`,
  `relationship/`) look under-covered by comparison. Picking one module
  and writing tests against its current behavior is a safe, well-scoped
  PR that doesn't require deep familiarity with the whole system.
- **`scripts/hardware_check.py` — support detection for more GPU/NPU
  configs.** Currently tiers S/A/B/C map hardware to model sizes; adding
  detection for a hardware class not yet covered (e.g. specific AMD or
  Apple Silicon configs) is scoped and testable.

## Higher effort, requires more context (label as `help wanted`, not
`good first issue`)

- **`hearth_ai/` label expansion** — adding new labels to
  `hearth_ai/labels/*.yaml` (emotion/intent/memory/relationship/strategy)
  and corresponding training data in `hearth_ai/data/prepare/` — needs a
  contributor comfortable with the annotation schemas.
- **`backend/app/cognitive/` complexity estimation tuning** — touches
  core response-routing behavior; good for a contributor who's already
  done 1-2 smaller PRs and understands the eval harness.

## Notes on triage

- Anything under `backend/app/safety/` or `backend/app/safety2/` should
  **not** be labeled `good first issue` regardless of apparent size — see
  CONTRIBUTING.md's "Safety-critical code" section.
- When you open these on GitHub, add a 2-3 line "what good looks like"
  note per issue — first-time contributors bounce off vague issues more
  than hard ones.
