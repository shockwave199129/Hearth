# Contributing to Hearth

Thanks for helping. Hearth is a privacy-first local companion — keep that
stance in mind for every change (see [`docs/privacy.md`](docs/privacy.md)).

## Before you start

```bash
git clone --recurse-submodules https://github.com/shockwave199129/Hearth.git
cd Hearth
```

- **Backend:** Python 3.12+, from `backend/`: install deps from
  `requirements-common.txt` (plus GPU/CPU extras only when you need them).
- **Frontend:** Node 20+, from `frontend/`: `pnpm install`.
- **Desktop:** see [`desktop/src-tauri/README.md`](desktop/src-tauri/README.md)
  for Tauri / system library prerequisites.

Design context lives in [`docs/architecture.md`](docs/architecture.md) and
[`docs/project-plan.md`](docs/project-plan.md).

## What we expect in a PR

1. **One concern per PR** when practical — easier to review and revert.
2. **Tests for the behaviour you change.** Backend: `pytest` under
   `backend/tests/`. Frontend: `pnpm test` / Playwright for user-visible
   flows you touch.
3. **No secrets** in the tree (`.env`, keychain material, crash dumps with
   personal data).
4. **Comments explain *why*** when the code isn't obvious — match the tone
   already in the repo.
5. Fill out the PR template (summary + test plan).

## Local checks (same spirit as CI)

```bash
# from repo root / backend
cd backend && python -m pytest -q

# lint (needs hearth_ai submodule initialized for the full CI command)
ruff check backend scripts

# frontend
cd frontend && pnpm typecheck && pnpm test
```

Packaged desktop builds are heavier — only required when you change the
Tauri shell, freeze, or installer path. See
[`.github/workflows/build.yml`](.github/workflows/build.yml).

## Issues

Use the Bug / Feature templates under `.github/ISSUE_TEMPLATE/`. Security-
sensitive reports about local data exposure or supply-chain issues should
not be filed publicly if they disclose an exploit path — open a private
security advisory on GitHub instead when possible.

## Code review

`CODEOWNERS` routes review requests. Address review comments or explain
why not; don't force-merge around unresolved threads on protected paths.
