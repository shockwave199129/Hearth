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
6. **Sign off your commits** (see below).

## Safety-critical code

Anything under `backend/app/safety/` or `backend/app/safety2/` gets extra
scrutiny regardless of how small the diff looks, because a regression there
can reach someone in crisis. Expect a slower review, a request for test
coverage of the specific path you changed, and questions about failure
modes rather than just correctness. These paths are never labelled
`good first issue`.

The same applies to changes that widen what leaves the device — see
[`docs/privacy.md`](docs/privacy.md). Crisis escalation is the only
outbound path, and it stays explicitly consented.

## Sign-off (DCO)

Hearth uses the [Developer Certificate of Origin](https://developercertificate.org/)
rather than a CLA: you keep copyright over your contribution, and you
confirm you have the right to submit it. There's nothing to sign — just add
a `Signed-off-by` line to each commit:

```bash
git commit -s -m "your message"
```

This appends `Signed-off-by: Your Name <your@email.com>` using your git
`user.name` / `user.email`. Forgot on the last commit?
`git commit --amend -s --no-edit`. For a whole branch:
`git rebase --signoff main`.

Why this matters rather than being paperwork: because no single entity
collects copyright over the codebase, nobody — including the maintainers —
can later claim sole ownership of it. See [`GOVERNANCE.md`](GOVERNANCE.md).

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
