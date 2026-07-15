# Hearth

A privacy-first emotional-support voice companion that runs entirely on
your own machine. No data leaves your device except one narrow, explicitly
consented path (crisis escalation) — see [`docs/privacy.md`](docs/privacy.md).

```
mic → Moonshine (STT) → LFM2.5 via a LangChain agent → Chatterbox/Kokoro (TTS) → speaker
```

For the full design history and rationale, see
[`project-plan.md`](project-plan.md) and [`project-phases.md`](project-phases.md).
For how the app actually works today, see [`docs/architecture.md`](docs/architecture.md).

## Stack

LFM2.5-1.2B (LLM, via `llama-server`) · Moonshine (STT) · Chatterbox
with a Kokoro-82M fallback (TTS) · EmbeddingGemma-300M + Chroma (long-term
memory) · LangChain `create_agent` (tool-calling agent + middleware) ·
FastAPI (backend) · React/Vite (frontend) · Tauri (desktop packaging).

## Project layout

```
backend/     FastAPI app, all model work (STT/LLM/TTS), REST + websocket API
frontend/    React/Vite web app (onboarding, chat, settings)
desktop/     Tauri wrapper — packages frontend + backend into an installer
scripts/     hardware_check.py (tier probe), setup.py (model downloads)
docs/        architecture.md, privacy.md
```

Hardware is auto-detected and mapped to a tier (S/A/B/C) that picks model
sizes and quantizations accordingly — see `docs/architecture.md`'s
"Hardware tiering" section. Tier B/C (CPU-only or low RAM) still works, just
with a smaller model and a lighter TTS engine.

## Getting started

### 1. Prerequisites

- Python 3.11+ and Node.js 20+
- [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server`
  binary, built with Jinja chat-template support (needed for tool calling),
  and on your `PATH` (or set `LLAMA_SERVER_BIN` — see `backend/app/config.py`)

### 2. Backend setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-common.txt
pip install -r requirements-gpu.txt   # tier S/A — Chatterbox TTS
# or: pip install -r requirements-cpu.txt   # tier B/C — Kokoro TTS
```

Not sure which tier you are? Run `python ../scripts/detect_tier_requirements.py`
after installing `requirements-common.txt` — it prints the right file.

Download the model files for your detected hardware tier:

```bash
python ../scripts/setup.py
```

This pulls the right LFM2.5 GGUF quantization and the EmbeddingGemma GGUF
for every tier, plus Kokoro's ONNX files on tier B/C. Moonshine and
Chatterbox auto-download their own weights on first use — nothing to do
for those.

Run the backend:

```bash
python -m app.main          # starts the FastAPI server on :8000
python -m app.main --cli    # or: talk to it directly via mic/speaker, no frontend needed
```

### 3. Frontend setup

Uses [pnpm](https://pnpm.io) (`npm install -g pnpm` if you don't have it):

```bash
cd frontend
pnpm install
pnpm run dev   # Vite dev server on :5173, proxies /api and /ws to :8000
```

Open `http://localhost:5173` — first launch walks you through onboarding
(you can create multiple profiles later from Settings → Profiles).

### 4. Desktop packaging (optional)

```bash
cd desktop
pnpm install
pnpm run tauri:dev     # run the desktop shell against the Vite dev server
pnpm run tauri:build   # build an installer for the current platform
```

See [`desktop/src-tauri/README.md`](desktop/src-tauri/README.md) for Linux
build prerequisites and what's still scaffold-only (frozen Python
interpreter, real app icons).

## Development notes

- `scripts/hardware_check.py` — see what tier your machine lands on without
  starting the full app.
- `backend/app/eval/llm_judge.py` — offline rubric-based regression harness;
  run it after changing the system prompt, skill library, or model tier.
- Everything under `backend/data/` is encrypted at rest and per-install
  (never commit it) — see `docs/privacy.md` for exactly what's stored and
  where.
