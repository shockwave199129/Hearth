# Architecture

This describes the actual current implementation, not the original
aspirational sketch in `project-plan.md` — a few things (the agent runtime,
multi-profile support) evolved past that first pass. See `project-plan.md`
and `project-phases.md` for the design history and rationale.

## Overview

A single local FastAPI process (`backend/app/main.py`) does all model work
— STT, LLM, TTS — and serves a websocket (`/ws`) for the voice loop plus a
REST API for everything else. The frontend (`frontend/`, React + Vite) is a
normal web app talking to that same process at `localhost:8000` in
development (proxied by Vite) and served from the same origin once
packaged. `desktop/src-tauri` wraps the built frontend and spawns the
backend as a child process for a double-clickable desktop build.

```
mic → Moonshine (STT) → agent (LFM2.5 via llama-server) → Parler-TTS-Tiny-v1/Kokoro (TTS) → speaker
```

## The agent

There's no hand-rolled tool-calling loop — `backend/app/main.py`'s
`Pipeline` builds a LangChain `create_agent` graph (`langchain.agents`)
around a `ChatOpenAI` client (`langchain_openai`) pointed at the local
`llama-server` process's OpenAI-compatible `/v1/chat/completions` endpoint.
Tool execution, the iteration bound, retries, and safety-net summarization
all live in that graph's middleware stack:

- `ModelCallLimitMiddleware` — bounds the tool-calling loop.
- `ModelRetryMiddleware` — retries transient local-server failures.
- `PIIMiddleware` (email, url) — redacts obvious PII from replies.
- `SummarizationMiddleware` — a safety net above `ShortTermMemory`'s own
  rolling window, guarding the one case that can still balloon a single
  call (the end-of-session maintenance pass's full-session dump).

Tools (memory, skills, check-ins) are LangChain `@tool`-decorated
functions, built fresh per active profile via a `make_tools(user_id)`
factory closure — `user_id` is deliberately never an LLM-fillable
argument, so the model can't name an arbitrary profile. `Pipeline.set_profile`
rebuilds the whole agent graph whenever the active profile changes.

## Hardware tiering

`hardware/detect.py` probes RAM/GPU/VRAM; `hardware/tier_manager.py` maps
that to a tier (S/A/B/C per `project-plan.md` §2), each naming a specific
LLM quantization, STT model size, and TTS engine. `scripts/setup.py`
downloads exactly what the detected tier needs.

## Multi-profile support

A single install can hold several named local profiles (`onboarding/`),
each with its own memory, check-in state, crisis/escalation history, and
chat history — but only one is *active* in a running `Pipeline` at a time
(`onboarding/active_profile.py`), since this is a single-process desktop
app, not concurrent multi-tenant serving. Switching profiles (Settings →
Profiles) rebuilds the agent's tools and system prompt for the new
`user_id` and triggers a frontend reload so every hook picks up the switch.

Every user-data table (`profiles`, `checkin`, `crisis_events`,
`escalations`, `chat_history`, and Chroma's long-term memory metadata) is
keyed by `user_id`. Deleting a profile (`DELETE /api/profiles/{user_id}`)
cascades across every one of those — never a partial delete.

## Memory — two layers

- **Short-term** (`memory/short_term.py`): an in-process rolling window of
  recent turns, summarized and dropped once it exceeds `SHORT_TERM_WINDOW`.
  Not persisted — this is per-session working memory.
- **Long-term** (`memory/long_term.py`): Chroma-backed, tool-based (not
  auto-injected into the system prompt) — the model calls
  `list_memories`/`get_memory`/`search_memories`/`create_memory`/
  `update_memory`/`delete_memory` explicitly. A silent end-of-session
  maintenance pass lets the model revise its own memory based on the whole
  session. Browsable/editable in Settings → Memory.

## Persisted chat history

`memory/chat_history.py` (backed by the `chat_history` table) stores every
turn, encrypted, per profile. It exists to back **replay**: rather than
caching synthesized audio anywhere, `GET /api/chat_history/{id}/audio`
re-synthesizes the stored text on demand via the normal TTS engine.
Browsable/deletable in Settings → Conversation history, and replayable
in-context from the live transcript during a session.

## Skills library

Static psychoeducational markdown files (`skills/library/`), exposed the
same way as memory — `list_skills`/`get_skill` tools, never dumped into the
system prompt. See each file's `source` front-matter field; this content
still needs review by a licensed mental health professional before being
relied on in practice.

## Safety layer

`safety/crisis_detector.py` runs a narrow, phrase-based regex check before
every LLM call. On a match, the LLM and live TTS are skipped entirely in
favor of a pre-synthesized response (`safety/safety_audio/`), and
`safety/escalation.py` checks whether to notify an emergency contact — only
if the user explicitly consented **and** a repeated-pattern threshold has
been crossed (never on a single ambiguous phrase). The actual "send" is
currently a logged stub (`LoggedNotifier`) — no real SMS/email provider is
wired in yet; see the module's docstring for what a real integration needs.

## Evaluation

Two distinct things: `eval/llm_judge.py` is a dev-only, offline
rubric-scoring script run against `eval/test_transcripts/` — the real
regression suite, never part of the live pipeline. `eval/self_check.py` is
a fast runtime heuristic (sentence count, list markers, clinical-language
check) run on every reply before TTS; if it flags something, the reply is
regenerated exactly once with a short nudge, then used regardless — a light
safety net, not a second LLM call.

## Encryption at rest

Everything in `backend/data/` is encrypted — the threat model is a shared
computer or device theft, not network interception (this app makes no
outbound network calls except the currently-stubbed escalation path). A
random key lives in the OS keychain (`security/crypto.py`); `profile.db`
(which holds profiles, checkin, crisis/escalation, and chat history) is
SQLCipher-encrypted, and Chroma's long-term-memory document text is
Fernet-encrypted before it touches disk (the embedding vector itself stays
unencrypted, since Chroma needs it for similarity search).

## Packaging

`desktop/src-tauri` wraps the built frontend (`frontend/dist`) and spawns
the Python backend as a child process. See its own README for what's
scaffolded versus real follow-up work (frozen Python interpreter, real app
icons, resource-path resolution for a packaged build).
