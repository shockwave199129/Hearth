# Implementing The Hearth Architecture Book (THF)

## Context

`The Book/` is a 10-volume design spec ("The Hearth Framework", THF) for a
privacy-first AI companion whose goal is **supportive presence over time**, not
answering questions. Its North Star: *"Does this help someone feel safe enough
to speak?"* The defining architectural idea is a split between a **Cognitive
Layer** (decides *what to do*, uses **no LLM**) and an **Execution Layer**
(reliably carries it out). A non-LLM **Cognitive Scheduler** is the sole
orchestrator; parallel **Workers** (Memory, Emotion, Relationship, Safety,
Goal/Intent, Reflection) feed an **Intervention Engine**, **Prompt Builder**,
**Compute Router**, and **Response Composer**. Invariant #5: *the LLM never
decides strategy.*

The existing `backend/app/` already ships the simpler `project-plan.md` voice
companion end-to-end, but as an **LLM-centric** design: a single LangChain
`create_agent` graph where the model decides everything and calls flat
memory/skills/checkin tools. That inverts the Book's core principle.

**This plan** does a **faithful re-architecture to THF**, following the Book's
own Volume 9 roadmap (Phases 0–5, with **Phase 4 Safety as a hard gate on real
users**). Decisions locked with the user:
- **Voice-first is preserved** — STT→LLM→TTS + S/A/B/C hardware tiering stay; the
  cognitive layer sits *behind* the voice pipeline.
- **Reuse & refactor in place** — existing storage, skills files, crisis
  detector, voice engines, and tiering become the Book's **Infrastructure /
  Execution layer**; the new cognitive layer is built on top. The app stays
  runnable throughout.

The end state replaces the LangChain `create_agent` core with a Scheduler +
Workers cognitive layer, while `llama-server`, Chroma, SQLCipher, Moonshine,
Parler/Kokoro, and the Tauri shell remain as the execution/infrastructure it
drives.

---

## Layer mapping: existing code → Book layers

| Book layer | Reuse (exists) | Build new |
|---|---|---|
| Infrastructure | `llm/server_manager.py`, `db/` (SQLCipher+Chroma), `stt/`, `tts/`, `hardware/`, `security/crypto.py`, `setup/` | DuckDB observation store; versioned safety resource store |
| Execution | `memory/embedder.py`, `skills/loader.py`, `safety/crisis_detector.py`, voice loop in `main.py` | Compute Router, LLM Runtime Adapter, Tool Runner, Cache Manager |
| Cognitive | *(none — this is the inversion)* | Scheduler, Complexity Estimator, Workers, Intervention Engine, Prompt Builder, Response Composer, Growth Engine, State Manager, MindState |

**Core refactor:** `main.py`'s `Pipeline`/`create_agent` becomes a thin
Execution-layer driver invoked *by* the Scheduler, not the orchestrator itself.
The `/ws` voice loop and REST API stay; only what sits between transcript-in and
text-out changes.

---

## Target backend structure (new modules under `backend/app/`)

```
cognitive/
  mind_state.py          # MindState (Pydantic v2, live/RAM) — Vol1 Ch4
  scheduler.py           # Cognitive Scheduler, sole orchestrator — Vol1 Ch5
  complexity.py          # Conversation Complexity Estimator (non-LLM) — Vol1 Ch6
  state_machine.py       # Idle→Greeting→…→Closing — Vol1 Ch13
  budget.py              # ThinkingBudget per runtime profile — Vol1 Ch16
workers/
  base.py                # CognitiveWorker interface (should_run/execute/update)
  memory.py emotion.py relationship.py goal.py safety.py reflection.py
intervention/
  engine.py              # strategy-first, then skill — Vol1 Ch8
  retrieval.py ranking.py
execution/
  compute_router.py      # route task→rules|classifier|vector|LLM — Vol1 Ch10
  prompt_builder.py      # PromptPlan assembler — Vol1 Ch9
  response_composer.py   # final gate: plan adherence, Hearth style, safety — Vol1 Ch11
  llm_adapter.py         # wraps existing llama-server client
growth/
  engine.py              # async post-response Reflection — Vol1 Ch12
state/
  manager.py snapshot.py # State Manager + RuntimeSnapshot, restart-safe — Vol1 Ch14
relationship/            # Vol3: trust, attachment, development, affinity, boundaries, life_model, shared_history
memory2/                 # Vol4: episodic/semantic + formation/promotion/decay/retrieval/consolidation
learning/
  observation_store.py   # DuckDB append-only — Vol7 Ch3
  recompute.py           # EWMA recomputation pipeline — Vol7 Ch4
  cold_start.py insights.py
evaluation/
  worker.py invariants.py benchmarks.py regression.py release_gate.py  # Vol8
safety2/                 # Vol6: Safety Worker, escalation, resources/, audit log
```

Data models are **Pydantic v2**, every persistent one carries `schema_version`
(Book invariant). Contract objects (`CognitiveTask`, `WorkerResult`,
`ResponsePlan`, `PromptPlan`) are data models, never ad-hoc strings (Vol1 Inv #8).

---

## Phased plan (follows Book Volume 9)

Each phase must satisfy the cross-phase DoD: relevant **Invariants pass**,
**benchmarks pass with zero regression from prior phases**, every deliverable
**traceable to a chapter**.

### Phase 0 — Runtime Foundation (Vol 1)
Stand up the cognitive layer skeleton driving the *existing* voice loop.
- Build `MindState`, `Scheduler` (start two-tier: fast-path vs full),
  `ComplexityEstimator`, `ComputeRouter`, `PromptBuilder`, `ResponseComposer`
  (at minimum "apply Hearth style"), `StateManager` + `RuntimeSnapshot`.
- Reroute `main.py`: transcript → Scheduler → PromptBuilder → `llm_adapter`
  (existing `llama-server`) → ResponseComposer → TTS. Growth Engine is an inert
  stub. No memory/relationship/skills wired yet.
- **DoD:** coherent multi-turn voice conversation; survives app restart via
  snapshot; meets Reflex + Conversation latency tiers on current hardware.
- **Risk gate (Book Risk #2):** validate the local model actually hits Vol 1
  latency targets here, before building on top.

### Phase 1 — Basic Conversation (Vol 2)
- Active Listening, Emotional Validation, Question Generation as prompt logic;
  Conversation Lifecycle (Greeting/Listening/Supporting/Closing);
  `CommunicationPreferences` layer only (name, formality, response length) —
  reuse onboarding fields; learned Traits stubbed neutral until Phase 5.
- Author anti-pattern avoidance into templates (Vol2 Ch24).
- **DoD:** passes Vol 2 Communication Invariants; preferences respected across a
  session.

### Phase 2 — Memory & Relationship (Vol 3 + 4, parallel tracks)
- **Memory (`memory2/`):** Working→Episodic→Semantic pipeline; rule-based
  Formation triggers; template Promotion (LLM fallback deferrable); Decay +
  deterministic Retrieval on **Chroma** (reuse `chroma_client.py`,
  `embedder.py`). **Privacy controls ship in this phase** — view/correct/delete
  (extend existing Settings→Memory UI).
- **Relationship (`relationship/`):** Trust + Development level via simple direct
  computation (temporary placeholder, flagged for Phase-5 replacement);
  Boundaries + Life Model; Attachment signals *computed & logged only* (no
  escalation response until Phase 4).
- **DoD:** a memory formed in one conversation is retrieved and restrainedly
  used in a later one; restart resumes full relationship context; user can
  view/correct/delete a specific memory (tested explicitly).

### Phase 3 — Intervention & Skills (Vol 5)
- Convert `skills/library/*.md` to the Book's `skills/<category>/<id>/`
  (`manifest.yaml` + `content.md`), `values_implemented` mapped to a Vol 0 value;
  load-time manifest validation (malformed = hard fail).
- Populate **all six**: Validation, Grounding, Journaling, Cognitive Reframing,
  Boundary Setting, Sleep Hygiene; author **Crisis Support content only** (not
  detection). Build Skill benchmarks *before* skills are "done".
- Intervention Engine proper: strategy-first, Candidate Retrieval (vector) →
  Ranking; Multi-Skill Composition + `_compatibility.yaml` (at least the
  Grounding+Reframing incompatibility). `SkillObservation` → DuckDB.
- **DoD:** each skill passes its benchmark; engine re-ranks as conversation
  evolves; no skill repeats without genuine re-escalation.
- ⚠️ **Content still requires licensed-clinician review before real users.**

### Phase 4 — Safety Framework (Vol 6) — **HARD GATE**
- **Safety Worker:** mandatory, always-on, wired so the Scheduler *cannot* skip
  or cache it (enforce Vol1 Ch7 guarantee in code). Layered detection: classifier
  + rule signals (reuse/upgrade `crisis_detector.py`) + contextual (Vol3/4) + LLM
  as one corroborating signal only.
- Detection for all **5 risk categories**; Escalation Protocol routing into
  Phase 3 Crisis Support; **Resource Provision** from versioned
  `safety2/resources/global.yaml` + `regions/*.yaml` (verified, current data);
  Audit log store with retention-window exemption reconciled against Phase 2
  deletion rights. Finish the **real escalation notifier** (current
  `escalation.py` is a logged stub).
- **Gate — real-user exposure allowed ONLY IF:** full safety benchmark suite
  passes; detection content **professionally validated (documented review, a
  process requirement)**; resource data verified; logging/deletion reconciled.
- **Book Risk #1 (largest):** this phase ships no new user-facing feature, so
  schedule pressure pushes to skip it — treat the gate as literally blocking at
  the release-process level.

### Phase 5 — Learning & Evaluation (Vol 7 + 8)
- **Observation Store** (`hearth.duckdb`, append-only) + generic **Recomputation
  Pipeline** (EWMA). Replace *every* temporary placeholder — Trust/Development
  (P2), Skill Affinity (P3), Communication Traits (P1) — verified by before/after
  comparison on the same accumulated data. Cold-start priors; trust starts at 0.
- **Evaluation Worker** (background, post-session): InvariantChecks,
  anti-pattern detection, success-metric proxies; results written as
  `evaluation_observations`; safety findings dual-written to the Vol6 log.
- **Benchmark library + Release Gating**: safety regression = BLOCK (no
  override); skill regression = BLOCK unless documented override.
- **DoD:** placeholders replaced & verified; Eval Worker runs without measurable
  latency impact; Release Gate wired into the real release process. (No clean
  "finished" state — ongoing infra verified working.)

---

## Cross-cutting reuse notes
- **Storage split (Vol1 Ch17):** SQLite/SQLCipher (profile, conversation,
  snapshots, cached traits) — extend `db/sqlite_models.py`; **DuckDB** (new,
  observations/derived traits); **Chroma** (episodic+semantic+skill embeddings) —
  reuse. Runtime never queries DuckDB in the critical path (reads cached traits
  from SQLite).
- **Runtime profiles** map onto existing tiers: Lite≈B/C, Balanced≈A,
  Advanced≈S — only the Scheduler's ThinkingBudget changes per profile.
- **Vector DB discrepancy** in the Book (Vol1 Ch3 says Qdrant; Ch17 & Vol4 Ch14
  say Chroma) — resolve to **Chroma**, matching existing code.
- **Growth Engine is the sole writer** to relationship/memory state (Vol3 Inv);
  workers never write MindState directly — only Scheduler-via-StateManager does.

---

## Verification
- **Per phase:** run that phase's benchmark suite + all prior suites (zero
  regression). Extend `eval/` into the Vol8 `benchmarks/{skills,safety,cross_volume}/`
  structure.
- **Phase 0 latency probe:** measure end-to-end voice round-trip against Vol 1
  Reflex/Conversation tiers on real hardware before proceeding (Risk #2).
- **End-to-end voice smoke test** each phase via `python -m app.main --cli`
  (mic→speaker, no frontend) and via the Tauri/web UI.
- **Restart-safety test** (Phase 0+): kill mid-conversation, relaunch, confirm
  RuntimeSnapshot restores context.
- **Memory privacy test** (Phase 2): create → retrieve later → correct → delete,
  each verified in Settings→Memory and on disk.
- **Safety gate (Phase 4):** full per-category safety benchmark pass with
  weighted recall; documented clinician review; over-escalation measured against
  realistic *non-crisis* transcripts too.
- **Release gate (Phase 5):** safety-regression BLOCK proven to actually block a
  release build.

---

## Key files to anchor on
- Refactor core: `backend/app/main.py` (`Pipeline`/`create_agent` → Scheduler-driven)
- Config/tiers/prompts: `backend/app/config.py`, `hardware/tier_manager.py`
- Reused infra: `db/sqlite_models.py`, `memory/embedder.py`, `skills/loader.py`,
  `safety/crisis_detector.py`, `safety/escalation.py` (finish stub in P4),
  `tts/tts_engines.py`, `stt/moonshine_engine.py`
- Frontend touch-points: `frontend/src/` Settings (Memory/Skills/Safety panels)
- Spec source of truth: `The Book/volume_1..9/` (own directory numbering;
  ignore Vol 5's stale forward-refs)
