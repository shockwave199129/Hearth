# Local Voice Companion — Project Plan

> **Historical document.** This records the original design and its
> rationale. The product's current positioning has changed — Hearth is no
> longer described as an "emotional-support" companion, for the reasons in
> [`compliance.md`](compliance.md). See [`positioning.md`](positioning.md)
> for current framing. Kept as written; the engineering rationale below is
> still accurate.

Privacy-first emotional-support voice companion. Runs entirely on the user's
machine. No data leaves the device except one narrow, consented path: crisis
escalation.

Stack: LFM2.5-1.2B (LLM) · Moonshine (STT) · Chatterbox-Turbo (TTS) ·
EmbeddingGemma-300M (long-term memory embeddings) · Chroma (vector store) ·
FastAPI (backend) · React/Vite (frontend) · Tauri (desktop shell)

---

## 1. Folder structure

```
voice-companion/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app, websocket entrypoint
│   │   ├── config.py                # paths, ports, tier config
│   │   │
│   │   ├── hardware/
│   │   │   ├── detect.py            # RAM, VRAM, CPU probe
│   │   │   └── tier_manager.py      # maps hardware -> model tier
│   │   │
│   │   ├── llm/
│   │   │   ├── server_manager.py    # spawns/monitors llama.cpp subprocess
│   │   │   └── chain.py             # LangChain/LangGraph orchestration
│   │   │
│   │   ├── stt/
│   │   │   └── moonshine_engine.py
│   │   │
│   │   ├── tts/
│   │   │   ├── chatterbox_engine.py
│   │   │   └── voice_profiles/      # bundled male.wav, female.wav refs
│   │   │
│   │   ├── memory/
│   │   │   ├── short_term.py        # rolling window + summarizer
│   │   │   ├── long_term.py         # create/update/delete/list/get store ops
│   │   │   ├── tools.py             # tool schemas exposed to the LLM
│   │   │   └── embedder.py          # embeddinggemma llama.cpp wrapper
│   │   │
│   │   ├── checkin/
│   │   │   ├── state.py             # tracks last_checkin_at, tiny and cheap to read
│   │   │   └── tools.py             # mark_checkin tool schema
│   │   │
│   │   ├── skills/
│   │   │   ├── library/             # psychoeducational reference markdown
│   │   │   ├── loader.py
│   │   │   └── tools.py             # list_skills / get_skill tool schemas
│   │   │
│   │   ├── eval/
│   │   │   ├── rubric.md            # scoring criteria for response quality
│   │   │   ├── llm_judge.py         # offline eval harness
│   │   │   └── test_transcripts/    # regression test conversations
│   │   │
│   │   ├── security/
│   │   │   └── crypto.py            # key management, encrypt/decrypt helpers
│   │   │
│   │   ├── safety/
│   │   │   ├── crisis_detector.py
│   │   │   ├── escalation.py        # the one internet-facing module
│   │   │   └── safety_audio/        # pre-synthesized wav
│   │   │
│   │   ├── onboarding/
│   │   │   ├── profile_schema.py
│   │   │   └── profile_store.py
│   │   │
│   │   ├── graph/
│   │   │   └── conversation_graph.py  # LangGraph state machine, ties it together
│   │   │
│   │   └── db/
│   │       ├── chroma_client.py
│   │       └── sqlite_models.py     # profile + chat history tables
│   │
│   ├── models/                      # gguf / onnx weights, downloaded on first run
│   │   ├── llm/
│   │   ├── embedding/
│   │   ├── stt/
│   │   └── tts/
│   │
│   ├── data/                        # per-user, gitignored, never synced, encrypted at rest
│   │   ├── profile.db               # SQLCipher-encrypted
│   │   ├── chat_history.db          # SQLCipher-encrypted
│   │   ├── vector_store/            # chroma persistent dir, documents encrypted before write
│   │   └── audio_cache/             # AES-encrypted
│   │
│   └── requirements.txt
│
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── Onboarding.tsx
│       │   ├── Chat.tsx
│       │   └── Settings.tsx
│       ├── components/
│       │   ├── VoiceOrb.tsx         # mic state / speaking indicator
│       │   └── TierBadge.tsx        # shows detected hardware tier
│       └── hooks/
│           ├── useAudioRecorder.ts
│           └── useCompanionSocket.ts
│
├── desktop/
│   └── src-tauri/                   # Tauri wrapper for installable app
│
├── scripts/
│   ├── setup.py                     # detects hardware, pulls right model files
│   └── hardware_check.py
│
└── docs/
    ├── architecture.md
    └── privacy.md
```

**Why this shape:** backend does all model work as a single local FastAPI
process (websocket for streaming audio/text both directions), frontend is a
normal web UI served from that same process at `localhost`, and Tauri wraps
the whole thing into a double-clickable installer later without changing any
of the backend code. You can develop and test purely in a browser tab against
`localhost:8000` before touching Tauri at all.

---

## 2. Hardware detection and model tiering

Run once on first launch, cached in `config.py`, re-checked if the user
changes machines or forces a re-scan from Settings.

```python
# hardware/detect.py
import psutil, subprocess, json

def detect_hardware():
    ram_gb = psutil.virtual_memory().total / (1024**3)
    gpu = detect_gpu()  # nvidia-smi / rocm-smi parse, None if absent
    return {
        "ram_gb": round(ram_gb, 1),
        "gpu_name": gpu["name"] if gpu else None,
        "vram_gb": gpu["vram_gb"] if gpu else 0,
    }
```

| Tier | Trigger | LLM | STT | TTS |
|---|---|---|---|---|
| S | GPU with ≥8GB VRAM | LFM2.5 BF16, GPU | Moonshine base, GPU | Chatterbox-Turbo, GPU, full exaggeration range |
| A | GPU 4-8GB VRAM, or CPU + RAM ≥16GB | LFM2.5 Q8 | Moonshine base | Chatterbox-Turbo, CPU |
| B | CPU only, RAM 8-16GB | LFM2.5 Q6_K | Moonshine tiny | Kokoro-82M fallback |
| C | RAM <8GB | LFM2.5 Q4_K_M | Moonshine tiny | Kokoro-82M fallback |

`tier_manager.py` picks the tier, `server_manager.py` launches
`llama-server` with the matching gguf and `--n-gpu-layers` set to `99` if a
GPU was detected, else `0`. Re-running the detector is cheap enough to do on
every app launch rather than trusting a stale cache.

Kokoro is the safe TTS fallback on tiers B/C — Chatterbox at those specs
would be usable but slow enough to hurt the conversational feel, so drop
paralinguistic tags and exaggeration control gracefully rather than forcing
a bad experience.

---

## 3. Encryption at rest

Everything in `data/` is sensitive, so it's encrypted even though it never
leaves the device — the threat model here is a shared computer, theft, or
casual access, not network interception.

**Key management:** on first launch, generate a random 256-bit key and store
it in the OS keychain (`keyring` library — Windows Credential Manager, macOS
Keychain, Linux Secret Service). No password prompt needed day-to-day; the
OS login itself gates access to the keychain entry. This is the practical
default. A user-set unlock PIN on top of this is a reasonable future
addition but adds friction worth deferring past v1.

```python
# security/crypto.py
import keyring
from cryptography.fernet import Fernet

SERVICE_NAME = "voice-companion"

def get_or_create_key() -> bytes:
    key = keyring.get_password(SERVICE_NAME, "data_key")
    if key is None:
        key = Fernet.generate_key().decode()
        keyring.set_password(SERVICE_NAME, "data_key", key)
    return key.encode()

_fernet = Fernet(get_or_create_key())

def encrypt(text: str) -> bytes:
    return _fernet.encrypt(text.encode())

def decrypt(token: bytes) -> str:
    return _fernet.decrypt(token).decode()
```

**Where it applies:**
- `profile.db` / `chat_history.db` — swap `sqlite3` for `sqlcipher3`, key
  from `get_or_create_key()`.
- Chroma documents — encrypt the fact text with `encrypt()` before
  `memory.add()`, decrypt after `memory.query()`. The embedding vector itself
  stays unencrypted (Chroma needs it for similarity search), but the
  human-readable fact text never sits on disk in plaintext.
- `audio_cache/` — encrypt files at write, decrypt on playback.

## 4. Onboarding flow

First launch only, before the first real conversation. Keep it short — a
handful of fields, optional skips, framed as "so I can support you better"
rather than a form.

```python
# onboarding/profile_schema.py
class UserProfile(BaseModel):
    name: str
    age_range: str            # "18-24", "25-34" etc — avoid exact DOB
    gender: str | None        # optional, used for voice/pronoun defaults
    profession: str | None
    stressors: list[str] = [] # e.g. "work deadlines", "family", "finances"
    preferred_voice: str      # "male" | "female"
    companion_name: str = "Assistant"
    created_at: datetime
```

Store as an encrypted local SQLite row (or a simple `age_range` bucket
rather than raw age, to minimize sensitive data at rest even though it never
leaves the device). This profile gets injected into the system prompt every
session — it's static context, not something to re-retrieve via search.

```python
SYSTEM_PROMPT_TEMPLATE = """You are {companion_name}, a warm, calm companion
for {name}, who works as a {profession} and has mentioned dealing with
{stressors}. Keep the persona consistent with what you know, but don't
recite these facts back at them — use them to inform tone, not to perform
familiarity."""
```

---

## 4. Conversation memory — two layers

### Short-term (within a session)

Rolling buffer of the last N raw turns (e.g. 20). Once the session exceeds
that, summarize the oldest chunk into a short paragraph and drop the raw
turns — same pattern as LangChain's summary-buffer memory, just built by
hand so it's easy to reason about with a 1.2B model doing the summarizing.

```python
# memory/short_term.py
def maybe_summarize(state):
    if len(state["messages"]) <= 20:
        return state
    old_chunk = state["messages"][:10]
    summary_prompt = f"Summarize this exchange in 2-3 sentences, keeping only what matters for future support:\n{format_messages(old_chunk)}"
    summary = llm.invoke(summary_prompt).content
    state["session_summary"] = (state.get("session_summary", "") + " " + summary).strip()
    state["messages"] = state["messages"][10:]
    return state
```

### Long-term (across sessions) — tool-based, not auto-injected

Nothing from long-term memory gets pasted into the system prompt. Instead
the LLM gets a small set of tools and decides when to use them — this keeps
the context lean (no stale, growing "here's what I know about you" block)
and gives the model explicit, deliberate control over its own memory rather
than a wall of facts it may or may not use well.

```python
# memory/embedder.py — separate lightweight llama.cpp embedding server
# ./llama-server --model embeddinggemma-300M-Q8_0.gguf --embedding --port 8002
import requests

def embed(text: str) -> list[float]:
    r = requests.post("http://127.0.0.1:8002/embedding", json={"content": text})
    return r.json()["embedding"]
```

```python
# memory/long_term.py
import chromadb
from security.crypto import encrypt, decrypt

client = chromadb.PersistentClient(path="./data/vector_store")
memory = client.get_or_create_collection("long_term_memory")

def create(text: str, category: str, user_id: str) -> str:
    mem_id = str(uuid4())
    memory.add(
        documents=[encrypt(text).decode("latin1")],
        embeddings=[embed(text)],
        metadatas=[{"user_id": user_id, "category": category, "updated_at": now_iso()}],
        ids=[mem_id],
    )
    return mem_id

def update(mem_id: str, new_text: str):
    memory.update(ids=[mem_id], documents=[encrypt(new_text).decode("latin1")],
                   embeddings=[embed(new_text)])

def delete(mem_id: str):
    memory.delete(ids=[mem_id])

def list_memories(user_id: str, category: str | None = None) -> list[dict]:
    # returns id + category + short label only — not full content
    where = {"user_id": user_id} if not category else {"user_id": user_id, "category": category}
    results = memory.get(where=where)
    return [{"id": i, "category": m["category"], "label": decrypt(d.encode("latin1"))[:40]}
             for i, d, m in zip(results["ids"], results["documents"], results["metadatas"])]

def get(mem_id: str) -> str:
    result = memory.get(ids=[mem_id])
    return decrypt(result["documents"][0].encode("latin1"))

def search(query: str, user_id: str, k: int = 5) -> list[dict]:
    results = memory.query(query_embeddings=[embed(query)], n_results=k, where={"user_id": user_id})
    return [{"id": i, "text": decrypt(d.encode("latin1"))}
             for i, d in zip(results["ids"][0], results["documents"][0])]
```

**Tool schemas exposed to the LLM** — this is the only part the model
actually sees; the raw memory content stays out of the system prompt
entirely:

```python
# memory/tools.py
MEMORY_TOOLS = [
    {"name": "list_memories", "description": "List what you remember about this user — id, category, and a short label only. Call this before assuming you don't know something.",
     "parameters": {"category": "optional filter: preference | stressor | life_event | relationship | other"}},
    {"name": "get_memory", "description": "Get the full text of a specific memory by id.",
     "parameters": {"id": "string"}},
    {"name": "search_memories", "description": "Semantic search over memories for a topic, when you don't know the exact id.",
     "parameters": {"query": "string"}},
    {"name": "create_memory", "description": "Save a new fact worth remembering long-term.",
     "parameters": {"text": "string", "category": "string"}},
    {"name": "update_memory", "description": "Correct or refresh an existing memory — e.g. a stressor that's resolved, a changed circumstance.",
     "parameters": {"id": "string", "text": "string"}},
    {"name": "delete_memory", "description": "Remove a memory that's no longer accurate or relevant.",
     "parameters": {"id": "string"}},
]
```

System prompt addition is one line, not a data dump:

```
You have memory tools (list_memories, get_memory, search_memories,
create_memory, update_memory, delete_memory). Use them to check what you
know before assuming, and to keep memory accurate as the person's situation
changes. Manage memory quietly in the background — don't narrate memory
operations to the user or announce what you're saving/updating/removing
unless they directly ask what you remember.
```

**Silent maintenance:** at the end of a session, run one extra LLM turn
(not shown to the user) with the session summary as context and the memory
tools available, prompted roughly as "review what you know about this
person against this session — create, update, or delete memories as
needed." This is how point 3 gets implemented: the model can revise or
retire outdated facts (e.g. a resolved stressor) as part of normal upkeep,
without it being a conversational event. Even though this happens quietly
in-chat, it's worth surfacing in a **Settings → Memory** screen where the
user can browse, edit, or delete anything themselves — quiet by default
during conversation, but never actually hidden from them if they go look.

**Resource footprint check:** unchanged from before — EmbeddingGemma Q8_0 is
~330MB, embedding calls are cheap, and both the embedding server and the
end-of-session maintenance pass only fire at session boundaries, not on
every message, so this is fine even on tier B/C hardware.

---

## 6. Skills library — psychoeducational reference content

Same access pattern as memory: a library of markdown files distilled from
psychological and emotional-support research (grounding techniques, active
listening/validation language, cognitive reframing basics, burnout recovery,
sleep hygiene, boundary-setting, journaling prompts), exposed via
`list_skills` / `get_skill` tools rather than dumped into the system prompt.
The model pulls in a technique when the conversation calls for it, instead
of carrying the whole library as context on every turn.

```
backend/app/skills/
├── library/
│   ├── grounding-techniques.md
│   ├── active-listening-validation.md
│   ├── cognitive-reframing-basics.md
│   ├── burnout-recovery-research.md
│   ├── sleep-hygiene.md
│   ├── boundary-setting.md
│   └── journaling-prompts.md
├── loader.py             # parses front-matter, builds the catalog
└── tools.py               # list_skills / get_skill tool schemas
```

Each file carries front-matter so the catalog can be listed cheaply without
loading full content:

```markdown
---
id: grounding-techniques
title: Grounding techniques for acute stress or racing thoughts
tags: [anxiety, overwhelm, night-time, panic]
summary: Short sensory and breathing techniques to interrupt spiraling thoughts.
source: Adapted from CBT/DBT grounding literature — see references section.
---

## 5-4-3-2-1 technique
Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can
smell, 1 you can taste...

## Box breathing
...
```

```python
# skills/tools.py
SKILL_TOOLS = [
    {"name": "list_skills", "description": "List available support techniques — id, title, tags, one-line summary only. Check this before reaching for a technique from general knowledge.",
     "parameters": {"tag": "optional filter, e.g. anxiety, sleep, boundaries"}},
    {"name": "get_skill", "description": "Get the full content of a specific technique by id.",
     "parameters": {"id": "string"}},
]
```

System prompt addition, again just a pointer, not the content itself:

```
You have skills tools (list_skills, get_skill) — reference material on
support techniques like grounding, validation language, and reframing.
Check list_skills when a technique might genuinely help, but adapt what you
find to a short, spoken, conversational reply — never read a skill file
back verbatim, and don't turn a reply into a lecture or numbered list.
```

**Content sourcing — this needs real care, not just drafting:** these files
should be adapted from established, citable frameworks (CBT/DBT grounding
exercises, motivational interviewing's active-listening patterns, published
burnout research) and kept at a self-help/psychoeducational register —
techniques and general information, not diagnostic criteria or clinical
treatment protocols. Before shipping, have a licensed mental health
professional review the library, and keep a `source` field in every file's
front-matter for provenance and future review. This is content the app's
credibility rests on — worth treating as seriously as the safety layer.

## 7. Evaluation

Two different things, don't conflate them:

**Offline eval harness (dev-facing, this is the main piece).** A rubric
derived from the skill library plus the safety requirements, run against a
set of test transcripts during development — this is how you catch
regressions before they reach a real conversation, not something that runs
live in the voice loop.

```
backend/app/eval/
├── rubric.md              # scoring criteria
├── llm_judge.py           # scores transcripts against the rubric
└── test_transcripts/      # sample conversations, including edge cases
```

Rubric dimensions worth scoring per response:
- **Validation before advice** — does it acknowledge the feeling before
  jumping to a fix?
- **Length/format** — 2-3 sentences, no lists, spoken-language register
- **Register** — psychoeducational, not clinical/diagnostic language
- **Tag misuse** — no more than one emotion tag, none on heavy content
- **Crisis handling** — did the crisis path correctly trigger (or correctly
  *not* trigger) on each test case
- **Memory/skill tool use** — called when relevant, not overused or forced

```python
# eval/llm_judge.py — run as a script, not part of the live pipeline
def score_response(user_msg: str, assistant_reply: str, rubric_path="rubric.md") -> dict:
    rubric = open(rubric_path).read()
    judge_prompt = f"""Score this reply against the rubric. Return JSON with
a 1-5 score and one-line reason per dimension.\n\nRubric:\n{rubric}\n\n
User: {user_msg}\nReply: {assistant_reply}"""
    return json.loads(llm.invoke(judge_prompt).content)
```

Run this over `test_transcripts/` (include crisis-trigger cases,
low-energy/burnout venting, ordinary chat, and edge cases like ambiguous
distress) every time you change the system prompt, skill library, or model
tier — this is your regression suite.

**Runtime self-check (optional, keep it light).** A pre-TTS pass that flags
obvious rubric violations (too long, too clinical, tag misuse) before a
reply is spoken. Worth having as a safety net, but keep it a fast heuristic
rather than a second full LLM call — voice latency matters, and the offline
harness is where the real quality assurance should happen. If it flags a
reply, regenerate once with a shorter/simpler prompt rather than blocking
the conversation.

## 8. Dynamic check-ins

Simpler than a scoring formula: the backend just tracks *when* the
companion last checked in, and the LLM decides *whether* to ask again,
using its own judgment plus whatever it already knows from the
conversation. No separate heaviness algorithm needed — the model already
reads the conversation content, so let it reason directly rather than
building a parallel scoring system to approximate what it can already see.

**State tracked (tiny, cheap):**

```python
# checkin/state.py
def get_last_checkin(user_id: str) -> datetime | None:
    # single row lookup in profile.db — encrypted, same as other tables
    ...

def set_last_checkin(user_id: str, ts: datetime):
    ...
```

At the start of every session, the backend computes `days_since_last_checkin`
and drops it into the system prompt as one small line of state — this isn't
a content dump like memory/skills, it's a single scalar, so it's fine to
include directly rather than gating behind a tool:

```
Today's date: {date}. It has been {days_since_last_checkin} days since you
last asked how they're feeling (or: you have never asked). If it's been a
day or more, or you never have, weave a genuine check-in about how they're
doing into your reply — not as a script, just naturally, the way you'd ask
a friend. Don't force it if you already asked recently, and don't let it
turn into a checklist question separate from the conversation.
```

**The LLM marks it done itself**, via one tool — this is what closes the
loop without any backend logic guessing whether a check-in "counted":

```python
# checkin/tools.py
CHECKIN_TOOLS = [
    {"name": "mark_checkin", "description": "Call this once, right after you've asked how the user is feeling/doing in your reply — not for every message, only when you've actually asked.",
     "parameters": {}}
]
```

```python
def handle_mark_checkin(user_id: str):
    set_last_checkin(user_id, datetime.now())
```

So the actual flow you described works like this: two days ago the
companion asked how you were doing and called `mark_checkin`. Today, your
first message of the new session arrives, the backend sees
`days_since_last_checkin = 2`, includes that in the system prompt, and the
LLM's reply naturally answers your message *and* asks how you've been —
then calls `mark_checkin` again, resetting the clock.

**In-conversation noticing** (a person seeming to be struggling mid-session,
not just at session start) doesn't need a separate heuristic either — it's
covered by the same judgment plus the skills library: the system prompt can
simply say the model should check in directly if the conversation itself
starts reading as heavy, rather than waiting to be asked. Worth watching in
practice whether a 1.2B model's judgment here is reliable enough, or
whether it needs a nudge from the eval harness — that's a good thing to
check with the rubric in §7 once you have real transcripts, rather than
something to over-engineer now.

## 9. Safety layer

Already designed in earlier passes — carries over unchanged:
- `crisis_detector.py`: regex-based, runs before every LLM call
- On trigger: skip LLM and TTS generation entirely, play the pre-synthesized
  `safety_audio/response.wav`
- `escalation.py`: the **only** module allowed to touch the network, and
  only after either a repeated/escalating crisis pattern or explicit user
  consent captured during onboarding (e.g. an opt-in emergency contact).
  This needs its own design pass — flagged as a separate milestone below
  rather than bolted on here.

---

## 10. Build roadmap

1. **Core loop, no memory, no onboarding** — mic → Moonshine → LFM2.5 →
   Chatterbox → speaker, hardcoded profile, working end to end on your own
   machine.
2. **Hardware detection + tiering** — swap in the tier manager, test on at
   least one low-spec target (CPU-only VM is a good stand-in) to confirm
   tier B/C actually feels usable.
3. **Encryption at rest** — keychain key management, SQLCipher swap, Chroma
   document encryption. Do this before any real user data exists, not after.
4. **Onboarding + static profile injection** — UI form, encrypted SQLite
   storage, system prompt templating.
5. **Short-term summarization** — rolling window behavior, verify it holds
   up past 100+ messages without drifting or losing thread.
6. **Long-term memory as tools** — EmbeddingGemma + Chroma, the five memory
   tools, end-of-session silent maintenance pass, and the Settings → Memory
   browse/edit screen. Test that the LLM actually calls the tools sensibly
   before trusting it unsupervised.
7. **Skills library + eval harness** — draft the first few skill files,
   `list_skills`/`get_skill` tools, and the offline rubric-based eval
   harness with a starter set of test transcripts. Get the skill content
   reviewed before it's relied on in real conversations.
8. **Dynamic check-ins** — `last_checkin_at` tracking, the `mark_checkin`
   tool, and the system prompt instruction. Let the LLM's own judgment drive
   timing rather than a formula; validate against real usage that it
   neither over-asks nor lets long gaps pass silently.
9. **Safety/crisis layer** — detector, pre-synth audio, escalation consent
   flow (separate design pass).
10. **Tauri packaging** — wrap for distribution once the web app is stable.

---

## 11. Open decisions worth settling before you start coding

- **Exact escalation trigger and channel** (SMS/email/call, which service,
  what consent UI) — needs its own design session.
- **Whether the LLM's check-in judgment needs a backstop** — start by
  trusting it with just the `days_since_last_checkin` context, but watch
  real transcripts (via the eval harness) for over-asking or, worse,
  silently letting a long gap pass without ever checking in; add a simple
  hard floor (e.g. force a check-in past N days regardless of judgment)
  only if that shows up in practice.
- **How much memory-tool activity to surface in Settings** — quiet during
  conversation is decided, but worth deciding whether the Memory screen
  shows a changelog (what got added/updated/removed and when) or just
  current state.
- **Who reviews the skills library** — the content quality/safety of the
  psychoeducational files matters as much as the code; line up a licensed
  mental health professional (or at minimum, well-sourced published
  frameworks with clear citations) before treating it as production-ready.
