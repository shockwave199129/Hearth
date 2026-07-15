# Privacy

This is a local-first app. Everything about you — what you say, what the
companion remembers, when it last checked in on you — stays on your device.
The only thing that can ever leave it is described in full below, and it's
off by default.

## What's stored, and where

| Data | Where | Encrypted? |
|---|---|---|
| Your profile (name, age range, stressors, etc.) | `backend/data/profile.db`, table `profiles` | Yes — SQLCipher |
| Conversation history (every turn, every session) | same file, table `chat_history` | Yes — text encrypted before insert |
| Long-term memories the companion has saved | `backend/data/vector_store/` (Chroma) | Yes — fact text encrypted before insert (the embedding vector itself is not, since similarity search needs it) |
| When you were last checked in on | same file, table `checkin` | Yes — whole file is SQLCipher-encrypted |
| Crisis-detector trigger history | same file, table `crisis_events` | Yes |
| Escalation history (see below) | same file, table `escalations` | Yes |

Nothing above is ever synced anywhere. There's no account, no server, no
telemetry.

## Multiple profiles

One install can hold several named profiles (Settings → Profiles) — each
has its own memory, history, and check-in state, fully isolated from the
others. Only one is active at a time. Deleting a profile deletes *all* of
its data across every table above — never a partial delete.

## The one thing that can leave your device

If, during onboarding, you explicitly opt in and provide an emergency
contact, the app *can* notify that person — but only if **both**: (a) you
consented, and (b) the crisis detector has triggered repeatedly within a
short window, not just once. A single ambiguous phrase never triggers
outreach.

**As of today, this is a stub.** No message is actually sent anywhere — the
app logs what *would* be sent and stops there. A real provider (SMS, email)
hasn't been wired in yet; see `backend/app/safety/escalation.py`'s
docstring. This will be reviewed and clearly re-documented before any real
outreach capability ships.

## What you can see and delete yourself

Nothing here is "quiet forever" — it's quiet during conversation (the
companion doesn't narrate its own memory operations), but always visible
and editable if you go look:

- **Settings → Memory** — browse, correct, or delete anything the
  companion has saved about you.
- **Settings → Conversation history** — browse, replay, or delete past
  turns. Replay re-synthesizes the stored text fresh each time through the
  normal TTS engine — no audio files are cached anywhere.
- **Settings → Safety** — see your emergency-contact status and a count of
  recent crisis-detector triggers/escalations.
- **Settings → Profiles** — delete an entire profile and everything tied
  to it in one action.

## What we deliberately don't do

- No analytics, no crash reporting to a third party, no update-check
  ping — this app doesn't know you exist.
- No cloud inference — the LLM, STT, and TTS all run on your machine.
- No plaintext data at rest, anywhere, for any of the categories above.
