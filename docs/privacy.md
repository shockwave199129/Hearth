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
| Your voiceprint, **only if you set one up** | same file, table `voiceprints` | Yes — the template is encrypted before insert |

Nothing above is ever synced anywhere. There's no account, no server, no
telemetry.

### Your voiceprint (off unless you turn it on)

If you set up voice recognition (**Settings → Your voice**), Hearth stores a
*voiceprint*: a 256-number mathematical summary of how your voice sounds. It
is not a recording, and it cannot be played back — but it does identify you,
which makes it **biometric data** and the most sensitive single thing Hearth
can hold. So:

- It exists only if you explicitly set it up. Nothing records a voiceprint in
  the background, and voice conversations work fine without one.
- It is encrypted on this device, never uploaded, never included in a crash
  report, and deliberately **not** included in a data export — an export is
  plaintext, and a copy of something that identifies you does not belong in a
  plain folder.
- You can delete it on its own, at any time, without affecting your profile,
  conversations, or memories: **Settings → Your voice → Delete my
  voiceprint**. Deleting your whole profile also deletes it.
- You have to agree to it explicitly first, in writing, before anything is
  recorded. That agreement is dated and stored, and if we ever change what
  we're asking you to agree to, you'll be asked again rather than carried
  over.

#### How long it's kept, and when it's destroyed

This is the schedule Hearth follows. It is deliberately short and specific,
because "we keep it as long as necessary" is not a schedule.

A voiceprint is destroyed at whichever of these comes **first**:

1. **When you ask.** Deleting it in Settings, or deleting your profile,
   destroys it immediately along with the agreement you gave.
2. **Three years after you last talked to Hearth.** Not three years from when
   you recorded it — three years from your most recent conversation. Deleting
   your voiceprint also withdraws your agreement, so setting it up again asks
   you afresh.

Settings → Your voice shows the exact date yours is due to be destroyed.

One honest limitation: Hearth has no background service, and nothing about it
runs while the app is closed. So the three-year deletion happens the next time
you open Hearth, not on the day itself — before any conversation can use it.
If you never open Hearth again, the voiceprint stays encrypted on your own
disk, where it has always been and where nothing else can reach it. We think
that is the right trade: the alternative is a program that runs when you did
not ask it to, which is exactly what this app is built not to be.

What it is used for, exactly: when someone else is talking near your
microphone, Hearth can tell it probably wasn't you, and then it will not add
what it heard to what it remembers about you. That is all. The conversation
still happens, the reply still comes, and the words are still saved to your
transcript where you can read and delete them. It is **not** a lock — it
cannot stop anyone from using Hearth, and it is not certain about who is
speaking. If you sound different than usual, it may not recognise you, which
is why it never blocks anything.

## Uninstalling or resetting local data

**Settings → Reset local data** removes downloaded models, Python packages,
conversation history, memories, learning data, and pending crash reports. It
keeps only profile identity and preferences (such as your name, companion,
voice, and emergency-contact settings), so reinstalling Hearth does not make
you onboard again. Setup downloads the required models and packages again.

The standard Windows NSIS uninstaller and Ubuntu `.deb` / `.rpm` package
removal do the same automatically. On macOS, use **Settings → Uninstall
Hearth…** before moving the app to Trash; macOS does not notify an app when
its `.app` bundle is dragged to Trash. If the app has already been removed,
download `scripts/uninstall-macos.sh` from the release source, mount a Hearth
DMG, and run `scripts/uninstall-macos.sh --app
/path/to/Hearth.app` to perform the profile-preserving cleanup.

## Multiple profiles

One install can hold several named profiles (Settings → Profiles) — each
has its own memory, history, and check-in state, fully isolated from the
others. Only one is active at a time. Deleting a profile deletes *all* of
its data across every table above — never a partial delete.

### The one retention exception: safety records

When a message trips a safety check (signs of crisis or acute distress), a
separate, minimal record is kept for **up to 30 days**, and it survives
deleting your memories or your chat history. After 30 days it is deleted
automatically.

Being precise about what that record is, because it is narrower than it
sounds: it stores the safety *category*, which internal signals fired, and
which response the app gave. **It does not store the message, or any part
of it.** There is no transcript in the safety log.

Why it exists: so a safety response can be checked for quality and improved
— the failure mode being guarded against is the app mishandling a crisis and
nobody being able to tell afterwards.

Two limits worth stating plainly:

- **Deleting the whole profile deletes these too, immediately.** The 30-day
  exception applies to deleting memories or history, not to erasing the
  profile. A profile deletion is honoured in full.
- **You can see the count.** Settings → Safety shows how many such records
  are currently held.

## The one thing that can leave your device

If, during onboarding, you explicitly opt in and provide an emergency
contact, the app *can* notify that person — but only if **both**: (a) you
consented, and (b) the crisis detector has triggered repeatedly within a
short window, not just once. A single ambiguous phrase never triggers
outreach.

**What actually happens today, by contact method:**

- **Email — really sends, but only if this install has been configured to.**
  Sending needs SMTP credentials supplied through environment variables
  (`SAFETY_SMTP_HOST`, `SAFETY_SMTP_FROM_ADDRESS`, and optionally
  username/password). Those are **empty in a normal install**, so a
  default build sends nothing and logs what it would have sent. If you or
  whoever deployed your install has set them, a real email goes to your
  chosen contact over SMTP with STARTTLS.
- **SMS — does not send.** It logs only. SMS needs a paid third-party
  provider that hasn't been chosen, so there is deliberately no silent
  fallback: Hearth never reports having contacted someone it hasn't.

The message itself contains your name, your companion's name, and the fact
that you opted in — **not** your conversation, memories, or what triggered
the concern. See `ESCALATION_MESSAGE_TEMPLATE` in
`backend/app/safety/escalation.py`.

Both conditions still apply in every case: you opted in *and* the crisis
detector has triggered repeatedly within the window. Neither alone is
enough.

### Crash reports (opt-in, only when you say yes)

If Hearth closes unexpectedly, a short diagnostic file is written on this
device under your local userdata (`crash-logs/`). On the next launch you
are asked whether to send it. Choosing **Don't send** deletes it. Choosing
**Send report** uploads that one file over HTTPS to
`hearth-sub.s3.ap-south-1.amazonaws.com/hearth_ai/crash-logs/` and needs
an internet connection. The file contains stack traces, app version, and
OS/hardware class — never conversations, memories, or profile fields.
Nothing is uploaded unless you confirm.

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
- **Settings → Your voice** — see whether a voiceprint is stored and when it
  was made, replace it, or delete it on its own.
- **Settings → Your data → Export my data** — write your profile, both
  memory stores, and your entire conversation to plain files in
  `~/Hearth-exports/`, readable without Hearth installed. It is a copy:
  nothing is removed from the app, and nothing is uploaded. Exports live
  outside the app's data directory on purpose, so a reset or an uninstall
  does not delete them.

**An export is not encrypted.** It cannot be: a copy you can't read would
defeat the purpose. Everything Hearth keeps for itself is encrypted at rest,
but the moment you export, that data exists in plaintext in a folder in your
home directory, and anyone who can read that folder can read all of it. The
app says so before it writes, and the export contains a `README.txt`
repeating it, because the warning needs to travel with the files rather than
stay in a dialog you saw once. If that exposure isn't what you want, delete
the folder when you're finished with it.

## What we deliberately don't do

- No analytics, and no automatic crash reporting — a crash log only leaves
  this device if you confirm the on-launch prompt (see above).
- No cloud inference — the LLM, STT, and TTS all run on your machine.
- No plaintext data at rest, anywhere, for any of the categories above —
  with one deliberate, user-initiated exception: a data export you asked for
  is written in plaintext, because an export you cannot read is not an
  export. See "What you can see and delete yourself" above.
