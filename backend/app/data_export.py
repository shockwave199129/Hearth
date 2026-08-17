"""User-owned data export — docs/roadmap-v1.md's 0.4 "data export" item.

Hearth's privacy claim is that your data is yours and stays on your machine.
That claim is only meaningful if you can actually *get it out*, in a form you
can read without Hearth installed. This module writes the profile, both
memory stores, and the full transcript to plain files in the user's home
directory.

Three decisions worth knowing, because each has a less-good obvious
alternative:

**A folder of plain files, not a zip.** The point is that a stranger — or the
user in five years, on a machine with no Hearth — can open these and read
them. A zip adds an unpacking step to every read for no gain at this size.

**A fixed destination, not a client-supplied path.** The API is loopback-only
and token-protected, but an endpoint that writes an arbitrary caller-named
path is an arbitrary-write primitive regardless, and the token lives in a
webview. Callers choose *whether* to export, never *where*. The `destination`
parameter exists for tests, which pass a tmp_path.

**Exported data is decrypted.** Everything here is encrypted at rest —
SQLCipher for the profile DB, Fernet for memory and transcript content. An
export the user cannot read would be useless, so the export is necessarily
plaintext. That is a real change in exposure, not an implementation detail:
the UI says so before writing, and `README.txt` inside the export says so
again next to the files themselves. Do not "improve" this by encrypting the
export unless you also ship a way to decrypt it that outlives the app.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import config
from app.memory import chat_history, long_term
from app.memory2.store import MemoryStore
from app.onboarding.profile_schema import UserProfile
from app.voice import store as voiceprint_store

logger = logging.getLogger("hearth.data_export")

EXPORTS_DIR_NAME = "Hearth-exports"

README_TEXT = """\
Your Hearth data
================

This folder was written by Hearth's Settings -> Your data -> Export button
on {stamp}. It is a copy, not a move: nothing was removed from the app.

IMPORTANT: these files are NOT encrypted.

Inside Hearth, your conversations and memories are encrypted on disk. This
export had to decrypt them so that you can read them, which means anyone who
can read this folder can read everything in it. If that matters to you, move
it somewhere you control (an encrypted drive, a password manager's file
vault) or delete it when you are done.

What each file is
-----------------

profile.json     Who you told Hearth you are, and your preferences.
memories.json    What Hearth remembers about you, from both memory stores.
transcript.json  Every message, oldest first, with timestamps.
transcript.txt   The same conversation as plain readable text.
voice.json       Whether a voiceprint is stored, and when it was made. The
                 voiceprint itself is NOT exported: it identifies you, and a
                 copy of it in a plain folder would be worse than useless.
manifest.json    What this export contains and which Hearth version wrote it.

Hearth is software for reflection and everyday conversation. It is not a
medical record, and nothing in here is a clinical assessment or a diagnosis.
"""


def exports_root() -> Path:
    """Where exports go: ``~/Hearth-exports``.

    The home directory rather than Downloads/Documents, whose names are
    localized and not guaranteed to exist. Outside ``USER_DATA_DIR`` on
    purpose, so `data_reset` and an uninstall leave exports alone — an
    export that a reset could delete would be worthless as a backup.
    """
    return Path.home() / EXPORTS_DIR_NAME


def _timestamp_slug(moment: datetime) -> str:
    return moment.strftime("%Y%m%d-%H%M%S")


def _profile_payload(profile: UserProfile) -> dict[str, Any]:
    """The profile as stored, minus nothing.

    The emergency contact is included: it is the user's own data, they
    supplied it, and an export that silently omitted a field would
    misrepresent what Hearth holds. `README.txt` covers the exposure.
    """
    return profile.model_dump(mode="json")


def _memories_payload(user_id: str, store: MemoryStore | None) -> dict[str, Any]:
    """Both memory stores, labelled, with per-store failures isolated.

    There are genuinely two: the flat `long_term` store (Chroma) and Book
    Vol 4's tiered `memory2` (episodic + semantic). Exporting one and
    calling the file `memories.json` would be a false claim about what
    Hearth remembers.

    Chroma or the memory2 sqlite file can be absent or unreadable on a
    half-set-up install. One store failing records an `error` for that
    store and still exports the other, rather than failing the whole export.
    """
    payload: dict[str, Any] = {}

    try:
        payload["long_term"] = long_term.export_all(user_id)
    except Exception as exc:
        logger.exception("data export: long_term store unreadable")
        payload["long_term"] = []
        payload["long_term_error"] = f"could not be read: {exc}"

    if store is None:
        payload["episodic"] = []
        payload["semantic"] = []
        payload["memory2_error"] = "tiered memory store was not available"
        return payload

    try:
        payload["episodic"] = [m.model_dump(mode="json") for m in store.list_episodic(user_id)]
        payload["semantic"] = [m.model_dump(mode="json") for m in store.list_semantic(user_id)]
    except Exception as exc:
        logger.exception("data export: memory2 store unreadable")
        payload.setdefault("episodic", [])
        payload.setdefault("semantic", [])
        payload["memory2_error"] = f"could not be read: {exc}"

    return payload


def _transcript_text(turns: list[dict], profile: UserProfile) -> str:
    """The transcript as prose, for reading rather than parsing.

    Speaker labels use the names the user chose, since that is how they
    experienced the conversation; `transcript.json` keeps the raw
    ``user``/``assistant`` roles for anything mechanical.
    """
    lines: list[str] = [
        f"Conversation with {profile.companion_name}",
        f"{len(turns)} messages, oldest first.",
        "",
    ]
    current_session: str | None = None
    for turn in turns:
        if turn["session_id"] != current_session:
            current_session = turn["session_id"]
            lines.append("")
            lines.append(f"--- session started {turn['created_at']} ---")
            lines.append("")
        speaker = profile.name if turn["role"] == "user" else profile.companion_name
        lines.append(f"[{turn['created_at']}] {speaker}: {turn['content']}")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_user_data(
    profile: UserProfile,
    store: MemoryStore | None = None,
    *,
    destination: Path | None = None,
) -> dict[str, Any]:
    """Write one export folder and return a summary for the UI.

    `destination` overrides `exports_root()` for tests only — it is never
    reachable from the API. See the module docstring.
    """
    now = datetime.now(timezone.utc)
    root = destination or exports_root()
    folder = root / f"hearth-export-{_timestamp_slug(now)}"
    # A second export inside the same second would otherwise silently
    # overwrite the first.
    suffix = 2
    while folder.exists():
        folder = root / f"hearth-export-{_timestamp_slug(now)}-{suffix}"
        suffix += 1
    folder.mkdir(parents=True)

    memories = _memories_payload(profile.user_id, store)

    try:
        turns = chat_history.export_all(profile.user_id)
        transcript_error: str | None = None
    except Exception as exc:
        logger.exception("data export: chat history unreadable")
        turns = []
        transcript_error = f"could not be read: {exc}"

    counts = {
        "long_term_memories": len(memories["long_term"]),
        "episodic_memories": len(memories["episodic"]),
        "semantic_memories": len(memories["semantic"]),
        "transcript_messages": len(turns),
    }
    problems = {
        key: value for key, value in memories.items() if key.endswith("_error")
    }
    if transcript_error:
        problems["transcript_error"] = transcript_error

    _write_json(folder / "profile.json", _profile_payload(profile))
    # Enrollment *metadata* only — deliberately never the embedding. The
    # export is plaintext by design (see the module docstring), and a
    # biometric template sitting unencrypted in a home-directory folder is a
    # materially worse thing to leak than a transcript. `voice.json` therefore
    # answers "does Hearth hold my voiceprint, and since when" without being
    # the voiceprint. Do not "complete" this by exporting the vector.
    _write_json(folder / "voice.json", voiceprint_store.metadata(profile.user_id))
    _write_json(folder / "memories.json", memories)
    _write_json(folder / "transcript.json", turns)
    (folder / "transcript.txt").write_text(_transcript_text(turns, profile), encoding="utf-8")
    _write_json(
        folder / "manifest.json",
        {
            "exported_at": now.isoformat(),
            "hearth_version": config.APP_VERSION,
            "user_id": profile.user_id,
            "encrypted": False,
            "counts": counts,
            # Present and non-empty only when a store could not be read, so
            # a consumer can tell "you have no memories" apart from "your
            # memories could not be exported".
            "incomplete": problems,
        },
    )
    (folder / "README.txt").write_text(
        README_TEXT.format(stamp=now.strftime("%d %B %Y at %H:%M UTC")), encoding="utf-8"
    )

    logger.info("exported user data to %s (%s)", folder, counts)
    return {"path": str(folder), "counts": counts, "incomplete": problems}
