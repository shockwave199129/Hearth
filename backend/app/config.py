"""Paths, ports, and tier config."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _bundle_dir() -> Path:
    """Read-only app/bundle root (requirements files, frozen package data).

    Dev: `backend/`. Frozen PyInstaller onedir: `_internal` (or `_MEIPASS`).
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent / "_internal"
    return Path(__file__).resolve().parent.parent


def _install_root() -> Path:
    """Packaged app install directory (folder that contains the Tauri shell).

    Frozen layout (see tauri.conf.json bundle.resources + installer.py):
      {install}/resources/backend/hearth-backend[.exe]  <- sys.executable
      {install}/resources/llama-cpp/
      {install}/resources/setup-python/   (archive)
    so three parents up from the backend exe is {install}/.
    """
    return Path(sys.executable).resolve().parent.parent.parent


def _os_app_data_hearth() -> Path:
    """Legacy / fallback location when the install dir is not writable."""
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "Hearth"


def _dir_is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".hearth_write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


# Only these were written under the old OS app-data root — never copy the
# whole folder (NSIS per-user installs often live at the same path).
_LEGACY_USER_DATA_NAMES = ("data", "models", "backend-deps", "setup-python", ".env")


def _migrate_legacy_user_data(target: Path) -> None:
    """One-time copy from %LOCALAPPDATA%\\Hearth (etc.) into {install}/userdata.

    Skips when target already has content or the legacy dir is missing.
    Best-effort only; copies known data names only.
    """
    legacy = _os_app_data_hearth()
    if not legacy.is_dir() or legacy.resolve() == target.resolve():
        return
    if any(target.iterdir()):
        return
    try:
        import shutil

        copied = False
        for name in _LEGACY_USER_DATA_NAMES:
            child = legacy / name
            if not child.exists():
                continue
            dest = target / name
            if child.is_dir():
                shutil.copytree(child, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(child, dest)
            copied = True
        if not copied:
            return
    except OSError:
        pass


def _user_data_dir() -> Path:
    """Writable persistent root for profiles, models, and pip-installed deps.

    Dev: same as `backend/` (repo checkout).

    Frozen: `{install}/userdata` next to the installed app (models,
    backend-deps, profile.db stay with the install). Falls back to the OS
    app-data dir (`%LOCALAPPDATA%\\Hearth`, etc.) only if the install
    folder is not writable (e.g. machine-wide Program Files).
    """
    if getattr(sys, "frozen", False):
        local = _install_root() / "userdata"
        if _dir_is_writable(local):
            _migrate_legacy_user_data(local)
            return local
        fallback = _os_app_data_hearth()
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    return Path(__file__).resolve().parent.parent


BACKEND_DIR = _bundle_dir()
USER_DATA_DIR = _user_data_dir()

# Local dev convenience — e.g. LLAMA_SERVER_BIN when running the backend
# directly via a venv rather than through the packaged desktop app (which
# sets env vars itself, see scripts/build_backend.sh/.ps1). Does nothing if
# the file doesn't exist; never overrides a var already set in the real
# environment (override=False, load_dotenv's own default). Packaged: under
# {install}/userdata (or OS app-data fallback); dev: backend/.env.
load_dotenv(USER_DATA_DIR / ".env")


def _configure_huggingface_env() -> None:
    """Keep Hugging Face caches under our userdata and avoid Windows hub
    symlinks. Parler/transformers opening a hub snapshot `config.json` that
    is a reparse point under a user path with spaces has failed in the field
    with OSError Errno 22 (Invalid argument). setdefault so an explicit
    shell/.env value still wins.
    """
    os.environ.setdefault("HF_HOME", str(USER_DATA_DIR / "hf-home"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")


_configure_huggingface_env()

MODELS_DIR = USER_DATA_DIR / "models"
DATA_DIR = USER_DATA_DIR / "data"

# Where app/setup/installer.py pip-installs the hardware-matched torch/
# onnxruntime build into, post-first-run (see project setup plan — CI ships
# a thin build with neither, since CI has no GPU to match against). Single
# definition shared by both call sites that need it: this module extends
# sys.path automatically below (covers a *second* launch, after a previous
# run already finished setup), and installer.py calls
# extend_backend_deps_path() again right after a fresh install completes
# (covers the *first* launch, extending the already-running process's
# sys.path without needing a restart).
BACKEND_DEPS_DIR = USER_DATA_DIR / "backend-deps"

# Extracted standalone Python used only for first-run `uv pip install`s
# (and a one-time `pip install uv` into this interpreter) — must be
# writable, so it lives under USER_DATA_DIR in the packaged app (not inside
# the read-only install / bundle tree).
SETUP_PYTHON_EXTRACT_DIR = USER_DATA_DIR / "setup-python"


def extend_backend_deps_path() -> None:
    path_str = str(BACKEND_DEPS_DIR)
    if BACKEND_DEPS_DIR.is_dir() and path_str not in sys.path:
        sys.path.insert(0, path_str)


extend_backend_deps_path()

LLM_MODELS_DIR = MODELS_DIR / "llm"
STT_MODELS_DIR = MODELS_DIR / "stt"
TTS_MODELS_DIR = MODELS_DIR / "tts"
EMBEDDING_MODELS_DIR = MODELS_DIR / "embedding"

# Parler-TTS weights live here as a plain directory (snapshot_download
# local_dir) — not the hub cache's blobs/snapshots symlink layout.
TTS_PARLER_REPO = "parler-tts/parler-tts-tiny-v1"
TTS_PARLER_DIR = TTS_MODELS_DIR / "parler-tts-tiny-v1"
TTS_KOKORO_REPO = "NeuML/kokoro-fp16-onnx"
TTS_KOKORO_DIR = TTS_MODELS_DIR / "kokoro-fp16-onnx"

LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
LLM_SERVER_HOST = "127.0.0.1"
LLM_SERVER_PORT = 48174

EMBEDDING_MODEL_FILE = "embeddinggemma-300M-Q8_0.gguf"
EMBEDDING_SERVER_HOST = "127.0.0.1"
EMBEDDING_SERVER_PORT = 48175

# Chroma's own local persistence (chromadb.PersistentClient) — a plain
# on-disk folder, no server process of its own. See db/chroma_client.py.
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

APP_HOST = "127.0.0.1"
APP_PORT = 48173

# Cached hardware/tier detection result, re-checked on every launch (cheap enough,
# see project-plan.md §2) rather than trusted across app versions/machine changes.
TIER_CACHE_FILE = DATA_DIR / "tier_cache.json"

SAMPLE_RATE = 16000  # shared by mic capture, Moonshine STT input, and playback

SYSTEM_PROMPT_TEMPLATE = """You are {companion_name}, a warm, calm companion \
for {name}. Everything you write is spoken aloud, not read — never use \
lists, headers, markdown, or emoji, just plain talk. Always validate their feelings first; only offer a \
suggestion if it fits naturally, don't force one onto every reply."""

MEMORY_SYSTEM_PROMPT_ADDITION = """
You have memory tools (list_memories, get_memory, search_memories,
create_memory, update_memory, delete_memory). Check what you already know
before assuming you don't, and before saving something new — update an
existing memory instead of creating a near-duplicate one. Keep memory
accurate as the person's situation changes (e.g. a resolved stressor).
Manage memory quietly in the background — don't narrate memory operations
to the user or announce what you're saving/updating/removing unless they
directly ask what you remember."""

# See project-plan.md §6 — pointer only, never the library content itself.
SKILLS_SYSTEM_PROMPT_ADDITION = """
You have skills tools (list_skills, get_skill) — reference material on
support techniques like grounding, validation language, and reframing. Call
list_skills only when a specific technique might genuinely help right now,
not on every turn. Rework whatever get_skill returns into a short, spoken,
conversational reply in your own words — never read a skill file back
verbatim, and never turn a reply into a lecture, script, or numbered list."""

# Templated fresh each turn with the current date and check-in status — see
# project-plan.md §8. Deliberately inlined as scalar state rather than gated
# behind a tool, unlike memory/skills, since it's a single fact not content.
CHECKIN_PROMPT_TEMPLATE = """
Today's date: {date}. {checkin_status} If it's been a day or more, or you
never have, weave one genuine check-in about how they're doing into this
reply — naturally, the way a friend would ask, not as a separate scripted
question. Skip it if you already asked recently. The moment you've asked,
call mark_checkin — once per check-in, not once per turn."""

# Rolling short-term window: summarize the oldest chunk once the session
# exceeds this many raw turns. See project-plan.md §4 (short-term memory).
SHORT_TERM_WINDOW = 20
SHORT_TERM_SUMMARIZE_CHUNK = 10

# Safety/crisis layer — see project-plan.md §9 and app/safety/. Spoken
# verbatim on a crisis-detector trigger, in place of any LLM-generated
# reply. NEEDS REVIEW by a licensed mental health professional before
# production use, same caveat as the skills library.
SAFETY_RESPONSE_TEXT = (
    "I want to pause here — it sounds like you're going through something "
    "really serious, and I care about your safety more than anything else "
    "right now. Please reach out to the 988 Suicide & Crisis Lifeline "
    "(call or text 988 in the US) or a trusted person near you right now — "
    "you don't have to go through this alone."
)
SAFETY_AUDIO_PATH = BACKEND_DIR / "app" / "safety" / "safety_audio" / "response.wav"

# Escalation gating (app/safety/escalation.py) — a repeated/escalating
# pattern within this window is required, on top of explicit onboarding
# consent, before the (currently stubbed) notifier ever fires. Tunable, and
# flagged as needing real review alongside the rest of the safety layer.
ESCALATION_WINDOW_DAYS = 1
ESCALATION_TRIGGER_COUNT = 2
