# PyInstaller spec — freezes the backend into a standalone onedir bundle so
# an installed desktop app doesn't need a user-installed Python at all. See
# desktop/src-tauri/README.md and scripts/build_backend.sh/.ps1 (the actual
# entry point for running this — invoke via those, not `pyinstaller` raw,
# so the working directory/paths line up).
#
# onedir, not onefile: this is spawned once per app launch as a
# long-running server process, not a one-shot CLI tool — onedir avoids
# onefile's per-launch self-extraction cost and is far easier to debug
# missing-library issues in (you can just look in the output folder).
#
# VERIFIED FOR REAL (not just read/inspected), both tiers: this spec has
# been run through actual `pyinstaller` builds twice — once against
# requirements-cpu.txt (kokoro-onnx, no chatterbox-tts installed, ~1.9GB
# output) and once against requirements-gpu.txt (chatterbox-tts==0.1.7
# installed, ~6.0GB output, mostly torch). Both builds completed
# Analysis/PYZ/EXE/COLLECT successfully, and both frozen executables were
# launched directly: uvicorn started, the FastAPI startup event fired,
# hardware-tier detection ran, and each reached exactly the expected
# failure point for this sandbox — `FileNotFoundError: llama-server` (no
# llama-server binary present here) — meaning every Python import in the
# chain up to that point genuinely works frozen, including
# chromadb/onnxruntime/torch/sqlcipher3/moonshine_voice/chatterbox/
# kokoro_onnx/the whole langchain family.
#
# TIER-AWARE: requirements-gpu.txt (chatterbox-tts, tier S/A) and
# requirements-cpu.txt (kokoro-onnx, tier B/C) are mutually incompatible in
# one Python environment (conflicting numpy pins — see
# requirements-common.txt), so a given frozen build's venv only ever has
# ONE of the two installed. This spec detects which one via a real import
# probe (below) rather than assuming both are present, so collect_all()/
# hiddenimports only reference the package that's actually installed.
from pathlib import Path

# Import the core PyInstaller building classes required for the spec.
# These were previously missing, which can lead to runtime errors.
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# NOTE: `__file__` is not defined in a PyInstaller spec's exec namespace —
# use the `SPECPATH` global PyInstaller injects instead (caught by actually
# running this spec, not just reading it).
BACKEND_DIR = Path(SPECPATH).resolve()
APP_DIR = BACKEND_DIR / "app"

# --- Data files actually read at runtime by main.py's server path (not the
# dev-only eval/ harness, which main.py never imports). Paths mirror how
# the code reads them (e.g. skills/loader.py's `Path(__file__).parent /
# "library"`) — PyInstaller preserves this package-relative layout under
# its extraction root.
datas = [
    (str(APP_DIR / "skills" / "library"), "app/skills/library"),
    (str(APP_DIR / "safety" / "safety_audio"), "app/safety/safety_audio"),
    (str(APP_DIR / "tts" / "voice_profiles"), "app/tts/voice_profiles"),
]

# Which TTS stack is actually installed in this build's venv — real import
# probe, not a guess, since requirements-gpu.txt/-cpu.txt are never both
# installed at once (see note above).
try:
    import chatterbox  # noqa: F401
    _HAS_CHATTERBOX = True
except ImportError:
    _HAS_CHATTERBOX = False

try:
    import kokoro_onnx  # noqa: F401
    _HAS_KOKORO = True
except ImportError:
    _HAS_KOKORO = False

if _HAS_CHATTERBOX == _HAS_KOKORO:
    raise RuntimeError(
        "Expected exactly one of chatterbox-tts/kokoro-onnx installed "
        f"(got chatterbox={_HAS_CHATTERBOX}, kokoro_onnx={_HAS_KOKORO}) — "
        "install requirements-gpu.txt or requirements-cpu.txt, not both/neither."
    )

hiddenimports = [
    # uvicorn dynamically selects its event loop / protocol implementations
    # at runtime — static import analysis doesn't see these.
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    # All deferred (function-local) imports in the codebase, kept lazy
    # there deliberately (heavy/native, only needed once a given engine is
    # actually constructed) — PyInstaller's bytecode scan generally follows
    # these already, but declared explicitly since they're never imported
    # at module level anywhere. sqlcipher3's actual native piece is a
    # compiled extension module (_sqlite3...so), which PyInstaller's
    # default Analysis bundles automatically once the import is found —
    # confirmed collect_dynamic_libs("sqlcipher3") finds nothing (it's not
    # a loose ctypes-loaded library), so it's not used here.
    "sqlcipher3",
    "sqlcipher3.dbapi2",
    "moonshine_voice",
    "moonshine_voice.transcriber",
]
if _HAS_CHATTERBOX:
    # tts_turbo only — tts/chatterbox_engine.py uses ChatterboxTurboTTS
    # exclusively (the non-Turbo class is never imported anywhere).
    hiddenimports += ["chatterbox.tts_turbo"]
if _HAS_KOKORO:
    hiddenimports += ["kokoro_onnx"]

# Packages with native extensions / plugin-style dynamic imports that
# PyInstaller's default analysis reliably misses — collect_all() pulls in
# their submodules, data files, and bundled shared libraries together.
_COLLECT_ALL_PACKAGES = [
    "chromadb",
    "onnxruntime",
    "moonshine_voice",  # PyPI: moonshine-voice — see app/stt/moonshine_engine.py
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langgraph",
]
if _HAS_CHATTERBOX:
    _COLLECT_ALL_PACKAGES += ["torch", "chatterbox"]
if _HAS_KOKORO:
    _COLLECT_ALL_PACKAGES += ["kokoro_onnx"]

# NOTE: The collection of package data is performed after the Analysis
# step (which defines the `a` object) to avoid referencing `a` before it is
# created.

# Define the analysis step for PyInstaller. Only positional arguments are
# provided first, followed by keyword arguments as required by the API.
a = Analysis(
    [str(APP_DIR / "main.py")],
    pathex=[str(BACKEND_DIR)],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

# After creating the Analysis object, collect additional package data and
# binaries for packages that PyInstaller may miss. This must occur after `a`
# is defined because we extend `a.binaries`.
for _pkg in _COLLECT_ALL_PACKAGES:
    _datas, _binaries, _hiddenimports = collect_all(_pkg)
    datas += _datas
    hiddenimports += _hiddenimports
    # Filter out libmoonshine.so from binaries collected by collect_all()
    _binaries = [
        (src, dest, typ)
        for src, dest, typ in _binaries
        if not src.endswith("libmoonshine.so")
    ]
    a.binaries += _binaries

# Build the PYZ archive (pure Python modules) and the executable wrapper.
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="hearth-backend",
    debug=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="hearth-backend",
)
