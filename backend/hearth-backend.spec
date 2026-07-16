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
# TIER-AWARE: requirements-gpu.txt (parler-tts, tier S/A, torch-based) and
# requirements-cpu.txt (NeuML/kokoro-fp16-onnx via onnxruntime +
# ttstokenizer, tier B/C, no torch) are never both installed in one
# environment — see requirements-common.txt for why. This spec detects
# which one is present via a real import probe (below) rather than
# assuming, so collect_all()/hiddenimports only reference the package
# that's actually installed.
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
]

# Which TTS stack is actually installed in this build's venv — real import
# probe, not a guess, since requirements-gpu.txt/-cpu.txt are never both
# installed at once (see note above).
try:
    import parler_tts  # noqa: F401
    _HAS_PARLER = True
except ImportError:
    _HAS_PARLER = False

try:
    import ttstokenizer  # noqa: F401
    _HAS_KOKORO = True
except ImportError:
    _HAS_KOKORO = False

if _HAS_PARLER == _HAS_KOKORO:
    raise RuntimeError(
        "Expected exactly one of parler-tts/ttstokenizer installed "
        f"(got parler_tts={_HAS_PARLER}, ttstokenizer={_HAS_KOKORO}) — "
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
if _HAS_PARLER:
    hiddenimports += ["parler_tts"]
if _HAS_KOKORO:
    hiddenimports += ["ttstokenizer"]

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
if _HAS_PARLER:
    _COLLECT_ALL_PACKAGES += ["torch", "parler_tts"]
if _HAS_KOKORO:
    # ttstokenizer pulls in nltk, which normally fetches its tokenizer data
    # packages (e.g. punkt) on first use over the network rather than
    # shipping them as package data — collect_all() only grabs what's
    # already installed as package data, so this has NOT been verified to
    # cover nltk's runtime data fetch inside a frozen, possibly offline
    # build. Flagged here rather than assumed to just work.
    _COLLECT_ALL_PACKAGES += ["ttstokenizer"]

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
#
# collect_dynamic_libs() (called internally by collect_all(), verified via
# its actual source — it always returns 2-tuples, never 3) returns
# (source_path, dest_DIR) — hook-API order, where dest_DIR is a directory
# (often several files share the same one, e.g. onnxruntime's capi/ has
# multiple .so/.dylib siblings), not a full file path. That's fine when
# passed into Analysis()'s own `binaries=`/`datas=` constructor kwargs
# (its __init__ expands dest_DIR + source's basename into a full file path
# via format_binaries_and_datas(), while also reversing hook-order into TOC
# order), which is why `datas` above is left in that same (source, dest)
# order. But appending directly to the already-constructed `a.binaries`
# bypasses BOTH of those steps:
#
# 1. `a.binaries` is a TOC and expects (dest_name, src_name, typecode), not
#    hook-order (source, dest) — matching what PyInstaller's own equivalent
#    internal code (depend.analysis.Analysis.make_hook_binaries_toc) does
#    for hook-contributed binaries. An earlier version of this loop kept
#    hook-order when appending — PyInstaller's COLLECT step then checked
#    os.path.exists() on what it read as src_name (actually the un-reversed
#    dest fragment, e.g. "torch/lib"), which never exists as a real path,
#    so it silently dropped every one of these binaries ("Ignoring
#    non-existent resource torch/lib, meant to be collected as
#    .../torch/lib/libc10.dylib" in CI logs) — torch/onnxruntime/
#    moonshine_voice's actual native libraries were never bundled at all.
# 2. dest_name must be dest_DIR + the source's own basename, not dest_DIR
#    alone — using dest_DIR alone (as a subsequent fix here did) treats
#    e.g. "onnxruntime/capi" as a literal target FILENAME rather than a
#    directory, so every file collect_dynamic_libs placed in that same
#    directory collides on the exact same dest_name: on Linux this
#    silently overwrote one file with another (last-one-wins, no error at
#    all); on macOS CI it failed hard with "Pyinstaller needs to create a
#    directory at '.../onnxruntime/capi', but there already exists a file
#    at that path" — both reproduced locally with a minimal PyInstaller
#    spec before landing this fix.
for _pkg in _COLLECT_ALL_PACKAGES:
    _datas, _binaries, _hiddenimports = collect_all(_pkg)
    datas += _datas
    hiddenimports += _hiddenimports
    # Filter out libmoonshine.so from binaries collected by collect_all()
    cleaned_binaries = []
    for source, dest_dir in _binaries:
        if source.endswith("libmoonshine.so"):
            continue
        source_name = Path(source).name
        dest_name = f"{dest_dir}/{source_name}" if dest_dir else source_name
        cleaned_binaries.append((dest_name, source, "BINARY"))
    a.binaries += cleaned_binaries

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
