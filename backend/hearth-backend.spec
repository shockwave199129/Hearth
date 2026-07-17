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
# THIN BUILD: this freezes requirements-common.txt only — no torch,
# onnxruntime, parler-tts, or ttstokenizer. Neither hardware tier's TTS
# stack is installed at freeze time anymore; CI has no GPU to match either
# one against, so that decision now happens on the user's own machine at
# first run instead (see backend/app/setup/ and its /api/setup/*
# endpoints). This spec used to probe which of the two was installed and
# fail if neither/both were — removed entirely, since neither ever is now.
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
    # dest "." places these at the root of the frozen bundle's data dir —
    # BACKEND_DIR (app/config.py) resolves to exactly that root at runtime
    # in both dev and frozen modes, so app/setup/orchestrator.py's
    # `BACKEND_DIR / "requirements-gpu.txt"` needs no frozen/dev branching,
    # unlike the setup-python Tauri resource (a sibling of this whole
    # "backend" bundle, not something inside it — see
    # app/setup/installer.py's _setup_python_archive_dir()).
    (str(BACKEND_DIR / "requirements-gpu.txt"), "."),
    (str(BACKEND_DIR / "requirements-cpu.txt"), "."),
]

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
    # Stdlib modules not referenced by the thin freeze's import graph, but
    # pulled in at first-run after parler-tts / descript-audiotools land in
    # backend-deps (audiotools -> ipython -> timeit / pickletools / …).
    # Without these, Pipeline() dies with "No module named '…'" after setup
    # packages succeed (seen in the field for timeit, then pickletools).
    "timeit",
    "pydoc",
    "doctest",
    "pickletools",
    "code",
    "codeop",
    "pdb",
    "bdb",
    "cmd",
    "profile",
    "cProfile",
    "pstats",
    "rlcompleter",
    "dis",
    "opcode",
]

# Packages with native extensions / plugin-style dynamic imports that
# PyInstaller's default analysis reliably misses — collect_all() pulls in
# their submodules, data files, and bundled shared libraries together.
# No torch/parler_tts/ttstokenizer here anymore — see the THIN BUILD note
# above. onnxruntime stays: it's chromadb's own transitive dependency
# (verified via chromadb's PyPI metadata: onnxruntime>=1.14.1, for its
# default embedding function), not moonshine_voice's — moonshine-voice's
# own metadata declares no onnxruntime dependency at all, so it's always
# installed via requirements-common.txt's chromadb regardless of tier.
_COLLECT_ALL_PACKAGES = [
    "chromadb",
    "onnxruntime",
    "moonshine_voice",  # PyPI: moonshine-voice — see app/stt/moonshine_engine.py
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langgraph",
]

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
