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
import sys

from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Stdlib names that are never useful in a headless server freeze (GUI /
# demos / the test suite). Everything else from sys.stdlib_module_names —
# plus package submodules (xml.dom, encodings.*, …) — is force-included
# below. sys.stdlib_module_names alone is only top-level names, which is
# why v0.2.9 still missed xml.dom after bundling "xml".
_STDLIB_EXCLUDE = {
    "antigravity",
    "this",
    "turtle",
    "turtledemo",
    "tkinter",
    "_tkinter",
    "idlelib",
    "test",
    "lib2to3",
    "ensurepip",
    "venv",
    "pydoc_data",
    "curses",
}


def _stdlib_hiddenimports() -> list[str]:
    names: set[str] = set()
    for top in sorted(sys.stdlib_module_names - _STDLIB_EXCLUDE):
        names.update(collect_submodules(top))
    # Drop anything under an excluded top-level (collect_submodules can
    # still surface odd edges on some platforms).
    return sorted(
        n for n in names if n.split(".", 1)[0] not in _STDLIB_EXCLUDE
    )

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

# Baked by scripts/build_backend.* from the v* git tag / HEARTH_APP_VERSION
# so crash reports and status can name the release without a .git tree.
_version_file = APP_DIR / "VERSION"
if _version_file.is_file():
    datas.append((str(_version_file), "app"))

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
    # Post-setup packages land in backend-deps (transformers, ipython via
    # audiotools, …) and import stdlib the thin Analysis graph never sees
    # (timeit, pickletools, filecmp, xml.dom, …). Top-level names alone are
    # not enough — collect_submodules pulls xml.dom, encodings.*, etc.
    # Missing names on a given OS just warn at freeze time.
    *_stdlib_hiddenimports(),
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

# This loop MUST run before Analysis(), not after it: Analysis() consumes
# `datas`/`hiddenimports` by value in its constructor and builds the import
# graph there and then, so appending to those lists afterwards is silently
# ignored. A previous version of this spec collected after Analysis() and
# only the `a.binaries` mutation took effect (that one edits the already-built
# TOC), which shipped chromadb's native libraries but none of the submodules
# it imports dynamically: the installed app died the first time anything
# touched memory with "ModuleNotFoundError: No module named
# 'chromadb.telemetry.product.posthog'" (then 'chromadb.api.rust'). Both are
# resolved through importlib by chromadb's own config.get_class() at runtime,
# so only collect_all()'s hiddenimports can put them in the bundle.
#
# Binaries are kept in collect_dynamic_libs()' hook order — (source_path,
# dest_DIR), where dest_DIR is a directory that several files routinely share
# (onnxruntime's capi/ has multiple .so/.dylib siblings) — and handed to
# Analysis(binaries=...), whose format_binaries_and_datas() joins dest_DIR
# with each source's basename and reverses the pair into TOC order. Doing
# that by hand against `a.binaries` is what the post-Analysis version had to
# do, and both halves of it have already failed here once: hook-order tuples
# left as-is made COLLECT treat the dest fragment as a source path and drop
# every binary ("Ignoring non-existent resource torch/lib, meant to be
# collected as .../torch/lib/libc10.dylib"), and using dest_DIR alone as the
# dest name made every sibling library in one directory collide — silently
# overwriting each other on Linux, failing hard on macOS with "there already
# exists a file at that path".
binaries = []
for _pkg in _COLLECT_ALL_PACKAGES:
    _datas, _binaries, _hiddenimports = collect_all(_pkg)
    datas += _datas
    hiddenimports += _hiddenimports
    # Filter out libmoonshine.so from binaries collected by collect_all() —
    # macOS only. moonshine_voice ships this Linux ELF alongside macOS
    # wheels' own native .dylib, and PyInstaller's macOS analysis fails
    # trying to parse it. Linux/Windows builds need this exact file — it's
    # the native STT lib app/stt/moonshine_engine.py loads at runtime, so
    # dropping it unconditionally (as a prior refactor did, losing the
    # platform guard this originally had) ships a Linux/Windows app with no
    # STT engine, failing on first launch with "Failed to load dynlib/dll
    # 'libmoonshine.so' ... Most likely this dynlib/dll was not found when
    # the application was frozen."
    for source, dest_dir in _binaries:
        if sys.platform == "darwin" and source.endswith("libmoonshine.so"):
            continue
        binaries.append((source, dest_dir))

# moonshine-voice's manylinux wheel vendors its onnxruntime shared lib in a
# sibling auditwheel dir (moonshine_voice.libs/), not inside the package.
# collect_all/collect_dynamic_libs only scan the package tree, so they miss
# it. libmoonshine.so's RPATH is $ORIGIN:$ORIGIN/../moonshine_voice.libs —
# without this folder the frozen Linux app fails at launch with
# "Failed to load dynlib/dll '.../libmoonshine.so'" even though that .so
# itself was bundled (its NEEDED dep libonnxruntime-*.so.1 is missing).
if sys.platform.startswith("linux"):
    import moonshine_voice as _moonshine_voice

    _moonshine_libs = (
        Path(_moonshine_voice.__file__).resolve().parent.parent / "moonshine_voice.libs"
    )
    if _moonshine_libs.is_dir():
        for _lib in sorted(_moonshine_libs.iterdir()):
            if _lib.is_file() and ".so" in _lib.name:
                binaries.append((str(_lib), "moonshine_voice.libs"))
    else:
        raise SystemExit(
            f"moonshine_voice.libs not found at {_moonshine_libs} — "
            "Linux freeze needs this auditwheel dir next to moonshine_voice "
            "(libonnxruntime-*.so.1). Reinstall moonshine-voice from the "
            "manylinux wheel."
        )

a = Analysis(
    [str(APP_DIR / "main.py")],
    pathex=[str(BACKEND_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

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
