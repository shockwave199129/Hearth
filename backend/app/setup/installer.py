"""Orchestrates the actual hardware-matched package install: extracts the
bundled standalone Python (once), runs its pip to install into
BACKEND_DEPS_DIR, and reports progress for /api/setup/* to poll.

Uses a real separate Python interpreter for this rather than trying to
invoke pip from within the already-frozen PyInstaller process — PyInstaller
apps don't reliably support installing new packages into themselves at
runtime (pip has many dynamic import patterns static analysis can miss),
while a genuinely separate interpreter with real pip sidesteps that
entirely. See fetch_setup_python.py for where it comes from.
"""
import platform
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

from app.config import BACKEND_DEPS_DIR, BACKEND_DIR, extend_backend_deps_path


def _setup_python_archive_dir() -> Path:
    """Where fetch_setup_python.py's downloaded archive lives — NOT
    computable as a fixed constant, since it differs between dev and the
    actual installed app:

    - Frozen (sys.frozen, set by PyInstaller): "setup-python" is a Tauri
      resource *sibling* of "backend" (see tauri.conf.json's
      bundle.resources), not something inside the repo tree at all —
      resource_root/backend/hearth-backend[.exe] is sys.executable, so
      resource_root is two parents up.
    - Dev (running straight from the repo checkout): no such install
      layout exists yet, so this falls back to the same repo-relative
      path scripts/fetch_setup_python.py itself downloads into.
    """
    if getattr(sys, "frozen", False):
        resource_root = Path(sys.executable).resolve().parent.parent
        return resource_root / "setup-python"
    return BACKEND_DIR.parent / "desktop" / "src-tauri" / "resources" / "setup-python"


SETUP_PYTHON_EXTRACT_DIR = BACKEND_DIR / "setup-python"


class SetupError(RuntimeError):
    pass


def _setup_python_bin() -> Path:
    """Extracts the bundled python-build-standalone archive on first use
    (idempotent — skips if already extracted), returns the path to its
    python executable."""
    if not SETUP_PYTHON_EXTRACT_DIR.exists():
        archive_dir = _setup_python_archive_dir()
        archives = list(archive_dir.glob("*.tar.gz"))
        if not archives:
            raise SetupError(f"no bundled setup-python archive found in {archive_dir}")
        SETUP_PYTHON_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archives[0]) as tf:
            tf.extractall(SETUP_PYTHON_EXTRACT_DIR)  # noqa: S202 — our own bundled, trusted archive

    # Exact known paths, not a glob — python-build-standalone's
    # `install_only` archives also contain a *second*, smaller stub copy
    # under Lib/venv/scripts/nt/python.exe (Windows) purely for venv
    # relocation, which an rglob("python.exe") would also match with no
    # guaranteed enumeration order, risking silently invoking the wrong
    # one. Confirmed both real paths below by inspecting each platform's
    # actual archive contents (`tar -tzf`), not assumed from docs alone.
    binary = SETUP_PYTHON_EXTRACT_DIR / "python" / (
        "python.exe" if platform.system() == "Windows" else "bin/python3"
    )
    if not binary.exists():
        raise SetupError(f"extracted {SETUP_PYTHON_EXTRACT_DIR} but expected binary {binary} is missing")
    return binary


class InstallProgress:
    """Shared, thread-safe progress state /api/setup/* polls. One instance
    per process — a fresh setup attempt resets it."""

    def __init__(self):
        self._lock = threading.Lock()
        self.step = "idle"  # idle | detecting | installing_packages | downloading_models | done | error
        self.log_tail: list[str] = []
        self.error: str | None = None

    def set_step(self, step: str) -> None:
        with self._lock:
            self.step = step

    def append_log(self, line: str) -> None:
        with self._lock:
            self.log_tail.append(line)
            self.log_tail = self.log_tail[-200:]

    def set_error(self, message: str) -> None:
        with self._lock:
            self.step = "error"
            self.error = message

    def snapshot(self) -> dict:
        with self._lock:
            return {"step": self.step, "log_tail": list(self.log_tail), "error": self.error}


def install_packages(packages: list[str], index_url: str | None, progress: InstallProgress) -> None:
    """Runs `<bundled-python> -m pip install --target BACKEND_DEPS_DIR
    <packages>` via subprocess, streaming output into `progress`. Re-running
    this (e.g. after a partial failure) is safe — pip --target skips
    already-satisfied packages on its own, same idempotency scripts/setup.py
    already relies on for model downloads."""
    python_bin = _setup_python_bin()
    BACKEND_DEPS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [str(python_bin), "-m", "pip", "install", "--target", str(BACKEND_DEPS_DIR)]
    if index_url:
        cmd += ["--index-url", index_url]
    cmd += packages

    progress.append_log(f"running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:  # type: ignore[union-attr]
        progress.append_log(line.rstrip())
    returncode = proc.wait()
    if returncode != 0:
        raise SetupError(f"pip install exited with code {returncode} — see log_tail for details")

    # Makes BACKEND_DEPS_DIR importable in the current process right away —
    # no restart needed: a not-yet-attempted import isn't negatively cached
    # anywhere in the Python import system, so this works even though
    # main.py's earlier `assert _pipeline is not None` calls already ran
    # and found it None. config.py's own module-level call only covers a
    # *second* launch (after a previous run already finished setup); this
    # covers the first one, within the same already-running process.
    extend_backend_deps_path()
