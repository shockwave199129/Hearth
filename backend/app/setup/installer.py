"""Orchestrates the actual hardware-matched package install: extracts the
bundled standalone Python (once), bootstraps `uv` into that interpreter,
runs `python -m uv pip install --target BACKEND_DEPS_DIR`, and reports
progress for /api/setup/* to poll.

Uses a real separate Python interpreter for this rather than trying to
invoke pip/uv from within the already-frozen PyInstaller process —
PyInstaller apps don't reliably support installing new packages into
themselves at runtime, while a genuinely separate interpreter sidesteps
that entirely. See fetch_setup_python.py for where it comes from.
"""

import platform
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

from app.config import (
    BACKEND_DEPS_DIR,
    BACKEND_DIR,
    SETUP_PYTHON_EXTRACT_DIR,
    extend_backend_deps_path,
)


def _windows_no_window_kwargs() -> dict:
    if platform.system() != "Windows":
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}


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


class SetupError(RuntimeError):
    pass


def _expected_setup_python_binary() -> Path:
    # Exact known paths, not a glob — python-build-standalone's
    # `install_only` archives also contain a *second*, smaller stub copy
    # under Lib/venv/scripts/nt/python.exe (Windows) purely for venv
    # relocation, which an rglob("python.exe") would also match with no
    # guaranteed enumeration order, risking silently invoking the wrong
    # one. Confirmed both real paths below by inspecting each platform's
    # actual archive contents (`tar -tzf`), not assumed from docs alone.
    return (
        SETUP_PYTHON_EXTRACT_DIR
        / "python"
        / ("python.exe" if platform.system() == "Windows" else "bin/python3")
    )


def _extract_setup_python() -> None:
    archive_dir = _setup_python_archive_dir()
    archives = list(archive_dir.glob("*.tar.gz"))
    if not archives:
        raise SetupError(f"no bundled setup-python archive found in {archive_dir}")
    SETUP_PYTHON_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archives[0]) as tf:
        tf.extractall(SETUP_PYTHON_EXTRACT_DIR)  # noqa: S202 — our own bundled, trusted archive


def _setup_python_bin() -> Path:
    """Extracts the bundled python-build-standalone archive on first use
    (idempotent — skips if already extracted), returns the path to its
    python executable. If a prior extract was interrupted (dir exists but
    binary missing), wipes and re-extracts."""
    binary = _expected_setup_python_binary()
    if binary.exists():
        return binary

    if SETUP_PYTHON_EXTRACT_DIR.exists():
        # Partial extract from a crashed previous attempt — not recoverable
        # by skipping extract, so start clean.
        shutil.rmtree(SETUP_PYTHON_EXTRACT_DIR)

    _extract_setup_python()
    if not binary.exists():
        raise SetupError(
            f"extracted {SETUP_PYTHON_EXTRACT_DIR} but expected binary {binary} is missing"
        )
    return binary


class InstallProgress:
    """Shared, thread-safe progress state /api/setup/* polls. One instance
    per process — a fresh setup attempt resets it via reset()."""

    def __init__(self):
        self._lock = threading.Lock()
        # idle | detecting | installing_packages | downloading_models |
        # starting_engines | done | error
        self.step = "idle"
        self.log_tail: list[str] = []
        self.error: str | None = None

    def reset(self) -> None:
        """Clears prior step/error/log so a Retry doesn't leave stale UI state."""
        with self._lock:
            self.step = "idle"
            self.log_tail = []
            self.error = None

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
            return {
                "step": self.step,
                "log_tail": list(self.log_tail),
                "error": self.error,
            }


def _run_logged(
    cmd: list[str], progress: InstallProgress, *, error_prefix: str
) -> None:
    progress.append_log(f"running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **_windows_no_window_kwargs(),
    )
    for line in proc.stdout:  # type: ignore[union-attr]
        progress.append_log(line.rstrip())
    returncode = proc.wait()
    if returncode != 0:
        raise SetupError(
            f"{error_prefix} (exit {returncode}) — see log_tail for details"
        )


def _ensure_uv(python_bin: Path, progress: InstallProgress) -> None:
    """Install `uv` into the bundled interpreter's own site-packages once.

    Not installed into BACKEND_DEPS_DIR — that tree is only for packages the
    frozen app imports at runtime. Re-check is cheap (`python -m uv --version`).
    """
    check = subprocess.run(
        [str(python_bin), "-m", "uv", "--version"],
        capture_output=True,
        text=True,
        **_windows_no_window_kwargs(),
    )
    if check.returncode == 0:
        version = (check.stdout or check.stderr or "").strip()
        progress.append_log(version or "uv already available")
        return

    _run_logged(
        [str(python_bin), "-m", "pip", "install", "--upgrade", "uv"],
        progress,
        error_prefix="bootstrapping uv into bundled Python failed",
    )


def install_packages(
    packages: list[str], index_url: str | None, progress: InstallProgress
) -> None:
    """Runs `<bundled-python> -m uv pip install --target BACKEND_DEPS_DIR
    <packages>` via subprocess, streaming output into `progress`. Re-running
    this (e.g. after a partial failure) is safe — uv/pip --target skips
    already-satisfied packages on its own, same idempotency scripts/setup.py
    already relies on for model downloads."""
    python_bin = _setup_python_bin()
    BACKEND_DEPS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_uv(python_bin, progress)

    cmd = [
        str(python_bin),
        "-m",
        "uv",
        "pip",
        "install",
        "--python",
        str(python_bin),
        "--target",
        str(BACKEND_DEPS_DIR),
    ]
    if index_url:
        cmd += ["--index-url", index_url]
    cmd += packages

    _run_logged(cmd, progress, error_prefix="uv pip install failed")

    # Makes BACKEND_DEPS_DIR importable in the current process right away —
    # no restart needed: a not-yet-attempted import isn't negatively cached
    # anywhere in the Python import system, so this works even though
    # main.py's earlier `assert _pipeline is not None` calls already ran
    # and found it None. config.py's own module-level call only covers a
    # *second* launch (after a previous run already finished setup); this
    # covers the first one, within the same already-running process.
    extend_backend_deps_path()
