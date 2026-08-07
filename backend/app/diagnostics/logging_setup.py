"""Root logging config: stderr plus a rotating file under USER_DATA_DIR.

The desktop shell spawns hearth-backend with CREATE_NO_WINDOW and never
redirects its stdio (desktop/src-tauri/src/main.rs), so on Windows every
line this process logs is written to a console nobody can see and then
lost. crash_log.py is not a substitute: it captures *unhandled* exceptions,
and the failures that actually strand a user are the handled ones — a TTS
error that aborts a turn, a model file that won't load — which log a
traceback and carry on. Those need to survive to a file we can ask for.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import LOGS_DIR

LOG_FILE_NAME = "hearth-backend.log"
_MAX_BYTES = 2 * 1024 * 1024
_BACKUP_COUNT = 3
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

_configured = False


def log_file_path() -> Path:
    return LOGS_DIR / LOG_FILE_NAME


def configure_logging(level: int = logging.INFO) -> None:
    """Install handlers on the root logger. Idempotent — safe to call from
    both the server entrypoint and a test/CLI path.

    Replaces logging.basicConfig(), which only ever reached stderr. Pass
    ``log_config=None`` to uvicorn.run so its own loggers propagate here
    instead of being redirected to handlers of their own.
    """
    global _configured
    if _configured:
        return
    _configured = True

    formatter = logging.Formatter(_FORMAT)
    root = logging.getLogger()
    root.setLevel(level)

    # PyInstaller gives a windowed exe None for stderr; the spec keeps this
    # one console-based precisely so that doesn't happen, but a caller that
    # rebuilds it differently should degrade to file-only, not crash on the
    # first log record.
    if sys.stderr is not None:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file_path(),
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
    except OSError:
        # A read-only install tree must never stop the server from booting.
        root.warning("could not open %s — file logging disabled", log_file_path())
        return

    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.info("hearth %s — logging to %s", sys.version.split()[0], log_file_path())
