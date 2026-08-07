"""Local crash-log buffer + opt-in upload to the public crash-logs prefix.

Nothing leaves the device until the user says yes in the UI. Reports hold
only diagnostic fields (stack, OS, app version, tier) — never transcript,
memory, or profile text. See docs/privacy.md.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from app.config import APP_VERSION, CRASH_LOGS_BUCKET_URL, CRASH_LOGS_DIR

log = logging.getLogger("hearth")

_MAX_STACK_CHARS = 32_000
_MAX_MESSAGE_CHARS = 2_000
_handlers_installed = False
_handlers_lock = threading.Lock()


def _pending_dir() -> Path:
    path = CRASH_LOGS_DIR / "pending"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n…[truncated]"


def _base_metadata() -> dict[str, Any]:
    return {
        "app_version": APP_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "frozen": bool(getattr(sys, "frozen", False)),
    }


def record_crash(
    *,
    source: str,
    message: str,
    stack: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a pending crash report to disk. Safe to call from any thread."""
    report_id = uuid.uuid4().hex
    payload: dict[str, Any] = {
        "id": report_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "message": _truncate(str(message), _MAX_MESSAGE_CHARS),
        "stack": _truncate(stack or "", _MAX_STACK_CHARS),
        **_base_metadata(),
    }
    if extra:
        # Only allow a small allow-listed surface so callers can't
        # accidentally ship journal contents into an uploadable file.
        for key in ("component", "tier", "path"):
            if key in extra and extra[key] is not None:
                payload[key] = _truncate(str(extra[key]), 200)

    path = _pending_dir() / f"{report_id}.json"
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    log.warning("crash report buffered locally id=%s source=%s", report_id, source)
    return {"id": report_id, "created_at": payload["created_at"], "source": source, "message": payload["message"]}


def list_pending() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pending = _pending_dir()
    for path in sorted(pending.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append(
            {
                "id": data.get("id", path.stem),
                "created_at": data.get("created_at"),
                "source": data.get("source"),
                "message": data.get("message", ""),
            }
        )
    return out


def _load_pending(report_id: str) -> tuple[Path, dict[str, Any]] | None:
    path = _pending_dir() / f"{report_id}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return path, data


def dismiss_report(report_id: str) -> bool:
    """User chose not to send — delete the local pending file."""
    path = _pending_dir() / f"{report_id}.json"
    if not path.is_file():
        return False
    path.unlink(missing_ok=True)
    return True


def internet_reachable(timeout: float = 3.0) -> bool:
    """Cheap reachability probe against the crash-log host (HTTPS only).

    An HTTP error from the host still counts as online — many S3 buckets
    reject anonymous HEAD with 403 while accepting PUT on a prefix. DNS /
    TLS / timeout failures mean offline.
    """
    if not CRASH_LOGS_BUCKET_URL:
        return False
    parsed = urlparse(CRASH_LOGS_BUCKET_URL)
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    probe = f"https://{parsed.netloc}/"
    try:
        req = Request(probe, method="HEAD")
        with urlopen(req, timeout=timeout) as resp:
            resp.read(0)
        return True
    except HTTPError:
        return True
    except (URLError, TimeoutError, OSError):
        return False


def send_report(report_id: str) -> dict[str, Any]:
    """Upload a pending report to ``CRASH_LOGS_BUCKET_URL`` via HTTPS PUT.

    Requires a reachable network and a bucket policy that allows PutObject
    on the crash-logs prefix (anonymous or otherwise reachable without
    shipping AWS credentials in the client).
    """
    if not CRASH_LOGS_BUCKET_URL:
        raise RuntimeError("CRASH_LOGS_BUCKET_URL is not configured")
    if urlparse(CRASH_LOGS_BUCKET_URL).scheme != "https":
        raise RuntimeError("refusing non-HTTPS crash log upload URL")

    loaded = _load_pending(report_id)
    if loaded is None:
        raise FileNotFoundError(f"no pending crash report {report_id}")
    path, data = loaded

    if not internet_reachable():
        raise ConnectionError("no internet connection — crash report was not sent")

    stamp = _utc_stamp()
    filename = f"{APP_VERSION}_{stamp}_{report_id}.json"
    url = f"{CRASH_LOGS_BUCKET_URL.rstrip('/')}/{filename}"
    body = json.dumps(data, indent=2).encode("utf-8")
    req = Request(
        url,
        data=body,
        method="PUT",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body)),
        },
    )
    try:
        with urlopen(req, timeout=30) as resp:
            status = getattr(resp, "status", 200)
            if status >= 400:
                raise RuntimeError(f"upload rejected with HTTP {status}")
    except HTTPError as exc:
        raise RuntimeError(f"upload failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise ConnectionError(f"upload failed: {exc.reason}") from exc

    path.unlink(missing_ok=True)
    log.info("crash report uploaded id=%s url=%s", report_id, url)
    return {"id": report_id, "uploaded_as": filename, "url": url}


def _format_exception(exc_type, exc, tb) -> tuple[str, str]:
    message = f"{getattr(exc_type, '__name__', exc_type)}: {exc}"
    stack = "".join(traceback.format_exception(exc_type, exc, tb))
    return message, stack


def _excepthook(exc_type, exc, tb) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    message, stack = _format_exception(exc_type, exc, tb)
    try:
        record_crash(source="python", message=message, stack=stack)
    except Exception:
        log.exception("failed to buffer crash report")
    sys.__excepthook__(exc_type, exc, tb)


def _threading_excepthook(args) -> None:
    # threading.ExceptHookArgs: exc_type, exc_value, exc_traceback, thread
    if args.exc_type and issubclass(args.exc_type, SystemExit):
        return
    message, stack = _format_exception(args.exc_type, args.exc_value, args.exc_traceback)
    try:
        record_crash(
            source="python-thread",
            message=message,
            stack=stack,
            extra={"component": getattr(args.thread, "name", None)},
        )
    except Exception:
        log.exception("failed to buffer thread crash report")


def install_crash_handlers() -> None:
    """Install process-wide exception hooks. Idempotent."""
    global _handlers_installed
    with _handlers_lock:
        if _handlers_installed:
            return
        sys.excepthook = _excepthook
        if hasattr(threading, "excepthook"):
            threading.excepthook = _threading_excepthook
        _handlers_installed = True
        log.debug("crash report handlers installed (dir=%s)", CRASH_LOGS_DIR)
