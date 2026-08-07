"""Opt-in crash reporting — see ``crash_log`` and ``docs/privacy.md``."""

from app.diagnostics.crash_log import (
    dismiss_report,
    install_crash_handlers,
    list_pending,
    record_crash,
    send_report,
)

__all__ = [
    "dismiss_report",
    "install_crash_handlers",
    "list_pending",
    "record_crash",
    "send_report",
]
