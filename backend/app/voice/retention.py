"""Retention and destruction schedule for voiceprints — docs/compliance.md §6.

BIPA-style statutes require a *published* schedule and adherence to it. The
published version is the "Your voiceprint" section of docs/privacy.md and the
in-app consent text; this is the enforcement.

The rule, in the order the statute frames it — destroy at the earlier of:

1. **Purpose satisfied.** Deleting a voiceprint, or the profile it belongs to,
   destroys it immediately (`api/voice.py`, and the cascade in
   `api/profile.py`). Turning the feature off is the user's own action, so
   there is nothing to schedule.
2. **Three years from the last interaction** (`VOICEPRINT_RETENTION_DAYS`).
   That is this module.

## What a local app can honestly promise

There is no daemon. Nothing runs while Hearth is closed, so destruction
happens on the next profile activation rather than on a timer. If someone
stops using Hearth for four years, their voiceprint sits encrypted on their
own disk until they next open the app, at which point it is destroyed before
any turn can use it.

This is a real limitation and it should be described accurately rather than
papered over — but note what it is not: the data never left the device, so
"deleted late, locally, on a machine only that user has" is a materially
different exposure from a retained server-side template. Do not "fix" it with
a background service; a companion app that runs when you did not ask it to is
a worse trade, and would undercut the no-engagement-optimization invariant.

The one gap worth naming: enforcement keys off conversation activity, so a
profile that is activated but never spoken to keeps deferring its own
deadline. `last_activity_at` returning None (never any conversation) falls
back to when the voiceprint was enrolled, so a voiceprint made and then
abandoned still expires.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.config import VOICEPRINT_RETENTION_DAYS
from app.memory import chat_history
from app.voice import consent, store

logger = logging.getLogger("hearth.speaker")


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def expiry_for(
    user_id: str, *, last_activity_at: datetime | None = None
) -> datetime | None:
    """When this profile's voiceprint is due for destruction, or None if there
    is nothing stored."""
    voiceprint = store.get(user_id)
    if voiceprint is None:
        return None
    # Enrollment time is the floor: a voiceprint created and then never used
    # must still expire, rather than waiting for activity that never comes.
    reference = last_activity_at or _parse(voiceprint.updated_at) or _parse(voiceprint.enrolled_at)
    if reference is None:  # pragma: no cover - both timestamps are NOT NULL
        return None
    return reference + timedelta(days=VOICEPRINT_RETENTION_DAYS)


def enforce(user_id: str, *, now: datetime | None = None) -> bool:
    """Destroy the voiceprint if the retention period has elapsed.

    Returns True when something was destroyed. Consent is revoked alongside
    it: permission to hold a template does not outlive the template, and
    re-enrolling later should ask again.

    Never raises. This runs on profile activation, and a retention sweep that
    could prevent the app from opening would be a worse failure than a late
    deletion.
    """
    moment = now or datetime.now(timezone.utc)
    try:
        last_activity = chat_history.last_activity_at(user_id)
        due = expiry_for(user_id, last_activity_at=last_activity)
        if due is None or moment < due:
            return False
        store.delete(user_id)
        consent.revoke(user_id)
        logger.info(
            "destroyed voiceprint for %s under the %s-day retention schedule (due %s)",
            user_id, VOICEPRINT_RETENTION_DAYS, due.isoformat(),
        )
        return True
    except Exception:
        logger.exception("voiceprint retention sweep failed for %s", user_id)
        return False
