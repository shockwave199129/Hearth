"""Computing Attachment Signals (Book Vol 7 Ch 8) — the missing middle
between Volume 3 Ch 4 (which named the warning signals qualitatively) and
Volume 6 Ch 8 (which defined how a flagged signal is acted on). Three
independent streams, each scored differently since they're qualitatively
different kinds of evidence, combined into one score that finally feeds
Phase 4's escalation (`app.safety2.worker`).

No LLM anywhere in this module (Vol 7 Design Goal 3) — reuses the same
rule-based patterns Volume 3's `evaluate_attachment_signals` already
defines for replacement language / unavailability distress, plus a genuine
statistical trend query for contact urgency."""
from __future__ import annotations

from datetime import datetime

from app.relationship.state import _REPLACEMENT_LANGUAGE, _UNAVAILABILITY_DISTRESS

# Higher than Communication Traits/Skill Affinity's alpha (Ch 4/Ch 8) — a
# genuine, escalating dependency pattern shouldn't take dozens of
# conversations to register, since Volume 6's response is already gradual;
# slow detection stacked on a slow response would let real drift go
# unaddressed too long.
ATTACHMENT_ALPHA = 0.25

CONTACT_URGENCY_WINDOW = 5


def contact_urgency_trend(session_timestamps: list[datetime], *, window: int = CONTACT_URGENCY_WINDOW) -> float:
    """0..1: how much the gap between sessions has shrunk over a sustained
    recent window compared to the window before it. This is the one
    legitimate use of frequency-adjacent data in this volume (Ch 8) — it's
    the *trend*, not the absolute frequency, and it's only ever combined
    with the other two streams below, never read alone."""
    ordered = sorted(session_timestamps)
    gaps = [(b - a).total_seconds() for a, b in zip(ordered, ordered[1:])]
    if len(gaps) < window * 2:
        return 0.0
    recent = gaps[-window:]
    prior = gaps[-2 * window : -window]
    recent_avg = sum(recent) / len(recent)
    prior_avg = sum(prior) / len(prior)
    if prior_avg <= 0:
        return 0.0
    shrink_ratio = (prior_avg - recent_avg) / prior_avg
    return max(0.0, min(1.0, shrink_ratio))


def replacement_language_score(user_messages: list[str]) -> float:
    if not user_messages:
        return 0.0
    hits = sum(1 for m in user_messages if _REPLACEMENT_LANGUAGE.search(m))
    return max(0.0, min(1.0, hits / len(user_messages) * 3))


def unavailability_distress_score(user_messages: list[str]) -> float:
    if not user_messages:
        return 0.0
    hits = sum(1 for m in user_messages if _UNAVAILABILITY_DISTRESS.search(m))
    return max(0.0, min(1.0, hits / len(user_messages) * 3))


def compute_combined_attachment_score(
    *,
    current_score: float,
    contact_urgency: float,
    replacement_language: float,
    unavailability_distress: float,
    alpha: float = ATTACHMENT_ALPHA,
) -> float:
    """Folds the strongest of the three streams into the cached score via a
    higher-alpha moving average — never optimized to increase (Vol 3 Ch 4/
    Vol 7 Ch 8's guardrail); this exists purely as a check."""
    latest = max(contact_urgency, replacement_language, unavailability_distress)
    return round(alpha * latest + (1 - alpha) * current_score, 4)
