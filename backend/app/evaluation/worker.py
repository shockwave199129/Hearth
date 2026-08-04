"""The Evaluation Worker (Book Vol 8 Ch 3) — background, non-blocking,
structurally distinct from the Growth Engine (Vol 1 Ch 12): the Growth
Engine asks "what should Hearth remember and how should the relationship
update"; this worker asks "did this conversation actually meet the
standards this book defines" — a quality-assurance question, never mixed
with the first.

Scores three categories (Ch 4): Invariant Adherence (`app.evaluation.
invariants`), Anti-Pattern Detection (reuses `app.eval.self_check`'s
already-built checker), and Success-Metric Proxies — the two proxies this
book honestly can't automate (felt understanding, reduction in expressed
isolation) are never forced into a fake score; they're flagged for the
human-review queue (Ch 6) instead.

Writes everything into the observation store as its own `evaluation`
observation type (Ch 5) — a fifth type alongside memory/skill/relationship/
communication — and additionally, immediately, dual-writes safety-relevant
findings into Volume 6's separate log, never waiting for this worker's own
normal cadence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.eval.self_check import flag_reply
from app.evaluation.invariants import InvariantCheckResult, run_invariant_checks
from app.learning.observation_store import ObservationStore
from app.safety2.audit import SafetyAuditEntry, record as record_safety

# Book Vol 6 Ch12: safety-relevant logs get a defined, bounded retention
# window, exempt from Volume 4's general deletion rights only within that
# window — never indefinitely, and never accidentally zero. Matches
# SafetyWorker.log's own default so there is one retention policy for
# safety-relevant entries, not two silently-diverging ones.
SAFETY_FINDINGS_RETENTION_DAYS = 30

# Ch 6: automated results in this confidence band are sampled into the
# human-review queue, weighted toward borderline/uncertain results rather
# than a uniform random sample.
BORDERLINE_LOW = 0.3
BORDERLINE_HIGH = 0.8

# Metrics this book honestly acknowledges resist reliable automated
# proxying (Ch 4) — never forced into a low-quality automated score.
HUMAN_REVIEW_ONLY_METRICS = ("felt_understanding", "reduction_in_expressed_isolation")


@dataclass(frozen=True)
class EvaluationResult:
    invariant_results: list[InvariantCheckResult]
    invariant_score: float
    anti_pattern_reason: str | None
    anti_pattern_score: float
    success_proxies: dict[str, float | None]
    human_review_flagged: bool
    notes: str
    timestamp: datetime


class EvaluationWorker:
    def __init__(self, store: ObservationStore | None = None):
        self.store = store or ObservationStore()

    def evaluate(
        self,
        user_id: str,
        transcript: str,
        reply_text: str,
        safety_findings: dict | None = None,
        *,
        recent_assistant_messages: list[str] | None = None,
        skill_id: str | None = None,
        composed_with: str | None = None,
        is_safety_response: bool = False,
        trust_snapshot: dict[str, float] | None = None,
        memory_referenced: bool | None = None,
    ) -> EvaluationResult:
        ctx = {
            "transcript": transcript,
            "reply_text": reply_text,
            "skill_id": skill_id,
            "composed_with": composed_with,
            "is_safety_response": is_safety_response,
        }

        # 1. Invariant Adherence (Ch 4.1) — score over only the
        # automatically-checkable ones; log-based invariants are reported,
        # never silently counted as passing.
        invariant_results = run_invariant_checks(ctx)
        checkable = [r for r in invariant_results if r.result != "not_automatically_checkable"]
        invariant_score = (sum(1 for r in checkable if r.result == "pass") / len(checkable)) if checkable else 1.0

        # 2. Anti-Pattern Detection (Ch 4.2) — reuses Phase 1's checker
        # directly rather than re-deriving Volume 2's named list here.
        anti_pattern_reason = flag_reply(reply_text, recent_assistant_messages=recent_assistant_messages)
        anti_pattern_score = 0.5 if anti_pattern_reason else 1.0

        # 3. Success-Metric Proxies (Ch 4.3) — honest approximations, with
        # the two genuinely unmeasurable ones routed to human review
        # instead of a fabricated automated score.
        success_proxies: dict[str, float | None] = {}
        if trust_snapshot:
            success_proxies["trust_consistency"] = round(sum(trust_snapshot.values()) / len(trust_snapshot), 4)
        else:
            success_proxies["trust_consistency"] = None
        success_proxies["memory_quality"] = float(memory_referenced) if memory_referenced is not None else None
        for metric in HUMAN_REVIEW_ONLY_METRICS:
            success_proxies[metric] = None

        # Ch 6: borderline automated confidence -> sample into human review,
        # plus the two metrics this book never even attempts to automate.
        borderline = BORDERLINE_LOW < invariant_score < BORDERLINE_HIGH
        human_review_flagged = borderline or any(v is None for k, v in success_proxies.items() if k in HUMAN_REVIEW_ONLY_METRICS)

        failed_rules = [r.rule for r in invariant_results if r.result == "fail"]
        notes = "; ".join(failed_rules) if failed_rules else (anti_pattern_reason or "no issues detected")
        now = datetime.now(timezone.utc)

        self._write_observations(
            invariant_results=invariant_results,
            invariant_score=invariant_score,
            anti_pattern_reason=anti_pattern_reason,
            anti_pattern_score=anti_pattern_score,
            success_proxies=success_proxies,
            human_review_flagged=human_review_flagged,
            reply_text=reply_text,
            now=now,
        )

        if safety_findings:
            self._dual_write_safety_findings(user_id, safety_findings, notes, now)

        return EvaluationResult(
            invariant_results=invariant_results,
            invariant_score=round(invariant_score, 4),
            anti_pattern_reason=anti_pattern_reason,
            anti_pattern_score=anti_pattern_score,
            success_proxies=success_proxies,
            human_review_flagged=human_review_flagged,
            notes=notes,
            timestamp=now,
        )

    def _write_observations(
        self,
        *,
        invariant_results: list[InvariantCheckResult],
        invariant_score: float,
        anti_pattern_reason: str | None,
        anti_pattern_score: float,
        success_proxies: dict[str, float | None],
        human_review_flagged: bool,
        reply_text: str,
        now: datetime,
    ) -> None:
        """Book Vol 8 Ch 5: evaluation results are themselves observations,
        written as their own `evaluation` type (not folded into
        `communication`) so they become real learning input for Volume 7's
        analytics rather than a standalone report nobody acts on."""
        self.store.append(
            "evaluation", "invariant_adherence", invariant_score,
            {"results": [r.__dict__ for r in invariant_results]}, "evaluation_worker",
        )
        self.store.append(
            "evaluation", "anti_pattern", anti_pattern_score,
            {"reason": anti_pattern_reason, "reply": reply_text}, "evaluation_worker",
        )
        for metric, value in success_proxies.items():
            if value is not None:
                self.store.append("evaluation", metric, value, {}, "evaluation_worker")
        if human_review_flagged:
            self.store.append(
                "evaluation", "human_review_flag", 1.0,
                {"reason": "borderline_or_unmeasurable"}, "evaluation_worker",
            )

    def _dual_write_safety_findings(self, user_id: str, safety_findings: dict, notes: str, now: datetime) -> None:
        record_safety(
            SafetyAuditEntry(
                user_id=user_id,
                timestamp=now.isoformat(),
                category=safety_findings.get("category", "evaluation"),
                confidence_signals=safety_findings,
                response_taken="evaluation",
                outcome_notes=notes,
                retention_expiry=(now + timedelta(days=SAFETY_FINDINGS_RETENTION_DAYS)).isoformat(),
            )
        )
