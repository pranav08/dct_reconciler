"""
compliance/hitl.py — Human-in-the-Loop routing.

THE PRINCIPLE
==============
For records the deterministic validator AND the judge LLM both flag as
ambiguous, the answer isn't "let the LLM decide." It's "stop, route to a
human." A clinical data manager reviews the ReviewPacket (raw input vs
canonical output side-by-side), then either approves, rejects, or amends.

This is the conservative complement to constrained decoding: when the
machine is unsure, it does not guess. It defers.

WHEN HITL FIRES
================
Three triggers, in order of severity:

  1. Validator returns ERROR  → mandatory HITL
  2. Judge returns "human_review" or plausibility < threshold → mandatory HITL
  3. Validator returns WARN + judge unavailable → optional HITL

The HITLQueue persists these as JSONL alongside the audit log.
The reviewer's eventual decision is itself an audit event (SIGNED or AMENDED).
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class HITLReason(str, Enum):
    VALIDATOR_ERROR     = "validator_error"
    JUDGE_REQUESTED     = "judge_requested_human_review"
    JUDGE_LOW_CONFIDENCE = "judge_low_confidence"
    JUDGE_UNAVAILABLE   = "judge_unavailable"


class ReviewerAction(str, Enum):
    APPROVE = "approve"     # canonical output is correct as-is
    REJECT  = "reject"      # canonical output is wrong; drop the record
    AMEND   = "amend"       # canonical output needs correction (creates Amendment)


@dataclass
class HITLItem:
    """One record awaiting human review."""
    source_id:          str
    queued_at_utc:      str
    reason:             HITLReason
    review_packet_hash: str        # binds to the ReviewPacket via SHA-256
    priority:           int = 0    # 0 = normal, 1 = urgent (e.g. patient safety)
    assigned_to:        str | None = None
    metadata:           dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewerDecision:
    """A human reviewer's outcome — itself a Part 11 e-signed event."""
    source_id:          str
    reviewer_user_id:   str          # Attributable
    action:             ReviewerAction
    rationale:          str          # why this decision
    decided_at_utc:     str
    review_packet_hash: str          # binds back to what they reviewed
    signature:          str | None = None    # ESignature if action=APPROVE


# ── HITL queue (append-only, like the audit log) ─────────────────────────────

class HITLQueue:
    """
    Append-only JSONL queue of records awaiting human review.

    Keeps things deliberately simple — production deployments would back this
    with PostgreSQL or a workflow engine (Camunda, Temporal). The queue
    interface is the same regardless of backend.
    """

    def __init__(self, path: str | Path = "hitl_queue.jsonl") -> None:
        self.path = Path(path)

    def enqueue(
        self, *, source_id: str, reason: HITLReason,
        review_packet_hash: str, priority: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> HITLItem:
        item = HITLItem(
            source_id=source_id, reason=reason,
            review_packet_hash=review_packet_hash,
            queued_at_utc=datetime.utcnow().isoformat() + "Z",
            priority=priority, metadata=metadata or {},
        )
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(item), default=str) + "\n")
        return item

    def pending(self) -> list[HITLItem]:
        if not self.path.exists():
            return []
        items: list[HITLItem] = []
        with self.path.open() as f:
            for line in f:
                d = json.loads(line)
                d["reason"] = HITLReason(d["reason"])
                items.append(HITLItem(**d))
        # sort: urgent first, then oldest first
        items.sort(key=lambda x: (-x.priority, x.queued_at_utc))
        return items


def should_route_to_hitl(
    val_report, judge_report,
    plausibility_threshold: float = 0.4,
) -> HITLReason | None:
    """Pure function: deterministic HITL routing logic."""
    if val_report and val_report.error_count > 0:
        return HITLReason.VALIDATOR_ERROR
    if judge_report:
        if judge_report.suggested_action == "human_review":
            return HITLReason.JUDGE_REQUESTED
        if judge_report.plausibility_score < plausibility_threshold:
            return HITLReason.JUDGE_LOW_CONFIDENCE
    if val_report and val_report.needs_judge and judge_report is None:
        return HITLReason.JUDGE_UNAVAILABLE
    return None


__all__ = [
    "HITLReason", "ReviewerAction", "HITLItem", "ReviewerDecision",
    "HITLQueue", "should_route_to_hitl",
]
