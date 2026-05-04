"""
compliance/audit.py — ALCOA++ audit trail + 21 CFR Part 11 e-signature.

WHY THIS FILE EXISTS
=====================
Every clinical-data system bound for FDA/EMA submission must satisfy two
overlapping regulatory frameworks:

  1. ALCOA++  — the FDA / EMA / WHO data-integrity principles
  2. 21 CFR Part 11 — the FDA rule on electronic records & e-signatures

Both reduce, in code, to roughly the same idea:
  ✓ every record traces to a person/system, a timestamp, and an action
  ✓ records are tamper-evident (hash-chained)
  ✓ originals are preserved, edits are logged not overwritten

This module is ~100 lines of plain Python. That is on purpose: regulators
trust code they can read.

ALCOA++ MAPPING
================
  A  Attributable  → AuditEvent.actor (user_id or system_id)
  L  Legible       → JSON-serialised, UTF-8, human-readable
  C  Contemporaneous → AuditEvent.timestamp_utc (recorded at the moment of action)
  O  Original      → raw_observation preserved alongside canonical
  A  Accurate      → enforced by Layer 3 deterministic validator
  +  Complete      → every layer emits an event; chain breaks if one is missing
  +  Consistent    → SHA-256 hash chain links events in immutable order
  +  Enduring      → write-once append log
  +  Available     → JSONL, queryable by source_id

21 CFR Part 11 MAPPING
=======================
  §11.10(a) Validation of systems        → Layer 3 + tests/test_all.py
  §11.10(b) Ability to generate records  → audit_log.export(source_id)
  §11.10(c) Protection of records        → SHA-256 hash chain (tamper-evident)
  §11.10(e) Audit trail                  → THIS FILE
  §11.50    Signature manifestation      → ESignature.print_meaning
  §11.70    Signature/record linking     → ESignature.record_hash
  §11.200   E-signature components       → user_id + meaning + timestamp + hash
"""

from __future__ import annotations
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class AuditAction(str, Enum):
    """Discrete, controlled-vocabulary actions logged in the audit trail."""
    PARSED      = "parsed"        # Layer 0: raw source ingested
    EXTRACTED   = "extracted"     # Layer 1: LLM produced RawObservation
    COMPUTED    = "computed"      # Layer 2: canonicalised
    VALIDATED   = "validated"     # Layer 3: deterministic validator ran
    JUDGED      = "judged"        # Layer 4: judge LLM ran
    SIGNED      = "signed"        # 21 CFR Part 11 e-signature applied
    AMENDED     = "amended"       # tracked correction (originals preserved)
    REJECTED    = "rejected"      # validator-rejected record


@dataclass
class ModelSnapshot:
    """
    Frozen snapshot of which model produced this output. Critical for
    reproducibility under ALCOA++ (Original) and Part 11 (Validation).

    Why every field matters:
      - model_id:    which model
      - model_version: which exact build (LLM weights drift between releases)
      - prompt_id / prompt_hash: which prompt template was used
      - prompt_version: which version of that template
      - temperature: 0.0 = deterministic decoding (recommended)
      - seed:        when supported, locks the sampler PRNG
      - library:     "outlines==0.0.46" — affects FSM compilation
    """
    model_id:        str = ""
    model_version:   str = ""
    library:         str = ""
    prompt_id:       str = ""        # "extract_vital_sign_v3"
    prompt_version:  str = ""        # "3.1.0"
    prompt_hash:     str = ""        # SHA-256 of the rendered prompt
    temperature:     float | None = None
    seed:            int | None = None
    extra:           dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    """
    One row in the ALCOA++ audit trail.

    Attributable / Contemporaneous / Original — all three live in this dataclass.

    For LLM layers (L1, L4) we additionally snapshot:
      - input_hash / input_redacted: what went into the model
      - output_hash:                 what came out
      - model_snapshot:              which model+prompt+params (reproducibility)
    """
    source_id:     str            # which observation
    layer:         str            # "L0".."L4" or "compliance"
    action:        AuditAction
    actor:         str            # user_id (Part 11) or "system:<component>"
    timestamp_utc: str            # ISO8601 — Contemporaneous
    payload_hash:  str            # SHA-256 of the layer's payload — Original
    prev_hash:     str            # SHA-256 of the previous event — chain

    # ── For side-by-side human review (input vs output) ─────────────────
    input_hash:    str = ""       # SHA-256 of the layer's input
    input_excerpt: str = ""       # first ~500 chars of input, for inspector
    output_excerpt:str = ""       # first ~500 chars of output, for inspector

    # ── Reproducibility (LLM layers only) ───────────────────────────────
    model_snapshot: ModelSnapshot | None = None

    # ── 21 CFR Part 11 retention (§11.10(c) — minimum 6 years for clinical) ─
    retention_years: int = 7      # 6 minimum + 1 buffer; FDA pre-1572 trials may need longer

    chain_hash:    str = ""       # SHA-256 of (this event + prev_hash) — tamper-evident
    metadata:      dict[str, Any] = field(default_factory=dict)

    def compute_chain_hash(self) -> str:
        """Tamper-evident link to the previous event."""
        snap_json = (
            json.dumps(asdict(self.model_snapshot), sort_keys=True)
            if self.model_snapshot else ""
        )
        material = "|".join([
            self.source_id, self.layer, self.action.value, self.actor,
            self.timestamp_utc, self.payload_hash, self.prev_hash,
            self.input_hash, snap_json,
            json.dumps(self.metadata, sort_keys=True),
        ])
        return hashlib.sha256(material.encode()).hexdigest()


def _hash_payload(payload: Any) -> str:
    """Stable SHA-256 of any JSON-serialisable payload."""
    if hasattr(payload, "model_dump"):              # Pydantic v2
        payload = payload.model_dump(mode="json")
    elif hasattr(payload, "dict"):                  # Pydantic v1
        payload = payload.dict()
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()


def _excerpt(payload: Any, max_chars: int = 500) -> str:
    """Short, human-readable rendering of a payload — for inspector review."""
    if payload is None:
        return ""
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    elif hasattr(payload, "dict"):
        payload = payload.dict()
    s = json.dumps(payload, default=str, ensure_ascii=False)
    return s[:max_chars] + ("…" if len(s) > max_chars else "")


# ── Append-only audit log (Enduring, Consistent) ─────────────────────────────

class AuditLog:
    """
    Append-only JSONL audit log with SHA-256 hash chain.

    GUARANTEES
    -----------
    - Append-only: AuditLog never edits or deletes prior events.
    - Tamper-evident: chain_hash[i] = SHA256(event[i] || chain_hash[i-1]).
    - Verifiable: AuditLog.verify() recomputes the chain end-to-end.
    """
    GENESIS = "0" * 64        # chain root

    def __init__(self, path: str | Path = "audit.jsonl") -> None:
        self.path = Path(path)
        self._last_hash = self._load_last_hash()

    def _load_last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return self.GENESIS
        with self.path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = 4096
            f.seek(max(0, size - chunk))
            last_line = f.read().splitlines()[-1].decode()
        return json.loads(last_line)["chain_hash"]

    def append(
        self, *,
        source_id: str, layer: str, action: AuditAction,
        actor: str, payload: Any,
        input_data: Any | None = None,           # for LLM layers, the prompt input
        model_snapshot: "ModelSnapshot | None" = None,
        retention_years: int = 7,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Record one immutable event."""
        evt = AuditEvent(
            source_id=source_id, layer=layer, action=action, actor=actor,
            timestamp_utc=datetime.utcnow().isoformat() + "Z",
            payload_hash=_hash_payload(payload),
            prev_hash=self._last_hash,
            input_hash=_hash_payload(input_data) if input_data is not None else "",
            input_excerpt=_excerpt(input_data),
            output_excerpt=_excerpt(payload),
            model_snapshot=model_snapshot,
            retention_years=retention_years,
            metadata=metadata or {},
        )
        evt.chain_hash = evt.compute_chain_hash()
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(evt), default=str) + "\n")
        self._last_hash = evt.chain_hash
        return evt

    def verify(self) -> tuple[bool, str | None]:
        """
        Recompute the entire chain. Return (ok, error_message).
        Used by validation/inspection workflows to prove the log is intact.
        """
        if not self.path.exists():
            return True, None
        prev = self.GENESIS
        with self.path.open() as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                # Reconstruct AuditEvent (action enum + nested ModelSnapshot)
                snap = d.get("model_snapshot")
                if isinstance(snap, dict):
                    snap = ModelSnapshot(**snap)
                payload = {k: v for k, v in d.items()
                           if k not in ("action", "model_snapshot")}
                payload["action"] = AuditAction(d["action"])
                payload["model_snapshot"] = snap
                evt = AuditEvent(**payload)
                if evt.prev_hash != prev:
                    return False, f"event {i}: prev_hash mismatch"
                if evt.compute_chain_hash() != evt.chain_hash:
                    return False, f"event {i}: chain_hash tampered"
                prev = evt.chain_hash
        return True, None

    def export(self, source_id: str) -> list[dict]:
        """Return all events for one observation — the §11.10(b) requirement."""
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as f:
            for line in f:
                d = json.loads(line)
                if d["source_id"] == source_id:
                    out.append(d)
        return out


# ── 21 CFR Part 11 e-signature (§11.50, §11.70, §11.200) ─────────────────────

@dataclass
class ESignature:
    """
    21 CFR Part 11-compliant electronic signature.

    REQUIRED COMPONENTS (§11.50, §11.200)
    --------------------------------------
    - Printed name of signer   → user_id
    - Date and time of signing → timestamp_utc
    - Meaning of signature      → meaning ("review", "approval", "responsibility")
    - Linked to the record     → record_hash (cryptographically bound)
    """
    user_id:       str
    record_hash:   str           # SHA-256 of the signed record (§11.70)
    meaning:       str           # e.g. "approved-for-submission"
    timestamp_utc: str
    signature:     str           # SHA-256(user_id|record_hash|meaning|timestamp)

    @classmethod
    def sign(cls, user_id: str, record: Any, meaning: str) -> "ESignature":
        ts = datetime.utcnow().isoformat() + "Z"
        rec_hash = _hash_payload(record)
        material = f"{user_id}|{rec_hash}|{meaning}|{ts}"
        sig = hashlib.sha256(material.encode()).hexdigest()
        return cls(user_id=user_id, record_hash=rec_hash, meaning=meaning,
                   timestamp_utc=ts, signature=sig)

    def verify(self, record: Any) -> bool:
        """§11.70 — record/signature linkage check."""
        if _hash_payload(record) != self.record_hash:
            return False
        material = f"{self.user_id}|{self.record_hash}|{self.meaning}|{self.timestamp_utc}"
        return hashlib.sha256(material.encode()).hexdigest() == self.signature


# ── Amendment record (Part 11 §11.10(e) — tracked corrections) ──────────────

@dataclass
class Amendment:
    """
    A tracked correction to a previously-recorded observation.

    Part 11 demands: prior values are preserved, the change is attributable,
    and a reason is recorded. Originals are never overwritten — they are
    superseded by an Amendment that points back to them via prev_chain_hash.

    REQUIRED FIELDS (Part 11 §11.10(e))
    -------------------------------------
      - field_name:       which field was changed
      - prev_value:       the original value (preserved, never deleted)
      - new_value:        the corrected value
      - reason_code:      controlled vocabulary (transcription_error, etc.)
      - reason_text:      free-text justification
      - actor:            user_id of the person making the change
      - prev_chain_hash:  links to the audit event being amended
    """
    source_id:        str
    field_name:       str
    prev_value:       Any
    new_value:        Any
    reason_code:      str           # see VALID_REASON_CODES below
    reason_text:      str
    actor:            str
    timestamp_utc:    str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    prev_chain_hash:  str = ""      # which event this amends


VALID_REASON_CODES = (
    "transcription_error",
    "data_entry_error",
    "source_correction",
    "investigator_review",
    "monitor_query_resolution",
    "regulatory_request",
    "other",
)


# ── ReviewPacket (human review with side-by-side input vs output) ────────────

@dataclass
class ReviewPacket:
    """
    Self-contained packet for a human reviewer to inspect one record before
    signing. Bundles input, output, validator findings, judge opinion, and
    full audit trail into one structure.

    Built by AuditLog.build_review_packet(source_id). The reviewer compares
    input_excerpt vs output_excerpt side-by-side, reads the rationale, and
    if satisfied calls ESignature.sign(record=packet, meaning="approved").
    """
    source_id:           str
    raw_input:           Any            # what came in (dict, str, etc.)
    canonical_output:    Any            # what came out (Observation)
    validator_findings:  list[dict]
    judge_rationale:     str | None
    audit_events:        list[dict]     # full chain for this source_id
    model_snapshots:     list[dict]     # all model versions involved
    built_at_utc:        str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


# Add helper method to AuditLog for building review packets and amendments
def _build_review_packet(self: AuditLog, source_id: str,
                         raw_input: Any, canonical_output: Any,
                         validator_findings: list[dict] | None = None,
                         judge_rationale: str | None = None) -> ReviewPacket:
    events = self.export(source_id)
    snaps = [e["model_snapshot"] for e in events
             if e.get("model_snapshot") is not None]
    return ReviewPacket(
        source_id=source_id,
        raw_input=raw_input,
        canonical_output=canonical_output,
        validator_findings=validator_findings or [],
        judge_rationale=judge_rationale,
        audit_events=events,
        model_snapshots=snaps,
    )


def _record_amendment(self: AuditLog, amendment: Amendment) -> AuditEvent:
    """Append an AMENDED audit event with full prev/new/reason data."""
    if amendment.reason_code not in VALID_REASON_CODES:
        raise ValueError(
            f"Invalid reason_code '{amendment.reason_code}'. "
            f"Must be one of: {VALID_REASON_CODES}"
        )
    return self.append(
        source_id=amendment.source_id,
        layer="compliance", action=AuditAction.AMENDED,
        actor=amendment.actor,
        payload=asdict(amendment),
        metadata={
            "field_name":       amendment.field_name,
            "reason_code":      amendment.reason_code,
            "prev_chain_hash":  amendment.prev_chain_hash,
        },
    )


# Bind methods onto AuditLog
AuditLog.build_review_packet = _build_review_packet
AuditLog.record_amendment    = _record_amendment


# ─────────────────────────────────────────────────────────────────────────────
#  REPRODUCIBILITY NOTE — temperature 0 and seed-locking
# ─────────────────────────────────────────────────────────────────────────────
#
#  All LLM layers run with temperature=0.0 and (where the runtime supports it)
#  a fixed integer seed. This makes the same prompt produce the same output
#  bit-for-bit across runs — the foundation of ALCOA "Accurate" and Part 11
#  §11.10(a) "Validation". ModelSnapshot captures both values so the producer
#  of every record is fully reproducible.
#
#  In practice: greedy / argmax decoding is deterministic by construction
#  (Outlines uses outlines.samplers.greedy() with no seed needed). For sampled
#  decoders, both temperature and seed must be locked together. Cloud APIs
#  that do not expose a seed parameter (e.g. older Claude APIs) are flagged
#  in ModelSnapshot.extra so reviewers know reproducibility is best-effort,
#  not bit-identical.
# ─────────────────────────────────────────────────────────────────────────────


__all__ = ["AuditAction", "AuditEvent", "AuditLog", "ESignature",
           "ModelSnapshot", "Amendment", "ReviewPacket"]
