"""
pipeline/run.py — The 5-layer sandwich orchestrator.

Plain function pipeline (no LangGraph). Each layer is a distinct call with
a clear input/output type. The shape of this file IS the architecture:

    Layer 0: parse_source     (deterministic)
    Layer 1: extract          (LLM — only here)
    Layer 2: canonicalise     (deterministic)
    Layer 3: validate         (deterministic)
    Layer 4: judge            (LLM — only if flagged)
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator

from parsers.sources       import SourceRecord, parse_source
from extractor.constrained import OutlinesExtractor, ExtractionConfig, verify_grounding
from computation.engine    import ComputationEngine
from validator.rules       import DeterministicValidator
from judge.llm_judge       import JudgeLLM
from compliance.audit      import AuditLog, AuditAction, ModelSnapshot
from compliance.hitl       import HITLQueue, should_route_to_hitl
from schemas.enums         import SourceFormat
from schemas.observation   import Observation, RawObservation
from schemas.reports       import ValidationReport, JudgeReport


@dataclass
class PipelineResult:
    source_record: SourceRecord
    raw:           RawObservation
    canonical:     Observation | None
    val_report:    ValidationReport | None
    judge_report:  JudgeReport | None
    dropped:       bool = False
    drop_reason:   str | None = None
    processing_ms: float = 0.0
    grounding_violations: list[str] = None        # set by grounding check
    routed_to_hitl: bool = False                  # set by HITL router
    hitl_reason:   str | None = None


@dataclass
class BatchSummary:
    total_records:     int = 0
    canonicalised:     int = 0
    dropped:           int = 0
    valid_e2e:         int = 0
    judge_calls:       int = 0
    judge_total_cost:  float = 0.0
    flagged_warn:      int = 0
    flagged_error:     int = 0


class Pipeline:
    """The 5-layer DCT pipeline."""

    def __init__(self, use_mock_extractor: bool = True,
                 judge_enabled: bool = True,
                 audit_log_path: str = "audit.jsonl",
                 hitl_queue_path: str = "hitl_queue.jsonl",
                 actor: str = "system:dct_simple") -> None:
        self.extractor   = OutlinesExtractor(ExtractionConfig(use_mock=use_mock_extractor))
        self.computation = ComputationEngine()
        self.validator   = DeterministicValidator()
        self.judge       = JudgeLLM(enabled=judge_enabled, only_on_warn=True)
        self.audit       = AuditLog(audit_log_path)        # ALCOA++ + Part 11 §11.10(e)
        self.hitl        = HITLQueue(hitl_queue_path)      # human-in-the-loop queue
        self.actor       = actor

    def process_one(self, record: SourceRecord) -> PipelineResult:
        t0 = datetime.utcnow()

        # ── Layer 0 audit event (Original — raw source preserved) ────────
        self.audit.append(
            source_id=record.source_id, layer="L0", action=AuditAction.PARSED,
            actor=self.actor, payload=record.raw_dict,
            metadata={"format": record.source_format.value},
        )

        # ── Layer 1: constrained extraction (LLM) ─────────────────────────
        raw = self.extractor.extract(record)
        l1_snap = ModelSnapshot(
            model_id=self.extractor.config.model_id,
            model_version="mock" if self.extractor.config.use_mock else "weights:sha256-pinned",
            library="outlines==0.0.46" if not self.extractor.config.use_mock else "mock",
            prompt_id="extract_vital_sign",
            prompt_version="1.0.0",
            prompt_hash=hashlib.sha256(b"extract_vital_sign:v1").hexdigest(),
            temperature=self.extractor.config.temperature,   # 0.0 → deterministic
            seed=42 if not self.extractor.config.use_mock else None,
        )
        self.audit.append(
            source_id=record.source_id, layer="L1", action=AuditAction.EXTRACTED,
            actor=self.actor, payload=raw,
            input_data=record.raw_text,                       # ← side-by-side input
            model_snapshot=l1_snap,                           # ← reproducibility
        )

        # ── Grounding check: every numeric value must trace to source ─────
        grounding_violations = verify_grounding(raw)
        if grounding_violations:
            self.audit.append(
                source_id=record.source_id, layer="L1",
                action=AuditAction.REJECTED,
                actor=self.actor, payload={"violations": grounding_violations},
                metadata={"check": "grounding"},
            )

        # ── Layer 2: computation (deterministic) ──────────────────────────
        canonical, anomalies = self.computation.canonicalise(raw)

        if canonical is None:
            self.audit.append(
                source_id=record.source_id, layer="L2", action=AuditAction.REJECTED,
                actor=self.actor, payload={"anomalies": [a.message for a in anomalies]},
            )
            ms = (datetime.utcnow() - t0).total_seconds() * 1000
            return PipelineResult(
                source_record=record, raw=raw, canonical=None,
                val_report=None, judge_report=None,
                dropped=True,
                drop_reason="; ".join(a.message for a in anomalies),
                processing_ms=round(ms, 1),
            )

        self.audit.append(
            source_id=record.source_id, layer="L2", action=AuditAction.COMPUTED,
            actor=self.actor, payload=canonical,
        )

        # ── Layer 3: deterministic validator ──────────────────────────────
        val_report = self.validator.validate(canonical)
        self.audit.append(
            source_id=record.source_id, layer="L3", action=AuditAction.VALIDATED,
            actor=self.actor, payload=val_report,
            metadata={"errors": val_report.error_count, "warns": val_report.warn_count},
        )

        # ── Layer 4: judge LLM (conditional) ──────────────────────────────
        judge_report = self.judge.judge(canonical, val_report)
        if judge_report:
            l4_snap = ModelSnapshot(
                model_id=judge_report.judge_model,
                model_version=judge_report.judge_model,        # API-pinned model id
                library="anthropic-sdk",
                prompt_id="judge_flagged_observation",
                prompt_version="1.0.0",
                prompt_hash=hashlib.sha256(b"judge_flagged_observation:v1").hexdigest(),
                temperature=0.0,
                seed=None,                                      # cloud API may not expose seed
                extra={"determinism": "best-effort (no seed param on this API)"},
            )
            self.audit.append(
                source_id=record.source_id, layer="L4", action=AuditAction.JUDGED,
                actor=f"system:{judge_report.judge_model}",
                payload=judge_report,
                input_data={"observation": canonical, "findings": val_report},
                model_snapshot=l4_snap,
                metadata={"action": judge_report.suggested_action,
                          "cost_usd": judge_report.judge_cost_usd},
            )

        # ── HITL routing (deterministic) ──────────────────────────────────
        hitl_reason = should_route_to_hitl(val_report, judge_report)
        routed_to_hitl = False
        if hitl_reason is not None:
            routed_to_hitl = True
            packet_hash = hashlib.sha256(
                f"{record.source_id}|{val_report.error_count}|{val_report.warn_count}".encode()
            ).hexdigest()
            self.hitl.enqueue(
                source_id=record.source_id, reason=hitl_reason,
                review_packet_hash=packet_hash,
                priority=1 if hitl_reason.value == "validator_error" else 0,
                metadata={"layer_summary": f"warns={val_report.warn_count}, errors={val_report.error_count}"},
            )

        ms = (datetime.utcnow() - t0).total_seconds() * 1000
        return PipelineResult(
            source_record=record, raw=raw, canonical=canonical,
            val_report=val_report, judge_report=judge_report,
            grounding_violations=grounding_violations or None,
            routed_to_hitl=routed_to_hitl,
            hitl_reason=hitl_reason.value if hitl_reason else None,
            processing_ms=round(ms, 1),
        )

    def process_batch(
        self, fmt: SourceFormat, raw_text: str, batch_id: str = "b",
    ) -> tuple[list[PipelineResult], BatchSummary]:
        """Layer 0 + run all records."""
        results: list[PipelineResult] = []
        summary = BatchSummary()

        for record in parse_source(fmt, raw_text, batch_id):
            summary.total_records += 1
            result = self.process_one(record)
            results.append(result)
            if result.dropped:
                summary.dropped += 1
            else:
                summary.canonicalised += 1
                if result.val_report and result.val_report.is_valid and \
                   result.val_report.warn_count == 0:
                    summary.valid_e2e += 1
                if result.val_report:
                    if result.val_report.error_count > 0:
                        summary.flagged_error += 1
                    elif result.val_report.warn_count > 0:
                        summary.flagged_warn += 1
                if result.judge_report:
                    summary.judge_calls += 1
                    summary.judge_total_cost += result.judge_report.judge_cost_usd

        return results, summary
