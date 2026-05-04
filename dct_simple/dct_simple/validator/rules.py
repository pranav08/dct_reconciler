"""
validator/rules.py — Layer 3: The Deterministic Validator (Concept 4).

THE BREAD OF THE SANDWICH
==========================
Every rule is a named Python function. Every finding cites a specific rule.
A regulator can re-run this validator and get bit-identical results.

Each rule:
  - Has a stable rule_id (auditable in submission packages)
  - Has a severity (INFO / WARN / ERROR)
  - Returns findings, not exceptions (allows full report on bad input)
  - Is registered automatically via the @rule decorator

Adding a rule = writing one function. No registration boilerplate.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from functools import wraps
from typing import Callable, Any

from schemas.observation import Observation, RawObservation
from schemas.reports import (
    ValidationReport, ValidationFinding, FindingSeverity,
)
from schemas.enums import VitalSignCode


_RULE_REGISTRY: list[dict[str, Any]] = []


def rule(rule_id: str, severity: FindingSeverity, field_path: str, description: str):
    """Decorator that registers a validation function."""
    def decorator(fn: Callable) -> Callable:
        _RULE_REGISTRY.append({
            "id": rule_id, "severity": severity,
            "field_path": field_path, "description": description, "fn": fn,
        })
        @wraps(fn)
        def wrapper(*a, **kw): return fn(*a, **kw)
        return wrapper
    return decorator


# ── Rules ─────────────────────────────────────────────────────────────────────

@rule("VS-001", FindingSeverity.ERROR, "subject_id", "Subject ID required")
def check_subject_id(obs: Observation, report: ValidationReport) -> None:
    if not obs.subject_id:
        report.add(ValidationFinding(
            rule_id="VS-001", severity=FindingSeverity.ERROR,
            field_path="subject_id",
            message="USUBJID is required for SDTM submission.",
        ))


@rule("VS-002", FindingSeverity.ERROR, "measured_at", "Timestamp required and not in future")
def check_timestamp(obs: Observation, report: ValidationReport) -> None:
    if obs.measured_at is None:
        report.add(ValidationFinding(
            rule_id="VS-002", severity=FindingSeverity.ERROR,
            field_path="measured_at", message="Timestamp required.",
        ))
    elif obs.measured_at > datetime.utcnow() + timedelta(minutes=5):
        report.add(ValidationFinding(
            rule_id="VS-002", severity=FindingSeverity.ERROR,
            field_path="measured_at",
            message=f"Timestamp {obs.measured_at} is in the future.",
            actual=str(obs.measured_at),
        ))


@rule("VS-003", FindingSeverity.WARN, "value_numeric", "Heart rate sanity range")
def check_hr_range(obs: Observation, report: ValidationReport) -> None:
    if obs.vs_code == VitalSignCode.HEART_RATE:
        if not (40 <= obs.value_numeric <= 200):
            report.add(ValidationFinding(
                rule_id="VS-003", severity=FindingSeverity.WARN,
                field_path="value_numeric",
                message=f"HR {obs.value_numeric} outside expected range [40, 200]",
                actual=str(obs.value_numeric), expected="[40, 200] /min",
            ))


@rule("VS-004", FindingSeverity.WARN, "value_numeric", "BP sanity ranges")
def check_bp_range(obs: Observation, report: ValidationReport) -> None:
    if obs.vs_code == VitalSignCode.SYSTOLIC_BP:
        if not (80 <= obs.value_numeric <= 200):
            report.add(ValidationFinding(
                rule_id="VS-004", severity=FindingSeverity.WARN,
                field_path="value_numeric",
                message=f"SBP {obs.value_numeric} outside expected range [80, 200]",
                actual=str(obs.value_numeric), expected="[80, 200] mmHg",
            ))
    elif obs.vs_code == VitalSignCode.DIASTOLIC_BP:
        if not (40 <= obs.value_numeric <= 120):
            report.add(ValidationFinding(
                rule_id="VS-004", severity=FindingSeverity.WARN,
                field_path="value_numeric",
                message=f"DBP {obs.value_numeric} outside expected range [40, 120]",
                actual=str(obs.value_numeric), expected="[40, 120] mmHg",
            ))


@rule("VS-005", FindingSeverity.WARN, "value_numeric", "Body temperature range")
def check_temp_range(obs: Observation, report: ValidationReport) -> None:
    if obs.vs_code == VitalSignCode.BODY_TEMP:
        if not (35.0 <= obs.value_numeric <= 41.0):
            report.add(ValidationFinding(
                rule_id="VS-005", severity=FindingSeverity.WARN,
                field_path="value_numeric",
                message=f"Temp {obs.value_numeric}°C outside expected range [35, 41]",
                actual=str(obs.value_numeric), expected="[35, 41] Cel",
            ))


@rule("VS-006", FindingSeverity.INFO, "anomalies", "Any anomalies attached?")
def report_anomalies(obs: Observation, report: ValidationReport) -> None:
    for a in obs.anomalies:
        sev = (FindingSeverity.WARN if a.kind.value in ("out_of_range", "duplicate")
               else FindingSeverity.INFO)
        report.add(ValidationFinding(
            rule_id=f"COMP-{a.rule_id}", severity=sev,
            field_path=a.field or "anomalies",
            message=f"[{a.kind.value}] {a.message}",
            actual=a.actual, expected=a.expected,
        ))


@rule("VS-007", FindingSeverity.ERROR, "vs_code", "Vital sign code is one of accepted LOINC codes")
def check_vs_code(obs: Observation, report: ValidationReport) -> None:
    if obs.vs_code is None:
        report.add(ValidationFinding(
            rule_id="VS-007", severity=FindingSeverity.ERROR,
            field_path="vs_code", message="VSTESTCD not mapped to LOINC.",
        ))


@rule("VS-008", FindingSeverity.INFO, "value_si", "SI conversion sanity")
def check_si_conversion(obs: Observation, report: ValidationReport) -> None:
    if obs.value_si is not None and obs.value_numeric is not None:
        if obs.value_si != obs.value_numeric:
            report.add(ValidationFinding(
                rule_id="VS-008", severity=FindingSeverity.INFO,
                field_path="value_si",
                message=f"Converted to SI: {obs.value_numeric} → {obs.value_si} {obs.unit_si.value if obs.unit_si else ''}",
                actual=str(obs.value_si),
            ))


# ── Validator orchestrator ────────────────────────────────────────────────────

class DeterministicValidator:
    """Runs all registered @rule functions on one Observation."""

    def __init__(self) -> None:
        self._rules = _RULE_REGISTRY

    def validate(self, obs: Observation) -> ValidationReport:
        report = ValidationReport(observation_id=obs.source_id)
        for r in self._rules:
            try:
                r["fn"](obs, report)
            except Exception as e:
                report.add(ValidationFinding(
                    rule_id=f"{r['id']}-EXCEPTION",
                    severity=FindingSeverity.ERROR,
                    field_path=r["field_path"],
                    message=f"Rule {r['id']} crashed: {e}",
                ))
        return report

    def rule_count(self) -> int:
        return len(self._rules)


__all__ = ["DeterministicValidator", "rule"]
