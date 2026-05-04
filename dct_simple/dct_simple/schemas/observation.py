"""
schemas/observation.py — Concept 2: The Lossy Parser.

THE INSIGHT
============
Every field in RawObservation is Optional. The LLM is allowed to say
"I don't know" — and that is the correct answer when the source data
doesn't contain the field.

The temptation when prompting LLMs is to demand complete output. But for
clinical data this produces hallucinations. By making the schema permissive
on input, we get HONEST extractions. The computation and validation layers
then either fill the gap deterministically or flag it.

Compare:
  RawObservation  — what the extractor produces (lossy, tolerant)
  Observation     — canonical, validated, all required fields present
"""

from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, model_validator
from schemas.enums import (
    VitalSignCode, UCUMUnit, ObservationStatus,
    SourceFormat, AnomalyKind,
)


class RawObservation(BaseModel):
    """
    The lossy parser's output — every field is Optional.

    This schema IS the grammar that Outlines compiles into a finite-state
    machine. The LLM CANNOT emit anything outside the constraints of this
    Pydantic class.
    """
    # ── Provenance (set by the parser, not the LLM) ─────────────────────
    source_id:       str
    source_format:   SourceFormat
    source_record:   str | None = None         # raw text, for audit

    # ── Subject identification ──────────────────────────────────────────
    subject_id:      str | None = None         # USUBJID in SDTM terms
    visit_id:        str | None = None

    # ── Observation core ────────────────────────────────────────────────
    vs_code:         VitalSignCode | None = None    # constrained to LOINC enum
    vs_code_text:    str | None = None              # raw vendor name (e.g. "PULSE")

    value_numeric:   float | None = None
    unit:            UCUMUnit | None = None         # constrained to UCUM enum
    unit_text:       str | None = None              # raw unit string (e.g. "deg F")

    # ── Time ────────────────────────────────────────────────────────────
    measured_at:     datetime | None = None

    # ── Status ──────────────────────────────────────────────────────────
    status:          ObservationStatus | None = None

    # ── Quality (set by the model if it knows; else by computation) ─────
    signal_quality:  float | None = Field(None, ge=0.0, le=1.0)

    class Config:
        extra = "allow"


class Anomaly(BaseModel):
    """One anomaly finding from the computation module or validator."""
    kind:        AnomalyKind
    rule_id:     str
    message:     str
    field:       str | None = None
    actual:      str | None = None
    expected:    str | None = None


class Observation(BaseModel):
    """
    Canonical, fully-validated observation.

    This is what the SDTM emitter consumes. All previously Optional fields
    are now required. Cross-field validators run automatically.
    """
    source_id:       str
    source_format:   SourceFormat
    subject_id:      str
    visit_id:        str | None = None

    vs_code:         VitalSignCode
    value_numeric:   float
    unit:            UCUMUnit
    measured_at:     datetime
    status:          ObservationStatus = ObservationStatus.FINAL

    # ── Computed fields ─────────────────────────────────────────────────
    value_si:        float | None = None       # standardised SI value
    unit_si:         UCUMUnit | None = None    # standardised SI unit
    anomalies:       list[Anomaly] = Field(default_factory=list)
    is_imputed:      bool = False

    @model_validator(mode="after")
    def physiological_range(self) -> "Observation":
        """Sanity-only range check (looser than the validator's clinical ranges)."""
        if self.vs_code == VitalSignCode.HEART_RATE:
            if not (10 <= self.value_numeric <= 350):
                raise ValueError(f"Heart rate {self.value_numeric} outside survivable bounds")
        return self


__all__ = ["RawObservation", "Observation", "Anomaly"]
