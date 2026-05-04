"""
schemas/enums.py — Concept 1: Finite vocabularies for constrained decoding.

THE INSIGHT
============
For Outlines/XGrammar to enforce a controlled vocabulary, that vocabulary
must be enumerable. Every Enum below corresponds to a CDISC SDTM-controlled
value set used in clinical trial data submissions.

When the LLM is generating, the constrained decoder masks every token that
would not lead to a valid Enum member. Hallucination is structurally
impossible.

This is the difference between "asking nicely" (prompt engineering)
and "physically preventing" (constrained decoding).
"""

from enum import Enum


class VitalSignCode(str, Enum):
    """LOINC-coded vital signs accepted in this pipeline."""
    HEART_RATE         = "8867-4"    # bpm
    SYSTOLIC_BP        = "8480-6"    # mmHg
    DIASTOLIC_BP       = "8462-4"    # mmHg
    BODY_TEMP          = "8310-5"    # Cel
    RESP_RATE          = "9279-1"    # /min
    SPO2               = "59408-5"   # %


class UCUMUnit(str, Enum):
    """Unified Code for Units of Measure — clinical units accepted."""
    BPM        = "/min"      # heart rate, resp rate
    MMHG       = "mm[Hg]"    # blood pressure
    CELSIUS    = "Cel"
    FAHRENHEIT = "[degF]"
    PERCENT    = "%"          # SpO2
    UNKNOWN    = "{unknown}"


class ObservationStatus(str, Enum):
    """SDTM observation status — finite controlled vocabulary."""
    FINAL       = "F"
    PRELIMINARY = "P"
    AMENDED     = "A"
    CANCELLED   = "X"


class SourceFormat(str, Enum):
    """The three messy source formats this pipeline accepts."""
    VENDOR_JSON   = "vendor_json"     # wearable vendor batch dump
    SITE_CRF_CSV  = "site_crf_csv"    # site investigator CSV
    HL7_FRAGMENT  = "hl7_fragment"    # device HL7 v2 OBX segments


class AnomalyKind(str, Enum):
    """Types of anomalies detected by the computation module."""
    NONE              = "none"
    OUT_OF_RANGE      = "out_of_range"
    DUPLICATE         = "duplicate"
    UNIT_CONVERTED    = "unit_converted"     # informational, not an error
    CODE_MAPPED       = "code_mapped"        # informational
    UNRECOGNISED_CODE = "unrecognised_code"


__all__ = [
    "VitalSignCode", "UCUMUnit", "ObservationStatus",
    "SourceFormat", "AnomalyKind",
]
