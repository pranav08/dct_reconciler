"""
parsers/sources.py — Layer 0: Pure-Python source parsers ($0).

Three messy formats normalised to a common SourceRecord. No LLM.
"""

from __future__ import annotations
import csv
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator

from schemas.enums import SourceFormat


@dataclass
class SourceRecord:
    """A single observation as it appeared in the source — pre-extraction."""
    source_id:     str
    source_format: SourceFormat
    raw_text:      str           # human-readable rendering of this record
    raw_dict:      dict[str, Any]  # structured fields, if any


# ── 1. Vendor JSON (wearable batch dump) ──────────────────────────────────────

def parse_vendor_json(json_text: str, batch_id: str = "vendor") -> Iterator[SourceRecord]:
    """
    Vendor batch dump. Format varies by vendor — we accept either:
      {"records": [{"subj":"S001","metric":"PULSE","val":72,"u":"bpm","ts":"..."}]}
    or a list at top level.

    Vendors use inconsistent codes ("PULSE" vs "HR" vs "heart_rate") and
    inconsistent units ("bpm" vs "/min"). We pass them through unchanged
    — the extractor will normalise.
    """
    data = json.loads(json_text)
    records = data["records"] if isinstance(data, dict) and "records" in data else data
    for i, rec in enumerate(records):
        sid = f"{batch_id}-{i:04d}"
        text = (
            f"subject={rec.get('subj') or rec.get('subject_id','?')} "
            f"metric={rec.get('metric') or rec.get('code','?')} "
            f"value={rec.get('val') or rec.get('value','?')} "
            f"unit={rec.get('u') or rec.get('unit','?')} "
            f"timestamp={rec.get('ts') or rec.get('timestamp','?')}"
        )
        yield SourceRecord(
            source_id=sid,
            source_format=SourceFormat.VENDOR_JSON,
            raw_text=text,
            raw_dict=rec,
        )


# ── 2. Site CRF CSV (site investigator export) ────────────────────────────────

def parse_site_crf_csv(csv_text: str, batch_id: str = "crf") -> Iterator[SourceRecord]:
    """
    Site CRF export — clean column names but human-entered values.

    Expected columns: subject_id, visit, parameter, value, unit, datetime
    Free text creeps in: "approx 70bpm", "98.6 F", "37 deg C".
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    for i, row in enumerate(reader):
        sid = f"{batch_id}-{i:04d}"
        text = " | ".join(f"{k}={v}" for k, v in row.items())
        yield SourceRecord(
            source_id=sid,
            source_format=SourceFormat.SITE_CRF_CSV,
            raw_text=text,
            raw_dict=row,
        )


# ── 3. HL7 v2 fragment (device OBX segments) ─────────────────────────────────

def parse_hl7_fragment(hl7_text: str, batch_id: str = "hl7") -> Iterator[SourceRecord]:
    """
    HL7 v2 OBX segments. Pipe-delimited, fields are positional.

    Example:
      PID|||SUBJ-001
      OBX|1|NM|8867-4^Heart Rate||72|/min|||F|||20250108120000
      OBX|2|NM|8480-6^Systolic BP||120|mm[Hg]|||F|||20250108120000
    """
    current_subject: str | None = None
    i = 0
    for line in hl7_text.strip().splitlines():
        parts = line.split("|")
        seg = parts[0] if parts else ""
        if seg == "PID" and len(parts) > 3:
            current_subject = parts[3].strip()
        elif seg == "OBX" and len(parts) >= 12:
            sid = f"{batch_id}-{i:04d}"
            i += 1
            code_field = parts[3] if len(parts) > 3 else ""
            code, *rest = code_field.split("^")
            value = parts[5] if len(parts) > 5 else ""
            unit = parts[6] if len(parts) > 6 else ""
            status = parts[8] if len(parts) > 8 else ""
            ts = parts[11] if len(parts) > 11 else ""
            text = (
                f"subject={current_subject} code={code} "
                f"name={rest[0] if rest else '?'} value={value} "
                f"unit={unit} status={status} timestamp={ts}"
            )
            yield SourceRecord(
                source_id=sid,
                source_format=SourceFormat.HL7_FRAGMENT,
                raw_text=text,
                raw_dict={
                    "subject_id": current_subject,
                    "code": code,
                    "code_name": rest[0] if rest else None,
                    "value": value,
                    "unit": unit,
                    "status": status,
                    "timestamp": ts,
                },
            )


# ── Dispatcher ────────────────────────────────────────────────────────────────

def parse_source(fmt: SourceFormat, text: str, batch_id: str = "src") -> Iterator[SourceRecord]:
    if fmt == SourceFormat.VENDOR_JSON:
        return parse_vendor_json(text, batch_id)
    if fmt == SourceFormat.SITE_CRF_CSV:
        return parse_site_crf_csv(text, batch_id)
    if fmt == SourceFormat.HL7_FRAGMENT:
        return parse_hl7_fragment(text, batch_id)
    return iter([])
