"""
tests/test_all.py — Unit tests for all deterministic layers.

Runs without GPU, without Anthropic key.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from datetime import datetime


# ── Source parsers ────────────────────────────────────────────────────────────

def test_vendor_json_parsing():
    from parsers.sources import parse_vendor_json
    js = '{"records":[{"subj":"S1","metric":"PULSE","val":72,"u":"bpm","ts":"2025-01-08T08:00:00"}]}'
    records = list(parse_vendor_json(js, "b"))
    assert len(records) == 1
    assert records[0].raw_dict["subj"] == "S1"
    assert "PULSE" in records[0].raw_text

def test_csv_parsing():
    from parsers.sources import parse_site_crf_csv
    csv = "subject_id,parameter,value,unit,datetime\nS1,Heart Rate,72,bpm,2025-01-08 08:00:00\n"
    records = list(parse_site_crf_csv(csv, "b"))
    assert len(records) == 1
    assert records[0].raw_dict["parameter"] == "Heart Rate"

def test_hl7_parsing():
    from parsers.sources import parse_hl7_fragment
    hl7 = "PID|||SUBJ-001\nOBX|1|NM|8867-4^Heart Rate||72|/min|||F|||20250108080000"
    records = list(parse_hl7_fragment(hl7, "b"))
    assert len(records) == 1
    assert records[0].raw_dict["subject_id"] == "SUBJ-001"
    assert records[0].raw_dict["code"] == "8867-4"


# ── Mock extractor (constrained-decoding stand-in) ────────────────────────────

def test_mock_extractor_only_emits_valid_enums():
    """The mock must respect enum constraints — same as real Outlines decoder."""
    from extractor.constrained import OutlinesExtractor, ExtractionConfig
    from parsers.sources import parse_vendor_json
    from schemas.enums import VitalSignCode, UCUMUnit

    ext = OutlinesExtractor(ExtractionConfig(use_mock=True))
    js = '{"records":[{"subj":"S1","metric":"PULSE","val":72,"u":"bpm","ts":"2025-01-08T08:00:00"}]}'
    records = list(parse_vendor_json(js, "b"))
    raw = ext.extract(records[0])

    if raw.vs_code is not None:
        assert isinstance(raw.vs_code, VitalSignCode)
    if raw.unit is not None:
        assert isinstance(raw.unit, UCUMUnit)
    assert raw.subject_id == "S1"
    assert raw.value_numeric == 72.0

def test_mock_extractor_leaves_unrecognised_code_as_none():
    """Lossy parser principle — never invent enum values."""
    from extractor.constrained import OutlinesExtractor, ExtractionConfig
    from parsers.sources import parse_vendor_json

    ext = OutlinesExtractor(ExtractionConfig(use_mock=True))
    js = '{"records":[{"subj":"S1","metric":"WeirdMetric","val":42,"u":"x","ts":"2025-01-08T08:00:00"}]}'
    records = list(parse_vendor_json(js, "b"))
    raw = ext.extract(records[0])
    # vs_code is None — extractor refused to invent
    assert raw.vs_code is None
    # vs_code_text preserved for downstream computation to attempt mapping
    assert "WeirdMetric" in (raw.vs_code_text or "")


# ── Computation module ────────────────────────────────────────────────────────

def test_loinc_lookup_aliases():
    from computation.engine import map_to_loinc
    from schemas.enums import VitalSignCode
    assert map_to_loinc("PULSE")    == VitalSignCode.HEART_RATE
    assert map_to_loinc("HR")        == VitalSignCode.HEART_RATE
    assert map_to_loinc("8867-4")   == VitalSignCode.HEART_RATE
    assert map_to_loinc("garbage")  is None

def test_fahrenheit_to_celsius():
    from computation.engine import to_si
    from schemas.enums import VitalSignCode, UCUMUnit
    val, unit = to_si(98.6, UCUMUnit.FAHRENHEIT, VitalSignCode.BODY_TEMP)
    assert abs(val - 37.0) < 0.05
    assert unit == UCUMUnit.CELSIUS

def test_no_temp_conversion_needed():
    from computation.engine import to_si
    from schemas.enums import VitalSignCode, UCUMUnit
    val, unit = to_si(36.8, UCUMUnit.CELSIUS, VitalSignCode.BODY_TEMP)
    assert val == 36.8
    assert unit == UCUMUnit.CELSIUS

def test_range_check_in_bounds():
    from computation.engine import check_range
    from schemas.enums import VitalSignCode
    assert check_range(72, VitalSignCode.HEART_RATE) is None

def test_range_check_out_of_bounds():
    from computation.engine import check_range
    from schemas.enums import VitalSignCode, AnomalyKind
    a = check_range(285, VitalSignCode.HEART_RATE)
    assert a is not None
    assert a.kind == AnomalyKind.OUT_OF_RANGE
    assert "285" in a.message

def test_duplicate_detection():
    from computation.engine import DuplicateDetector
    from schemas.observation import Observation
    from schemas.enums import VitalSignCode, UCUMUnit, ObservationStatus, SourceFormat

    dd = DuplicateDetector()
    obs1 = Observation(
        source_id="r1", source_format=SourceFormat.HL7_FRAGMENT,
        subject_id="S1", vs_code=VitalSignCode.HEART_RATE,
        value_numeric=72, unit=UCUMUnit.BPM,
        measured_at=datetime(2025,1,8,8,0,0),
        status=ObservationStatus.FINAL,
    )
    assert dd.check(obs1) is None       # first sighting
    obs2 = Observation(
        source_id="r2", source_format=SourceFormat.HL7_FRAGMENT,
        subject_id="S1", vs_code=VitalSignCode.HEART_RATE,
        value_numeric=72, unit=UCUMUnit.BPM,
        measured_at=datetime(2025,1,8,8,0,0),
        status=ObservationStatus.FINAL,
    )
    dup = dd.check(obs2)
    assert dup is not None              # duplicate caught

def test_full_canonicalise_with_unit_conversion():
    from computation.engine import ComputationEngine
    from schemas.observation import RawObservation
    from schemas.enums import (VitalSignCode, UCUMUnit, ObservationStatus, SourceFormat)

    engine = ComputationEngine()
    raw = RawObservation(
        source_id="r1", source_format=SourceFormat.VENDOR_JSON,
        subject_id="S1", vs_code=VitalSignCode.BODY_TEMP, vs_code_text="BodyTemp",
        value_numeric=98.6, unit=UCUMUnit.FAHRENHEIT, unit_text="deg F",
        measured_at=datetime(2025,1,8,8,0,0), status=ObservationStatus.FINAL,
    )
    obs, anomalies = engine.canonicalise(raw)
    assert obs is not None
    assert obs.unit == UCUMUnit.CELSIUS
    assert abs(obs.value_numeric - 37.0) < 0.05
    # An informational anomaly is recorded
    assert any(a.rule_id == "UCUM-1" for a in anomalies)


# ── Deterministic validator ───────────────────────────────────────────────────

def test_validator_clean_record_passes():
    from validator.rules import DeterministicValidator
    from schemas.observation import Observation
    from schemas.enums import (VitalSignCode, UCUMUnit, ObservationStatus, SourceFormat)
    v = DeterministicValidator()
    obs = Observation(
        source_id="r1", source_format=SourceFormat.HL7_FRAGMENT,
        subject_id="S1", vs_code=VitalSignCode.HEART_RATE,
        value_numeric=72, unit=UCUMUnit.BPM,
        measured_at=datetime(2025,1,8,8,0,0),
        status=ObservationStatus.FINAL,
    )
    report = v.validate(obs)
    assert report.error_count == 0
    assert report.warn_count == 0

def test_validator_catches_extreme_hr():
    from validator.rules import DeterministicValidator
    from schemas.observation import Observation
    from schemas.enums import (VitalSignCode, UCUMUnit, ObservationStatus, SourceFormat)
    v = DeterministicValidator()
    obs = Observation(
        source_id="r1", source_format=SourceFormat.HL7_FRAGMENT,
        subject_id="S1", vs_code=VitalSignCode.HEART_RATE,
        value_numeric=210, unit=UCUMUnit.BPM,    # within Pydantic range, outside clinical range
        measured_at=datetime(2025,1,8,8,0,0),
        status=ObservationStatus.FINAL,
    )
    report = v.validate(obs)
    assert any(f.rule_id == "VS-003" for f in report.findings)
    assert report.warn_count > 0
    assert report.needs_judge        # judge SHOULD be called

def test_validator_rule_count():
    from validator.rules import DeterministicValidator
    v = DeterministicValidator()
    assert v.rule_count() >= 8


# ── End-to-end pipeline (mock extractor, judge disabled) ──────────────────────

def test_pipeline_end_to_end_vendor_json(tmp_path):
    from pipeline.run import Pipeline
    from schemas.enums import SourceFormat
    from data.synthetic import VENDOR_JSON

    p = Pipeline(use_mock_extractor=True, judge_enabled=False,
                 audit_log_path=str(tmp_path / "audit.jsonl",
                 hitl_queue_path=str(tmp_path / "hitl.jsonl")))
    results, summary = p.process_batch(SourceFormat.VENDOR_JSON, VENDOR_JSON, "v")
    assert summary.total_records == 10
    # 9 should canonicalise (1 has unrecognised code "WeirdCode")
    assert summary.canonicalised >= 8
    assert summary.dropped >= 1
    # at least one record should be flagged (the HR=285 spike)
    assert summary.flagged_warn + summary.flagged_error >= 1
    # judge disabled → no calls
    assert summary.judge_calls == 0

def test_pipeline_end_to_end_all_sources(tmp_path):
    from pipeline.run import Pipeline
    from schemas.enums import SourceFormat
    from data.synthetic import VENDOR_JSON, SITE_CRF_CSV, HL7_FRAGMENT

    p = Pipeline(use_mock_extractor=True, judge_enabled=False,
                 audit_log_path=str(tmp_path / "audit.jsonl"),
                 hitl_queue_path=str(tmp_path / "hitl.jsonl"))
    total = 0
    for fmt, raw in [(SourceFormat.VENDOR_JSON, VENDOR_JSON),
                     (SourceFormat.SITE_CRF_CSV, SITE_CRF_CSV),
                     (SourceFormat.HL7_FRAGMENT, HL7_FRAGMENT)]:
        results, summary = p.process_batch(fmt, raw, fmt.value[:6])
        total += summary.total_records
    assert total >= 28        # 10 + 10 + ~10


# ── Compliance: ALCOA++ audit trail + 21 CFR Part 11 e-signature ─────────────

def test_audit_log_chain_intact(tmp_path):
    from compliance.audit import AuditLog, AuditAction
    log = AuditLog(tmp_path / "audit.jsonl")
    e1 = log.append(source_id="r1", layer="L1", action=AuditAction.EXTRACTED,
                    actor="system:test", payload={"v": 72})
    e2 = log.append(source_id="r1", layer="L2", action=AuditAction.COMPUTED,
                    actor="system:test", payload={"v": 72})
    assert e2.prev_hash == e1.chain_hash
    ok, err = log.verify()
    assert ok and err is None

def test_audit_log_detects_tampering(tmp_path):
    from compliance.audit import AuditLog, AuditAction
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append(source_id="r1", layer="L1", action=AuditAction.EXTRACTED,
               actor="system:test", payload={"v": 72})
    # Tamper: rewrite the file with a different payload hash
    raw = path.read_text().replace('"v": 72', '"v": 99')
    path.write_text(raw)
    ok, err = AuditLog(path).verify()
    assert not ok
    assert "tamper" in err or "mismatch" in err

def test_audit_log_export_per_source(tmp_path):
    from compliance.audit import AuditLog, AuditAction
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(source_id="r1", layer="L1", action=AuditAction.EXTRACTED,
               actor="u", payload={})
    log.append(source_id="r2", layer="L1", action=AuditAction.EXTRACTED,
               actor="u", payload={})
    log.append(source_id="r1", layer="L3", action=AuditAction.VALIDATED,
               actor="u", payload={})
    events = log.export("r1")
    assert len(events) == 2     # §11.10(b) — accurate copies for inspection

def test_e_signature_round_trip():
    from compliance.audit import ESignature
    record = {"subject": "S1", "value": 72}
    sig = ESignature.sign("dr.smith@trial.org", record, meaning="approved-for-submission")
    assert sig.verify(record) is True
    # tampered record fails verification
    assert sig.verify({"subject": "S1", "value": 99}) is False

def test_pipeline_emits_audit_events(tmp_path):
    from pipeline.run import Pipeline
    from schemas.enums import SourceFormat
    from data.synthetic import VENDOR_JSON

    log_path = tmp_path / "audit.jsonl"
    p = Pipeline(use_mock_extractor=True, judge_enabled=False,
                 audit_log_path=str(log_path,
                 hitl_queue_path=str(tmp_path / "hitl.jsonl")))
    p.process_batch(SourceFormat.VENDOR_JSON, VENDOR_JSON, "v")
    # Audit log should have many events; chain must verify
    ok, err = p.audit.verify()
    assert ok, f"Chain verification failed: {err}"
    events = log_path.read_text().strip().splitlines()
    assert len(events) >= 10        # at least L0+L1 per record


def test_audit_event_captures_model_snapshot(tmp_path):
    """L1 must record model_id, prompt_hash, temperature for reproducibility."""
    import json
    from pipeline.run import Pipeline
    from schemas.enums import SourceFormat
    from data.synthetic import VENDOR_JSON

    log_path = tmp_path / "audit.jsonl"
    p = Pipeline(use_mock_extractor=True, judge_enabled=False,
                 audit_log_path=str(log_path,
                 hitl_queue_path=str(tmp_path / "hitl.jsonl")))
    p.process_batch(SourceFormat.VENDOR_JSON, VENDOR_JSON, "v")
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    l1_events = [e for e in events if e["layer"] == "L1"]
    assert len(l1_events) > 0
    snap = l1_events[0]["model_snapshot"]
    assert snap is not None
    assert snap["model_id"]
    assert snap["prompt_hash"]
    assert snap["prompt_version"] == "1.0.0"
    assert snap["temperature"] == 0.0       # determinism

def test_audit_event_records_input_and_output_excerpts(tmp_path):
    """For human review, every LLM event must store both input and output excerpts."""
    import json
    from pipeline.run import Pipeline
    from schemas.enums import SourceFormat
    from data.synthetic import VENDOR_JSON

    log_path = tmp_path / "audit.jsonl"
    p = Pipeline(use_mock_extractor=True, judge_enabled=False,
                 audit_log_path=str(log_path,
                 hitl_queue_path=str(tmp_path / "hitl.jsonl")))
    p.process_batch(SourceFormat.VENDOR_JSON, VENDOR_JSON, "v")
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    l1 = next(e for e in events if e["layer"] == "L1")
    assert l1["input_excerpt"]                    # input visible to reviewer
    assert l1["output_excerpt"]                   # output visible to reviewer
    assert l1["input_hash"]                       # cryptographically bound
    assert l1["payload_hash"]

def test_amendment_validates_reason_code(tmp_path):
    """Tracked corrections must carry prev/new/reason — reason from controlled vocab."""
    from compliance.audit import AuditLog, Amendment

    log = AuditLog(tmp_path / "audit.jsonl")
    bad = Amendment(
        source_id="r1", field_name="value_numeric",
        prev_value=72, new_value=70,
        reason_code="i_changed_my_mind",        # not in vocabulary
        reason_text="Felt like it.",
        actor="dr.smith@trial.org",
    )
    import pytest
    with pytest.raises(ValueError):
        log.record_amendment(bad)

def test_amendment_preserves_prev_value(tmp_path):
    """Original value is never overwritten — Amendment links back via prev_chain_hash."""
    import json
    from compliance.audit import AuditLog, Amendment, AuditAction

    log = AuditLog(tmp_path / "audit.jsonl")
    e1 = log.append(source_id="r1", layer="L2", action=AuditAction.COMPUTED,
                    actor="system", payload={"value_numeric": 72})
    log.record_amendment(Amendment(
        source_id="r1", field_name="value_numeric",
        prev_value=72, new_value=70,
        reason_code="transcription_error",
        reason_text="Source CRF says 70; OCR misread as 72.",
        actor="dr.smith@trial.org",
        prev_chain_hash=e1.chain_hash,
    ))
    events = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    amendment = next(e for e in events if e["action"] == "amended")
    # Prev value preserved alongside new value
    payload = json.loads(amendment["output_excerpt"]) if amendment["output_excerpt"].startswith("{") else None
    assert payload is not None
    assert payload["prev_value"] == 72
    assert payload["new_value"] == 70
    assert payload["reason_code"] == "transcription_error"
    # Chain still verifies
    ok, err = log.verify()
    assert ok, err

def test_review_packet_bundles_input_output_for_signing(tmp_path):
    """Reviewer sees raw input vs canonical output side-by-side before signing."""
    from compliance.audit import AuditLog, AuditAction, ESignature

    log = AuditLog(tmp_path / "audit.jsonl")
    raw_input = {"subj": "S1", "metric": "PULSE", "val": 72, "u": "bpm"}
    canonical = {"subject_id": "S1", "vs_code": "8867-4", "value": 72, "unit": "/min"}
    log.append(source_id="r1", layer="L1", action=AuditAction.EXTRACTED,
               actor="system", payload=canonical, input_data=raw_input)

    packet = log.build_review_packet(
        source_id="r1", raw_input=raw_input, canonical_output=canonical,
    )
    assert packet.raw_input == raw_input
    assert packet.canonical_output == canonical
    assert len(packet.audit_events) >= 1

    # Sign the review packet — record_hash binds the entire packet
    sig = ESignature.sign("dr.smith@trial.org", packet, meaning="approved-for-submission")
    assert sig.verify(packet) is True

def test_retention_metadata_is_recorded(tmp_path):
    """21 CFR Part 11 §11.10(c) — clinical records retained for ≥6 years."""
    import json
    from compliance.audit import AuditLog, AuditAction
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(source_id="r1", layer="L1", action=AuditAction.EXTRACTED,
               actor="system", payload={"v": 1}, retention_years=10)
    events = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert events[0]["retention_years"] >= 6


# ── Grounding (post-extraction verification) ──────────────────────────────────

def test_grounding_passes_for_value_in_source():
    from extractor.constrained import verify_grounding
    from schemas.observation import RawObservation
    from schemas.enums import VitalSignCode, UCUMUnit, SourceFormat
    raw = RawObservation(
        source_id="r1", source_format=SourceFormat.VENDOR_JSON,
        source_record="subject=SUBJ-001 metric=PULSE value=72 unit=bpm",
        subject_id="SUBJ-001",
        vs_code=VitalSignCode.HEART_RATE,
        value_numeric=72.0,
        unit=UCUMUnit.BPM,
    )
    assert verify_grounding(raw) == []     # value 72 IS in the source

def test_grounding_catches_hallucinated_value():
    """Even if Outlines compiles a valid schema, a numeric value not present
    in the source text is flagged as a possible hallucination."""
    from extractor.constrained import verify_grounding
    from schemas.observation import RawObservation
    from schemas.enums import VitalSignCode, UCUMUnit, SourceFormat
    raw = RawObservation(
        source_id="r1", source_format=SourceFormat.VENDOR_JSON,
        source_record="subject=SUBJ-001 metric=PULSE",   # no value in source!
        subject_id="SUBJ-001",
        vs_code=VitalSignCode.HEART_RATE,
        value_numeric=72.0,                              # hallucinated
        unit=UCUMUnit.BPM,
    )
    violations = verify_grounding(raw)
    assert len(violations) == 1
    assert "72" in violations[0]


# ── HITL routing ──────────────────────────────────────────────────────────────

def test_hitl_router_routes_validator_errors():
    from compliance.hitl import should_route_to_hitl, HITLReason
    from schemas.reports import ValidationReport, ValidationFinding, FindingSeverity
    rep = ValidationReport(observation_id="r1")
    rep.add(ValidationFinding(rule_id="VS-001", severity=FindingSeverity.ERROR,
                              field_path="x", message="missing"))
    assert should_route_to_hitl(rep, judge_report=None) == HITLReason.VALIDATOR_ERROR

def test_hitl_router_routes_judge_request():
    from compliance.hitl import should_route_to_hitl, HITLReason
    from schemas.reports import ValidationReport, JudgeReport
    rep = ValidationReport(observation_id="r1")
    judge = JudgeReport(observation_id="r1", judge_model="claude-haiku",
                        plausibility_score=0.7, suggested_action="human_review",
                        rationale="ambiguous")
    assert should_route_to_hitl(rep, judge) == HITLReason.JUDGE_REQUESTED

def test_hitl_router_low_confidence():
    from compliance.hitl import should_route_to_hitl, HITLReason
    from schemas.reports import ValidationReport, JudgeReport
    rep = ValidationReport(observation_id="r1")
    judge = JudgeReport(observation_id="r1", judge_model="claude-haiku",
                        plausibility_score=0.2, suggested_action="amend",
                        rationale="low conf")
    assert should_route_to_hitl(rep, judge) == HITLReason.JUDGE_LOW_CONFIDENCE

def test_hitl_router_no_route_when_clean():
    from compliance.hitl import should_route_to_hitl
    from schemas.reports import ValidationReport
    rep = ValidationReport(observation_id="r1")
    assert should_route_to_hitl(rep, judge_report=None) is None

def test_hitl_queue_persists_and_sorts(tmp_path):
    """Urgent items come first; ties broken by FIFO."""
    from compliance.hitl import HITLQueue, HITLReason
    q = HITLQueue(tmp_path / "hitl.jsonl")
    q.enqueue(source_id="r1", reason=HITLReason.JUDGE_REQUESTED,
              review_packet_hash="h1", priority=0)
    q.enqueue(source_id="r2", reason=HITLReason.VALIDATOR_ERROR,
              review_packet_hash="h2", priority=1)         # urgent
    items = q.pending()
    assert items[0].source_id == "r2"     # urgent first
    assert items[1].source_id == "r1"
