"""
extractor/constrained.py — Layer 1: The Lossy Parser (Concept 1 + 2).

THE CENTREPIECE OF THE WHOLE PIPELINE
======================================
This is the only file in the project that calls an LLM during steady-state
operation. Everything else is deterministic Python.

We feature TWO libraries:

  1. OUTLINES   — Token-level FSM constraint. Strongest guarantee.
                  Compiles a Pydantic model into a finite-state machine
                  and masks logits at every generation step.
                  The model PHYSICALLY CANNOT emit invalid output.

  2. INSTRUCTOR — Validation-retry. Pydantic-native, simpler API.
                  Asks the model for JSON, validates it against the schema,
                  retries if invalid. Model CAN emit invalid output but it
                  gets caught and re-asked.

When to pick which:
  - Outlines: production, strict guarantees, local models, high throughput
  - Instructor: prototypes, cloud APIs (OpenAI/Anthropic), schema evolution

Both achieve the same end goal: Pydantic-conformant output. Outlines does
it at the token level (impossible to be wrong), Instructor at the response
level (caught and retried if wrong).

For this POC we wire Outlines as the primary path and document Instructor
as an alternative. A MockExtractor lets the full pipeline run without GPU
or API keys for testing and demos.
"""

from __future__ import annotations
import re
from datetime import datetime
from pydantic import BaseModel

from parsers.sources import SourceRecord
from schemas.observation import RawObservation
from schemas.enums import VitalSignCode, UCUMUnit, ObservationStatus, SourceFormat


SYSTEM_PROMPT = """You are a clinical data parser for a decentralised clinical trial.
Extract a single vital-sign observation from the input text.

GROUNDING RULES — these are non-negotiable:
- The source text is your ONLY evidence. You have no other knowledge of the patient.
- Extract only what is EXPLICITLY in the text. Quote-equivalent grounding.
- If a field cannot be located in the source, leave it null. NEVER invent.
- Never infer values from clinical knowledge ("normal HR is ~70" is forbidden).
- Never carry context across records — each record is parsed independently.

EXTRACTION RULES:
- For vs_code: pick the LOINC code that matches the named metric.
- For unit: use UCUM units. If the source says "deg F", use [degF]; "bpm" → /min.
- Numeric values: use the exact number written; no rounding, no normalising.

Output ONLY the JSON.
"""


class ExtractionConfig(BaseModel):
    model_id:       str   = "mistralai/Mistral-7B-Instruct-v0.3"
    device:         str   = "cpu"
    max_new_tokens: int   = 512
    temperature:    float = 0.0
    use_mock:       bool  = True


# ─────────────────────────────────────────────────────────────────────────────
#  PRIMARY PATH: OUTLINES (token-level FSM constraint)
# ─────────────────────────────────────────────────────────────────────────────

class OutlinesExtractor:
    """
    The Outlines-based constrained decoder.

    EXAMPLE OF THE KEY LINE:
        generator = outlines.generate.json(model, RawObservation)

    This call compiles the RawObservation Pydantic schema into a finite-state
    machine. From that point on, the generator can ONLY produce JSON that
    deserialises into a valid RawObservation — every enum field is masked
    to its allowed values at the token level.
    """

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        self.config = config or ExtractionConfig()
        self._generator = None
        if not self.config.use_mock:
            self._init_outlines()

    def _init_outlines(self) -> None:
        try:
            import outlines
            import outlines.models as models
            import outlines.generate as generate

            model = models.transformers(self.config.model_id, device=self.config.device)

            # ★★★ THE KEY LINE — compile schema → FSM
            self._generator = generate.json(
                model, RawObservation,
                sampler=outlines.samplers.greedy(),
            )
        except ImportError as e:
            raise ImportError(
                "Outlines not installed. Run: pip install outlines transformers torch"
            ) from e

    def extract(self, record: SourceRecord) -> RawObservation:
        if self.config.use_mock:
            return _mock_extract(record)

        prompt = (
            f"<s>[INST] {SYSTEM_PROMPT}\n\n"
            f"INPUT:\n{record.raw_text}\n\n"
            f"Output the RawObservation JSON. [/INST]"
        )
        raw: RawObservation = self._generator(prompt, max_tokens=self.config.max_new_tokens)
        # Inject provenance the model can't know
        raw.source_id = record.source_id
        raw.source_format = record.source_format
        raw.source_record = record.raw_text
        return raw


# ─────────────────────────────────────────────────────────────────────────────
#  ALTERNATIVE PATH: INSTRUCTOR (validation-retry constraint)
# ─────────────────────────────────────────────────────────────────────────────

class InstructorExtractor:
    """
    Alternative implementation using the `instructor` library.

    DIFFERENCES FROM OUTLINES
    ==========================
    Outlines:   FSM at the token level. Model CANNOT emit invalid output.
    Instructor: Schema validated AFTER generation. Retries on failure.

    Both paths produce Pydantic-validated output. Outlines is stronger
    technically (impossible to be wrong); Instructor is simpler and works
    with cloud APIs out of the box.

    EXAMPLE OF THE KEY CALL (Instructor):
        client = instructor.from_anthropic(anthropic.Anthropic())
        raw_obs: RawObservation = client.messages.create(
            model="claude-haiku-4-5",
            messages=[...],
            response_model=RawObservation,
            max_retries=3,            # retry on validation failure
        )

    Wired here as a documented alternative. The POC defaults to Outlines.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic, instructor
            self._client = instructor.from_anthropic(anthropic.Anthropic())
        return self._client

    def extract(self, record: SourceRecord) -> RawObservation:
        client = self._get_client()
        raw: RawObservation = client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {"role": "user", "content": (
                    SYSTEM_PROMPT + f"\n\nINPUT:\n{record.raw_text}"
                )},
            ],
            response_model=RawObservation,
            max_retries=3,
        )
        raw.source_id = record.source_id
        raw.source_format = record.source_format
        raw.source_record = record.raw_text
        return raw


# ─────────────────────────────────────────────────────────────────────────────
#  MOCK PATH: deterministic regex (testing/demo without GPU or API)
# ─────────────────────────────────────────────────────────────────────────────

def _mock_extract(record: SourceRecord) -> RawObservation:
    """
    Regex-based mock that simulates a constrained decoder's output.

    Crucially: ONLY produces valid VitalSignCode and UCUMUnit enum values,
    just like the real constrained decoder would.
    """
    text = record.raw_text
    rd = record.raw_dict

    subject = rd.get("subject_id") or rd.get("subj") or _extract(r"subject[=:]\s*(\S+)", text)
    visit = rd.get("visit") or _extract(r"visit[=:]\s*(\S+)", text)

    # ── Vital sign code (lossy parser: leave None if uncertain) ────────────
    code_text = (rd.get("metric") or rd.get("code_name") or rd.get("parameter")
                 or rd.get("code") or "")
    vs_code = _map_vs_code(code_text + " " + text)

    # ── Numeric value ──────────────────────────────────────────────────────
    val_raw = rd.get("val") or rd.get("value") or _extract(r"value[=:]\s*([0-9.]+)", text)
    try:
        value = float(val_raw) if val_raw is not None else None
    except (ValueError, TypeError):
        value = None

    # ── Unit (lossy: keep raw text, mapped to enum if possible) ────────────
    unit_text = rd.get("u") or rd.get("unit") or _extract(r"unit[=:]\s*(\S+)", text)
    unit = _map_unit(unit_text or "")

    # ── Timestamp ──────────────────────────────────────────────────────────
    ts_raw = (rd.get("ts") or rd.get("timestamp") or rd.get("datetime")
              or _extract(r"timestamp[=:]\s*(\S+)", text))
    measured_at = _parse_ts(ts_raw)

    # ── Status ─────────────────────────────────────────────────────────────
    status_raw = rd.get("status") or "F"
    status_map = {"F": ObservationStatus.FINAL, "P": ObservationStatus.PRELIMINARY,
                  "A": ObservationStatus.AMENDED, "X": ObservationStatus.CANCELLED}
    status = status_map.get(status_raw, ObservationStatus.FINAL)

    return RawObservation(
        source_id=record.source_id,
        source_format=record.source_format,
        source_record=record.raw_text,
        subject_id=subject,
        visit_id=visit,
        vs_code=vs_code,
        vs_code_text=code_text or None,
        value_numeric=value,
        unit=unit,
        unit_text=unit_text or None,
        measured_at=measured_at,
        status=status,
    )


def _extract(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1) if m else None


def _map_vs_code(text: str) -> VitalSignCode | None:
    """
    Lossy mapping from vendor names to LOINC codes.
    Returns None if the code isn't recognised — extractor must not invent.
    """
    t = text.lower()
    if any(k in t for k in ("8867-4", "heart rate", "pulse", "hr", "heartrate")):
        return VitalSignCode.HEART_RATE
    if any(k in t for k in ("8480-6", "systolic", "sysbp", "sbp")):
        return VitalSignCode.SYSTOLIC_BP
    if any(k in t for k in ("8462-4", "diastolic", "diabp", "dbp")):
        return VitalSignCode.DIASTOLIC_BP
    if any(k in t for k in ("8310-5", "temp", "body temperature")):
        return VitalSignCode.BODY_TEMP
    if any(k in t for k in ("9279-1", "respiratory rate", "resp rate", "rr")):
        return VitalSignCode.RESP_RATE
    if any(k in t for k in ("59408-5", "spo2", "oxygen sat", "sao2")):
        return VitalSignCode.SPO2
    return None


def _map_unit(unit_text: str) -> UCUMUnit | None:
    u = unit_text.strip().lower()
    if u in ("/min", "bpm", "beats/min", "beats per minute"):
        return UCUMUnit.BPM
    if u in ("mmhg", "mm[hg]", "mm hg"):
        return UCUMUnit.MMHG
    if u in ("cel", "c", "deg c", "degc", "celsius", "°c"):
        return UCUMUnit.CELSIUS
    if u in ("[degf]", "f", "deg f", "degf", "fahrenheit", "°f"):
        return UCUMUnit.FAHRENHEIT
    if u in ("%", "percent", "pct"):
        return UCUMUnit.PERCENT
    return None


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts: return None
    s = str(ts).strip()
    # HL7-style YYYYMMDDHHMMSS
    if len(s) == 14 and s.isdigit():
        try:
            return datetime.strptime(s, "%Y%m%d%H%M%S")
        except ValueError:
            return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


__all__ = [
    "ExtractionConfig", "OutlinesExtractor", "InstructorExtractor",
    "_mock_extract", "verify_grounding",
]


# ─────────────────────────────────────────────────────────────────────────────
#  GROUNDING VERIFICATION (deterministic, post-extraction)
# ─────────────────────────────────────────────────────────────────────────────

def verify_grounding(raw: RawObservation) -> list[str]:
    """
    Deterministic check: every numeric value the LLM emitted must be locatable
    in the source text. Returns a list of grounding violations (empty list = OK).

    This is the belt-and-braces complement to constrained decoding. Outlines
    prevents structurally-invalid output; this catches semantically-invented
    output (e.g. model emits value=72 when the source says nothing about HR).

    NOTE: enums (vs_code, unit) are intentionally NOT grounding-checked here —
    they undergo legitimate vocabulary mapping (e.g. "PULSE" → "8867-4").
    Only the *numeric value* and *subject_id* must trace literally to source.
    """
    violations: list[str] = []
    src = (raw.source_record or "").lower()

    if raw.value_numeric is not None:
        # match the number with up to 4 decimal places, ignoring ".0" trailing
        v = raw.value_numeric
        candidates = {f"{v:g}", f"{v:.0f}", f"{v:.1f}", f"{v:.2f}"}
        if not any(c in src for c in candidates):
            violations.append(
                f"value_numeric={v} not found in source text — possible hallucination"
            )

    if raw.subject_id and raw.subject_id.lower() not in src:
        violations.append(
            f"subject_id={raw.subject_id} not found in source text"
        )

    return violations
