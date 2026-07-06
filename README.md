# Designing LLM Pipelines for Clinical Data: A Pattern for ALCOA++ and 21 CFR Part 11 Compliance

https://medium.com/towards-artificial-intelligence/designing-llm-pipelines-for-clinical-data-a-pattern-for-alcoa-and-21-cfr-part-11-compliance-84f8c91d8d28

## The LLM is one component, not the system. Notes on architecture, cost discipline, and human review from a working DCT pipeline.

---

Most teams shipping LLM features into clinical-data workflows discover the
same problem on the same timeline. The first prototype is fast and convincing
— a model reads a messy clinical note and produces a clean structured output.
Then the questions start arriving. *Can you reproduce the run from last
Tuesday? Where's the audit trail? Why did the same input give a different
output? What's the cost at one million records a day? What happens when the
model is wrong, and who is accountable?*

The prototype that answered the first question well rarely survives the rest.
Not because the underlying model is bad, but because the architecture put the
LLM in a role that doesn't fit a regulated environment: as the system, rather
than as a component within one.

I want to share a pattern that I've found holds up under those questions —
the architecture for clinical-data pipelines that need to satisfy ALCOA++
and 21 CFR Part 11, while still benefitting from modern language models. It's a careful arrangement of pieces that already exist:
constrained decoding, Pydantic schemas, deterministic validators, conditional
LLM judging, append-only audit logs, and human-in-the-loop routing. The
contribution, such as it is, is in how they fit together.

The architecture has five layers. Only two of them call an LLM, and one of
those is conditional. The result is a pipeline where roughly 85% of records
never trigger an LLM call at all, every output traces back to a named
Pydantic schema and a named Python function, and the per-record cost lands
around 100s of time below what a naive cloud-LLM approach would
produce.

The mental model that makes the rest of the architecture obvious is this:
**the LLM is a lossy parser.** A function that takes unstructured chaos and
emits structured, schema-conformant facts. Nothing more. The logic, the
math, the rules, the decisions — those are plain Python, written by people,
testable like any other code, auditable on a regulator's first ask.

What follows is the five concepts that hold the architecture together, the
compliance work that falls out of the design, and the human-review patterns
that make the whole thing safe to operate.

---

## Concept 1 — Constrained Decoding: The LLM Cannot Hallucinate

The trick: the **schema is the grammar**.

Libraries like **Outlines** and **XGrammar** compile a Pydantic model into a
finite-state machine. At every token-generation step, the FSM masks the
logits — only tokens that lead to a valid schema-conformant continuation
are allowed.

This is structurally different from prompt engineering. Prompt engineering
is *asking nicely*. Constrained decoding is *physically preventing*.

```python
import outlines
import outlines.models as models
import outlines.generate as generate
from schemas.observation import RawObservation

model = models.transformers("mistralai/Mistral-7B-Instruct-v0.3")

# ★ The key line: compile the Pydantic schema into a constrained generator.
# From here on, the model CANNOT emit invalid JSON or out-of-vocabulary enums.
generator = generate.json(model, RawObservation, sampler=outlines.samplers.greedy())

raw_obs: RawObservation = generator(prompt, max_tokens=512)
# raw_obs.vs_code is GUARANTEED to be a valid VitalSignCode enum member —
# never "PULSE" or "heart_rate" or "hr" — only "8867-4" or null.
```

If you're on a cloud API, the same idea exists at the response level:
**Instructor** uses Pydantic for retry-on-validation, and Anthropic's
`tool_use` enforces a JSON schema on tool arguments. Different mechanism,
same goal: structure is non-negotiable.

---

## Concept 2 — The Lossy Parser: Letting the Model Say "I Don't Know"

The temptation when prompting LLMs is to demand complete output. *Fill in
every field.* This is exactly what produces hallucinations.

The fix is to make the schema itself permissive on the way in:

```python
class RawObservation(BaseModel):
    """The lossy parser's output — every field is Optional."""
    subject_id:    str | None = None
    vs_code:       VitalSignCode | None = None
    value_numeric: float | None = None
    unit:          UCUMUnit | None = None
    measured_at:   datetime | None = None
    # ...everything Optional
```

Now the LLM is *allowed* to leave fields blank. And because we use
constrained decoding, when the model does fill a field, the value is
guaranteed schema-valid.

The result is **honest extraction**. A blank field is a feature, not a bug.
The downstream computation module either fills the gap deterministically or
flags it for review.

---

## Concept 3 — Compiled AI: Use the LLM to Extract, Use Python to Compute

This is where most of the cost reduction lives.

> "Use the LLM once to generate a deterministic rule-set in Python.
> Then run that code on millions of records."

In my pipeline, the LLM never:
- Looks up LOINC codes (it's a dict)
- Converts Fahrenheit to Celsius (it's `(F-32) × 5/9`)
- Checks if a heart rate of 285 is plausible (it's a range comparison)
- Detects duplicates (it's a SHA-1 fingerprint)

```python
def to_si(value: float, unit: UCUMUnit, vs_code: VitalSignCode):
    """Pure Python. Runs in microseconds. Costs $0."""
    if vs_code == VitalSignCode.BODY_TEMP and unit == UCUMUnit.FAHRENHEIT:
        return (round((value - 32) * 5.0 / 9.0, 2), UCUMUnit.CELSIUS)
    return (value, unit)
```

The LLM helped author this function once. From now on it's just code. Per
1,000 records: ~$0 in compute, deterministic output, trivially testable.

This pattern has names — *Compiled AI*, *Code Factory*, *Hybrid AI* — but the
key idea is simple: the LLM is a code generator, not a runtime.

---

## Concept 4 — The Deterministic Validator: The "Bread" of the Sandwich

A validator is the layer no auditor will ever complain about. Pure Python (obviously written by LLM itself in first place).
Each rule is a named function. Each finding cites a stable rule_id.

```python
@rule("VS-003", FindingSeverity.WARN, "value_numeric", "Heart rate sanity range")
def check_hr_range(obs: Observation, report: ValidationReport) -> None:
    if obs.vs_code == VitalSignCode.HEART_RATE:
        if not (40 <= obs.value_numeric <= 200):
            report.add(ValidationFinding(
                rule_id="VS-003",
                severity=FindingSeverity.WARN,
                field_path="value_numeric",
                message=f"HR {obs.value_numeric} outside expected range [40, 200]",
            ))
```

A regulator can re-run this validator and get bit-identical results. They
can read each rule. They can write a unit test for it. They can ask "why
was this record flagged?" and get a one-line answer with a citation.

This is the "bread" in the sandwich. The LLM goes between two layers of
deterministic code, never around them.

---

## Concept 5 — The Conditional Judge: Summoned, Not on Retainer

The judge LLM never sees clean records.

After the deterministic validator runs, it sets a `needs_judge` flag based
on whether any WARN or ERROR findings were produced. The judge fires only
on flagged records — empirically about 15% of input.

```python
def judge(self, obs: Observation, val_report: ValidationReport):
    # Cost gate: skip clean records entirely
    if not val_report.needs_judge:
        return None        # 85% of records exit here at $0

    # Constrained output even at the API level — Anthropic tool_use
    response = self.client.messages.create(
        model="claude-haiku-4-5-20251001",
        tools=[self._tool_schema()],
        tool_choice={"type": "tool", "name": "submit_judgement"},
        messages=[{"role": "user", "content": prompt}],
    )
```

For the 15% that do reach Haiku, the output is *also* constrained — via
`tool_use`. The judge cannot return prose; it must call the tool with valid
arguments matching the JSON schema.

Net cost on 1,000 records: ~150 calls × ~$0.001 = **~$0.15**.

---

## The Sandwich

```
Messy clinical input
       │
       ▼
Layer 0: Source parsers ─────────── stdlib   $0
       │
       ▼
Layer 1: Constrained extraction ─── LLM (Outlines)
       │
       ▼
Layer 2: Compiled AI computation ── pure Python   $0
       │
       ▼
Layer 3: Deterministic validator ── pure Python   $0
       │
       ├── clean? (≈85%) ─────────── done. $0.
       │
       └── flagged? (≈15%) ─────────┐
                                    ▼
Layer 4: Judge LLM (Haiku tool_use) ─ ~$0.001/record
       │
       ▼
SDTM-compliant canonical observation
```

Two LLM-using layers. One of them is conditional. Everything in between
is plain Python you can lint, test, and audit.

---

## The Numbers

| Approach | LLM calls per 1k records | Cost per 1k | Auditable? |
|---|---:|---:|:---:|
| Naive: Opus does everything | 1,000 | ~$200 | ❌ |
| Hybrid: Sonnet extract + validate | 2,000 | ~$10 | ⚠ |
| **This: Outlines + conditional judge** | **0 + ~150** | **~$0.15** | ✓ |

The 1000× isn't marketing. It falls directly out of the architecture.

---

## Grounding and Human-in-the-Loop — The Two Honest Layers

Two more pieces sit on top of the sandwich. Both make the pipeline *more
conservative*, not more clever.

### Grounding: extract what's in the source, never invent

The LLM in this pipeline has no RAG corpus, no retrieval, no external
context. Its only evidence is the source text in front of it. The system
prompt makes the rule blunt:

> The source text is your ONLY evidence. You have no other knowledge of the patient.
> Extract only what is EXPLICITLY in the text. Quote-equivalent grounding.
> Never infer values from clinical knowledge.

But prompts alone don't enforce anything. Constrained decoding handles
schema-validity; we add a *deterministic* post-extraction check that every
numeric value the LLM emitted can be located as a substring of the source:

```python
def verify_grounding(raw: RawObservation) -> list[str]:
    """Every numeric value the LLM emitted must appear in the source text.
    This catches semantic hallucinations that constrained decoding can't."""
    violations = []
    src = raw.source_record.lower()
    if raw.value_numeric is not None:
        candidates = {f"{raw.value_numeric:g}", f"{raw.value_numeric:.0f}",
                      f"{raw.value_numeric:.1f}", f"{raw.value_numeric:.2f}"}
        if not any(c in src for c in candidates):
            violations.append(f"value_numeric={raw.value_numeric} not in source")
    if raw.subject_id and raw.subject_id.lower() not in src:
        violations.append(f"subject_id={raw.subject_id} not in source")
    return violations
```

Belt and braces. Outlines prevents *structural* hallucinations (invalid
enums, malformed JSON); `verify_grounding` catches *semantic* hallucinations
(plausible values that aren't in the source). Failures generate an
`AuditAction.REJECTED` event — the canonical record is dropped, not stored.

### Human-in-the-Loop: when the machine isn't sure, it stops

The conditional judge is half the story. The other half is what happens
when the judge *itself* isn't confident. Three triggers route a record
out of automation entirely:

1. Validator returns `ERROR` → mandatory HITL (urgent priority)
2. Judge says `suggested_action="human_review"` or `plausibility_score < 0.4` → HITL
3. Validator flagged the record but the judge was unavailable → HITL

```python
def should_route_to_hitl(val_report, judge_report, threshold=0.4):
    if val_report.error_count > 0:           return HITLReason.VALIDATOR_ERROR
    if judge_report:
        if judge_report.suggested_action == "human_review":
            return HITLReason.JUDGE_REQUESTED
        if judge_report.plausibility_score < threshold:
            return HITLReason.JUDGE_LOW_CONFIDENCE
    if val_report.needs_judge and judge_report is None:
        return HITLReason.JUDGE_UNAVAILABLE
    return None
```

The `HITLQueue` is just an append-only JSONL — same shape as the audit log,
sorted by priority and FIFO. A clinical data manager picks up the next item,
opens the `ReviewPacket` (input vs output side-by-side), and either approves
(creates an `ESignature`), rejects (drops the record), or amends (creates an
`Amendment` with `prev_value` / `new_value` / `reason_code`).

Empirically: ~85% records pass through fully automated, ~15% reach the
judge LLM, ~2% reach a human. The pipeline is fast because most records
need nothing more than the deterministic layers — **and safe because the
last 2% always reach a human, never a guess.**

---

## ALCOA++ and 21 CFR Part 11 - Inherent to design

For a regulated industry, "auditable" can be a deal-breaker.
The good news is that the same architecture that makes the pipeline cheap also
makes it compliant. Both ALCOA++ and 21 CFR Part 11 reduce, in code, to one
short Python module.

### What gets recorded for every LLM-touched record

For each record passing through the pipeline, the audit log captures:

```python
AuditEvent(
    source_id      = "v-0007",                              # which observation
    layer          = "L1",                                  # which step
    action         = AuditAction.EXTRACTED,
    actor          = "system:dct_simple",                   # Attributable
    timestamp_utc  = "2026-05-03T08:00:01.234Z",            # Contemporaneous

    input_hash     = sha256(raw_input),                     # what went IN
    input_excerpt  = "subject=S1 metric=PULSE value=72…",   # for human review
    output_excerpt = '{"vs_code":"8867-4","value":72…}',    # for human review
    payload_hash   = sha256(output),                        # what came OUT

    model_snapshot = ModelSnapshot(                         # reproducibility
        model_id        = "Mistral-7B-Instruct-v0.3",
        model_version   = "weights:sha256-pinned",
        library         = "outlines==0.0.46",
        prompt_id       = "extract_vital_sign",
        prompt_version  = "1.0.0",
        prompt_hash     = sha256(rendered_prompt),
        temperature     = 0.0,                              # deterministic
        seed            = 42,
    ),

    retention_years = 7,                                    # Part 11 §11.10(c)
    prev_hash       = "<chain link to previous event>",
    chain_hash      = "<sha256 of all of the above>",
)
```

Every LLM-touched record has both its input and output preserved in the
chain. A reviewer opens the audit log and sees the raw clinical text
on the left, the canonical Pydantic structure on the right, and the
exact model+prompt+seed that made the leap.

### Tracked corrections — Part 11 §11.10(e)

If anyone changes a value after the fact, the original is *never*
overwritten. An `Amendment` is appended:

```python
log.record_amendment(Amendment(
    source_id      = "v-0007",
    field_name     = "value_numeric",
    prev_value     = 72,                              # preserved forever
    new_value      = 70,                              # the correction
    reason_code    = "transcription_error",           # controlled vocabulary
    reason_text    = "Source CRF says 70; OCR misread.",
    actor          = "dr.smith@trial.org",
    prev_chain_hash = "<links back to event being amended>",
))
```

Reason codes are a controlled vocabulary: `transcription_error`,
`data_entry_error`, `source_correction`, `investigator_review`,
`monitor_query_resolution`, `regulatory_request`, `other`. Free text
without a valid code is rejected at the API boundary.

### Side-by-side review before signing

When a clinical data manager signs a record, they don't sign blindly. The
audit log builds a `ReviewPacket` — raw input, canonical output, validator
findings, judge rationale, and the full audit chain — and the e-signature
binds the entire packet:

```python
packet = log.build_review_packet(
    source_id = "v-0007",
    raw_input = original_vendor_json,
    canonical_output = canonical_observation,
)
# Reviewer sees raw_input vs canonical_output side-by-side, reads
# the validator findings and judge rationale, then signs the whole packet.
sig = ESignature.sign(
    user_id  = "dr.smith@trial.org",
    record   = packet,                                # ← binds entire packet
    meaning  = "approved-for-submission",
)
```

If anyone tampers with either side of the packet after signing,
`sig.verify(packet)` returns False.

### The mappings

**ALCOA++ → code:**

| Principle | Implementation |
|---|---|
| **A**ttributable | `AuditEvent.actor` (user_id or system:component) |
| **L**egible | JSONL, UTF-8, `model_dump(mode="json")` |
| **C**ontemporaneous | timestamp recorded at the moment of action |
| **O**riginal | raw source preserved in `source_record` + `input_excerpt` |
| **A**ccurate | Layer 3 deterministic validator + 8 `@rule` functions |
| **C**omplete | every layer emits an event; gaps break the chain |
| **C**onsistent | SHA-256 hash chain, monotonic order |
| **E**nduring | append-only log; `retention_years ≥ 7` |
| **A**vailable | `log.export(source_id)` returns full history |

**21 CFR Part 11 → code:**

| Section | Implementation |
|---|---|
| §11.10(a) Validation | unit tests + the deterministic validator |
| §11.10(b) Accurate copies | `AuditLog.export(source_id)` |
| §11.10(c) Record protection | hash chain + retention metadata (≥6 yrs) |
| §11.10(e) Audit trail w/ change reason | `Amendment.reason_code` + `prev_value` |
| §11.50, §11.70, §11.200 E-signature | `ESignature.sign()` over `ReviewPacket` |

### One last thing: temperature 0 and seed-locking

Every LLM layer runs with `temperature=0.0`. Where the runtime supports it
(local Outlines/Mistral), we also pin a fixed integer `seed`. Together they
make the same prompt produce bit-identical output across runs — the
foundation of "Accurate" in ALCOA and §11.10(a) in Part 11. Cloud APIs that
don't expose a seed parameter are flagged in `ModelSnapshot.extra` so
reviewers know reproducibility is best-effort, not bit-identical.

The whole compliance module is **~250 lines of Python**. That's on purpose —
regulators trust code they can read.

The principle that ties it together: **traceability isn't a document, it's a
data structure.** When every layer emits a hashed event with a named actor,
input/output excerpts, model snapshot, retention metadata, and a controlled
reason code on every change, "show me your audit trail" becomes "tail this
JSONL file."

---

## A Note on Where This Sits Relative to "Agents"

The big shift on LinkedIn timelines right now is "agents." Long-running
LLM loops, tool calls, planners, reflection, evaluation. The framing has
real applications, and I'm not going to argue against it in general.

But for a regulated industry — clinical trials, medical devices,
pharmacovigilance, payer adjudication — the agent framing inverts the
authority gradient. You don't want the LLM in the driver's seat. You
want it as a component, doing one well-defined job, with deterministic
Python doing the load-bearing work and humans on the loop where the
machine isn't sure.

Most of what I've described is not new. Constrained decoding, Pydantic,
hash-chained logs, e-signatures, controlled-vocabulary amendments — all
of these exist independently. The contribution is in the arrangement: a
pattern where compliance and cost discipline are emergent properties of
the architecture, not a layer of governance bolted on after the fact.

If you're building something similar, or if your team is wrestling with
the same questions on a different problem domain, I'd be interested to
hear what's worked for you.

---

**Code:** [link to repo]
**Stack:** Pydantic v2 · Outlines · Instructor · Anthropic Haiku · pure Python

#ClinicalTrials #MedTech #LLM #ALCOAplus #CFR21Part11 #DataIntegrity #PlatformEngineering #SystemDesign
