# DCT Simple — The 5-Layer LLM Sandwich

A simplified, article-ready clinical trial data pipeline that demonstrates
how to tame an LLM into emitting deterministic, schema-conformant data
from messy clinical inputs — at ~170× lower cost than naive cloud-LLM
pipelines.

## Domain

**Decentralised Clinical Trial (DCT) device data quality & SDTM reconciliation.**

Three messy formats arrive from trial sites:
- **Vendor JSON** — wearable batch dumps (inconsistent codes & units)
- **Site CRF CSV** — investigator-entered values (free text creeping into structured fields)
- **HL7 v2 fragments** — device feeds (clean codes, lots of noise)

All must converge to **SDTM-compliant canonical observations** for FDA / EMA
submission. The hard part isn't the LLM — it's keeping the LLM in its lane.

## The 5 Concepts → The 5 Layers

| Concept | Layer | File | What it does |
|---|---|---|---|
| **1. Constrained Decoding** | Layer 1 | `extractor/constrained.py` | Outlines compiles a Pydantic schema → FSM. The LLM cannot emit invalid enum values. Ever. |
| **2. The Lossy Parser** | Schema | `schemas/observation.py` | Every extracted field is `Optional`. Honest extraction, never fabrication. |
| **3. Compiled AI** | Layer 2 | `computation/engine.py` | LOINC dict, UCUM math, range checks, fingerprint hash. Zero LLM. |
| **4. Deterministic Validator** | Layer 3 | `validator/rules.py` | 8 rules, each a named Python function with a stable rule_id. The "bread" of the sandwich. |
| **5. Conditional Judge** | Layer 4 | `judge/llm_judge.py` | Haiku via `tool_use` — fires only on WARN/ERROR. ~85% of records cost $0. |

## The Sandwich

```
Vendor JSON / CRF CSV / HL7 fragment   ← messy in
        │
        ▼
[Layer 0] Source parsers ─────────────── pure Python ($0)
        │
        ▼
[Layer 1] Constrained extraction (LLM) ── Outlines / Instructor
        │   ↓
        ↓   produces RawObservation (lossy, all Optional)
        ▼
[Layer 2] Computation module ──────────── pure Python ($0)
        │   • LOINC mapping (dict)
        │   • UCUM conversion (math)
        │   • Range checks (rules)
        │   • Duplicate detection (hash)
        ↓
        ▼
[Layer 3] Deterministic validator ─────── pure Python ($0)
        │   8 named rule functions
        │   ↓
        │   ValidationReport
        ▼
   ╔════════════════════════╗
   ║ needs_judge?           ║
   ║   no  → done ($0)      ║   ← ~85% exit here
   ║   yes → Layer 4        ║
   ╚════════════════════════╝
        │
        ▼
[Layer 4] Conditional judge LLM ───────── Haiku tool_use
        │   ↓
        │   JudgeReport
        ▼
SDTM-compliant Observation ← canonical out
```

## Cost Math (per 1,000 records)

| Approach | LLM calls | Cost |
|---|---:|---:|
| **Naive: Opus does everything** | 1,000 | **~$200.00** |
| **Hybrid: Sonnet extract + Sonnet validate** | 2,000 | ~$10.00 |
| **This pipeline (Outlines + judge on flag)** | 0 + ~150 | **~$0.15** |

The 1000× number isn't marketing — it falls directly out of the architecture.
The LLM is asked to do exactly one job: parse text into a Pydantic model.
Everything else is Python.

## Quickstart

```bash
# Install
pip install -e .

# Run the demo (no GPU, no API key required — uses mock extractor)
python scripts/run_demo.py

# Run with judge LLM enabled (requires ANTHROPIC_API_KEY)
python scripts/run_demo.py --judge

# Run tests
pytest tests/
```

## Project Layout

```
dct_simple/
├── schemas/
│   ├── enums.py            ← Concept 1: finite vocabularies for FSM
│   ├── observation.py      ← Concept 2: RawObservation (lossy) + Observation (canonical)
│   └── reports.py          ← ValidationReport + JudgeReport
│
├── parsers/sources.py      ← Layer 0: 3 source format parsers (stdlib only)
│
├── extractor/constrained.py ← Layer 1: OutlinesExtractor + InstructorExtractor + Mock
│
├── computation/engine.py   ← Layer 2: Compiled AI — LOINC, UCUM, range, dedup, LOCF
│
├── validator/rules.py      ← Layer 3: 8 @rule-decorated functions, the "bread"
│
├── judge/llm_judge.py      ← Layer 4: Conditional Haiku via tool_use
│
├── compliance/
│   ├── audit.py            ← ALCOA++ + 21 CFR Part 11 audit trail
│   └── hitl.py             ← Human-in-the-Loop queue + routing
│
├── pipeline/run.py         ← The 5-layer sandwich orchestrator
│
├── data/synthetic.py       ← 30 synthetic records (3 formats, 2 subjects)
│
├── scripts/run_demo.py     ← Rich CLI showing each layer's output
│
└── tests/test_all.py       ← Unit tests, all run without GPU or API key
```

## Why This Pattern Wins for Clinical Data

1. **Auditability** — every output traces to a named rule. Show the validator code to a regulator.
2. **Reproducibility** — same input → same output. The deterministic layers don't drift.
3. **Cost discipline** — the LLM is summoned, not on retainer.
4. **Schema evolution** — change a Pydantic field; constrained decoding adapts automatically.
5. **Lossy is honest** — `None` is allowed; fabrication is not.

## Regulatory Compliance: ALCOA++ and 21 CFR Part 11

The architecture is designed around the FDA / EMA / WHO data-integrity
framework. Every LLM-touched record carries forward:

- **What went in** — `input_hash` + `input_excerpt` (for human review)
- **What came out** — `payload_hash` + `output_excerpt` (for human review)
- **Which model produced it** — `ModelSnapshot` with `model_id`,
  `model_version`, `library`, `prompt_id`, `prompt_version`, `prompt_hash`,
  `temperature`, and `seed`
- **Retention** — `retention_years` default 7 (Part 11 §11.10(c) minimum 6)

Tracked corrections use `Amendment` records — originals are never
overwritten. Each amendment carries `prev_value`, `new_value`, an
`actor`, and a controlled-vocabulary `reason_code`.

Reviewers sign a `ReviewPacket` that bundles raw input + canonical output
+ validator findings + judge rationale; the e-signature binds the *whole
packet*, so tampering with either side breaks `sig.verify()`.

| ALCOA++ principle | How the pipeline satisfies it |
|---|---|
| **A** Attributable | `AuditEvent.actor` |
| **L** Legible | JSONL audit log, UTF-8 |
| **C** Contemporaneous | `timestamp_utc` at the moment of action |
| **O** Original | `source_record` + `input_excerpt` preserved |
| **A** Accurate | Layer 3 deterministic validator + 8 `@rule` functions |
| **+** Complete | Every layer emits an event; chain breaks if one is missing |
| **+** Consistent | SHA-256 hash chain links events in immutable order |
| **+** Enduring | Append-only JSONL; `retention_years ≥ 7` |
| **+** Available | `audit_log.export(source_id)` returns full history |

| Part 11 § | Implementation |
|---|---|
| §11.10(a) Validation | `tests/test_all.py` + Layer 3 validator |
| §11.10(b) Accurate copies | `AuditLog.export(source_id)` |
| §11.10(c) Record protection + retention | hash chain + `retention_years ≥ 6` |
| §11.10(e) Audit trail with change reason | `Amendment` + controlled `reason_code` |
| §11.50, §11.70, §11.200 E-signature | `ESignature.sign()` over `ReviewPacket` |

**Determinism:** every LLM layer runs with `temperature=0.0` and (where
supported) a fixed integer `seed`. Same prompt → bit-identical output —
the foundation of "Accurate" in ALCOA and §11.10(a) Validation.

**Grounding:** the LLM has no RAG corpus, no retrieval, no external context.
The source text is its only evidence. After extraction, `verify_grounding()`
checks that every numeric value the LLM emitted appears in the source. Catches
semantic hallucinations that constrained decoding can't.

**Human-in-the-loop:** when the validator says ERROR, or the judge requests
review, or judge confidence drops below 0.4, the record routes to a human
queue (`compliance/hitl.py`). ~85% of records pass through fully automated;
~15% reach the judge LLM; ~2% reach a human. The pipeline is fast because
most records skip the LLM, and safe because the unsure ones always reach a
human, never a guess.

The whole compliance module is **~560 lines of Python** (`compliance/audit.py` + `compliance/hitl.py`).
Regulators trust code they can read.

## The Two Constrained-Decoding Libraries

| Aspect | Outlines (primary) | Instructor (alternative) |
|---|---|---|
| Constraint level | **Token-level FSM** | Response-level retry |
| Can model emit invalid output? | **No, structurally impossible** | Yes, but caught and re-asked |
| Best for | Local models, strict guarantees, high throughput | Cloud APIs (OpenAI/Anthropic), prototypes |
| API surface | `outlines.generate.json(model, Schema)` | `client.create(response_model=Schema)` |

Both produce Pydantic-conformant output. Pick Outlines when you need a cast-iron
guarantee; pick Instructor when you want a 5-minute integration with a cloud API.

## Read Next

- `extractor/constrained.py` for the actual constrained-decoding code (both libraries side-by-side)
- `computation/engine.py` for the "Compiled AI" pattern in concrete form
- `validator/rules.py` for what auditable validation looks like
- The article (`ARTICLE.md`) walks through all 5 concepts with prose
