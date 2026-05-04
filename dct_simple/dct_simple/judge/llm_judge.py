"""
judge/llm_judge.py — Layer 4: The Conditional Judge LLM (Concept 5).

THE INSIGHT: SUMMONED, NOT ON RETAINER
========================================
The judge LLM does NOT see every record. It only sees records flagged
WARN or ERROR by the deterministic validator. ~85% of records skip the
judge entirely.

This is what keeps cost flat:
  - Clean records: $0  (no LLM call at all)
  - Flagged records: ~$0.001 each (Haiku via tool_use)

OUTPUT IS ALSO CONSTRAINED
===========================
We don't ask the judge "what do you think?" and parse its prose. We give
it a tool with a JSON schema; the model MUST call the tool with valid
arguments. This is the cloud-API equivalent of constrained decoding —
structure enforced at the API level instead of the token level.
"""

from __future__ import annotations
from datetime import datetime

from schemas.observation import Observation
from schemas.reports import (
    ValidationReport, JudgeReport, FindingSeverity,
)


_HAIKU_INPUT_PRICE_PER_1M  = 1.0
_HAIKU_OUTPUT_PRICE_PER_1M = 5.0


class JudgeLLM:
    """Conditional judge — fires only on WARN/ERROR validation findings."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        enabled: bool = True,
        only_on_warn: bool = True,
    ) -> None:
        self.model = model
        self.enabled = enabled
        self.only_on_warn = only_on_warn
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def judge(
        self, obs: Observation, val_report: ValidationReport,
    ) -> JudgeReport | None:
        # Cost gate: skip clean records entirely
        if not self.enabled:
            return None
        if self.only_on_warn and not val_report.needs_judge:
            return None

        prompt = self._build_prompt(obs, val_report)
        tool_schema = self._build_tool_schema()

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=512,
                tools=[tool_schema],
                tool_choice={"type": "tool", "name": "submit_judgement"},
                messages=[{"role": "user", "content": prompt}],
            )
            tool_block = next(
                (b for b in response.content if b.type == "tool_use"), None
            )
            if not tool_block:
                return None
            usage = response.usage
            cost = (
                (usage.input_tokens  / 1_000_000) * _HAIKU_INPUT_PRICE_PER_1M
              + (usage.output_tokens / 1_000_000) * _HAIKU_OUTPUT_PRICE_PER_1M
            )
            out = tool_block.input
            return JudgeReport(
                observation_id=obs.source_id,
                judge_model=self.model,
                plausibility_score=float(out.get("plausibility_score", 0.5)),
                needs_human_review=bool(out.get("needs_human_review", False)),
                rationale=str(out.get("rationale", "")),
                suggested_action=str(out.get("suggested_action", "human_review")),
                judge_cost_usd=round(cost, 6),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
        except Exception:
            return None

    @staticmethod
    def _build_prompt(obs: Observation, val_report: ValidationReport) -> str:
        findings_summary = "\n".join(
            f"  [{f.severity.value}] {f.rule_id}: {f.message}"
            for f in val_report.findings[:8]
        ) or "  (no findings)"
        return (
            f"You are a clinical data manager reviewing a flagged observation.\n\n"
            f"OBSERVATION:\n"
            f"  subject:     {obs.subject_id}\n"
            f"  vital_sign:  {obs.vs_code.name} ({obs.vs_code.value})\n"
            f"  value:       {obs.value_numeric} {obs.unit.value}\n"
            f"  measured_at: {obs.measured_at}\n"
            f"  source:      {obs.source_format.value}\n\n"
            f"VALIDATOR FINDINGS:\n{findings_summary}\n\n"
            f"Decide: is this observation plausible despite the findings? "
            f"Should it be accepted, rejected, amended, or sent for human review?"
        )

    @staticmethod
    def _build_tool_schema() -> dict:
        """The cloud-API equivalent of constrained decoding — structure
        enforced at the API level instead of the token level."""
        return {
            "name": "submit_judgement",
            "description": "Submit your judgement on this flagged observation",
            "input_schema": {
                "type": "object",
                "required": [
                    "plausibility_score", "needs_human_review",
                    "rationale", "suggested_action",
                ],
                "properties": {
                    "plausibility_score":  {
                        "type": "number", "minimum": 0, "maximum": 1,
                        "description": "0=clearly bad, 1=clearly fine",
                    },
                    "needs_human_review":  {"type": "boolean"},
                    "rationale":           {"type": "string"},
                    "suggested_action":    {
                        "type": "string",
                        "enum": ["accept", "reject", "amend", "human_review"],
                    },
                },
            },
        }


__all__ = ["JudgeLLM"]
