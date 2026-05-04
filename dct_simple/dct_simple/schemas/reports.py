"""schemas/reports.py — ValidationReport and JudgeReport schemas."""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class FindingSeverity(str, Enum):
    INFO  = "INFO"
    WARN  = "WARN"
    ERROR = "ERROR"


class ValidationFinding(BaseModel):
    rule_id:    str
    severity:   FindingSeverity
    field_path: str
    message:    str
    expected:   str | None = None
    actual:     str | None = None


class ValidationReport(BaseModel):
    observation_id: str
    validated_at:   datetime = Field(default_factory=datetime.utcnow)
    findings:       list[ValidationFinding] = Field(default_factory=list)
    is_valid:       bool = True
    error_count:    int = 0
    warn_count:     int = 0
    info_count:     int = 0

    def add(self, finding: ValidationFinding) -> None:
        self.findings.append(finding)
        if finding.severity == FindingSeverity.ERROR:
            self.error_count += 1
            self.is_valid = False
        elif finding.severity == FindingSeverity.WARN:
            self.warn_count += 1
        else:
            self.info_count += 1

    @property
    def needs_judge(self) -> bool:
        """True if the judge LLM should review this record."""
        return self.warn_count > 0 or self.error_count > 0


class JudgeReport(BaseModel):
    observation_id:     str
    judged_at:          datetime = Field(default_factory=datetime.utcnow)
    judge_model:        str
    plausibility_score: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool = False
    rationale:          str
    suggested_action:   str   # "accept" | "reject" | "amend" | "human_review"
    judge_cost_usd:     float = 0.0
    input_tokens:       int = 0
    output_tokens:      int = 0


__all__ = [
    "FindingSeverity", "ValidationFinding", "ValidationReport", "JudgeReport",
]
