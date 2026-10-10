"""Small schemas owned by the active raw-history runtime."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FactCheckReport(StrictModel):
    """Human-readable, observation-bounded terminal report."""

    headline: str = Field(min_length=1, max_length=240)
    claim_under_review: str = Field(min_length=1, max_length=1600)
    verdict_summary: str = Field(min_length=1, max_length=1800)
    key_findings: List[str] = Field(min_length=1, max_length=5)
    evidence_summary: str = Field(min_length=1, max_length=1800)
    remaining_uncertainties: List[str] = Field(default_factory=list, max_length=6)


class FactCheckEvidenceCitation(StrictModel):
    """Optional runtime-owned citation attached to the terminal report."""

    evidence_id: str = Field(min_length=1, max_length=100)
    source_url: str = Field(default="", max_length=4000)
    source_family: str = Field(min_length=1, max_length=300)
    evidence_kind: Literal[
        "web_span",
        "image_region",
        "reference_comparison",
    ]
    relation_stance: Literal[
        "supports",
        "contradicts",
        "background",
        "unclear",
    ]
    excerpt: str = Field(min_length=1, max_length=2400)


class RawHistoryJudgmentOutput(StrictModel):
    """Provider judgment grounded directly in retained tool observations."""

    verdict: Literal["real", "fake"]
    confidence: float = Field(ge=0.0, le=1.0)
    verdict_observation_ids: List[str] = Field(default_factory=list, max_length=12)
    overall_assessment: str = Field(min_length=1, max_length=2000)
    fact_check_report: FactCheckReport


class DiscrepancyJudgment(StrictModel):
    """Canonical terminal judgment retained for the public trace contract."""

    verdict: Literal["real", "fake"]
    confidence: float = Field(ge=0.0, le=1.0)
    policy_rule_id: Literal["unified-react-v1"] = "unified-react-v1"
    selected_observation_ids: List[str] = Field(default_factory=list, max_length=40)
    verdict_observation_ids: List[str] = Field(default_factory=list, max_length=12)
    overall_assessment: str = Field(min_length=1, max_length=2000)
    fact_check_report: Optional[FactCheckReport] = None
    evidence_citations: List[FactCheckEvidenceCitation] = Field(
        default_factory=list,
        max_length=40,
    )


class InvestigationSegmentOutput(StrictModel):
    """Deterministic boundary object for one accepted ReAct action."""

    segment_summary: str = Field(default="", max_length=1200)
    action_completed: bool = True
