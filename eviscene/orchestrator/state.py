"""Canonical inputs and persisted state for the raw-history runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eviscene.orchestrator.investigation_models import DiscrepancyJudgment
from eviscene.redaction import sanitize_for_persistence


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


TEXT_ROLE_VALUES = (
    "scene_text",
    "overlay_text",
    "watermark",
    "caption",
    "identity_label",
    "claim_text",
    "unknown",
    "not_applicable",
)
TextRole = Literal[
    "scene_text",
    "overlay_text",
    "watermark",
    "caption",
    "identity_label",
    "claim_text",
    "unknown",
    "not_applicable",
]


class ImageOnlyRuntimeCase(StrictModel):
    """The complete public v0.3 runtime input."""

    case_id: str = Field(min_length=1, max_length=200)
    image_path: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Entity(StrictModel):
    """A literal entity reported by image perception."""

    name: str = ""
    entity_type: str = ""
    bbox: List[float] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    attributes: Dict[str, str] = Field(default_factory=dict)
    text_role: TextRole = "not_applicable"

    @model_validator(mode="after")
    def validate_bbox(self) -> "Entity":
        if not self.bbox:
            return self
        if len(self.bbox) != 4:
            raise ValueError("entity bbox must be [x1,y1,x2,y2]")
        x1, y1, x2, y2 = self.bbox
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("entity bbox must be normalized and non-empty")
        return self


class TextRegion(StrictModel):
    """A literal OCR observation with normalized pixel geometry."""

    text: str = ""
    bbox_quad: List[List[float]] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    language: str = "unknown"
    text_role: TextRole = "unknown"


class PerceptionRelation(StrictModel):
    """A literal visible relation reported by image perception."""

    subject: str = Field(default="", max_length=160)
    predicate: str = Field(default="", max_length=160)
    object: str = Field(default="", max_length=160)
    description: str = Field(default="", max_length=600)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class PerceptionReport(StrictModel):
    """Merged Gemini perception and deterministic OCR output."""

    entities: List[Entity] = Field(default_factory=list)
    relations: List[PerceptionRelation] = Field(default_factory=list)
    notable_details: List[str] = Field(default_factory=list, max_length=16)
    uncertainties: List[str] = Field(default_factory=list, max_length=8)
    text_regions: List[TextRegion] = Field(default_factory=list)
    scene_description: str = ""
    image_type: str = "photo"


@dataclass
class VerificationState:
    """Mechanical runtime state persisted with the raw provider history."""

    image_path: str = ""
    image_id: str = ""
    runtime_case: Optional[ImageOnlyRuntimeCase] = None
    input_mode: str = "image_only"
    decision_policy_version: str = "unified-react-v1"
    investigation_state: Optional[Any] = None
    judgment: Optional[DiscrepancyJudgment] = None
    all_steps: List[Any] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    total_tool_calls: int = 0
    total_tool_subcalls: int = 0
    tool_subcalls_by_kind: Dict[str, int] = field(default_factory=dict)
    llm_api_calls: int = 0
    token_usage: Dict[str, int] = field(
        default_factory=lambda: {"prompt": 0, "completion": 0, "thought": 0}
    )
    termination: str = ""
    errors: List[str] = field(default_factory=list)
    tool_health: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    runtime_store: Optional[Any] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        steps_data: List[Dict[str, Any]] = []
        for step in self.all_steps:
            step_dict = {
                "round": getattr(step, "round", 0),
                "stage": (
                    getattr(step, "stage_name", "")
                    or getattr(step, "metadata", {}).get("stage", "")
                ),
                "action_type": getattr(step, "action_type", ""),
                "tool_name": getattr(step, "tool_name", ""),
                "tool_args": getattr(step, "tool_args", {}),
                "tool_result": getattr(step, "tool_result", ""),
                "tokens": getattr(step, "tokens", {}),
            }
            thought = getattr(step, "thought", "")
            if thought:
                step_dict["thought"] = thought
            output = getattr(step, "output", None)
            if output is not None:
                step_dict["output"] = output
            metadata = getattr(step, "metadata", None)
            if metadata:
                step_dict["metadata"] = metadata
            steps_data.append(step_dict)

        return sanitize_for_persistence(
            {
                "image_path": self.image_path,
                "image_id": self.image_id,
                "runtime_case": (
                    self.runtime_case.model_dump() if self.runtime_case else None
                ),
                "input_mode": self.input_mode,
                "decision_policy_version": self.decision_policy_version,
                "investigation_state": (
                    self.investigation_state.model_dump(mode="json")
                    if self.investigation_state
                    else None
                ),
                "judgment": (
                    self.judgment.model_dump(mode="json")
                    if self.judgment
                    else None
                ),
                "all_steps": steps_data,
                "stage_timings": self.stage_timings,
                "total_tool_calls": self.total_tool_calls,
                "total_tool_subcalls": self.total_tool_subcalls,
                "tool_subcalls_by_kind": self.tool_subcalls_by_kind,
                "llm_api_calls": self.llm_api_calls,
                "token_usage": self.token_usage,
                "termination": self.termination,
                "errors": self.errors,
                "tool_health": self.tool_health,
                "runtime_store": (
                    self.runtime_store.descriptor
                    if self.runtime_store is not None
                    else None
                ),
            }
        )
