"""Raw-history unified ReAct runtime control.

The provider interaction history and ``VerificationState.all_steps`` are the
only sources of tool-observation truth. This module deliberately does not
summarize, classify, promote, or reject successful tool results.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from eviscene.orchestrator.evidence_policy import query_policy_violation
from eviscene.orchestrator.source_access import SourceAccessPolicy
from eviscene.orchestrator.source_provenance import canonicalize_url
from eviscene.orchestrator.state import ImageOnlyRuntimeCase
from eviscene.orchestrator.tool_result import parse_tool_result
from eviscene.tools.base import BaseTool


REACT_RUNTIME_SCHEMA_VERSION = "ifv-unified-react-raw-history-v1"
UNIFIED_REACT_RUNTIME_POLICY_VERSION = "unified-react-v1"
UNIFIED_REACT_RUNTIME_STAGE = "unified_react"
MAX_REACT_ACTIONS = 24

REACT_RUNTIME_TOOLS = (
    "perceive_scene",
    "ocr_with_position",
    "current_time",
    "reverse_image_search",
    "text_image_search",
    "text_search",
    "visit",
    "compare_with_reference",
    "check_consistency",
    "analyze_visual_anomalies",
    "crop_and_inspect",
    "focused_visual_inspection",
    "count_objects",
)

# Keep the hard action budgets next to the runtime tool contract so training
# conversion, rollout execution, and offline audits cannot silently drift.
REACT_RUNTIME_TOOL_CALL_LIMITS = {
    "current_time": 1,
    "ocr_with_position": 3,
    "reverse_image_search": 2,
    "text_image_search": 6,
    "text_search": 16,
    "visit": 16,
    "compare_with_reference": 6,
    "crop_and_inspect": 4,
    "focused_visual_inspection": 2,
    "check_consistency": 3,
    "analyze_visual_anomalies": 3,
}

_INTERNAL_FIELDS = {
    "image_input",
    "image_claim",
    "retrieval_goal",
    "visual_question_id",
    "active_fact",
    "evidence_context",
    "trace_stage",
    "trace_purpose",
    "source_evidence_id",
    "source_discovery_id",
    "before_understanding_version",
    "investigation_progress",
}


def _one_line(value: Any, limit: int = 800) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


class UnifiedReactState(BaseModel):
    """Mechanical runtime state; observations live only in the raw trace."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = REACT_RUNTIME_SCHEMA_VERSION
    case_id: str = Field(min_length=1, max_length=200)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    objective: str = Field(
        default=(
            "Verify the factual content expressed by the image and decide "
            "whether the image should be labeled real or fake."
        ),
        min_length=1,
        max_length=800,
    )
    action_count: int = Field(default=0, ge=0, le=MAX_REACT_ACTIONS)
    stop_reason: str = Field(default="", max_length=200)
    finish_rationale: str = Field(default="", max_length=1200)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


def new_unified_react_runtime_state(
    runtime_case: ImageOnlyRuntimeCase,
) -> UnifiedReactState:
    return UnifiedReactState(
        case_id=runtime_case.case_id,
        image_sha256=runtime_case.image_sha256,
    )


def available_unified_react_runtime_tools(
    state: UnifiedReactState,
) -> list[str]:
    if state.action_count >= MAX_REACT_ACTIONS:
        return ["finish_investigation"]
    return [*REACT_RUNTIME_TOOLS, "finish_investigation"]


def is_unified_react_runtime_budget_action(step: Mapping[str, Any]) -> bool:
    return (
        str(step.get("stage", "")).strip() == UNIFIED_REACT_RUNTIME_STAGE
        and str(step.get("action_type", "")).strip() == "tool_call"
        and str(step.get("tool_name", "")).strip() != "finish_investigation"
    )


def _public_parameters(delegate: BaseTool) -> Dict[str, Any]:
    parameters = copy.deepcopy(delegate.parameters or {})
    properties = parameters.setdefault("properties", {})
    for name in _INTERNAL_FIELDS:
        properties.pop(name, None)
    parameters["required"] = [
        name
        for name in parameters.get("required", []) or []
        if name not in _INTERNAL_FIELDS
    ]
    name = str(delegate.name).strip()
    if name == "visit":
        properties["question"] = {
            "type": "string",
            "description": (
                "What concrete image fact or relationship this action is "
                "checking on the selected page."
            ),
        }
        properties["context"] = {
            "type": "string",
            "description": "Short relevant context from the retained tool history.",
        }
    if name == "focused_visual_inspection":
        properties["question"] = {
            "type": "string",
            "description": "One focused visual question about the original image.",
        }
        properties["expected_property"] = {
            "type": "string",
            "description": "The concrete visible property to inspect.",
        }
        properties["scope"] = {
            "type": "string",
            "enum": ["subject", "relation", "scene", "text", "integrity"],
        }
        properties["anchor_regions"] = {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
            "maxItems": 4,
        }
    parameters["additionalProperties"] = False
    return parameters


@dataclass
class RuntimeToolAdapter(BaseTool):
    """Hide internal fields and bind only execution-time transport data."""

    delegate: BaseTool = field(repr=False)
    state: UnifiedReactState = field(repr=False)
    image_path: str = ""
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = self.delegate.name
        self.description = self.delegate.description
        if self.name == "visit":
            self.description = (
                "Inspect a selected web candidate for the concrete image "
                "fact or relationship in the current question."
            )
        elif self.name == "focused_visual_inspection":
            self.description = (
                "Reinspect the original image to answer one concrete visual "
                "question raised by the retained tool history."
            )
        self.parameters = _public_parameters(self.delegate)

    def _bound_delegate(self) -> BaseTool:
        if not self.image_path or not hasattr(self.delegate, "image_path"):
            return self.delegate
        bound = copy.copy(self.delegate)
        bound.image_path = self.image_path
        return bound

    def _provider_args(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        args = {
            str(key): value
            for key, value in dict(params).items()
            if str(key) not in _INTERNAL_FIELDS
        }
        focus = _one_line(
            args.get("question")
            or args.get("context")
            or args.get("focus")
            or args.get("focus_question")
            or args.get("aspect")
            or args.get("goal")
            or args.get("queries")
            or args.get("target_object")
        ) or "Check the most relevant factual detail in the image."
        if self.name == "visit":
            args.pop("question", None)
            context = args.pop("context", "")
            args["image_claim"] = focus
            args["retrieval_goal"] = _one_line(context or focus)
        elif self.name == "focused_visual_inspection":
            args["image_input"] = self.image_path
            args.setdefault("question", focus)
            args.setdefault("expected_property", focus)
            args.setdefault("scope", "scene")
            args.setdefault("anchor_regions", [])
            args["visual_question_id"] = _stable_id(
                "visual-question",
                self.state.case_id,
                self.state.action_count,
                focus,
            )
            args["active_fact"] = focus
            args["evidence_context"] = ""
            args["trace_stage"] = UNIFIED_REACT_RUNTIME_STAGE
            args["trace_purpose"] = "image_grounded_react_observation"
        elif self.name in {
            "perceive_scene",
            "reverse_image_search",
            "check_consistency",
            "count_objects",
            "ocr_with_position",
            "crop_and_inspect",
        }:
            args["image_input"] = self.image_path
        return args

    def call(self, params: Dict[str, Any]) -> Any:
        return self._bound_delegate().call(self._provider_args(params))

    async def call_async(self, params: Dict[str, Any]) -> Any:
        delegate = self._bound_delegate()
        method = getattr(delegate, "call_async", None)
        args = self._provider_args(params)
        if callable(method):
            result = method(args)
            if hasattr(result, "__await__"):
                return await result
            return result
        import asyncio

        return await asyncio.to_thread(delegate.call, args)


@dataclass
class FinishInvestigationTool(BaseTool):
    name: str = "finish_investigation"
    description: str = (
        "End the investigation when additional available actions are unlikely "
        "to change the final factual judgment."
    )
    parameters: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "rationale": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Why the retained raw investigation history is ready "
                        "for final Judgment."
                    ),
                }
            },
            "required": ["rationale"],
            "additionalProperties": False,
        }
    )

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "success",
            "control": self.name,
            "rationale": _one_line(params.get("rationale"), 1200),
        }


def build_react_runtime_tools(
    state: UnifiedReactState,
    all_tools: Mapping[str, BaseTool],
    *,
    image_path: str = "",
    excluded_tool_names: Iterable[str] = (),
) -> list[BaseTool]:
    excluded = {str(name).strip() for name in excluded_tool_names}
    result: list[BaseTool] = []
    for name in available_unified_react_runtime_tools(state):
        if name in excluded:
            continue
        if name == "finish_investigation":
            result.append(FinishInvestigationTool())
            continue
        delegate = all_tools.get(name)
        if delegate is None:
            continue
        result.append(
            RuntimeToolAdapter(
                delegate=delegate,
                state=state,
                image_path=image_path,
            )
        )
    return result


def _requested_urls(tool_name: str, args: Mapping[str, Any]) -> list[str]:
    fields = (
        ("reference_url", "source_page_url")
        if tool_name == "compare_with_reference"
        else ("url",)
    )
    values: list[str] = []
    for field_name in fields:
        raw = args.get(field_name, [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            continue
        values.extend(
            canonicalize_url(str(value))
            for value in raw
            if canonicalize_url(str(value))
        )
    return list(dict.fromkeys(values))


def validate_react_action(
    state: UnifiedReactState,
    *,
    tool_name: str,
    tool_args: Mapping[str, Any],
    source_access_policy: SourceAccessPolicy | None = None,
    allowed_tool_names: Iterable[str] | None = None,
) -> str:
    if (
        tool_name != "finish_investigation"
        and allowed_tool_names is not None
        and tool_name not in {
            str(name).strip() for name in allowed_tool_names
        }
    ):
        return f"{tool_name} is disabled by the active tool-family configuration"
    if tool_name not in set(available_unified_react_runtime_tools(state)):
        return f"{tool_name} is not available in the current unified ReAct state"
    if tool_name == "finish_investigation":
        if not _one_line(tool_args.get("rationale")):
            return "finish_investigation requires a rationale"
        return ""
    if any(
        str(key)
        in {"claim_id", "task_id", "question_id", "__claim_id", "__question_id"}
        for key in tool_args
    ):
        return "claim/task/question IDs are not part of the unified ReAct contract"
    policy = source_access_policy or SourceAccessPolicy()
    if tool_name in {"text_search", "text_image_search"}:
        query = (
            tool_args.get("queries", "")
            if tool_name == "text_search"
            else tool_args.get("query", "")
        )
        if isinstance(query, list):
            query = query[0] if query else ""
        query = _one_line(query, 1200)
        if not query:
            return f"{tool_name} requires one non-empty query"
        violation = query_policy_violation(query, source_access_policy=policy)
        if violation:
            return f"{tool_name} query violates source policy: {violation}"
    if tool_name in {"visit", "compare_with_reference"}:
        urls = _requested_urls(tool_name, tool_args)
        if not urls:
            return f"{tool_name} requires a non-empty URL"
        if any(not policy.allows(url) for url in urls):
            return f"{tool_name} URL is blocked by the active source policy"
    if tool_name == "reverse_image_search":
        branch = str(tool_args.get("branch", "lens")).strip().lower()
        if branch not in {"lens", "semantic"}:
            return "reverse_image_search branch must be lens or semantic"
    return ""


def record_react_action(
    state: UnifiedReactState,
    *,
    tool_name: str,
    tool_args: Mapping[str, Any],
    call_id: str,
) -> Dict[str, Any]:
    """Advance mechanical control state without inspecting the tool result."""

    if tool_name == "finish_investigation":
        state.stop_reason = "model_finished"
        state.finish_rationale = _one_line(tool_args.get("rationale"), 1200)
    else:
        state.action_count = min(MAX_REACT_ACTIONS, state.action_count + 1)
    return {
        "schema_version": "ifv-react-control-update-v1",
        "function_call_id": call_id,
        "tool_name": tool_name,
        "action_count": state.action_count,
        "stop_reason": state.stop_reason,
    }


def render_react_runtime_context(state: UnifiedReactState) -> str:
    payload = {
        "phase": "unified_react_investigation",
        "objective": state.objective,
        "original_image": {
            "attached_to_this_request": True,
            "note": "Use the pixels together with the retained raw tool history.",
        },
        "observation_delivery": (
            "Every prior tool result remains verbatim in the provider interaction "
            "history. No runtime reducer summarizes or classifies those results."
        ),
        "budget": {
            "actions_used": state.action_count,
            "actions_remaining": max(0, MAX_REACT_ACTIONS - state.action_count),
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _step_value(step: Any, name: str, default: Any = None) -> Any:
    if isinstance(step, Mapping):
        return step.get(name, default)
    return getattr(step, name, default)


def _step_metadata(step: Any) -> Mapping[str, Any]:
    value = _step_value(step, "metadata", {})
    return value if isinstance(value, Mapping) else {}


def _observation_locator(step: Any) -> Dict[str, Any] | None:
    stage_name = _step_value(step, "stage_name", _step_value(step, "stage", ""))
    if str(stage_name).strip() != UNIFIED_REACT_RUNTIME_STAGE:
        return None
    if str(_step_value(step, "action_type", "")).strip() != "tool_call":
        return None
    tool_name = str(_step_value(step, "tool_name", "")).strip()
    if not tool_name or tool_name == "finish_investigation":
        return None
    call_id = str(_step_metadata(step).get("function_call_id", "")).strip()
    if not call_id:
        return None
    raw_result = str(_step_value(step, "tool_result", ""))
    try:
        _payload, succeeded = parse_tool_result(raw_result)
        status = "success" if succeeded else "error"
    except Exception:
        status = "malformed"
    tool_args = _step_value(step, "tool_args", {})
    if not isinstance(tool_args, Mapping):
        tool_args = {}
    query = tool_args.get("queries", tool_args.get("query", ""))
    if isinstance(query, list):
        query = query[0] if query else ""
    return {
        "observation_id": call_id,
        "function_call_id": call_id,
        "tool_name": tool_name,
        "status": status,
        "query": _one_line(query, 1200),
        "source_urls": _requested_urls(tool_name, tool_args)[:4],
    }


def compile_react_judgment_basis(
    state: UnifiedReactState,
    steps: Sequence[Any],
) -> Dict[str, Any]:
    observations = [
        locator
        for step in steps
        if (locator := _observation_locator(step)) is not None
    ]
    observation_ids = [
        item["observation_id"]
        for item in observations
        if item["status"] == "success"
    ]
    return {
        "schema_version": "ifv-raw-history-judgment-basis-v1",
        "decision_mode": "bounded_binary_judgment",
        "objective": state.objective,
        "observation_ids": observation_ids,
        "observations": observations,
        "action_count": state.action_count,
        "stop_reason": state.stop_reason,
        "finish_rationale": state.finish_rationale,
    }


def render_react_judgment_context(
    state: UnifiedReactState,
    basis: Mapping[str, Any],
) -> str:
    payload = {
        "stage": "final_judgment",
        "objective": state.objective,
        "original_image": {
            "attached_to_this_request": True,
            "instruction": (
                "Use the image only with observations already present in the "
                "retained interaction history. Do not start a new investigation."
            ),
        },
        "observation_locator": list(basis.get("observations", [])),
        "investigation_history": (
            "The complete raw tool history is retained in this interaction and "
            "is the only observation record. Successful empty searches are valid "
            "observations of no matches for the submitted query; tool errors are "
            "limitations, not factual evidence."
        ),
        "output_requirements": {
            "verdict": ["real", "fake"],
            "report": [
                "headline",
                "claim_under_review",
                "verdict_summary",
                "key_findings",
                "evidence_summary",
                "remaining_uncertainties",
            ],
            "verdict_observation_ids": (
                "List only successful observation_id values from "
                "observation_locator that the verdict actually relies on."
            ),
            "rules": [
                "Keep the claim faithful to what the image expresses.",
                "Use raw tool observations without inventing facts.",
                "Visible artifacts or image quality alone are not a factual verdict.",
                "Search result pages are leads unless returned content directly answers the question.",
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
