# -*- coding: utf-8 -*-
"""Provider-neutral visual consistency diagnostic tool."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from eviscene.integrations.gemini import (
    RUNTIME_METRICS_KEY,
    exception_runtime_metrics,
    normalize_json_schema,
)
from eviscene.tools.base import BaseTool


CHECK_TYPES = (
    "physical_consistency",
    "logical_consistency",
    "all",
)
_LEGACY_CHECK_TYPE_ALIASES = {
    "ai_generation": "all",
    "manipulation": "all",
}

ShortRequiredText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]
RequiredText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1200),
]
EntityText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=160),
]
NotesText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, max_length=1200),
]


class _VisualAnomaly(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: ShortRequiredText
    region: ShortRequiredText
    phenomenon: RequiredText
    reasoning: RequiredText
    severity: int = Field(ge=0, le=100)
    anomaly_type: Literal[
        "physical_inconsistency",
        "logical_inconsistency",
        "relation_mismatch",
        "text_mismatch",
    ] = Field(alias="type")
    entities_involved: list[EntityText] = Field(min_length=1, max_length=12)


class _VisualAnomalyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    anomalies: list[_VisualAnomaly] = Field(max_length=12)
    target_relation_status: Literal["observed", "not_observed", "ambiguous"]
    confidence: float = Field(ge=0.0, le=1.0)
    notes: NotesText


VISUAL_ANOMALY_RESPONSE_SCHEMA = normalize_json_schema(
    _VisualAnomalyResponse.model_json_schema(by_alias=True),
    require_all_properties=True,
)
VISUAL_ANOMALY_MAX_OUTPUT_TOKENS = 8192
VISUAL_ANOMALY_SYSTEM_INSTRUCTION = (
    "You are an expert visual-consistency analyst. Follow the supplied JSON schema "
    "exactly and return only one JSON object without markdown or commentary."
)


TARGETED_ANALYSIS_PROMPT = """\
You are an expert visual-consistency analyst. Inspect this image for concrete,
locatable visual properties that bear on the supplied target relation.

## Focus Areas
{focus_areas}

## Context
{context}

## Check Type: {check_type}

## Instructions
For each focus area:
1. Identify the entities involved.
2. Describe the visible relationship, geometry, text, or physical configuration.
3. When the observed configuration differs from the expected target relation,
   describe that difference precisely.
4. Explain the observation using visible physics, anatomy, geometry, or image
   structure.
5. Score severity from 0 to 100.

## Output Format
Return exactly one JSON object:
{{
  "anomalies": [
    {{
      "name": "short descriptive name",
      "region": "where in the image",
      "phenomenon": "visible issue",
      "reasoning": "why it is anomalous",
      "severity": 0,
      "type": "physical_inconsistency | logical_inconsistency | relation_mismatch | text_mismatch",
      "entities_involved": ["entity A", "entity B"]
    }}
  ],
  "target_relation_status": "observed | not_observed | ambiguous",
  "confidence": 0.0,
  "notes": "additional target-relation observations"
}}

Report only concrete observations that are clearly supported by the image.
"""


BROAD_SCAN_PROMPT = """\
You are an expert visual-consistency analyst. Perform a broad scan of the image
for concrete relationships and physical configurations relevant to the target.

## Analysis Steps
1. Inventory major entities, text, and structures.
2. Check internal consistency of each entity.
3. Check relationships between interacting entities.
4. Check perspective, lighting, and background structure where they bear on a
   visible target relation.

## Check Type: {check_type}

## Output Format
Return exactly one JSON object:
{{
  "anomalies": [
    {{
      "name": "short descriptive name",
      "region": "where in the image",
      "phenomenon": "visible issue",
      "reasoning": "why it is anomalous",
      "severity": 0,
      "type": "physical_inconsistency | logical_inconsistency | relation_mismatch | text_mismatch",
      "entities_involved": ["entity A", "entity B"]
    }}
  ],
  "target_relation_status": "observed | not_observed | ambiguous",
  "confidence": 0.0,
  "notes": "additional target-relation observations"
}}

Report only concrete observations that are clearly supported by the image.
"""


@dataclass
class VisualAnomalyTool(BaseTool):
    """Analyze images for visual anomalies."""

    name: str = "analyze_visual_anomalies"
    description: str = (
        "Inspect the image for concrete visual inconsistencies in a supplied "
        "entity, relationship, text value, or physical configuration. This is "
        "a diagnostic observation tool; its output does not own the verdict."
    )
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "focus_areas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific areas or aspects to inspect. Leave empty for a broad scan.",
            },
            "context": {
                "type": "string",
                "description": "Optional context from other tools to guide the analysis.",
            },
            "check_type": {
                "type": "string",
                "enum": list(CHECK_TYPES),
                "description": "Which anomaly family to emphasize.",
            },
        },
        "required": [],
    })

    client: Any = None
    provider: str = "gemini"
    model_name: str = "gemini-3.7-flash"
    image_path: str = ""

    def call(self, params: Dict[str, Any]) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.call_async(params))

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            future = pool.submit(asyncio.run, self.call_async(params))
            return future.result(timeout=120)

    async def call_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client:
            return {
                "status": "error",
                "error": "VLM client not configured for visual anomaly analysis.",
            }

        raw_focus_areas = params.get("focus_areas", [])
        if not isinstance(raw_focus_areas, list) or any(
            not isinstance(item, str) or not item.strip() for item in raw_focus_areas
        ):
            return {
                "status": "error",
                "error": "focus_areas must be an array of non-empty strings.",
            }
        focus_areas = [item.strip() for item in raw_focus_areas]

        context = params.get("context", "")
        if not isinstance(context, str):
            return {"status": "error", "error": "context must be a string."}
        context = context.strip()

        check_type = params.get("check_type", "all")
        if (
            check_type not in CHECK_TYPES
            and check_type not in _LEGACY_CHECK_TYPE_ALIASES
        ):
            return {
                "status": "error",
                "error": f"check_type must be one of: {', '.join(CHECK_TYPES)}.",
            }

        create_image_json = getattr(self.client, "create_image_json", None)
        if not callable(create_image_json):
            return {
                "status": "error",
                "error": (
                    "VLM client does not support structured single-image analysis."
                ),
            }

        normalized_check_type = _LEGACY_CHECK_TYPE_ALIASES.get(
            check_type,
            check_type,
        )
        if focus_areas:
            prompt = TARGETED_ANALYSIS_PROMPT.format(
                focus_areas="\n".join(f"- {item}" for item in focus_areas),
                context=context or "No additional context provided.",
                check_type=normalized_check_type,
            )
        else:
            prompt = BROAD_SCAN_PROMPT.format(check_type=normalized_check_type)
            if context:
                prompt += f"\n\n## Additional Context\n{context}"

        try:
            parsed = await asyncio.to_thread(
                create_image_json,
                system_prompt=VISUAL_ANOMALY_SYSTEM_INSTRUCTION,
                user_text=prompt,
                image_input=self.image_path,
                max_tokens=VISUAL_ANOMALY_MAX_OUTPUT_TOKENS,
                model_name=self.model_name,
                temperature=0.0,
                response_schema=VISUAL_ANOMALY_RESPONSE_SCHEMA,
            )
            if not isinstance(parsed, Mapping):
                raise TypeError("VLM response must be a JSON object.")
            response = dict(parsed)
            raw_runtime_metrics = response.pop(RUNTIME_METRICS_KEY, {})
            runtime_metrics = (
                dict(raw_runtime_metrics)
                if isinstance(raw_runtime_metrics, Mapping)
                else {}
            )
        except Exception as exc:
            error = {
                "status": "error",
                "error": self._error_message("Visual anomaly request failed", exc),
            }
            metrics = exception_runtime_metrics(exc)
            if metrics:
                error[RUNTIME_METRICS_KEY] = metrics
            return error

        try:
            result = _VisualAnomalyResponse.model_validate(response, strict=True)
        except ValidationError as exc:
            details = []
            for error in exc.errors(include_input=False, include_url=False)[:6]:
                location = ".".join(str(part) for part in error.get("loc", ())) or "$"
                details.append(f"{location}: {error.get('msg', 'invalid value')}")
            return {
                "status": "error",
                "error": (
                    "Visual anomaly response failed schema validation: "
                    + "; ".join(details)
                ),
                RUNTIME_METRICS_KEY: runtime_metrics,
            }

        return {
            "status": "success",
            "focus_areas": focus_areas,
            **result.model_dump(mode="json", by_alias=True),
            RUNTIME_METRICS_KEY: runtime_metrics,
        }

    @staticmethod
    def _error_message(prefix: str, exc: Exception) -> str:
        detail = str(exc).strip() or "<no message>"
        return f"{prefix}: {type(exc).__name__}: {detail}"
