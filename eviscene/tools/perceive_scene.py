# -*- coding: utf-8 -*-
"""Scene perception using VLM with structured output."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from eviscene.tools.base import BaseTool
from eviscene.integrations.gemini import RUNTIME_METRICS_KEY, exception_runtime_metrics
from eviscene.tools.vision_utils import controlled_image_to_data_url


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


PERCEIVE_SCENE_PROMPT = """\
You are the perception module of an image verification system.
Inspect the image carefully and return exactly one JSON object:
{
  "scene_description": "two to four literal sentences describing the whole visible scene",
  "image_type": "photo|screenshot|document|illustration|meme",
  "entities": [
    {
      "name": "literal visible label or generic descriptor",
      "entity_type": "person|object|building|logo|animal|scene_element",
      "bbox": [x_min, y_min, x_max, y_max],
      "confidence": 0.9,
      "text_role": "scene_text|overlay_text|watermark|caption|identity_label|claim_text|unknown|not_applicable",
      "attributes": {
        "appearance": "literal visible appearance",
        "role_or_action": "literal visible action or role",
        "spatial_context": "where this entity is located relative to the scene"
      }
    }
  ],
  "relations": [
    {
      "subject": "entity name",
      "predicate": "literal visible relation or action",
      "object": "entity name or scene element",
      "description": "one literal sentence describing the visible relation",
      "confidence": 0.9
    }
  ],
  "notable_details": ["other concrete visible detail"],
  "uncertainties": ["specific detail that the pixels do not resolve"]
}

Rules:
1. Inventory the whole image, listing up to 16 decision-relevant visible entities.
   List an entity only when you can give it a reliable non-empty bounding box.
   If a visible entity cannot be localized, mention it in scene_description or
   uncertainties instead of returning an entity with an empty bbox.
2. For people, do not assign a proper-name identity from appearance alone;
   use a generic visible descriptor such as "pilot", "man in dark suit", or
   "unidentified person" unless visible text explicitly labels the person.
3. For each entity, record concrete visible attributes, actions, roles, and
   spatial context when they are visible. Do not leave all attributes empty
   when the image clearly shows them.
4. Describe visible relations between people, objects, text-bearing surfaces,
   and scene elements. Use the literal relation, such as "person holds object",
   "person stands behind podium", or "vehicle is parked beside building".
5. Include visible logos and text-bearing surfaces, but do not transcribe text;
   a separate OCR stage handles exact visible text.
   For every text-bearing surface, classify its visible layout role as
   scene_text, overlay_text, watermark, caption, identity_label, claim_text,
   or unknown. This is a visual layout classification, not a provenance claim.
6. Use normalized [x_min, y_min, x_max, y_max] bounding boxes in [0,1].
   If no reliable box is available, use [].
7. Keep every entity name under 100 characters and every attribute under 300
   characters.
8. Keep scene_description to two to four literal, objective sentences under
   1200 characters. Cover the main subjects, actions, relationships, setting,
   and decision-relevant visible details.
9. Keep relations concrete and image-grounded. Do not use relations to infer
   provenance, authenticity, authorship, or how the image was made.
10. Use notable_details for concrete details not naturally represented by an
   entity or relation. Use uncertainties only for genuine pixel ambiguity.
11. Do not include explanations, hidden-state reasoning, history, biographies, or
   information that is not directly visible in the pixels.
12. Output JSON only.
"""

PERCEIVE_SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_description": {"type": "string", "maxLength": 1200},
        "image_type": {
            "type": "string",
            "enum": ["photo", "screenshot", "document", "illustration", "meme"],
        },
        "entities": {
            "type": "array",
            "maxItems": 16,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 100},
                    "entity_type": {
                        "type": "string",
                        "enum": [
                            "person",
                            "object",
                            "building",
                            "logo",
                            "animal",
                            "scene_element",
                        ],
                    },
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "confidence": {"type": "number"},
                    "text_role": {
                        "type": "string",
                        "enum": list(TEXT_ROLE_VALUES),
                    },
                    "attributes": {
                        "type": "object",
                        "properties": {
                            "appearance": {
                                "type": "string",
                                "maxLength": 300,
                            },
                            "role_or_action": {
                                "type": "string",
                                "maxLength": 300,
                            },
                            "spatial_context": {
                                "type": "string",
                                "maxLength": 300,
                            },
                        },
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "name",
                    "entity_type",
                    "bbox",
                    "confidence",
                    "text_role",
                    "attributes",
                ],
            },
        },
        "relations": {
            "type": "array",
            "maxItems": 16,
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "maxLength": 160},
                    "predicate": {"type": "string", "maxLength": 160},
                    "object": {"type": "string", "maxLength": 160},
                    "description": {"type": "string", "maxLength": 600},
                    "confidence": {"type": "number"},
                },
            },
        },
        "notable_details": {
            "type": "array",
            "maxItems": 16,
            "items": {"type": "string", "maxLength": 400},
        },
        "uncertainties": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string", "maxLength": 400},
        },
    },
}

PERCEIVE_SCENE_FALLBACK_SCHEMA = {
    "type": "object",
}


@dataclass
class PerceiveSceneTool(BaseTool):
    """VLM-based scene perception that outputs a structured entity list."""

    name: str = "perceive_scene"
    description: str = (
        "Observe the image and extract a structured list of visible entities "
        "(people, objects, logos, buildings, animals), their types, and approximate "
        "positions. Also determine the image type and provide a "
        "scene description."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {
                    "type": "string",
                    "description": "Local path to the image file.",
                },
            },
            "required": ["image_input"],
        }
    )

    client: Optional[Any] = field(default=None, repr=False)
    provider: str = "gemini"
    model_name: str = "gemini-3.7-flash"

    def _get_client(self):
        if self.client is None:
            from eviscene.integrations.vlm.factory import build_vlm_client

            self.client = build_vlm_client(
                provider=self.provider,
                model_name=self.model_name,
            )
        return self.client

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_input = params["image_input"]

        attempts: list[Dict[str, Any]] = []
        try:
            compressed_input, compression_metadata = controlled_image_to_data_url(
                image_input,
            )
            request_inputs = [compressed_input, compressed_input]
        except Exception as exc:
            compression_metadata = {
                "error": f"{type(exc).__name__}: {exc}",
            }
            error = {
                "status": "error",
                "error": (
                    "Scene perception image preparation failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "entities": [],
                "scene_description": "",
                "image_type": "unknown",
                "compression": compression_metadata,
            }
            metrics = exception_runtime_metrics(exc)
            if metrics:
                error[RUNTIME_METRICS_KEY] = metrics
            return error

        request_schemas = [PERCEIVE_SCENE_SCHEMA]
        if len(request_inputs) > 1:
            request_schemas.append(PERCEIVE_SCENE_FALLBACK_SCHEMA)

        try:
            client = self._get_client()
            parsed = None
            for attempt_index, (request_image, request_schema) in enumerate(
                zip(request_inputs, request_schemas),
                start=1,
            ):
                try:
                    candidate = client.create_image_json(
                        system_prompt=PERCEIVE_SCENE_PROMPT,
                        user_text=(
                            "Inspect the complete image, preserve the relationships "
                            "between the visible entities, and output the structured "
                            "scene JSON."
                        ),
                        image_input=request_image,
                        max_tokens=4000,
                        model_name=self.model_name,
                        response_schema=request_schema,
                    )
                    if not _has_usable_perception_payload(candidate):
                        raise RuntimeError(
                            "Scene perception returned no usable structured observation."
                        )
                    parsed = candidate
                    if attempt_index > 1 and isinstance(parsed, dict):
                        parsed["perception_recovery"] = {
                            "recovered": True,
                            "attempt": attempt_index,
                            "reason": attempts[-1].get("error", ""),
                            "compression": compression_metadata,
                        }
                    break
                except Exception as exc:
                    attempts.append(
                        {
                            "attempt": attempt_index,
                            "error": (
                                f"{type(exc).__name__}: {exc or '<no message>'}"
                            ),
                            "retryable": self._is_recoverable_provider_error(exc),
                        }
                    )
                    if (
                        attempt_index >= len(request_inputs)
                        or not self._is_recoverable_provider_error(exc)
                    ):
                        raise
            if parsed is None:
                raise RuntimeError("Scene perception returned no result.")
        except Exception as exc:
            error = {
                "status": "error",
                "error": (
                    "Scene perception failed: "
                    f"{type(exc).__name__}: {exc or '<no message>'}"
                ),
                "entities": [],
                "scene_description": "",
                "image_type": "unknown",
            }
            if attempts:
                error["perception_attempts"] = attempts
            metrics = exception_runtime_metrics(exc)
            if metrics:
                error[RUNTIME_METRICS_KEY] = metrics
            return error

        entities = []
        bbox_warnings = []
        for index, ent in enumerate(parsed.get("entities", [])):
            if not isinstance(ent, dict):
                continue
            try:
                bbox = normalize_entity_bbox(
                    ent.get("bbox", []),
                    thousand_scale_order=(
                        "yxyx" if self.provider.strip().lower() == "gemini" else "xyxy"
                    ),
                )
            except ValueError as exc:
                # Keep unlocalized entities with an empty bbox; downstream crops
                # must require a non-empty spatial anchor.
                bbox = []
                bbox_warnings.append(
                    {
                        "entity_index": index,
                        "entity_name": str(ent.get("name", "")).strip()[:100],
                        "error": str(exc),
                    }
                )
            text_role = str(
                ent.get("text_role", "not_applicable")
            ).strip().lower()
            if text_role not in TEXT_ROLE_VALUES:
                text_role = "unknown"
            attributes = {}
            raw_attributes = ent.get("attributes", {})
            if isinstance(raw_attributes, dict):
                for key, value in list(raw_attributes.items())[:8]:
                    key_text = str(key).strip()[:80]
                    value_text = " ".join(str(value).split()).strip()[:300]
                    if key_text and value_text:
                        attributes[key_text] = value_text
            entities.append(
                {
                    "name": str(ent.get("name", "")).strip(),
                    "entity_type": str(ent.get("entity_type", "object")).strip(),
                    "bbox": bbox,
                    "confidence": float(ent.get("confidence", 0.8)),
                    "attributes": attributes,
                    "text_role": text_role,
                }
            )

        relations = []
        for relation in parsed.get("relations", []):
            if not isinstance(relation, dict):
                continue
            subject = " ".join(str(relation.get("subject", "")).split()).strip()[:160]
            predicate = " ".join(str(relation.get("predicate", "")).split()).strip()[:160]
            object_name = " ".join(str(relation.get("object", "")).split()).strip()[:160]
            description = " ".join(
                str(relation.get("description", "")).split()
            ).strip()[:600]
            if not subject or not predicate or not object_name or not description:
                continue
            confidence = relation.get("confidence", 0.0)
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                confidence = 0.0
            relations.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_name,
                    "description": description,
                    "confidence": round(max(0.0, min(1.0, float(confidence))), 4),
                }
            )

        notable_details = [
            " ".join(str(item).split()).strip()[:400]
            for item in parsed.get("notable_details", [])
            if " ".join(str(item).split()).strip()
        ][:16]
        uncertainties = [
            " ".join(str(item).split()).strip()[:400]
            for item in parsed.get("uncertainties", [])
            if " ".join(str(item).split()).strip()
        ][:8]

        return {
            "status": "success",
            "entities": entities[:16],
            "relations": relations[:16],
            "notable_details": notable_details,
            "uncertainties": uncertainties,
            "scene_description": str(parsed.get("scene_description", "")).strip(),
            "image_type": str(parsed.get("image_type", "photo")).strip(),
            "total_entities": len(entities[:16]),
            "total_relations": len(relations[:16]),
            "bbox_warnings": bbox_warnings[:16],
            RUNTIME_METRICS_KEY: parsed.get(RUNTIME_METRICS_KEY, {}),
            **(
                {"perception_attempts": attempts}
                if attempts
                else {}
            ),
            **(
                {"perception_recovery": parsed["perception_recovery"]}
                if isinstance(parsed.get("perception_recovery"), dict)
                else {}
            ),
        }

    @staticmethod
    def _is_recoverable_provider_error(exc: BaseException) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code in {400, 429, 500, 502, 503, 504}:
            return True
        name = type(exc).__name__.lower()
        message = str(exc).lower()
        return (
            "timeout" in name
            or "timeout" in message
            or "transport" in name
            or "readerror" in name
            or "connection" in name
            or "no usable structured observation" in message
        )


def _has_usable_perception_payload(value: Any) -> bool:
    """Do not turn an empty fallback object into a successful observation."""

    if not isinstance(value, dict):
        return False
    if str(value.get("scene_description", "")).strip():
        return True
    for key in ("entities", "relations", "notable_details", "uncertainties"):
        rows = value.get(key)
        if isinstance(rows, list) and any(item not in (None, "", {}) for item in rows):
            return True
    return False


def normalize_entity_bbox(
    raw_bbox: Any,
    *,
    thousand_scale_order: str = "yxyx",
) -> List[float]:
    """Convert a model bbox to normalized project-order XYXY coordinates.

    The public tool contract is ``[x_min, y_min, x_max, y_max]`` in ``[0, 1]``.
    Gemini vision can nevertheless emit its native ``[y_min, x_min, y_max,
    x_max]`` coordinates on a 0..1000 grid. Non-empty boxes outside these two
    documented forms are rejected so incorrect regions cannot enter ReInspect.
    """

    if raw_bbox in (None, []):
        return []
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise ValueError("bbox must contain exactly four coordinates")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in raw_bbox):
        raise ValueError("bbox coordinates must be numeric")
    values = [float(value) for value in raw_bbox]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox coordinates must be finite")
    if any(0.0 < value < 1.0 for value in values) and any(
        value > 1.0 for value in values
    ):
        raise ValueError(
            "bbox mixes normalized and Gemini 0..1000 coordinate scales"
        )

    if all(0.0 <= value <= 1.0 for value in values):
        x1, y1, x2, y2 = values
    elif all(0.0 <= value <= 1000.0 for value in values) and any(
        value > 1.0 for value in values
    ):
        if thousand_scale_order == "yxyx":
            y1, x1, y2, x2 = (value / 1000.0 for value in values)
        elif thousand_scale_order == "xyxy":
            x1, y1, x2, y2 = (value / 1000.0 for value in values)
        else:
            raise ValueError("thousand_scale_order must be 'xyxy' or 'yxyx'")
    else:
        raise ValueError("bbox must use normalized XYXY or Gemini 0..1000 YXYX coordinates")

    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("bbox must be ordered, normalized, and non-empty")
    return [round(value, 6) for value in (x1, y1, x2, y2)]
