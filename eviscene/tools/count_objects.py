"""Count visible objects with schema-constrained vision output."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from eviscene.integrations.gemini import RUNTIME_METRICS_KEY, exception_runtime_metrics
from eviscene.tools.base import BaseTool


COUNT_PROMPT_TEMPLATE = """\
You are the object-counting module of an image verification system.
Count only the visible instances of the requested target.

Target: {target_object}

Rules:
1. Inspect the full supplied image or crop systematically.
2. Count a partially occluded instance only when its presence is visually supported.
3. Return zero when no target instance is visible.
4. Do not infer instances outside the frame or hidden behind objects.
5. Keep the explanation literal and concise.
6. Return exactly one JSON object matching the response schema.
"""


COUNT_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "target": {"type": "string", "maxLength": 200},
        "count": {"type": "integer", "minimum": 0},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "details": {"type": "string", "maxLength": 800},
        "locations": {
            "type": "array",
            "items": {"type": "string", "maxLength": 240},
            "maxItems": 50,
        },
    },
}


@dataclass
class CountObjectsTool(BaseTool):
    """Count a requested visible object in the image or a normalized crop."""

    name: str = "count_objects"
    description: str = (
        "Count a specific visible object in the image or in a normalized crop. "
        "Provide a precise target such as 'fingers on the left hand' or "
        "'people in the background'. Returns a count, confidence, and locations."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {
                    "type": "string",
                    "description": "Local path to the image file.",
                },
                "target_object": {
                    "type": "string",
                    "description": "A precise description of what to count.",
                },
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "Optional normalized crop [x1, y1, x2, y2], with every "
                        "coordinate in [0, 1]."
                    ),
                },
                "visual_question_id": {"type": "string"},
                "source_evidence_id": {"type": "string"},
                "source_discovery_id": {"type": "string"},
                "expected_property": {"type": "string"},
            },
            "required": ["image_input", "target_object"],
        }
    )

    client: Optional[Any] = field(default=None, repr=False)
    provider: str = "gemini"
    model_name: str = "gemini-3.7-flash"

    def _get_client(self) -> Any:
        if self.client is None:
            from eviscene.integrations.vlm.factory import build_vlm_client

            self.client = build_vlm_client(
                provider=self.provider,
                model_name=self.model_name,
            )
        return self.client

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_input = str(params.get("image_input", "")).strip()
        target_object = str(params.get("target_object", "")).strip()
        if not image_input:
            return {"status": "error", "error": "image_input is required."}
        if not target_object:
            return {"status": "error", "error": "target_object is required."}

        actual_image = image_input
        temporary_path: Optional[str] = None
        bbox = params.get("bbox")
        if bbox is not None:
            try:
                coordinates = self._validate_bbox(bbox)
                actual_image, temporary_path = self._crop(image_input, coordinates)
            except Exception as exc:
                return {
                    "status": "error",
                    "error": f"Requested count region could not be cropped: {exc}",
                }

        try:
            parsed = self._get_client().create_image_json(
                system_prompt=COUNT_PROMPT_TEMPLATE.format(
                    target_object=target_object,
                ),
                user_text=f"Count the visible instances of: {target_object}",
                image_input=actual_image,
                max_tokens=1000,
                model_name=self.model_name,
                response_schema=COUNT_RESPONSE_SCHEMA,
            )
        except Exception as exc:
            error = {
                "status": "error",
                "error": f"Counting failed: {type(exc).__name__}: {exc or '<no message>'}",
            }
            metrics = exception_runtime_metrics(exc)
            if metrics:
                error[RUNTIME_METRICS_KEY] = metrics
            return error
        finally:
            if temporary_path:
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

        result = self._validate_response(parsed, target_object)
        runtime_metrics = parsed.get(RUNTIME_METRICS_KEY)
        if runtime_metrics:
            result[RUNTIME_METRICS_KEY] = runtime_metrics
        return result

    @staticmethod
    def _validate_bbox(value: Any) -> tuple[float, float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("bbox must contain exactly four normalized coordinates")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
            raise ValueError("bbox coordinates must be numeric")
        coordinates = tuple(float(item) for item in value)
        if any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in coordinates):
            raise ValueError("bbox coordinates must be finite values in [0, 1]")
        x1, y1, x2, y2 = coordinates
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")
        return x1, y1, x2, y2

    @staticmethod
    def _crop(
        image_input: str,
        bbox: tuple[float, float, float, float],
    ) -> tuple[str, str]:
        from PIL import Image

        with Image.open(image_input) as image:
            width, height = image.size
            x1, y1, x2, y2 = bbox
            pixel_box = (
                max(0, min(int(x1 * width), width - 1)),
                max(0, min(int(y1 * height), height - 1)),
                max(1, min(int(math.ceil(x2 * width)), width)),
                max(1, min(int(math.ceil(y2 * height)), height)),
            )
            if pixel_box[0] >= pixel_box[2] or pixel_box[1] >= pixel_box[3]:
                raise ValueError("bbox resolves to an empty crop")
            cropped = image.crop(pixel_box)
            handle, temporary_path = tempfile.mkstemp(prefix="count_objects_", suffix=".png")
            os.close(handle)
            try:
                cropped.save(temporary_path)
            except Exception:
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass
                raise
        return temporary_path, temporary_path

    @staticmethod
    def _validate_response(parsed: Any, target_object: str) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            return {"status": "error", "error": "Counting response must be a JSON object."}

        count = parsed.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return {
                "status": "error",
                "error": "Counting response must contain a non-negative integer 'count'.",
            }
        confidence = parsed.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return {
                "status": "error",
                "error": "Counting response must contain numeric 'confidence'.",
            }
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            return {
                "status": "error",
                "error": "Counting confidence must be a finite value in [0, 1].",
            }

        details = parsed.get("details")
        locations = parsed.get("locations")
        if not isinstance(details, str):
            return {"status": "error", "error": "Counting details must be a string."}
        if not isinstance(locations, list) or any(
            not isinstance(location, str) for location in locations
        ):
            return {"status": "error", "error": "Counting locations must be an array of strings."}

        return {
            "status": "success",
            "target": target_object,
            "count": count,
            "confidence": confidence,
            "details": details.strip(),
            "locations": [location.strip() for location in locations],
        }
