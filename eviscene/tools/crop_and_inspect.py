# -*- coding: utf-8 -*-
"""Crop a region from the image and inspect it with VLM."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import os
import tempfile
from typing import Any, Dict, Optional

from eviscene.integrations.gemini import RUNTIME_METRICS_KEY, exception_runtime_metrics
from eviscene.orchestrator.runtime_events import current_case_runtime_store
from eviscene.tools.base import BaseTool
from eviscene.tools.vision_utils import bounded_pil_image_to_jpeg_bytes


INSPECT_PROMPT_TEMPLATE = """\
You are inspecting a cropped image region for image verification.
Focus question: {focus_question}

Return exactly one JSON object:
{{
  "description": "literal description of the cropped region",
  "observations": ["literal visible observation 1", "literal visible observation 2"],
  "anomalies": ["specific visible irregularity, if any"],
  "limitations": ["what the crop cannot establish, if relevant"]
}}

Be concrete and conservative. This response is a visual record:
- description summarizes the cropped region;
- each observation names a visible region or object and its visible property;
- each anomaly names a visible region or object and its visible irregularity;
- each limitation names an unresolved question and the missing visual or
  external information.

Use quoted on-image text and concrete spatial details when they are visible.
Output JSON only.
"""

INSPECT_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "maxLength": 700},
        "observations": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
            "maxItems": 8,
        },
        "anomalies": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
            "maxItems": 8,
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
            "maxItems": 4,
        },
    },
}


@dataclass
class CropAndInspectTool(BaseTool):
    """Crop a region from the image and analyze it in detail."""

    name: str = "crop_and_inspect"
    description: str = (
        "Crop a specific region from the image and analyze it in detail. "
        "Provide a normalized bounding box [x1, y1, x2, y2] and a focus question."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {
                    "type": "string",
                    "description": "Local path to the image file.",
                },
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Bounding box [x1, y1, x2, y2] normalized to 0-1.",
                },
                "focus_question": {
                    "type": "string",
                    "description": "What to inspect in this cropped region.",
                },
                "visual_question_id": {"type": "string"},
                "source_evidence_id": {"type": "string"},
                "source_discovery_id": {"type": "string"},
                "expected_property": {"type": "string"},
            },
            "required": ["image_input", "bbox", "focus_question"],
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
        image_path = str(params.get("image_input", "")).strip()
        bbox = params.get("bbox")
        focus_question = str(params.get("focus_question", "")).strip()

        if not image_path:
            return {
                "status": "error",
                "error": "image_input is required.",
            }

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return {
                "status": "error",
                "error": "bbox must be [x1, y1, x2, y2] with 4 normalized values.",
            }
        if not focus_question:
            return {
                "status": "error",
                "error": "focus_question is required.",
            }
        try:
            self._validate_bbox(bbox)
        except ValueError as exc:
            return {
                "status": "error",
                "error": f"Invalid bbox: {exc}",
            }

        try:
            from PIL import Image

            with Image.open(image_path) as img:
                w, h = img.size
                x1, y1, x2, y2 = self._bbox_to_pixels(bbox, w, h)
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))
                cropped = img.crop((x1, y1, x2, y2))
            crop_bytes, crop_view = bounded_pil_image_to_jpeg_bytes(cropped)
            with tempfile.NamedTemporaryFile(
                prefix="eviscene_crop_inspect_",
                suffix=".jpg",
                delete=False,
            ) as handle:
                tmp_path = handle.name
                handle.write(crop_bytes)
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Crop failed: {exc}",
            }

        try:
            client = self._get_client()
            parsed = client.create_image_json(
                system_prompt=INSPECT_PROMPT_TEMPLATE.format(focus_question=focus_question),
                user_text=f"Inspect this cropped region. Focus: {focus_question}",
                image_input=tmp_path,
                max_tokens=1000,
                model_name=self.model_name,
                response_schema=INSPECT_SCHEMA,
            )
        except Exception as exc:
            error = {
                "status": "error",
                "error": f"VLM inspection failed: {exc}",
            }
            metrics = exception_runtime_metrics(exc)
            if metrics:
                error[RUNTIME_METRICS_KEY] = metrics
            return error
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        observations = parsed.get("observations")
        if not isinstance(observations, list):
            # Older cached/model responses called these literal visual items
            # "findings". Keep them usable as observations, but never revive
            # the removed free-form "answer" field as a verdict.
            observations = parsed.get("findings", [])
        if not isinstance(observations, list):
            observations = []

        result = {
            "status": "success",
            "description": str(parsed.get("description", "")),
            "observations": observations,
            "anomalies": parsed.get("anomalies", []),
            "limitations": parsed.get("limitations", []),
            "crop_bbox": bbox,
            "focus_question": focus_question,
            "crop_view": {
                "original_size": crop_view["original_size"],
                "sent_size": crop_view["sent_size"],
                "max_long_edge": crop_view["max_long_edge"],
                "jpeg_quality": crop_view["jpeg_quality"],
            },
            RUNTIME_METRICS_KEY: parsed.get(RUNTIME_METRICS_KEY, {}),
        }
        store = current_case_runtime_store()
        if store is not None:
            try:
                descriptor = store.artifacts.put_bytes(
                    crop_bytes,
                    media_type="image/jpeg",
                    suffix=".jpg",
                    metadata={
                        "kind": "crop_and_inspect_view",
                        "crop_bbox": list(bbox),
                        "focus_question": focus_question[:400],
                        "source_image_sha256": _sha256_file(image_path),
                        **crop_view,
                    },
                )
                view_artifact = {
                    "kind": "crop_and_inspect_view",
                    "region": list(bbox),
                    "artifact": descriptor,
                }
                result["view_artifacts"] = [view_artifact]
                store.append_event(
                    "image_view",
                    {
                        "stage": "unified_react",
                        "purpose": "crop_and_inspect_follow_up",
                        "image_id": "input-image",
                        "crop": list(bbox),
                        "view_artifacts": [view_artifact],
                        "decision_impact": "pending",
                        **crop_view,
                    },
                )
            except Exception as exc:
                result["crop_artifact_error"] = f"{type(exc).__name__}: {exc}"
        return result

    @staticmethod
    def _validate_bbox(bbox: Any) -> None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("bbox must contain exactly four coordinates")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in bbox
        ):
            raise ValueError("bbox coordinates must be numeric")
        values = [float(value) for value in bbox]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("bbox coordinates must be finite")
        x1, y1, x2, y2 = values
        if x1 < 0 or y1 < 0:
            raise ValueError("bbox coordinates cannot be negative")
        if max(values) <= 1.5:
            if any(value > 1.0 for value in values):
                raise ValueError("normalized bbox coordinates must be in [0, 1]")
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")

    @staticmethod
    def _bbox_to_pixels(bbox: Any, width: int, height: int) -> tuple[int, int, int, int]:
        values = [float(v) for v in bbox]
        max_value = max(values)

        if max_value <= 1.5:
            x1 = int(values[0] * width)
            y1 = int(values[1] * height)
            x2 = int(values[2] * width)
            y2 = int(values[3] * height)
        elif max_value <= 1000.0:
            x1 = int(values[0] / 1000.0 * width)
            y1 = int(values[1] / 1000.0 * height)
            x2 = int(values[2] / 1000.0 * width)
            y2 = int(values[3] / 1000.0 * height)
        else:
            x1, y1, x2, y2 = [int(v) for v in values]

        return x1, y1, x2, y2


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
