# -*- coding: utf-8 -*-
"""OCR tool with normalized positioned text observations."""

from __future__ import annotations

import hashlib
import io
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from eviscene.integrations.ocr.baidu import BaiduOCRError, BaiduOCRClient
from eviscene.tools.base import BaseTool


@dataclass
class OCRWithPositionTool(BaseTool):
    """Extract positioned text using the Baidu OCR API only."""

    name: str = "ocr_with_position"
    description: str = (
        "Extract text from the image with position information through the "
        "configured OCR backend. Returns text regions, bounding boxes, "
        "confidence scores, and language."
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
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "Optional target region [x1,y1,x2,y2], using normalized "
                        "0-1 coordinates or absolute pixels."
                    ),
                },
                "goal": {
                    "type": "string",
                    "description": "What visible text or property this OCR should check.",
                },
                "source_evidence_id": {
                    "type": "string",
                    "description": "Evidence id that motivated this visual revisit.",
                },
                "expected_property": {
                    "type": "string",
                    "description": "Discriminative text or property expected in the target region.",
                },
                "visual_question_id": {
                    "type": "string",
                    "description": "Pending visual question this regional OCR resolves.",
                },
                "source_discovery_id": {
                    "type": "string",
                    "description": "Discovery id that motivated this visual revisit.",
                },
            },
            "required": ["image_input"],
        }
    )
    baidu_client: Optional[BaiduOCRClient] = field(default=None, repr=False)
    backend: Optional[str] = None
    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        self._backend()

    def _backend(self) -> str:
        value = (
            (self.backend or os.getenv("OCR_BACKEND", "baidu") or "baidu")
            .strip()
            .lower()
        )
        if value not in {"baidu", "baidu_ocr", "baidu_general"}:
            raise ValueError(
                "OCR_BACKEND must be baidu; other OCR providers are not supported"
            )
        return "baidu"

    def _ocr_model(self) -> str:
        return "baidu-general"

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_path = str(params.get("image_input", "")).strip()
        artifact_bytes = b""
        requested_bbox: List[float] = []
        attempts: List[Dict[str, Any]] = []
        ocr_input_metadata: Dict[str, Any] = {}
        backend = "unknown"

        try:
            from PIL import Image

            if not image_path:
                raise ValueError("image_input is required")
            backend = self._backend()
            with Image.open(image_path).convert("RGB") as image:
                image_width, image_height = image.size
                requested_bbox = self._normalize_bbox(
                    params.get("bbox"),
                    image_width,
                    image_height,
                )
                if requested_bbox:
                    px1 = int(round(requested_bbox[0] * image_width))
                    py1 = int(round(requested_bbox[1] * image_height))
                    px2 = int(round(requested_bbox[2] * image_width))
                    py2 = int(round(requested_bbox[3] * image_height))
                    ocr_image = image.crop((px1, py1, px2, py2))
                    offset_x, offset_y = px1, py1
                else:
                    ocr_image = image.copy()
                    offset_x = offset_y = 0
                ocr_width, ocr_height = ocr_image.size
                (
                    artifact_bytes,
                    upload_width,
                    upload_height,
                    ocr_input_metadata,
                ) = self._baidu_image_bytes(ocr_image)
                attempts.append(
                    {
                        "backend": "baidu_input",
                        "status": "success",
                        "request_count": 0,
                        **ocr_input_metadata,
                    }
                )
                return self._call_baidu(
                    params,
                    artifact_bytes=artifact_bytes,
                    requested_bbox=requested_bbox,
                    image_width=image_width,
                    image_height=image_height,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    crop_width=ocr_width,
                    crop_height=ocr_height,
                    upload_width=upload_width,
                    upload_height=upload_height,
                    ocr_input_metadata=ocr_input_metadata,
                    attempts=attempts,
                )
        except Exception as exc:
            if not attempts:
                attempts.append(
                    {
                        "backend": backend,
                        "status": "error",
                        "request_count": (
                            int(getattr(exc, "request_count", 0))
                            if isinstance(exc, BaiduOCRError)
                            else 0
                        ),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            try:
                model = self._ocr_model()
            except Exception:
                model = backend
            return {
                "status": "error",
                "error": f"Baidu OCR failed: {exc}",
                "text_regions": [],
                "total_regions": 0,
                "full_text": "",
                "rejected_text_regions": [],
                "requested_bbox": requested_bbox,
                "goal": str(params.get("goal", "")),
                "source_evidence_id": str(params.get("source_evidence_id", "")),
                "expected_property": str(params.get("expected_property", "")),
                "artifact_sha256": (
                    hashlib.sha256(artifact_bytes).hexdigest() if artifact_bytes else ""
                ),
                "ocr_backend": backend,
                "ocr_model": model,
                "ocr_input": ocr_input_metadata,
                "backend_attempts": attempts,
                "subcalls": self._ocr_subcalls(attempts),
            }

    def _call_baidu(
        self,
        params: Dict[str, Any],
        *,
        artifact_bytes: bytes,
        requested_bbox: List[float],
        image_width: int,
        image_height: int,
        offset_x: int,
        offset_y: int,
        crop_width: int,
        crop_height: int,
        upload_width: int,
        upload_height: int,
        ocr_input_metadata: Dict[str, Any],
        attempts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        client = self._get_baidu_client()
        token_started = time.perf_counter()
        try:
            token, cache_hit = client.get_access_token()
        except BaiduOCRError as exc:
            attempts.append(
                {
                    "backend": "baidu_token",
                    "status": "error",
                    "request_count": exc.request_count,
                    "duration_ms": round(
                        (time.perf_counter() - token_started) * 1000,
                        2,
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        if not cache_hit:
            attempts.append(
                {
                    "backend": "baidu_token",
                    "status": "success",
                    "request_count": 1,
                    "duration_ms": round(
                        (time.perf_counter() - token_started) * 1000,
                        2,
                    ),
                }
            )

        ocr_started = time.perf_counter()
        try:
            payload, request_count = client.recognize_with_metadata(
                artifact_bytes,
                access_token=token,
            )
        except BaiduOCRError as exc:
            attempts.append(
                {
                    "backend": "baidu_ocr",
                    "status": "error",
                    "request_count": exc.request_count,
                    "duration_ms": round(
                        (time.perf_counter() - ocr_started) * 1000,
                        2,
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        attempts.append(
            {
                "backend": "baidu_ocr",
                "status": "success",
                "request_count": request_count,
                "duration_ms": round(
                    (time.perf_counter() - ocr_started) * 1000,
                    2,
                ),
            }
        )
        text_regions: List[Dict[str, Any]] = []
        rejected_text_regions: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []
        for index, item in enumerate(payload.get("words_result", [])):
            try:
                region = self._baidu_region(
                    item,
                    index=index,
                    image_width=image_width,
                    image_height=image_height,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    crop_width=crop_width,
                    crop_height=crop_height,
                    upload_width=upload_width,
                    upload_height=upload_height,
                )
            except (TypeError, ValueError) as exc:
                # A provider can return one malformed or out-of-bounds box
                # alongside otherwise usable OCR.  Keep the OCR call
                # successful and preserve the rejected item for diagnostics;
                # perception can still use scene observations and valid text.
                raw_text = (
                    str(item.get("words", "")).strip() if isinstance(item, dict) else ""
                )
                rejected_text_regions.append(
                    {
                        "status": "rejected",
                        "index": index,
                        "text": raw_text,
                        "error": str(exc),
                    }
                )
                continue
            if region["text"].strip() and float(region["confidence"]) >= float(
                self.min_confidence
            ):
                text_regions.append(region)
                full_text_parts.append(region["text"])
            else:
                rejected_text_regions.append(region)
        return self._success(
            params,
            backend="baidu",
            model="baidu-general",
            requested_bbox=requested_bbox,
            artifact_bytes=artifact_bytes,
            text_regions=text_regions,
            rejected_text_regions=rejected_text_regions,
            full_text=" ".join(full_text_parts),
            attempts=attempts,
            ocr_input_metadata=ocr_input_metadata,
        )

    @staticmethod
    def _baidu_region(
        item: Any,
        *,
        index: int,
        image_width: int,
        image_height: int,
        offset_x: int,
        offset_y: int,
        crop_width: int,
        crop_height: int,
        upload_width: int,
        upload_height: int,
    ) -> Dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError(f"Baidu OCR result {index} must be an object")
        text = str(item.get("words", "")).strip()
        location = item.get("location")
        if not isinstance(location, dict):
            raise ValueError(f"Baidu OCR result {index} lacks location")
        try:
            left = float(location["left"])
            top = float(location["top"])
            width = float(location["width"])
            height = float(location["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Baidu OCR result {index} has invalid location") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"Baidu OCR result {index} has empty location")
        vertices = item.get("vertexes_location")
        quad: List[List[float]] = []
        if isinstance(vertices, list) and len(vertices) == 4:
            for point in vertices:
                if not isinstance(point, dict):
                    quad = []
                    break
                try:
                    quad.append([float(point["x"]), float(point["y"])])
                except (KeyError, TypeError, ValueError):
                    quad = []
                    break
        if not quad:
            quad = [
                [left, top],
                [left + width, top],
                [left + width, top + height],
                [left, top + height],
            ]
        if upload_width <= 0 or upload_height <= 0:
            raise ValueError("Baidu OCR upload dimensions must be positive")
        scale_x = float(crop_width) / float(upload_width)
        scale_y = float(crop_height) / float(upload_height)
        global_quad = [
            [
                point[0] * scale_x + offset_x,
                point[1] * scale_y + offset_y,
            ]
            for point in quad
        ]
        xs = [point[0] for point in global_quad]
        ys = [point[1] for point in global_quad]
        if (
            min(xs) < 0
            or min(ys) < 0
            or max(xs) > image_width
            or max(ys) > image_height
        ):
            raise ValueError(f"Baidu OCR result {index} is outside the image")
        probability = item.get("probability", 1.0)
        if isinstance(probability, dict):
            probability = probability.get(
                "average",
                probability.get("min", 1.0),
            )
        try:
            confidence = float(probability)
        except (TypeError, ValueError):
            confidence = 1.0
        return {
            "text": text,
            "bbox_quad": [
                [
                    round(float(point[0]) / image_width, 4),
                    round(float(point[1]) / image_height, 4),
                ]
                for point in global_quad
            ],
            "bbox": [
                round(min(xs) / image_width, 4),
                round(min(ys) / image_height, 4),
                round(max(xs) / image_width, 4),
                round(max(ys) / image_height, 4),
            ],
            "confidence": round(confidence, 3),
            "language": (
                "zh" if any(0x4E00 <= ord(char) <= 0x9FFF for char in text) else "en"
            ),
        }

    def _success(
        self,
        params: Dict[str, Any],
        *,
        backend: str,
        model: str,
        requested_bbox: List[float],
        artifact_bytes: bytes,
        text_regions: List[Dict[str, Any]],
        rejected_text_regions: List[Dict[str, Any]],
        full_text: str,
        attempts: List[Dict[str, Any]],
        ocr_input_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "status": "success",
            "text_regions": text_regions,
            "total_regions": len(text_regions),
            "full_text": full_text,
            "rejected_text_regions": rejected_text_regions,
            "requested_bbox": requested_bbox,
            "goal": str(params.get("goal", "")),
            "source_evidence_id": str(params.get("source_evidence_id", "")),
            "expected_property": str(params.get("expected_property", "")),
            "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "ocr_backend": backend,
            "ocr_model": model,
            "ocr_input": dict(ocr_input_metadata or {}),
            "backend_attempts": attempts,
            "subcalls": self._ocr_subcalls(attempts),
        }

    def _get_baidu_client(self) -> BaiduOCRClient:
        if self.baidu_client is not None:
            return self.baidu_client
        # Keep the transport owned by this isolated rollout child.  The OCR
        # token itself is process-cached, so sharing the requests session is
        # unnecessary and would make per-case cleanup impossible.
        self.baidu_client = BaiduOCRClient()
        return self.baidu_client

    def close(self) -> None:
        client = self.baidu_client
        self.baidu_client = None
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _baidu_image_bytes(
        image: Any,
    ) -> tuple[bytes, int, int, Dict[str, Any]]:
        """Encode a bounded JPEG payload accepted by Baidu general OCR."""
        from PIL import Image

        source_width, source_height = image.size
        max_edge = OCRWithPositionTool._positive_int_env(
            "BAIDU_OCR_MAX_EDGE", 4096, minimum=64, maximum=4096
        )
        max_upload_bytes = OCRWithPositionTool._positive_int_env(
            "BAIDU_OCR_MAX_UPLOAD_BYTES",
            4_500_000,
            minimum=250_000,
            maximum=5_500_000,
        )
        initial_quality = OCRWithPositionTool._positive_int_env(
            "BAIDU_OCR_JPEG_QUALITY", 90, minimum=55, maximum=95
        )
        current = image.copy()
        largest_edge = max(source_width, source_height)
        if largest_edge > max_edge:
            scale = float(max_edge) / float(largest_edge)
            resampling = getattr(Image, "Resampling", Image)
            current = current.resize(
                (
                    max(1, int(round(source_width * scale))),
                    max(1, int(round(source_height * scale))),
                ),
                resample=resampling.LANCZOS,
            )

        quality = initial_quality
        for _ in range(10):
            buffer = io.BytesIO()
            current.save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=False,
                progressive=False,
            )
            payload = buffer.getvalue()
            if len(payload) <= max_upload_bytes:
                return (
                    payload,
                    current.width,
                    current.height,
                    {
                        "format": "jpeg",
                        "source_width": source_width,
                        "source_height": source_height,
                        "upload_width": current.width,
                        "upload_height": current.height,
                        "upload_bytes": len(payload),
                        "max_upload_bytes": max_upload_bytes,
                        "jpeg_quality": quality,
                    },
                )
            if quality > 55:
                quality = max(55, quality - 8)
                continue
            if min(current.width, current.height) <= 64:
                break
            resampling = getattr(Image, "Resampling", Image)
            current = current.resize(
                (
                    max(64, int(round(current.width * 0.8))),
                    max(64, int(round(current.height * 0.8))),
                ),
                resample=resampling.LANCZOS,
            )
            quality = initial_quality
        raise ValueError(
            f"Baidu OCR image cannot be compressed below {max_upload_bytes} bytes"
        )

    @staticmethod
    def _positive_int_env(
        name: str,
        default: int,
        *,
        minimum: int,
        maximum: int,
    ) -> int:
        try:
            value = int(os.getenv(name, str(default)).strip())
        except ValueError:
            value = default
        return max(minimum, min(maximum, value))

    @staticmethod
    def _ocr_subcalls(
        attempts: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [
            {
                "kind": "ocr",
                "provider": str(item.get("backend", "baidu")),
                "status": str(item.get("status", "error")),
                "request_count": int(item.get("request_count", 1)),
            }
            for item in attempts
            if int(item.get("request_count", 1)) > 0
        ]

    @staticmethod
    def _normalize_bbox(
        raw_bbox: Any,
        width: int,
        height: int,
    ) -> List[float]:
        if raw_bbox in (None, []):
            return []
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2].")
        if not all(isinstance(value, (int, float)) for value in raw_bbox):
            raise ValueError("bbox coordinates must be numeric.")
        values = [float(value) for value in raw_bbox]
        if all(0.0 <= value <= 1.0 for value in values):
            x1, y1, x2, y2 = values
        else:
            x1, y1, x2, y2 = (
                values[0] / width,
                values[1] / height,
                values[2] / width,
                values[3] / height,
            )
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError("bbox must be ordered and remain inside the image")
        if x2 - x1 <= 0.001 or y2 - y1 <= 0.001:
            raise ValueError("bbox must describe a non-empty region inside the image.")
        return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]
