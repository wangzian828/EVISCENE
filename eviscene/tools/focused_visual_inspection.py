# -*- coding: utf-8 -*-
"""Question-driven multi-view inspection of the original input image."""
from __future__ import annotations

from eviscene.environment import get_env

from dataclasses import dataclass, field
import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from eviscene.integrations.gemini import RUNTIME_METRICS_KEY, exception_runtime_metrics
from eviscene.tools.base import BaseTool
from eviscene.orchestrator.runtime_events import current_case_runtime_store


FOCUSED_VISUAL_INSPECTION_PROMPT = """\
You are the visual observation stage of an image fact-search agent.

Inspect the supplied views of one original input image to answer one focused
visual question that arose during web investigation. Views may arrive as separate
images in listed order or as one labeled contact sheet whose panels carry their
view_index. View 0 is always the complete original. Later views are deterministic
details derived from pixel/OCR anchors in that same original.

Treat names, identities, places, events, and expected properties in the request
as hypotheses to be checked against the supplied pixels. Produce a visual
observation record for the focused question: identify the relevant entity or
relationship, describe its visible property, and distinguish the observed value
from any competing value. Separate literal observations from interpretation,
preserve ambiguity, and use the supplied views as the complete visual basis.
The downstream runtime combines this observation with source evidence and the
target fact.

Report relative scale unless the views contain a usable reference for an absolute
measurement. For any property that the supplied pixels cannot resolve, use
answer_status=ambiguous and state the specific visual limitation.
When the question concerns an interaction between multiple people or objects,
first identify which participant owns the queried visible property. Do not
transfer a property from the other participant. For either/or questions, state
which alternative is visible in the summary while keeping answer_status relative
to the single expected_property.

Return:
- answer_status=observed only when the expected visible property is actually
  present in the supplied pixels;
- answer_status=not_observed only when the relevant region is visible enough and
  materially incompatible with that property;
- answer_status=ambiguous when resolution, occlusion, framing, or visual
  similarity prevents a reliable answer.

Every observation must identify the supplied view where it is visible.
"""


FOCUSED_VISUAL_INSPECTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_status": {
            "type": "string",
            "enum": ["observed", "not_observed", "ambiguous"],
        },
        "summary": {"type": "string", "maxLength": 1200},
        "observations": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "view_index": {"type": "integer"},
                    "statement": {"type": "string", "maxLength": 600},
                    "property_status": {
                        "type": "string",
                        "enum": ["observed", "not_observed", "ambiguous"],
                    },
                    "confidence": {"type": "number"},
                },
            },
        },
        "limitations": {
            "type": "array",
            "maxItems": 6,
            "items": {"type": "string", "maxLength": 400},
        },
    },
}


@dataclass
class FocusedVisualInspectionTool(BaseTool):
    """Inspect the original image and deterministic anchor views as one packet."""

    name: str = "focused_visual_inspection"
    description: str = (
        "Reinspect the complete original image and deterministic anchor-derived "
        "detail views to answer one focused visual question discovered during "
        "the investigation."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {
                    "type": "string",
                    "description": "Local path to the original input image.",
                },
                "visual_question_id": {"type": "string"},
                "question": {"type": "string"},
                "expected_property": {"type": "string"},
                "scope": {
                    "type": "string",
                    "enum": [
                        "subject",
                        "relation",
                        "scene",
                        "text",
                        "integrity",
                    ],
                },
                "anchor_regions": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                "active_fact": {"type": "string"},
                "evidence_context": {"type": "string"},
                "trace_stage": {
                    "type": "string",
                    "description": "Internal trace stage for this visual call.",
                },
                "trace_purpose": {
                    "type": "string",
                    "description": "Internal purpose label for this visual call.",
                },
            },
            "required": [
                "image_input",
                "visual_question_id",
                "question",
                "expected_property",
                "scope",
                "anchor_regions",
                "active_fact",
                "evidence_context",
            ],
        }
    )

    client: Optional[Any] = field(default=None, repr=False)
    provider: str = "gemini"
    model_name: str = "gemini-3.7-flash"
    max_images_per_prompt: int = field(
        default_factory=lambda: _positive_env_int(
            "EVISCENE_FOCUSED_VISUAL_MAX_IMAGES_PER_PROMPT"
        )
    )

    def _get_client(self) -> Any:
        if self.client is None:
            from eviscene.integrations.vlm.factory import build_vlm_client

            self.client = build_vlm_client(
                provider=self.provider,
                model_name=self.model_name,
            )
        method = getattr(self.client, "create_images_json", None)
        if not callable(method):
            raise RuntimeError(
                "The configured visual model client does not implement the "
                "multi-view create_images_json contract."
            )
        return self.client

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_path = str(params.get("image_input", "")).strip()
        if not image_path:
            return {"status": "error", "error": "image_input is required."}
        try:
            anchor_regions = _normalize_regions(params.get("anchor_regions", []))
            image_inputs, views, temporary_paths = _prepare_views(
                image_path,
                anchor_regions,
                scope=str(params.get("scope", "scene")).strip(),
            )
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Visual view preparation failed: {type(exc).__name__}: {exc}",
            }

        request_context = {
            "visual_question_id": str(params.get("visual_question_id", "")).strip(),
            "question": str(params.get("question", "")).strip(),
            "expected_property": str(params.get("expected_property", "")).strip(),
            "scope": str(params.get("scope", "")).strip(),
            "active_fact": str(params.get("active_fact", "")).strip(),
            "evidence_context": str(params.get("evidence_context", "")).strip(),
            "views": views,
        }
        trace_stage = (
            str(params.get("trace_stage") or "image_only_visual_reinspection")
            .strip()
            or "image_only_visual_reinspection"
        )
        trace_purpose = (
            str(
                params.get("trace_purpose")
                or "evidence_motivated_reinspection"
            ).strip()
            or "evidence_motivated_reinspection"
        )
        view_artifacts: List[Dict[str, Any]] = []
        try:
            before_version = str(params.get("before_understanding_version", "")).strip()
            client = self._get_client()
            request_image_inputs, image_packet_mode = _single_image_packet_if_needed(
                client,
                image_inputs,
                views,
                temporary_paths,
                max_images_per_prompt=self.max_images_per_prompt,
            )
            request_context["image_packet_mode"] = image_packet_mode
            parsed = client.create_images_json(
                system_prompt=FOCUSED_VISUAL_INSPECTION_PROMPT,
                user_text=(
                    "Answer the focused visual question using the supplied views "
                    "in their listed order.\n\n"
                    + json.dumps(request_context, ensure_ascii=False, indent=2)
                ),
                image_inputs=request_image_inputs,
                max_tokens=2400,
                model_name=self.model_name,
                response_schema=FOCUSED_VISUAL_INSPECTION_SCHEMA,
            )
            normalized = _normalize_model_output(parsed, views)
            view_artifacts = _persist_view_artifacts(
                request_image_inputs,
                views,
                image_packet_mode=image_packet_mode,
                visual_question_id=request_context["visual_question_id"],
                question=request_context["question"],
            )
        except Exception as exc:
            error = {
                "status": "error",
                "error": (
                    "Focused visual inspection failed: "
                    f"{type(exc).__name__}: {exc or '<no message>'}"
                ),
            }
            metrics = exception_runtime_metrics(exc)
            if metrics:
                error[RUNTIME_METRICS_KEY] = metrics
            return error
        finally:
            for path in temporary_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

        after_version = hashlib.sha256(
            json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        store = current_case_runtime_store()
        if store is not None:
            store.append_event(
                "image_view",
                {
                    "stage": trace_stage,
                    "purpose": trace_purpose,
                    "image_id": "input-image",
                    "visual_question_id": request_context["visual_question_id"],
                    "question": request_context["question"],
                    "views": views,
                    "view_artifacts": view_artifacts,
                    "before_understanding_version": before_version or None,
                    "after_understanding_version": after_version,
                    "decision_impact": "pending",
                    "answer_status": normalized["answer_status"],
                    "image_packet_mode": image_packet_mode,
                },
            )
        return {
            "status": "success",
            "visual_question_id": request_context["visual_question_id"],
            "question": request_context["question"],
            "expected_property": request_context["expected_property"],
            "scope": request_context["scope"],
            "answer_status": normalized["answer_status"],
            "summary": normalized["summary"],
            "observations": normalized["observations"],
            "limitations": normalized["limitations"],
            "views": views,
            "view_artifacts": view_artifacts,
            "before_understanding_version": before_version or None,
            "after_understanding_version": after_version,
            "image_packet_mode": image_packet_mode,
            "trace_stage": trace_stage,
            "trace_purpose": trace_purpose,
            RUNTIME_METRICS_KEY: parsed.get(RUNTIME_METRICS_KEY, {}),
        }


def _normalize_regions(value: Any) -> List[List[float]]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise ValueError("anchor_regions must be a list of normalized bboxes")
    regions: List[List[float]] = []
    for raw in value:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise ValueError("every anchor region must be [x1,y1,x2,y2]")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in raw
        ):
            raise ValueError("anchor region coordinates must be numeric")
        x1, y1, x2, y2 = [float(item) for item in raw]
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("anchor regions must be normalized and non-empty")
        region = [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]
        if region not in regions:
            regions.append(region)
    return regions[:4]


def _prepare_views(
    image_path: str,
    anchor_regions: Sequence[Sequence[float]],
    *,
    scope: str,
) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    original_view = _bounded_view(image)
    original_handle = tempfile.NamedTemporaryFile(
        prefix="eviscene_visual_original_",
        suffix=".jpg",
        delete=False,
    )
    original_handle.close()
    original_view.save(original_handle.name, format="JPEG", quality=95, optimize=True)
    image_inputs: List[str] = [original_handle.name]
    views: List[Dict[str, Any]] = [
        {
            "view_index": 0,
            "kind": "original",
            "region": [0.0, 0.0, 1.0, 1.0],
        }
    ]
    temporary_paths: List[str] = [original_handle.name]
    regions = [list(item) for item in anchor_regions]
    derived: List[tuple[str, List[float]]] = []
    if scope == "relation" and len(regions) >= 2:
        derived.append(("relation_context", _expand_region(_union_region(regions), 0.08)))
    for region in regions:
        derived.append(("anchor_detail", _expand_region(region, 0.06)))

    seen = {(0.0, 0.0, 1.0, 1.0)}
    for kind, region in derived:
        key = tuple(region)
        if key in seen:
            continue
        seen.add(key)
        x1 = max(0, min(width - 1, int(region[0] * width)))
        y1 = max(0, min(height - 1, int(region[1] * height)))
        x2 = max(x1 + 1, min(width, int(region[2] * width)))
        y2 = max(y1 + 1, min(height, int(region[3] * height)))
        crop = _bounded_view(image.crop((x1, y1, x2, y2)))
        handle = tempfile.NamedTemporaryFile(
            prefix="eviscene_visual_view_",
            suffix=".png",
            delete=False,
        )
        handle.close()
        crop.save(handle.name)
        temporary_paths.append(handle.name)
        image_inputs.append(handle.name)
        views.append(
            {
                "view_index": len(views),
                "kind": kind,
                "region": region,
            }
        )
        if len(views) >= 5:
            break
    return image_inputs, views, temporary_paths


def _single_image_packet_if_needed(
    client: Any,
    image_inputs: Sequence[str],
    views: Sequence[Mapping[str, Any]],
    temporary_paths: List[str],
    *,
    max_images_per_prompt: int = 0,
) -> tuple[List[str], str]:
    """Return a one-image contact sheet for clients that cannot accept multi-view packets."""

    max_images = max_images_per_prompt or _client_max_images_per_prompt(client)
    if max_images != 1 or len(image_inputs) <= 1:
        return list(image_inputs), "separate_images"
    contact_sheet_path = _build_contact_sheet(image_inputs, views)
    temporary_paths.append(contact_sheet_path)
    return [contact_sheet_path], "labeled_contact_sheet"


def _client_max_images_per_prompt(client: Any) -> Optional[int]:
    for name in (
        "max_images_per_prompt",
        "limit_mm_per_prompt",
        "limit_multimodal_per_prompt",
        "max_multimodal_per_prompt",
    ):
        value = getattr(client, name, None)
        if isinstance(value, int) and value > 0:
            return value
    provider = str(getattr(client, "provider", "")).strip().lower()
    # The locally hosted Qwen service used for replay is configured with a
    # single-image multimodal limit. Avoid a failing multi-image request.
    if provider == "qwen_local":
        return 1
    return None


def _positive_env_int(name: str) -> int:
    raw = get_env(name, "").strip()
    try:
        value = int(raw)
    except ValueError:
        return 0
    return value if value > 0 else 0


def _build_contact_sheet(
    image_inputs: Sequence[str],
    views: Sequence[Mapping[str, Any]],
) -> str:
    from PIL import Image, ImageDraw, ImageFont
    from eviscene.tools.vision_utils import bounded_pil_image_to_jpeg_bytes

    loaded = []
    for path in image_inputs:
        with Image.open(path) as source:
            loaded.append(source.convert("RGB"))
    if not loaded:
        raise ValueError("contact sheet requires at least one image")
    thumb_width = 640
    label_height = 44
    gap = 16
    thumbs = []
    for image in loaded:
        thumb = image.copy()
        thumb.thumbnail((thumb_width, thumb_width), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
    columns = 2 if len(thumbs) > 1 else 1
    rows = (len(thumbs) + columns - 1) // columns
    cell_width = thumb_width
    cell_height = max(thumb.height for thumb in thumbs) + label_height
    sheet = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * gap,
            rows * cell_height + (rows + 1) * gap,
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    for index, thumb in enumerate(thumbs):
        row = index // columns
        col = index % columns
        left = gap + col * (cell_width + gap)
        top = gap + row * (cell_height + gap)
        view = views[index] if index < len(views) else {}
        label = f"view_index {view.get('view_index', index)}: {view.get('kind', 'view')}"
        draw.text((left, top), label[:80], fill=(0, 0, 0), font=font)
        image_top = top + label_height
        sheet.paste(thumb, (left, image_top))
        draw.rectangle(
            [left, image_top, left + thumb.width - 1, image_top + thumb.height - 1],
            outline=(0, 0, 0),
            width=2,
        )
    # The grid can be larger than each individual view. Apply the same
    # model-image contract to the completed packet so single-image clients
    # never receive a contact sheet above the 1024-pixel bound.
    sheet_bytes, _ = bounded_pil_image_to_jpeg_bytes(sheet)
    with tempfile.NamedTemporaryFile(
        prefix="eviscene_visual_contact_sheet_",
        suffix=".jpg",
        delete=False,
    ) as handle:
        handle.write(sheet_bytes)
        return handle.name


def _persist_view_artifacts(
    request_image_inputs: Sequence[str],
    views: Sequence[Mapping[str, Any]],
    *,
    image_packet_mode: str,
    visual_question_id: str,
    question: str,
) -> List[Dict[str, Any]]:
    store = current_case_runtime_store()
    if store is None:
        return []

    entries: List[tuple[str, Mapping[str, Any]]] = []
    if image_packet_mode == "labeled_contact_sheet":
        if request_image_inputs:
            entries.append(
                (
                    str(request_image_inputs[0]),
                    {
                        "view_index": 0,
                        "kind": "labeled_contact_sheet",
                        "region": [0.0, 0.0, 1.0, 1.0],
                    },
                )
            )
    else:
        for path, view in zip(request_image_inputs, views):
            try:
                if int(view.get("view_index", 0)) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            entries.append((str(path), view))

    artifacts: List[Dict[str, Any]] = []
    for path, view in entries[:4]:
        file_path = Path(path)
        descriptor = store.artifacts.put_bytes(
            file_path.read_bytes(),
            media_type=_media_type_for_path(file_path),
            suffix=file_path.suffix or ".bin",
            metadata={
                "kind": "focused_visual_reinspection_view",
                "visual_question_id": visual_question_id,
                "question": question[:400],
                "view_index": int(view.get("view_index", 0) or 0),
                "view_kind": str(view.get("kind", "")),
            },
        )
        artifacts.append(
            {
                "view_index": int(view.get("view_index", 0) or 0),
                "kind": str(view.get("kind", "")),
                "region": list(view.get("region", [0.0, 0.0, 1.0, 1.0])),
                "artifact": descriptor,
                "packet_mode": image_packet_mode,
            }
        )
    return artifacts


def _media_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "application/octet-stream"


def _bounded_view(image: Any) -> Any:
    from PIL import Image

    bounded = image.copy()
    configured = max(256, int(get_env("EVISCENE_IMAGE_MAX_LONG_EDGE", "1024")))
    long_edge = min(1024, configured)
    bounded.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
    return bounded


def _expand_region(region: Sequence[float], margin: float) -> List[float]:
    x1, y1, x2, y2 = [float(item) for item in region]
    width = x2 - x1
    height = y2 - y1
    return [
        round(max(0.0, x1 - max(margin, width * 0.12)), 6),
        round(max(0.0, y1 - max(margin, height * 0.12)), 6),
        round(min(1.0, x2 + max(margin, width * 0.12)), 6),
        round(min(1.0, y2 + max(margin, height * 0.12)), 6),
    ]


def _union_region(regions: Sequence[Sequence[float]]) -> List[float]:
    return [
        min(float(item[0]) for item in regions),
        min(float(item[1]) for item in regions),
        max(float(item[2]) for item in regions),
        max(float(item[3]) for item in regions),
    ]


def _normalize_model_output(
    parsed: Mapping[str, Any],
    views: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    allowed_status = {"observed", "not_observed", "ambiguous"}
    answer_status = str(parsed.get("answer_status", "")).strip()
    if answer_status not in allowed_status:
        raise ValueError("answer_status is missing or invalid")
    summary = " ".join(str(parsed.get("summary", "")).split())
    if not summary:
        raise ValueError("summary is required")
    view_by_index = {
        int(item["view_index"]): item
        for item in views
    }
    observations: List[Dict[str, Any]] = []
    for raw in parsed.get("observations", []) or []:
        if not isinstance(raw, Mapping):
            continue
        view_index = raw.get("view_index")
        if isinstance(view_index, bool) or not isinstance(view_index, int):
            continue
        view = view_by_index.get(view_index)
        if view is None:
            continue
        statement = " ".join(str(raw.get("statement", "")).split())
        property_status = str(raw.get("property_status", "")).strip()
        if not statement or property_status not in allowed_status:
            continue
        confidence = raw.get("confidence", 0.0)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            confidence = 0.0
        observations.append(
            {
                "view_index": view_index,
                "view_kind": str(view.get("kind", "")),
                "region": list(view.get("region", [])),
                "statement": statement[:600],
                "property_status": property_status,
                "confidence": round(max(0.0, min(1.0, float(confidence))), 4),
            }
        )
    limitations = [
        " ".join(str(item).split())[:400]
        for item in parsed.get("limitations", []) or []
        if " ".join(str(item).split())
    ][:6]
    return {
        "answer_status": answer_status,
        "summary": summary[:1200],
        "observations": observations[:8],
        "limitations": limitations,
    }
