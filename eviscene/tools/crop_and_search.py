from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional
from uuid import uuid4

from eviscene.integrations.browse.jina_reader import JinaReaderClient
from eviscene.integrations.gemini import RUNTIME_METRICS_KEY, add_runtime_metrics, exception_runtime_metrics
from eviscene.integrations.search.serper import SerperImageSearchClient, SerperLensSearchClient
from eviscene.integrations.search.visual_search import VisualReverseSearchClient
from eviscene.integrations.vlm.factory import build_vlm_client
from eviscene.orchestrator.source_access import SourceAccessPolicy
from eviscene.tools.base import BaseTool
from eviscene.tools.vision_utils import bounded_pil_image_to_jpeg_bytes


CROP_QUERY_PROMPT = """You will see a cropped region from an image.
Generate a concise web image search query to identify this region.

Return one JSON object:
{
  "query": "short search query",
  "keywords": ["keyword1", "keyword2"]
}

Rules:
1. Focus on names, places, products, landmarks, logos, text phrases, or distinctive objects.
2. Keep the query short and specific.
3. Use the user goal as extra context when helpful.
4. Output JSON only.
"""

CROP_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "maxLength": 240},
        "keywords": {
            "type": "array",
            "items": {"type": "string", "maxLength": 100},
            "maxItems": 6,
        },
    },
}


@dataclass
class CropAndSearchTool(BaseTool):
    """Crop local regions, run visual search on those crops, then visit result pages."""

    lens_client: Optional[SerperLensSearchClient] = None
    image_search_client: Optional[SerperImageSearchClient] = None
    browse_client: Optional[JinaReaderClient] = None
    vlm_client: Optional[Any] = None
    visual_search_client: Optional[VisualReverseSearchClient] = None
    provider: str = "gemini"
    model_name: str = "gemini-3.7-flash"
    top_k: int = 5
    visit_top_k: int = 3
    saved_crop_dir: str = "outputs/trace_artifacts/crops"
    source_access_policy: Optional[SourceAccessPolicy] = None
    name: str = "crop_and_search"
    description: str = (
        "Crop one or more local regions from the image, run visual search on those crops, "
        "and visit result pages to extract goal-conditioned evidence."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {
                    "type": "string",
                    "description": "Local path to the source image.",
                },
                "bbox": {
                    "type": "array",
                    "description": (
                        "One bounding box [x1,y1,x2,y2]. "
                        "Coordinates may be normalized 0-1 or absolute pixels."
                    ),
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "image_claim": {
                    "type": "string",
                    "description": "Runtime-bound image claim used for evidence stance.",
                },
                "retrieval_goal": {
                    "type": "string",
                    "description": "Information sought by this action for retrieval and extraction.",
                },
                "visual_question_id": {"type": "string"},
                "source_evidence_id": {"type": "string"},
                "source_discovery_id": {"type": "string"},
                "expected_property": {"type": "string"},
            },
            "required": [
                "image_input",
                "bbox",
                "image_claim",
                "retrieval_goal",
            ],
        }
    )

    def __post_init__(self) -> None:
        if self.lens_client is None:
            self.lens_client = SerperLensSearchClient()
        if self.image_search_client is None:
            self.image_search_client = SerperImageSearchClient()
        if self.browse_client is None:
            self.browse_client = JinaReaderClient()
        if self.vlm_client is None:
            self.vlm_client = build_vlm_client(
                provider=self.provider,
                model_name=self.model_name,
            )
        if self.visual_search_client is None:
            self.visual_search_client = VisualReverseSearchClient(
                serper_lens_client=self.lens_client,
            )
        if self.source_access_policy is not None:
            self.set_source_access_policy(self.source_access_policy)

    def set_source_access_policy(self, policy: SourceAccessPolicy) -> None:
        self.source_access_policy = policy
        setter = getattr(self.browse_client, "set_source_access_policy", None)
        if callable(setter):
            setter(policy)

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        total_t0 = time.perf_counter()
        image_input = str(params["image_input"])
        raw_bbox = params["bbox"]
        image_claim = str(params["image_claim"]).strip()
        retrieval_goal = str(params["retrieval_goal"]).strip()

        bboxes = self._normalize_bboxes(raw_bbox)
        if not bboxes:
            return {
                "status": "error",
                "error": "bbox must be one box [x1,y1,x2,y2] with numeric coordinates.",
            }

        if not Path(image_input).exists():
            return {
                "status": "error",
                "error": f"Image not found: {image_input}",
            }

        regions: List[Dict[str, Any]] = []
        for idx, bbox in enumerate(bboxes):
            region = self._process_region(
                image_input=image_input,
                bbox=bbox,
                image_claim=image_claim,
                retrieval_goal=retrieval_goal,
                region_index=idx,
            )
            regions.append(region)

        runtime_metrics: Dict[str, Any] = {}
        for region in regions:
            add_runtime_metrics(runtime_metrics, region.get(RUNTIME_METRICS_KEY))

        best_region = self._pick_best_region(regions)
        candidate_page_urls = self._merge_region_candidate_urls(regions)
        reference_image_candidates = self._merge_region_reference_candidates(regions)
        failed_regions = [region for region in regions if region.get("error")]
        if regions and len(failed_regions) == len(regions):
            return {
                "status": "error",
                "error": "; ".join(
                    str(region.get("error", "crop search failed"))
                    for region in failed_regions
                ),
                "regions": regions,
            }
        return {
            "status": "success",
            "image_input": image_input,
            "image_claim": image_claim,
            "retrieval_goal": retrieval_goal,
            "regions": regions,
            "best_region": best_region,
            "candidate_page_urls": candidate_page_urls,
            "reference_image_candidates": reference_image_candidates,
            "reference_image_url": reference_image_candidates[0] if reference_image_candidates else "",
            "summary": str(best_region.get("summary", "")),
            "evidence": str(best_region.get("evidence", "")),
            "relevance": str(best_region.get("relevance", "low")),
            "stance": str(best_region.get("stance", "unclear")),
            "relation_scope": str(
                best_region.get("relation_scope", "unclear")
            ),
            "relation_stance": str(
                best_region.get("relation_stance", "unclear")
            ),
            "directness": str(best_region.get("directness", "none")),
            "context_only": bool(best_region.get("context_only", False)),
            "temporal_alignment": str(
                best_region.get("temporal_alignment", "not_applicable")
            ),
            "artifact_sha256": str(best_region.get("artifact_sha256", "")),
            "evidence_span": best_region.get("evidence_span", {}),
            "retrieved_at": str(best_region.get("retrieved_at", "")),
            "injection_flags": best_region.get("injection_flags", []),
            "evidence_eligible": bool(best_region.get("evidence_eligible", False)),
            "selected_url": str(best_region.get("selected_url", "")),
            "timings": {
                "total_ms": round((time.perf_counter() - total_t0) * 1000, 2),
                "num_regions": len(regions),
            },
            RUNTIME_METRICS_KEY: runtime_metrics.get(RUNTIME_METRICS_KEY, {}),
        }

    def _process_region(
        self,
        *,
        image_input: str,
        bbox: List[float],
        image_claim: str,
        retrieval_goal: str,
        region_index: int,
    ) -> Dict[str, Any]:
        crop_path = None
        region_t0 = time.perf_counter()
        try:
            crop_t0 = time.perf_counter()
            crop_path, normalized_bbox = self._crop_region(image_input, bbox, region_index)
            saved_crop_path = self._persist_crop(crop_path, image_input, region_index)
            crop_duration_ms = round((time.perf_counter() - crop_t0) * 1000, 2)

            visual_t0 = time.perf_counter()
            visual = self.visual_search_client.search(crop_path, top_k=self.top_k)
            visual_duration_ms = round((time.perf_counter() - visual_t0) * 1000, 2)
            crop_url = str(visual.get("image_url", "")).strip()
            lens_results = visual.get("results", [])

            semantic_t0 = time.perf_counter()
            semantic_query = self._generate_query(crop_path, retrieval_goal)
            query_metrics = dict(getattr(self, "_last_query_runtime_metrics", {}) or {})
            query_error = str(getattr(self, "_last_query_error", "") or "").strip()
            semantic_results = []
            if semantic_query:
                semantic_results = self.image_search_client.search(
                    query=semantic_query,
                    top_k=self.top_k,
                )
            if self.source_access_policy is not None:
                lens_results, lens_blocked = self.source_access_policy.filter_rows(lens_results)
                semantic_results, semantic_blocked = self.source_access_policy.filter_rows(semantic_results)
            else:
                lens_blocked = semantic_blocked = 0
            semantic_duration_ms = round((time.perf_counter() - semantic_t0) * 1000, 2)

            candidate_urls = self._collect_candidate_urls(lens_results, semantic_results)
            reference_image_candidates = self._collect_reference_image_candidates(lens_results, semantic_results)
            visit_t0 = time.perf_counter()
            visit_result = (
                self.browse_client.visit_many(
                    candidate_urls[: self.visit_top_k],
                    image_claim=image_claim,
                    retrieval_goal=retrieval_goal,
                )
                if candidate_urls
                else {
                    "image_claim": image_claim,
                    "retrieval_goal": retrieval_goal,
                    "provider": "jina_reader",
                    "visits": [],
                    "evidence": "",
                    "summary": "",
                    "rationale": "No result pages available from visual search.",
                    "relevance": "low",
                    "stance": "unclear",
                    "relation_scope": "unclear",
                    "relation_stance": "unclear",
                    "directness": "none",
                    "temporal_alignment": "unknown",
                    "artifact_sha256": "",
                    "evidence_span": {},
                    "retrieved_at": "",
                    "injection_flags": [],
                    "evidence_eligible": False,
                }
            )
            visit_duration_ms = round((time.perf_counter() - visit_t0) * 1000, 2)

            result = {
                "bbox": normalized_bbox,
                "image_claim": image_claim,
                "retrieval_goal": retrieval_goal,
                "saved_crop_path": saved_crop_path,
                "crop_query": semantic_query,
                "vlm_query_error": query_error,
                "image_url_for_search": crop_url,
                "visual_search_provider": visual.get("provider", ""),
                "upload": visual.get("upload", {}),
                "visual_search_errors": visual.get("errors", {}),
                "lens_results": lens_results,
                "semantic_results": semantic_results,
                "policy_filtered_count": lens_blocked + semantic_blocked,
                "candidate_page_urls": candidate_urls,
                "reference_image_candidates": reference_image_candidates,
                "reference_image_url": reference_image_candidates[0] if reference_image_candidates else "",
                "visited_pages": visit_result.get("visits", []),
                "rationale": visit_result.get("rationale", ""),
                "evidence": visit_result.get("evidence", ""),
                "summary": visit_result.get("summary", ""),
                "relevance": visit_result.get("relevance", "low"),
                "stance": visit_result.get("stance", "unclear"),
                "relation_scope": visit_result.get(
                    "relation_scope",
                    "unclear",
                ),
                "relation_stance": visit_result.get(
                    "relation_stance",
                    "unclear",
                ),
                "directness": visit_result.get("directness", "none"),
                "context_only": bool(visit_result.get("context_only", False)),
                "temporal_alignment": visit_result.get(
                    "temporal_alignment",
                    "not_applicable",
                ),
                "artifact_sha256": visit_result.get("artifact_sha256", ""),
                "evidence_span": visit_result.get("evidence_span", {}),
                "retrieved_at": visit_result.get("retrieved_at", ""),
                "injection_flags": visit_result.get("injection_flags", []),
                "evidence_eligible": bool(visit_result.get("evidence_eligible", False)),
                "selected_url": visit_result.get("selected_url", ""),
                "blocked_pages": visit_result.get("blocked_pages", []),
                "timings": {
                    "crop_ms": crop_duration_ms,
                    "visual_search_ms": visual_duration_ms,
                    "semantic_ms": semantic_duration_ms,
                    "visit_ms": visit_duration_ms,
                    "region_total_ms": round((time.perf_counter() - region_t0) * 1000, 2),
                    **(visual.get("timings", {}) if isinstance(visual.get("timings"), dict) else {}),
                    **(visit_result.get("timings", {}) if isinstance(visit_result.get("timings"), dict) else {}),
                },
            }
            add_runtime_metrics(result, query_metrics)
            add_runtime_metrics(result, visit_result.get(RUNTIME_METRICS_KEY))
            return result
        except Exception as exc:
            return {
                "bbox": bbox,
                "saved_crop_path": "",
                "error": str(exc),
                "rationale": f"Crop and search failed: {exc}",
                "evidence": "",
                "summary": "",
                "relevance": "low",
                "stance": "unclear",
                "relation_scope": "unclear",
                "relation_stance": "unclear",
                "directness": "none",
                "temporal_alignment": "unknown",
                "artifact_sha256": "",
                "evidence_span": {},
                "retrieved_at": "",
                "injection_flags": [],
                "evidence_eligible": False,
                "candidate_page_urls": [],
                "reference_image_candidates": [],
                "reference_image_url": "",
                "timings": {
                    "region_total_ms": round((time.perf_counter() - region_t0) * 1000, 2),
                },
            }
        finally:
            if crop_path:
                try:
                    os.remove(crop_path)
                except OSError:
                    pass

    def _normalize_bboxes(self, raw_bbox: Any) -> List[List[float]]:
        if not isinstance(raw_bbox, list) or not raw_bbox:
            return []
        if len(raw_bbox) == 4 and all(isinstance(v, (int, float)) for v in raw_bbox):
            return [[float(v) for v in raw_bbox]]
        return []

    def _crop_region(
        self,
        image_input: str,
        bbox: List[float],
        region_index: int,
    ) -> tuple[str, List[float]]:
        from PIL import Image

        with Image.open(image_input) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            width, height = img.size

            x1, y1, x2, y2 = self._bbox_to_pixels(bbox, width, height)
            x1, y1, x2, y2 = self._expand_small_region(x1, y1, x2, y2, width, height)
            cropped = img.crop((x1, y1, x2, y2))

            crop_bytes, _ = bounded_pil_image_to_jpeg_bytes(cropped)
            with tempfile.NamedTemporaryFile(
                prefix=f"eviscene_crop_search_{region_index}_",
                suffix=".jpg",
                delete=False,
            ) as handle:
                tmp_path = handle.name
                handle.write(crop_bytes)

            return tmp_path, [
                round(x1 / width, 4),
                round(y1 / height, 4),
                round(x2 / width, 4),
                round(y2 / height, 4),
            ]

    def _persist_crop(self, crop_path: str, image_input: str, region_index: int) -> str:
        if not crop_path or not os.path.exists(crop_path):
            return ""
        os.makedirs(self.saved_crop_dir, exist_ok=True)
        stem = Path(image_input).stem or "image"
        filename = f"{stem}_region{region_index}_{uuid4().hex[:8]}.jpg"
        target = os.path.join(self.saved_crop_dir, filename)
        shutil.copyfile(crop_path, target)
        return target

    @staticmethod
    def _bbox_to_pixels(bbox: List[float], width: int, height: int) -> tuple[int, int, int, int]:
        max_value = max(bbox)
        if max_value <= 1.5:
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
        else:
            x1, y1, x2, y2 = [int(v) for v in bbox]

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return x1, y1, x2, y2

    @staticmethod
    def _expand_small_region(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        box_w = x2 - x1
        box_h = y2 - y1
        min_w = max(64, int(width * 0.12))
        min_h = max(64, int(height * 0.12))

        if box_w >= min_w and box_h >= min_h:
            return x1, y1, x2, y2

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        target_w = max(box_w, min_w)
        target_h = max(box_h, min_h)

        new_x1 = max(0, int(round(cx - target_w / 2.0)))
        new_y1 = max(0, int(round(cy - target_h / 2.0)))
        new_x2 = min(width, int(round(cx + target_w / 2.0)))
        new_y2 = min(height, int(round(cy + target_h / 2.0)))

        if new_x2 - new_x1 < min_w:
            if new_x1 == 0:
                new_x2 = min(width, min_w)
            elif new_x2 == width:
                new_x1 = max(0, width - min_w)
        if new_y2 - new_y1 < min_h:
            if new_y1 == 0:
                new_y2 = min(height, min_h)
            elif new_y2 == height:
                new_y1 = max(0, height - min_h)

        return new_x1, new_y1, max(new_x1 + 1, new_x2), max(new_y1 + 1, new_y2)

    def _generate_query(self, crop_path: str, goal: str) -> str:
        self._last_query_runtime_metrics = {}
        self._last_query_error = ""
        try:
            payload = self.vlm_client.create_image_json(
                system_prompt=CROP_QUERY_PROMPT,
                user_text=f"Goal: {goal}",
                image_input=crop_path,
                max_tokens=200,
                model_name=self.model_name,
                response_schema=CROP_QUERY_SCHEMA,
            )
        except Exception as exc:
            self._last_query_runtime_metrics = exception_runtime_metrics(exc)
            self._last_query_error = f"{type(exc).__name__}: {exc}"
            return ""

        self._last_query_runtime_metrics = dict(
            payload.get(RUNTIME_METRICS_KEY, {}) or {}
        )

        query = str(payload.get("query", "")).strip()
        if query and not self._is_low_value_query(query):
            return query
        keywords = payload.get("keywords", [])
        if isinstance(keywords, list) and keywords:
            keyword_query = " ".join(
                str(item).strip() for item in keywords[:3] if str(item).strip()
            )
            if keyword_query and not self._is_low_value_query(keyword_query):
                return keyword_query
        return ""

    @staticmethod
    def _collect_candidate_urls(
        lens_results: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]],
    ) -> List[str]:
        urls: List[str] = []
        seen = set()
        for item in (lens_results or []) + (semantic_results or []):
            url = str(item.get("url", "")).strip()
            if not url or not CropAndSearchTool._is_preferred_candidate_url(url):
                continue
            if url not in seen:
                seen.add(url)
                urls.append(url)
        return urls

    @staticmethod
    def _collect_reference_image_candidates(
        lens_results: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]],
    ) -> List[str]:
        urls: List[str] = []
        seen = set()
        for item in (lens_results or []) + (semantic_results or []):
            image_url = str(item.get("image_url", "")).strip()
            if not image_url or image_url in seen or not CropAndSearchTool._is_image_candidate_url(image_url):
                continue
            seen.add(image_url)
            urls.append(image_url)
        return urls

    @staticmethod
    def _is_preferred_candidate_url(url: str) -> bool:
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False

        host = parsed.netloc.lower()
        path = parsed.path.lower()
        blocked_hosts = (
            "google.com",
            "www.google.com",
            "images.google.com",
            "facebook.com",
            "m.facebook.com",
            "instagram.com",
            "www.instagram.com",
            "pinterest.com",
            "www.pinterest.com",
        )
        blocked_suffixes = (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
            ".bmp",
            ".svg",
        )
        if any(host == blocked or host.endswith("." + blocked) for blocked in blocked_hosts):
            return False
        if any(path.endswith(ext) for ext in blocked_suffixes):
            return False
        return True

    @staticmethod
    def _is_image_candidate_url(url: str) -> bool:
        """Accept opaque provider image endpoints; verify bytes on download."""

        try:
            parsed = urlparse(url)
        except Exception:
            return False
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _merge_region_candidate_urls(regions: List[Dict[str, Any]]) -> List[str]:
        urls: List[str] = []
        seen = set()
        for region in regions:
            for value in region.get("candidate_page_urls", []) or []:
                url = str(value or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                urls.append(url)
        return urls

    @staticmethod
    def _merge_region_reference_candidates(regions: List[Dict[str, Any]]) -> List[str]:
        urls: List[str] = []
        seen = set()
        for region in regions:
            for value in region.get("reference_image_candidates", []) or []:
                url = str(value or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                urls.append(url)
        return urls

    @staticmethod
    def _pick_best_region(regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        order = {"high": 3, "medium": 2, "low": 1}
        best: Dict[str, Any] = {}
        best_score = -1.0
        for region in regions:
            relevance = str(region.get("relevance", "low")).lower()
            score = float(order.get(relevance, 0))
            evidence = str(region.get("evidence", ""))
            if evidence:
                score += min(len(evidence), 500) / 500.0
            if region.get("lens_results"):
                score += 0.5
            if score > best_score:
                best = region
                best_score = score
        return best

    @staticmethod
    def _is_low_value_query(query: str) -> bool:
        text = re.sub(r"\s+", " ", str(query or "").strip().lower())
        if not text:
            return True
        generic_patterns = (
            "identify the event and source of this image",
            "identify the source of this image",
            "identify the event in this image",
            "source of this image",
            "event in this image",
            "this image",
            "image source",
        )
        if text in generic_patterns:
            return True

        tokens = re.findall(r"[a-z0-9]+", text)
        if not tokens:
            return True
        stopwords = {
            "identify", "event", "source", "image", "photo", "picture", "find",
            "this", "that", "from", "with", "about", "what", "where", "which",
            "who", "when", "scene",
        }
        informative = [tok for tok in tokens if tok not in stopwords]
        return len(informative) < 2
