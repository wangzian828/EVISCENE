# -*- coding: utf-8 -*-
"""Compare the current image with one reference image using a structured VLM."""
from __future__ import annotations

import asyncio
import os
import re
import base64
import io
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from eviscene.integrations.gemini import (
    RUNTIME_METRICS_KEY,
    exception_runtime_metrics,
)
from eviscene.integrations.http_sessions import (
    close_response,
    close_tracked_sessions,
    get_tracked_session,
    init_tracked_sessions,
)
from eviscene.tools.base import BaseTool
from eviscene.orchestrator.evidence_semantics import derive_edit_evidence_summary
from eviscene.orchestrator.source_access import SourceAccessPolicy


DIFFERENCE_TYPES = (
    "crop",
    "color_adjust",
    "watermark",
    "perspective",
    "lighting",
    "compression",
    "occlusion",
    "addition",
    "removal",
    "modification",
    "different_capture",
    "unrelated_content",
    "uncertain",
)
EDIT_DIFFERENCE_TYPES = frozenset({"addition", "removal", "modification"})
EDIT_STRENGTHS = ("none", "weak", "moderate", "strong")
SIGNIFICANCE_LEVELS = ("high", "medium", "low")
DEFAULT_REFERENCE_COMPARE_MAX_OUTPUT_TOKENS = 8192

COMPARE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "same_subject_or_scene": {"type": "boolean"},
        "same_capture_or_near_duplicate": {"type": "boolean"},
        "likely_different_original_capture": {"type": "boolean"},
        "differences": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "maxLength": 300},
                    "description": {"type": "string", "maxLength": 800},
                    "type": {"type": "string", "enum": list(DIFFERENCE_TYPES)},
                    "significance": {
                        "type": "string",
                        "enum": list(SIGNIFICANCE_LEVELS),
                    },
                },
            },
        },
        "overall_observation": {"type": "string", "maxLength": 1200},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

COMPARE_PROMPT = """\
Compare the two supplied images for evidence extraction, not final fact-check judgment.

Image 1 is a candidate reference image found through search.
Image 2 is the current image being verified.

Focus: {focus}

Describe concrete visual relationships and differences only.
1. Decide whether the images show the same subject, event, object, or scene.
2. Distinguish the same original capture or a near-duplicate from different original captures.
3. Use difference type addition, removal, or modification only for directly visible
   edit evidence. The runtime derives both the edit-evidence flag and its strength
   from these difference types and their significance; do not output either field.
4. Treat crop, resizing, compression, lighting, perspective, watermark, occlusion, and
   color shifts as benign unless they clearly alter factual content.
5. If the images are unrelated, report unrelated content without inferring manipulation.
6. Explain uncertainty in the observation rather than guessing manipulation.
"""

SYSTEM_INSTRUCTION = (
    "You are an image comparison analyst. Return exactly one JSON object matching "
    "the supplied response schema, without markdown or commentary."
)


@dataclass
class CompareWithReferenceTool(BaseTool):
    """Compare the current image with a reference image for visual evidence."""

    name: str = "compare_with_reference"
    description: str = (
        "Compare the current image with a reference image found via search and return "
        "low-level evidence about whether the images are near-duplicates, different "
        "original captures, or show direct edit evidence."
    )
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "reference_url": {
                "type": "string",
                "description": "URL of the reference image to compare against.",
            },
            "focus": {
                "type": "string",
                "description": (
                    "What to focus on in the comparison. For example: "
                    "'number of windows', 'person on the left', or 'banner text'."
                ),
            },
            "source_page_url": {
                "type": "string",
                "description": (
                    "Optional surrounding page URL used only to recover a blocked "
                    "or expired direct image URL."
                ),
            },
            "visual_question_id": {"type": "string"},
            "source_evidence_id": {"type": "string"},
            "source_discovery_id": {"type": "string"},
            "expected_property": {"type": "string"},
        },
        "required": ["reference_url"],
    })

    client: Any = None
    provider: str = "gemini"
    model_name: str = "gemini-3.7-flash"
    image_path: str = ""
    source_access_policy: Any = None
    _reference_cache: OrderedDict[str, tuple[float, Dict[str, Any], int]] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )
    _reference_cache_bytes: int = field(default=0, init=False, repr=False)
    _reference_cache_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _reference_cache_ttl_seconds: float = field(default=1800.0, init=False, repr=False)
    _reference_cache_max_bytes: int = field(
        default=64 * 1024 * 1024,
        init=False,
        repr=False,
    )
    _reference_thread_local: threading.local = field(
        default_factory=threading.local,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._reference_cache_ttl_seconds = self._env_float(
            "REFERENCE_IMAGE_CACHE_TTL_SECONDS",
            1800.0,
            minimum=0.0,
        )
        self._reference_cache_max_bytes = int(
            self._env_float(
                "REFERENCE_IMAGE_CACHE_MAX_BYTES",
                64 * 1024 * 1024,
                minimum=0.0,
            )
        )
        init_tracked_sessions(self)

    def close(self) -> None:
        close_tracked_sessions(self)

    def set_source_access_policy(self, policy: SourceAccessPolicy) -> None:
        self.source_access_policy = policy
        with self._reference_cache_lock:
            self._reference_cache.clear()
            self._reference_cache_bytes = 0

    def call(self, params: Dict[str, Any]) -> Any:
        """Synchronous compatibility entry point."""
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.call_async(params))

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            future = pool.submit(asyncio.run, self.call_async(params))
            return future.result(timeout=120)

    async def call_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Download the reference and compare both images through the configured VLM."""
        reference_url = str(params.get("reference_url", "")).strip()
        source_page_url = str(params.get("source_page_url", "")).strip()
        focus = str(params.get("focus", "general comparison")).strip() or "general comparison"

        if not reference_url:
            return self._error("reference_url is required.")
        if self.source_access_policy is not None and not self.source_access_policy.allows(reference_url):
            return self._error("Reference URL blocked by the active source access policy.")
        if (
            source_page_url
            and self.source_access_policy is not None
            and not self.source_access_policy.allows(source_page_url)
        ):
            return self._error(
                "Reference source-page URL blocked by the active source access policy."
            )
        if self.client is None:
            return self._error("VLM client not configured for image comparison.")
        create_images_json = getattr(self.client, "create_images_json", None)
        if not callable(create_images_json):
            return self._error(
                "VLM client does not support structured multi-image comparison."
            )

        try:
            runtime_metrics: Dict[str, Any] = {}
            download: Dict[str, Any] = {}
            download_subcalls: list[Dict[str, Any]] = []
            comparison_attempted = False
            comparison_started: Optional[float] = None
            comparison_duration_ms: Optional[float] = None
            download = (
                await self._download_reference(
                    reference_url,
                    source_page_url=source_page_url,
                )
                if source_page_url
                else await self._download_reference(reference_url)
            )
            if isinstance(download, str):
                download = {
                    "data_url": download,
                    "resolved_url": reference_url,
                    "download_method": "direct",
                    "attempted_urls": [reference_url],
                }
            download_subcalls = self._download_subcalls(download or {})
            reference_data_url = str(
                (download or {}).get("data_url", "")
            ).strip()
            if not reference_data_url:
                error = self._error(
                    f"Reference image access failed for {reference_url}"
                )
                error["comparison_status"] = "invalid_reference"
                error["error_class"] = "external_unavailable"
                error["attempted_urls"] = list(
                    (download or {}).get("attempted_urls", [])
                )
                error["subcalls"] = download_subcalls
                return error

            from eviscene.tools.vision_utils import (
                controlled_image_to_data_url,
                image_to_data_url,
                vision_tool_image_to_data_url,
            )

            current_data_url = image_to_data_url(self.image_path)
            deterministic = self._deterministic_exact_match(
                reference_data_url,
                current_data_url,
            )
            if deterministic is not None:
                return {
                    "status": "success",
                    "comparison_status": "same_capture_or_near_duplicate",
                    "reference_url": reference_url,
                    "resolved_reference_url": str(
                        (download or {}).get("resolved_url", reference_url)
                    ),
                    "source_page_url": source_page_url,
                    "download_method": str(
                        (download or {}).get("download_method", "direct")
                    ),
                    "attempted_urls": list(
                        (download or {}).get("attempted_urls", [reference_url])
                    ),
                    "reference_cache_hit": bool(
                        (download or {}).get("cache_hit", False)
                    ),
                    "comparison_method": "deterministic_exact_pixels",
                    "subcalls": [
                        *download_subcalls,
                        {
                            "kind": "image_compare",
                            "provider": "deterministic_pixels",
                            "status": "success",
                            "request_count": 1,
                        },
                    ],
                    **deterministic,
                    RUNTIME_METRICS_KEY: {},
                }
            reference_data_url, _ = controlled_image_to_data_url(
                reference_data_url
            )
            vision_current_data_url = vision_tool_image_to_data_url(self.image_path)
            comparison_attempted = True
            comparison_started = time.perf_counter()
            parsed = await asyncio.to_thread(
                create_images_json,
                system_prompt=SYSTEM_INSTRUCTION,
                user_text=COMPARE_PROMPT.format(focus=focus),
                image_inputs=[reference_data_url, vision_current_data_url],
                max_tokens=self._configured_max_output_tokens(),
                model_name=self.model_name,
                temperature=0.0,
                response_schema=COMPARE_RESPONSE_SCHEMA,
            )
            comparison_duration_ms = round(
                (time.perf_counter() - comparison_started) * 1000,
                2,
            )
            if not isinstance(parsed, Mapping):
                raise TypeError("VLM comparison response must be a JSON object.")
            parsed = dict(parsed)
            raw_runtime_metrics = parsed.pop(RUNTIME_METRICS_KEY, {})
            runtime_metrics = (
                dict(raw_runtime_metrics)
                if isinstance(raw_runtime_metrics, Mapping)
                else {}
            )
            validated = self._validate_response(parsed)
        except Exception as exc:
            error = self._error(
                "VLM comparison failed: "
                f"{type(exc).__name__}: {exc or '<no message>'}"
            )
            error["subcalls"] = [
                *locals().get("download_subcalls", []),
                *(
                    [
                        {
                            "kind": "image_compare",
                            "provider": str(
                                getattr(self.client, "provider", self.provider)
                            ),
                            "status": "error",
                            "request_count": 1,
                            "duration_ms": (
                                round(
                                    (
                                        time.perf_counter() - comparison_started
                                    )
                                    * 1000,
                                    2,
                                )
                                if comparison_started is not None
                                else None
                            ),
                        }
                    ]
                    if locals().get("comparison_attempted", False)
                    else []
                ),
            ]
            if runtime_metrics:
                error[RUNTIME_METRICS_KEY] = runtime_metrics
            else:
                metrics = exception_runtime_metrics(exc)
                if metrics:
                    error[RUNTIME_METRICS_KEY] = metrics
            return error

        return {
            "status": "success",
            "comparison_status": self._comparison_status(validated),
            "reference_url": reference_url,
            "resolved_reference_url": str(
                (download or {}).get("resolved_url", reference_url)
            ),
            "source_page_url": source_page_url,
            "download_method": str(
                (download or {}).get("download_method", "direct")
            ),
            "attempted_urls": list(
                (download or {}).get("attempted_urls", [reference_url])
            ),
            "reference_cache_hit": bool(
                (download or {}).get("cache_hit", False)
            ),
            "comparison_method": "vlm",
            "subcalls": [
                *self._download_subcalls(download or {}),
                {
                    "kind": "image_compare",
                    "provider": str(
                        getattr(self.client, "provider", self.provider)
                    ),
                    "status": "success",
                    "request_count": 1,
                    "duration_ms": comparison_duration_ms,
                },
            ],
            **validated,
            RUNTIME_METRICS_KEY: runtime_metrics,
        }

    @staticmethod
    def _configured_max_output_tokens() -> int:
        raw = os.getenv(
            "VLM_REFERENCE_COMPARE_MAX_OUTPUT_TOKENS",
            str(DEFAULT_REFERENCE_COMPARE_MAX_OUTPUT_TOKENS),
        )
        try:
            return max(1024, int(raw))
        except ValueError:
            return DEFAULT_REFERENCE_COMPARE_MAX_OUTPUT_TOKENS

    @staticmethod
    def _comparison_status(value: Mapping[str, Any]) -> str:
        """Classify a valid comparison for downstream evidence handling."""

        if value.get("same_capture_or_near_duplicate") is True:
            return "same_capture_or_near_duplicate"
        if value.get("same_subject_or_scene") is True:
            return "same_subject_or_scene"
        if value.get("likely_different_original_capture") is True:
            for difference in value.get("differences", []) or []:
                if (
                    isinstance(difference, Mapping)
                    and str(difference.get("type", "")).strip()
                    == "unrelated_content"
                ):
                    return "unrelated_candidate"
            return "different_original_capture"
        return "insufficient_comparison"

    @staticmethod
    def _env_float(name: str, default: float, *, minimum: float) -> float:
        try:
            return max(minimum, float(os.getenv(name, str(default))))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _deterministic_exact_match(
        reference_data_url: str,
        current_data_url: str,
    ) -> Optional[Dict[str, Any]]:
        """Return an exact-pixel match without spending a VLM call."""

        from PIL import Image, ImageChops

        try:
            reference_bytes = CompareWithReferenceTool._decode_data_url(
                reference_data_url
            )
            current_bytes = CompareWithReferenceTool._decode_data_url(
                current_data_url
            )
            with Image.open(io.BytesIO(reference_bytes)) as reference_image:
                reference = reference_image.convert("RGB")
            with Image.open(io.BytesIO(current_bytes)) as current_image:
                current = current_image.convert("RGB")
            if reference.size != current.size:
                return None
            if ImageChops.difference(reference, current).getbbox() is not None:
                return None
        except Exception:
            return None
        return {
            "same_subject_or_scene": True,
            "same_capture_or_near_duplicate": True,
            "likely_different_original_capture": False,
            "edit_evidence_present": False,
            "edit_evidence_strength": "none",
            "differences": [],
            "overall_observation": (
                "The reference and current image are pixel-identical after "
                "deterministic RGB normalization."
            ),
            "confidence": 1.0,
        }

    @staticmethod
    def _download_subcalls(download: Mapping[str, Any]) -> list[Dict[str, Any]]:
        if bool(download.get("cache_hit", False)):
            return []
        diagnostics = [
            dict(item)
            for item in download.get("download_diagnostics", []) or []
            if isinstance(item, Mapping)
        ]
        if diagnostics:
            return [
                {
                    "kind": "reference_download",
                    "provider": "reference_download",
                    "status": (
                        "success"
                        if item.get("outcome")
                        in {"image_downloaded", "page_images_extracted"}
                        else "error"
                    ),
                    "request_count": 1,
                    "stage": str(item.get("stage", "")),
                    "outcome": str(item.get("outcome", "")),
                    **(
                        {"http_status": int(item["http_status"])}
                        if isinstance(item.get("http_status"), int)
                        else {}
                    ),
                }
                for item in diagnostics
            ]
        attempted = [
            str(url).strip()
            for url in download.get("attempted_urls", []) or []
            if str(url).strip()
        ]
        resolved = str(download.get("resolved_url", "")).strip()
        return [
            {
                "kind": "page_fetch",
                "provider": "reference_download",
                "status": (
                    "success"
                    if resolved and index == len(attempted) - 1
                    else "error"
                ),
                "request_count": 1,
            }
            for index, _url in enumerate(attempted)
        ]

    @staticmethod
    def _decode_data_url(data_url: str) -> bytes:
        if not (data_url.startswith("data:") and ";base64," in data_url):
            raise ValueError("Image must be a base64 data URL.")
        return base64.b64decode(data_url.split(",", 1)[1], validate=True)

    async def _download_reference(
        self,
        url: str,
        *,
        source_page_url: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Download a reference image through direct, URL, and page fallbacks."""
        cache_key = self._reference_cache_key(url, source_page_url)
        cached = self._get_reference_cache(cache_key)
        if cached is not None:
            return cached
        return await asyncio.to_thread(
            self._download_reference_sync,
            url,
            source_page_url,
            cache_key,
        )

    def _download_reference_sync(
        self,
        url: str,
        source_page_url: str,
        cache_key: str,
    ) -> Optional[Dict[str, Any]]:
        """Download through a thread-local session with connection reuse."""
        import requests

        attempted: list[str] = []
        diagnostics: list[dict[str, Any]] = []
        pending = [
            (
                candidate,
                "direct" if candidate == url else "url_variant",
            )
            for candidate in self._reference_url_variants(url)
        ]
        if source_page_url:
            pending.append((source_page_url, "source_page"))
        seen: set[str] = set()
        try:
            client = self._get_reference_session()
            while pending and len(attempted) < 12:
                raw_candidate, stage = pending.pop(0)
                candidate = str(raw_candidate or "").strip()
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                if (
                    self.source_access_policy is not None
                    and not self.source_access_policy.allows(candidate)
                ):
                    diagnostics.append(
                        {
                            "stage": stage,
                            "url": candidate,
                            "outcome": "policy_blocked",
                        }
                    )
                    continue
                attempted.append(candidate)
                headers = {}
                if source_page_url and candidate != source_page_url:
                    headers["Referer"] = source_page_url
                try:
                    response = client.get(
                        candidate,
                        headers=headers,
                        timeout=30,
                        allow_redirects=True,
                    )
                except requests.RequestException as exc:
                    diagnostics.append(
                        {
                            "stage": stage,
                            "url": candidate,
                            "outcome": "request_error",
                            "error_type": type(exc).__name__,
                        }
                    )
                    continue
                try:
                    try:
                        self._validate_download_redirects(response)
                    except PermissionError:
                        diagnostics.append(
                            {
                                "stage": stage,
                                "url": candidate,
                                "outcome": "redirect_policy_blocked",
                            }
                        )
                        continue
                    if response.status_code != 200:
                        diagnostics.append(
                            {
                                "stage": stage,
                                "url": candidate,
                                "outcome": "http_error",
                                "http_status": int(response.status_code),
                            }
                        )
                        continue
                    if not response.content:
                        diagnostics.append(
                            {
                                "stage": stage,
                                "url": candidate,
                                "outcome": "empty_response",
                            }
                        )
                        continue
                    content_type = response.headers.get(
                        "content-type",
                        "",
                    ).split(";", 1)[0].strip().lower()
                    image_mime = (
                        content_type
                        if content_type.startswith("image/")
                        else self._sniff_image_mime(response.content)
                    )
                    if image_mime:
                        valid_image, invalid_reason = (
                            self._validate_reference_image_bytes(
                                response.content,
                                content_type=content_type,
                            )
                        )
                        if not valid_image:
                            diagnostics.append(
                                {
                                    "stage": stage,
                                    "url": candidate,
                                    "outcome": "invalid_image_payload",
                                    "reason": invalid_reason,
                                    "content_type": content_type,
                                }
                            )
                            continue
                        encoded = base64.b64encode(response.content).decode(
                            "ascii"
                        )
                        result = {
                            "data_url": (
                                f"data:{image_mime};base64,{encoded}"
                            ),
                            "resolved_url": str(response.url),
                            "download_origin": stage,
                            "download_method": (
                                "direct"
                                if candidate == url
                                else "url_or_page_fallback"
                            ),
                            "attempted_urls": attempted,
                            "download_diagnostics": [
                                *diagnostics,
                                {
                                    "stage": stage,
                                    "url": candidate,
                                    "outcome": "image_downloaded",
                                    "content_type": content_type,
                                },
                            ],
                        }
                        self._put_reference_cache(cache_key, result)
                        return result
                    if "html" not in content_type:
                        diagnostics.append(
                            {
                                "stage": stage,
                                "url": candidate,
                                "outcome": "not_image_content",
                                "content_type": content_type,
                            }
                        )
                        continue
                    html = response.text[:2_000_000]
                    page_images = self._extract_page_image_urls(
                        html,
                        base_url=str(response.url),
                    )
                    diagnostics.append(
                        {
                            "stage": stage,
                            "url": candidate,
                            "outcome": "page_images_extracted",
                            "candidate_count": len(page_images),
                        }
                    )
                    for image_url in page_images:
                        if image_url not in seen:
                            pending.append((image_url, "page_image"))
                finally:
                    close_response(response)
        except Exception as exc:
            diagnostics.append(
                {
                    "stage": "download_runtime",
                    "url": "",
                    "outcome": "runtime_error",
                    "error_type": type(exc).__name__,
                }
            )
        return {
            "data_url": "",
            "resolved_url": "",
            "download_method": "",
            "attempted_urls": attempted,
            "download_diagnostics": diagnostics,
        }

    def _get_reference_session(self) -> Any:
        session = get_tracked_session(
            self,
            self._reference_thread_local,
        )
        if not getattr(session, "_eviscene_reference_headers", False):
            session.headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/126.0 Safari/537.36"
                    ),
                    "Accept": (
                        "image/avif,image/webp,image/apng,image/svg+xml,"
                        "image/*,*/*;q=0.8"
                    ),
                    "Accept-Language": "en-US,en;q=0.8",
                }
            )
            session._eviscene_reference_headers = True
        return session

    @staticmethod
    def _reference_cache_key(url: str, source_page_url: str) -> str:
        return f"{str(url or '').strip()}\n{str(source_page_url or '').strip()}"

    def _get_reference_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if self._reference_cache_ttl_seconds <= 0 or self._reference_cache_max_bytes <= 0:
            return None
        now = time.time()
        with self._reference_cache_lock:
            entry = self._reference_cache.get(cache_key)
            if entry is None:
                return None
            created_at, result, _size = entry
            if now - created_at > self._reference_cache_ttl_seconds:
                self._reference_cache.pop(cache_key, None)
                self._reference_cache_bytes -= _size
                return None
            self._reference_cache.move_to_end(cache_key)
            cached = dict(result)
            cached["attempted_urls"] = list(
                cached.get("attempted_urls", []) or []
            )
            cached["download_diagnostics"] = [
                dict(item)
                for item in cached.get("download_diagnostics", []) or []
                if isinstance(item, Mapping)
            ]
            cached["cache_hit"] = True
            return cached

    def _put_reference_cache(
        self,
        cache_key: str,
        result: Mapping[str, Any],
    ) -> None:
        if self._reference_cache_ttl_seconds <= 0 or self._reference_cache_max_bytes <= 0:
            return
        data_url = str(result.get("data_url", ""))
        size = len(data_url.encode("utf-8"))
        if not data_url or size > self._reference_cache_max_bytes:
            return
        cached = dict(result)
        cached["attempted_urls"] = list(cached.get("attempted_urls", []) or [])
        cached["download_diagnostics"] = [
            dict(item)
            for item in cached.get("download_diagnostics", []) or []
            if isinstance(item, Mapping)
        ]
        with self._reference_cache_lock:
            previous = self._reference_cache.pop(cache_key, None)
            if previous is not None:
                self._reference_cache_bytes -= previous[2]
            while (
                self._reference_cache
                and self._reference_cache_bytes + size > self._reference_cache_max_bytes
            ):
                _old_key, (_created_at, _old_result, old_size) = (
                    self._reference_cache.popitem(last=False)
                )
                self._reference_cache_bytes -= old_size
            self._reference_cache[cache_key] = (time.time(), cached, size)
            self._reference_cache_bytes += size

    def _validate_download_redirects(self, response: Any) -> None:
        if self.source_access_policy is None:
            return
        for redirect in response.history:
            if not self.source_access_policy.allows(str(redirect.url)):
                raise PermissionError(
                    "Reference redirect hop blocked by the active source access policy."
                )
        if not self.source_access_policy.allows(str(response.url)):
            raise PermissionError(
                "Reference redirect target blocked by the active source access policy."
            )

    @staticmethod
    def _reference_url_variants(url: str) -> tuple[str, ...]:
        raw = str(url or "").strip()
        if not raw:
            return ()
        variants = [raw]
        try:
            parsed = urlsplit(raw)
        except ValueError:
            return tuple(variants)
        filtered_query = urlencode(
            [
                (key, value)
                for key, value in parse_qsl(
                    parsed.query,
                    keep_blank_values=True,
                )
                if key.casefold()
                not in {
                    "width",
                    "height",
                    "quality",
                    "resize",
                    "crop",
                    "format",
                    "w",
                    "h",
                }
            ]
        )
        without_transform = urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                filtered_query,
                "",
            )
        )
        if without_transform != raw:
            variants.append(without_transform)
        match = re.search(
            r"^(?P<prefix>.+?/commons)/thumb/(?P<rest>.+?)/"
            r"(?P<size>\d+px-[^/?#]+)$",
            without_transform,
        )
        if match:
            variants.append(
                f"{match.group('prefix')}/{match.group('rest')}"
            )
        return tuple(dict.fromkeys(variants))

    @staticmethod
    def _extract_page_image_urls(
        html: str,
        *,
        base_url: str,
    ) -> tuple[str, ...]:
        patterns = (
            r"""<meta[^>]+(?:property|name)=["'](?:og:image|twitter:image|twitter:image:src)["'][^>]+content=["']([^"']+)""",
            r"""<meta[^>]+content=["']([^"']+)["'][^>]+(?:property|name)=["'](?:og:image|twitter:image|twitter:image:src)["']""",
            r"""<img[^>]+src=["']([^"']+)""",
        )
        urls: list[str] = []
        for pattern in patterns:
            for value in re.findall(pattern, html, flags=re.IGNORECASE):
                candidate = urljoin(base_url, value.strip())
                if candidate.startswith(("http://", "https://")):
                    urls.append(candidate)
        return tuple(dict.fromkeys(urls[:20]))

    @staticmethod
    def _sniff_image_mime(content: bytes) -> str:
        if content.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if content.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
            return "image/webp"
        return ""

    @staticmethod
    def _validate_reference_image_bytes(
        content: bytes,
        *,
        content_type: str = "",
    ) -> tuple[bool, str]:
        """Reject HTML/SVG/login payloads that advertise themselves as images."""

        lowered_type = str(content_type or "").split(";", 1)[0].strip().lower()
        prefix = bytes(content[:512]).lstrip().lower()
        if lowered_type == "image/svg+xml" or prefix.startswith(
            (b"<svg", b"<?xml", b"<!doctype html", b"<html")
        ):
            return False, "reference payload is SVG or HTML, not a raster image"
        try:
            from PIL import Image

            with Image.open(io.BytesIO(content)) as image:
                image.verify()
        except Exception as exc:
            return False, f"reference payload could not be decoded as an image: {type(exc).__name__}"
        return True, ""

    @classmethod
    def _validate_response(cls, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("comparison output must be a JSON object")

        expected = set(COMPARE_RESPONSE_SCHEMA["properties"])
        # Older provider adapters emitted these derived fields. Accept them
        # when replaying such a response, but do not include them in the live
        # VLM schema; both values are determined from ``differences``.
        tolerated_legacy_fields = {
            "edit_evidence_present",
            "edit_evidence_strength",
        }
        actual = set(value)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected - tolerated_legacy_fields)
        if missing:
            raise ValueError("comparison output is missing fields: " + ", ".join(missing))
        if extra:
            raise ValueError("comparison output has unexpected fields: " + ", ".join(extra))

        # Only these three booleans are provider fields; edit_evidence_present
        # is derived from typed differences, not read from the response.
        boolean_fields = (
            "same_subject_or_scene",
            "same_capture_or_near_duplicate",
            "likely_different_original_capture",
        )
        for name in boolean_fields:
            if not isinstance(value[name], bool):
                raise ValueError(f"comparison output field '{name}' must be boolean")

        raw_edit_present = value.get("edit_evidence_present")
        if raw_edit_present is not None and not isinstance(raw_edit_present, bool):
            raise ValueError(
                "legacy comparison output field 'edit_evidence_present' must be boolean"
            )
        raw_edit_strength = value.get("edit_evidence_strength")
        if raw_edit_strength is not None and raw_edit_strength not in EDIT_STRENGTHS:
            raise ValueError(
                "legacy comparison output field 'edit_evidence_strength' is invalid"
            )

        overall = value["overall_observation"]
        if not isinstance(overall, str) or not overall.strip():
            raise ValueError(
                "comparison output field 'overall_observation' must be a non-empty string"
            )

        confidence = value["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("comparison output field 'confidence' must be a number")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("comparison output field 'confidence' must be between 0 and 1")

        differences = value["differences"]
        if not isinstance(differences, list):
            raise ValueError("comparison output field 'differences' must be an array")
        if len(differences) > 20:
            raise ValueError("comparison output field 'differences' exceeds 20 items")

        validated_differences = []
        for index, item in enumerate(differences):
            validated_differences.append(cls._validate_difference(item, index))

        same_subject = value["same_subject_or_scene"]
        same_capture = value["same_capture_or_near_duplicate"]
        different_capture = value["likely_different_original_capture"]
        # Derive edit evidence from addition/removal/modification differences,
        # so a redundant provider summary cannot contradict the typed result.
        edit_present, edit_strength = derive_edit_evidence_summary(
            validated_differences
        )
        contract_repairs: list[str] = []
        if raw_edit_present is not None:
            if raw_edit_present != edit_present:
                contract_repairs.append(
                    "edit_evidence_present_derived_from_difference_types"
                )
            else:
                contract_repairs.append("legacy_edit_evidence_present_ignored")
        if raw_edit_strength is not None:
            if raw_edit_strength != edit_strength:
                contract_repairs.append(
                    "edit_evidence_strength_derived_from_edit_difference_significance"
                )
            else:
                contract_repairs.append("legacy_edit_evidence_strength_ignored")

        if same_capture and not same_subject:
            raise ValueError("same capture requires same_subject_or_scene=true")
        if same_capture and different_capture:
            raise ValueError("same capture and different original capture cannot both be true")
        normalized = {
            "same_subject_or_scene": same_subject,
            "same_capture_or_near_duplicate": same_capture,
            "likely_different_original_capture": different_capture,
            "edit_evidence_present": edit_present,
            "edit_evidence_strength": edit_strength,
            "differences": validated_differences,
            "overall_observation": overall.strip(),
            "confidence": float(confidence),
        }
        if contract_repairs:
            normalized["contract_repairs"] = contract_repairs
            normalized["raw_edit_evidence_summary"] = {
                "edit_evidence_present": raw_edit_present,
                "edit_evidence_strength": raw_edit_strength,
            }
        return normalized

    @staticmethod
    def _validate_difference(value: Any, index: int) -> Dict[str, Any]:
        path = f"differences[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"comparison output {path} must be an object")

        expected = {"region", "description", "type", "significance"}
        if set(value) != expected:
            raise ValueError(f"comparison output {path} has invalid fields")
        for name in ("region", "description"):
            if not isinstance(value[name], str) or not value[name].strip():
                raise ValueError(f"comparison output {path}.{name} must be a non-empty string")

        difference_type = value["type"]
        if difference_type not in DIFFERENCE_TYPES:
            raise ValueError(f"comparison output {path}.type is invalid")
        significance = value["significance"]
        if significance not in SIGNIFICANCE_LEVELS:
            raise ValueError(f"comparison output {path}.significance is invalid")
        return {
            "region": value["region"].strip(),
            "description": value["description"].strip(),
            "type": difference_type,
            "significance": significance,
            "is_edit_evidence": difference_type in EDIT_DIFFERENCE_TYPES,
        }

    @staticmethod
    def _error(message: str) -> Dict[str, str]:
        return {"status": "error", "error": message.strip() or "Image comparison failed."}
