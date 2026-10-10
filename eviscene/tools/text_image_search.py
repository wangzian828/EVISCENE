"""Text-query image search for image-grounded investigation."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from eviscene.integrations.search.serper import SerperImageSearchClient
from eviscene.orchestrator.source_access import SourceAccessPolicy
from eviscene.tools.base import BaseTool


@dataclass
class TextImageSearchTool(BaseTool):
    """Search image results from one concrete text query.

    The result is Discovery only. Candidate images are unverified until the
    ReAct policy inspects or compares them in a later action.
    """

    client: Optional[SerperImageSearchClient] = None
    top_k: int = 3
    source_access_policy: Optional[SourceAccessPolicy] = None
    name: str = "text_image_search"
    description: str = (
        "Search web images from one precise text query and return candidate "
        "image/page pairs as unverified Discovery. Use this when a named "
        "event, person, place, object, or visible text can locate relevant "
        "images; inspect or compare a candidate before treating it as evidence."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Exactly one concrete image-search query tied to the "
                        "current image fact."
                    ),
                },
                "gl": {
                    "type": "string",
                    "description": "Country code such as us or cn.",
                },
                "hl": {
                    "type": "string",
                    "description": "Language code such as en or zh-cn.",
                },
            },
            "required": ["query"],
        }
    )

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = SerperImageSearchClient()
        if self.source_access_policy is not None:
            self.set_source_access_policy(self.source_access_policy)

    def set_source_access_policy(self, policy: SourceAccessPolicy) -> None:
        self.source_access_policy = policy
        setter = getattr(self.client, "set_source_access_policy", None)
        if callable(setter):
            setter(policy)

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = str(params.get("query", "")).strip()
        if not query:
            return {
                "status": "error",
                "error": "text_image_search requires one non-empty query.",
            }
        policy = self.source_access_policy
        blocked = (
            policy.blocked_query_reference(query)
            if policy is not None
            else ""
        )
        if blocked:
            return {
                "status": "error",
                "error": (
                    "Image-search query explicitly targets a source excluded "
                    "by the active evaluation policy."
                ),
            }
        started = time.perf_counter()
        try:
            rows = self.client.search(
                query,
                top_k=self.top_k,
                gl=params.get("gl"),
                hl=params.get("hl"),
            )
        except Exception as exc:
            return {
                "status": "error",
                "query": query,
                "error": f"{type(exc).__name__}: {exc}",
                "subcalls": [
                    {
                        "kind": "image_search_query",
                        "provider": "serper_image",
                        "status": "error",
                        "request_count": 1,
                        "result_count": 0,
                        "duration_ms": round(
                            (time.perf_counter() - started) * 1000,
                            2,
                        ),
                    }
                ],
            }

        filtered_count = 0
        if policy is not None:
            rows, filtered_count = policy.filter_rows(rows)
        rows = [dict(row) for row in rows[: self.top_k] if isinstance(row, dict)]
        image_urls = [
            str(row.get("image_url", "")).strip()
            for row in rows
            if str(row.get("image_url", "")).strip()
        ]
        page_urls = [
            str(row.get("url", "")).strip()
            for row in rows
            if str(row.get("url", "")).strip()
        ]
        result: Dict[str, Any] = {
            "status": "success",
            "query": query,
            "provider": "serper_image",
            "results": rows,
            "candidate_page_urls": list(dict.fromkeys(page_urls)),
            "reference_image_candidates": list(dict.fromkeys(image_urls)),
            "observation_status": "has_results" if rows else "empty_results",
            "subcalls": [
                {
                    "kind": "image_search_query",
                    "provider": "serper_image",
                    "status": "success",
                    "request_count": 1,
                    "result_count": len(rows),
                    "duration_ms": round(
                        (time.perf_counter() - started) * 1000,
                        2,
                    ),
                }
            ],
        }
        if filtered_count:
            result["policy_filtered_count"] = filtered_count
        if not rows:
            result["observation_note"] = (
                "The image-search request completed but returned no usable "
                "candidate images. The current question remains unresolved."
            )
        return result

    async def call_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return await asyncio.to_thread(self.call, params)
