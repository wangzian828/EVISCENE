from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from eviscene.integrations.browse.jina_reader import JinaReaderClient
from eviscene.integrations.gemini import RUNTIME_METRICS_KEY, exception_runtime_metrics
from eviscene.orchestrator.source_access import SourceAccessPolicy
from eviscene.tools.base import BaseTool


MAX_VISIT_URLS_PER_ACTION = 3


@dataclass
class VisitTool(BaseTool):
    """Goal-conditioned webpage visit using Jina Reader."""

    client: Optional[JinaReaderClient] = None
    source_access_policy: Optional[SourceAccessPolicy] = None
    name: str = "visit"
    description: str = (
        "Visit up to three candidate webpages concurrently and extract an "
        "independent exact passage from each for the runtime-bound image claim "
        "and retrieval goal."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "url": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": MAX_VISIT_URLS_PER_ACTION,
                    "description": (
                        "One to three candidate webpage URLs. Each page is "
                        "fetched and extracted independently."
                    ),
                },
                "image_claim": {
                    "type": "string",
                    "description": "Runtime-bound image claim used only for evidence stance.",
                },
                "retrieval_goal": {
                    "type": "string",
                    "description": "Passage sought by this action; used only for extraction.",
                },
            },
            "required": ["url", "image_claim", "retrieval_goal"],
        }
    )

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = JinaReaderClient()
        if self.source_access_policy is not None:
            self.set_source_access_policy(self.source_access_policy)

    def set_source_access_policy(self, policy: SourceAccessPolicy) -> None:
        self.source_access_policy = policy
        setter = getattr(self.client, "set_source_access_policy", None)
        if callable(setter):
            setter(policy)

    def visit(
        self,
        url: Any,
        *,
        image_claim: str,
        retrieval_goal: str,
    ) -> dict:
        try:
            if isinstance(url, list):
                urls = list(dict.fromkeys(
                    str(item).strip()
                    for item in url
                    if str(item).strip()
                ))
                if not urls or len(urls) > MAX_VISIT_URLS_PER_ACTION:
                    return {
                        "status": "error",
                        "error": (
                            "visit accepts one to three unique URLs per action."
                        ),
                    }
                if self.source_access_policy is not None:
                    blocked = [
                        item
                        for item in urls
                        if not self.source_access_policy.allows(item)
                    ]
                    if blocked:
                        return {
                            "status": "error",
                            "error": (
                                "One or more requested URLs are blocked by the "
                                "active source access policy."
                            ),
                        }
                if not urls:
                    return {
                        "status": "error",
                        "error": "All requested URLs are blocked by the active source access policy.",
                    }
                if len(urls) == 1:
                    result = self.client.visit(
                        urls[0],
                        image_claim=image_claim,
                        retrieval_goal=retrieval_goal,
                    )
                else:
                    result = self.client.visit_many(
                        urls,
                        image_claim=image_claim,
                        retrieval_goal=retrieval_goal,
                    )
            else:
                if self.source_access_policy is not None and not self.source_access_policy.allows(str(url)):
                    return {
                        "status": "error",
                        "error": "URL blocked by the active source access policy.",
                    }
                result = self.client.visit(
                    str(url),
                    image_claim=image_claim,
                    retrieval_goal=retrieval_goal,
                )
        except Exception as exc:
            result = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            metrics = exception_runtime_metrics(exc)
            if metrics:
                result[RUNTIME_METRICS_KEY] = metrics
            return result

        status = str(result.get("status", "")).strip().lower()
        visits = result.get("visits", [])
        all_visits_failed = bool(visits) and all(
            isinstance(visit, dict)
            and (
                str(visit.get("status", "")).strip().lower() == "error"
                or bool(str(visit.get("error", "")).strip())
                or bool(visit.get("blocked"))
            )
            for visit in visits
        )
        if status == "error" or result.get("error") or all_visits_failed:
            nested_errors = [
                str(visit.get("error", "")).strip()
                for visit in visits
                if isinstance(visit, dict) and str(visit.get("error", "")).strip()
            ]
            error = (
                str(result.get("error", "")).strip()
                or "; ".join(nested_errors)
                or "All webpage visits failed."
            )
            return {**result, "status": "error", "error": error}
        return {**result, "status": "success"}

    def call(self, params: dict) -> dict:
        return self.visit(
            url=params["url"],
            image_claim=params["image_claim"],
            retrieval_goal=params["retrieval_goal"],
        )
