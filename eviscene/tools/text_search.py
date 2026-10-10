from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from eviscene.integrations.search.jina_reranker import JinaRerankerClient
from eviscene.integrations.search.serper import SerperTextSearchClient
from eviscene.orchestrator.source_access import SourceAccessPolicy
from eviscene.tools.base import BaseTool


@dataclass
class TextSearchTool(BaseTool):
    """Serper search with optional internal candidate reranking."""

    client: Optional[SerperTextSearchClient] = None
    candidate_reranker: Optional[JinaRerankerClient] = None
    top_k: int = 10
    source_access_policy: Optional[SourceAccessPolicy] = None
    name: str = "text_search"
    description: str = (
        "Search the web and return candidate titles, URLs, and snippets as "
        "Discovery only. Candidate ordering may be internally reranked after "
        "the Serper search; use visit to inspect a selected page and create Evidence."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Exactly one non-empty query string. One text_search "
                        "action maps to one provider request."
                    ),
                },
                "gl": {"type": "string", "description": "Country code such as us or cn."},
                "hl": {"type": "string", "description": "Language code such as en or zh-cn."},
                "goal": {
                    "type": "string",
                    "description": (
                        "Immutable task goal recorded with the search response; "
                        "text_search does not extract or adjudicate Evidence."
                    ),
                },
            },
            "required": ["queries"],
        }
    )

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = SerperTextSearchClient()
        if self.source_access_policy is not None:
            self.set_source_access_policy(self.source_access_policy)

    def set_source_access_policy(self, policy: SourceAccessPolicy) -> None:
        self.source_access_policy = policy
        setter = getattr(self.client, "set_source_access_policy", None)
        if callable(setter):
            setter(policy)

    def search(
        self,
        queries: Union[List[str], str],
        *,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        goal: Optional[str] = None,
    ) -> Dict[str, Any]:
        if isinstance(queries, str):
            queries = [queries]
        queries = [
            str(query).strip()
            for query in queries
            if str(query).strip()
        ]
        if not queries:
            return {"status": "error", "error": "At least one non-empty search query is required."}
        if len(queries) != 1:
            return {
                "status": "error",
                "error": (
                    "text_search accepts exactly one query per action; submit "
                    "additional queries as later actions only if the core gap remains open."
                ),
            }
        query = queries[0]
        blocked_domain = (
            self.source_access_policy.blocked_query_reference(query)
            if self.source_access_policy is not None
            else ""
        )
        if blocked_domain:
            return {
                "status": "error",
                "error": (
                    "Search query explicitly targets a source excluded by the active "
                    "evaluation policy. Use independent open-web sources instead."
                ),
            }
        return self._build_result(
            [self._run_single_query(query, gl=gl, hl=hl, goal=goal)]
        )

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.search(
            queries=params["queries"],
            gl=params.get("gl"),
            hl=params.get("hl"),
            goal=params.get("goal"),
        )

    async def call_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(
            self.search,
            params["queries"],
            gl=params.get("gl"),
            hl=params.get("hl"),
            goal=params.get("goal"),
        )

    @staticmethod
    def _build_result(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        errors = [
            str(response.get("search_error", "")).strip()
            for response in responses
            if str(response.get("search_error", "")).strip()
        ]
        subcalls: List[Dict[str, Any]] = [
            {
                "kind": "search_query",
                "provider": str(
                    response.get("provider", "serper")
                ),
                "status": (
                    "error"
                    if response.get("search_error")
                    else "success"
                ),
                "request_count": 1,
                "result_count": len(
                    response.get("results", []) or []
                ),
                "duration_ms": float(
                    (response.get("timings", {}) or {}).get(
                        "search_ms",
                        0.0,
                    )
                    or 0.0
                ),
            }
            for response in responses
        ]
        for response in responses:
            selection = response.get("candidate_selection")
            if not isinstance(selection, dict):
                continue
            if selection.get("status") == "skipped":
                continue
            subcalls.append(
                {
                    "kind": "candidate_rerank",
                    "provider": str(
                        selection.get("provider", "jina_reranker")
                    ),
                    "status": str(selection.get("status", "error")),
                    "request_count": 1,
                    "result_count": int(
                        selection.get("output_count", 0) or 0
                    ),
                    "duration_ms": float(
                        selection.get("duration_ms", 0.0) or 0.0
                    ),
                }
            )
        result: Dict[str, Any] = {
            "status": "success",
            "queries": responses,
            "subcalls": subcalls,
        }
        if errors:
            result.update({"status": "error", "error": "; ".join(errors)})
        return result

    def _run_single_query(
        self,
        query: str,
        *,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        goal: Optional[str] = None,
    ) -> Dict[str, Any]:
        search_t0 = time.perf_counter()
        try:
            response = self.client.search(
                query,
                top_k=self.top_k,
                gl=gl,
                hl=hl,
            )
        except Exception as exc:
            search_duration_ms = round(
                (time.perf_counter() - search_t0) * 1000,
                2,
            )
            return {
                "query": query,
                "results": [],
                "goal": str(goal or query),
                "search_error": f"{type(exc).__name__}: {exc}",
                "timings": {"search_ms": search_duration_ms},
            }
        policy = self.source_access_policy
        if policy is not None:
            filtered, blocked_count = policy.filter_rows(response.get("results", []))
            response = dict(response)
            response["results"] = filtered
            # These provider aggregates may quote a filtered result without a URL.
            response["answer_box"] = None
            response["knowledge_graph"] = None
            if blocked_count:
                response["policy_filtered_count"] = blocked_count
        response = dict(response)
        response["results"] = list(response.get("results", []) or [])[: self.top_k]
        reranked_results, candidate_selection = self._rerank_candidates(
            query=query,
            goal=str(goal or query),
            response=response,
        )
        response["results"] = reranked_results
        response["candidate_selection"] = candidate_selection
        search_duration_ms = round((time.perf_counter() - search_t0) * 1000, 2)
        result = dict(response)
        result["goal"] = str(goal or query)
        result["timings"] = {"search_ms": search_duration_ms}
        return result

    def _rerank_candidates(
        self,
        *,
        query: str,
        goal: str,
        response: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        rows = [
            dict(item)
            for item in response.get("results", []) or []
            if isinstance(item, dict)
        ]
        if not rows:
            return [], {
                "provider": "jina_reranker",
                "status": "skipped",
                "reason": "no_serper_candidates",
            }
        if self.candidate_reranker is None:
            return rows, {
                "provider": "jina_reranker",
                "status": "skipped",
                "reason": "JINA_API_KEY is not configured",
            }

        documents = [
            "\n".join(
                part
                for part in (
                    str(row.get("title", "")).strip(),
                    str(row.get("snippet", "")).strip(),
                )
                if part
            )
            or str(row.get("url", "")).strip()
            for row in rows
        ]
        started = time.perf_counter()
        try:
            ranked = self.candidate_reranker.rerank(
                goal or query,
                documents,
                top_n=min(self.top_k, len(rows)),
            )
        except Exception as exc:
            return rows, {
                "provider": "jina_reranker",
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "duration_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
            }

        ordered: List[Dict[str, Any]] = []
        used_indexes: set[int] = set()
        for rerank_rank, item in enumerate(ranked, 1):
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= len(rows) or index in used_indexes:
                continue
            candidate = dict(rows[index])
            candidate["rerank_rank"] = rerank_rank
            if item.get("relevance_score") is not None:
                candidate["rerank_score"] = item.get("relevance_score")
            ordered.append(candidate)
            used_indexes.add(index)
        ordered.extend(
            dict(rows[index])
            for index in range(len(rows))
            if index not in used_indexes
        )
        return ordered, {
            "provider": "jina_reranker",
            "status": "success",
            "input_count": len(rows),
            "output_count": len(ordered),
            "duration_ms": round(
                (time.perf_counter() - started) * 1000,
                2,
            ),
        }
