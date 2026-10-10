"""Stage tools for the unified ReAct runtime."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from eviscene.integrations.gemini import exception_runtime_metrics, take_runtime_metrics
from eviscene.orchestrator.route_policy import routes_semantically_equivalent
from eviscene.orchestrator.tool_cache import WEB_EVIDENCE_CONTRACT_VERSION
from eviscene.orchestrator.tool_result import (
    ToolResultContractError,
    parse_tool_result,
    serialize_tool_result,
)
from eviscene.orchestrator.tool_execution import (
    ToolActionRecord,
    ToolExecutionStatus,
    run_tool_with_timeout,
)
from eviscene.tools.base import BaseTool
from .stage_types import StageStep


class StageToolMixin:
    """Internal execution component; StageRunner owns shared case state."""

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute one tool with cache-miss single-flight protection."""

        tool_args = self._execution_tool_args(tool_args)

        if self.tool_cache and tool_name in self.cacheable_tools:
            cache_tool_args = dict(tool_args)
            properties = self.tools[tool_name].parameters.get(
                "properties",
                {},
            )
            if "image_input" in properties and not cache_tool_args.get("image_input"):
                cache_tool_args["image_input"] = self.image_path
            cache_args = self._build_cache_args(tool_name, cache_tool_args)
            async with self.tool_cache.singleflight(tool_name, cache_args):
                return await self._execute_tool_uncached(tool_name, tool_args)
        return await self._execute_tool_uncached(tool_name, tool_args)

    async def _execute_tool_uncached(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        tool = self.tools[tool_name]
        properties = tool.parameters.get("properties", {})
        if "image_input" in properties:
            tool_args["image_input"] = self.image_path
        if hasattr(tool, "image_path") and self.image_path:
            tool.image_path = self.image_path

        cache_args = self._build_cache_args(tool_name, tool_args)
        started = time.perf_counter()
        action = ToolActionRecord.start(
            tool_name,
            timeout_seconds=self.tool_timeout_seconds,
            continuation_id=self._session_previous_interaction_id(),
        )
        if self.runtime_store is not None:
            self.runtime_store.append_event(
                "tool_action_started",
                action.to_dict(),
            )

        def finish_metadata(
            status: ToolExecutionStatus,
            metadata: Dict[str, Any],
            *,
            error: str = "",
        ) -> Dict[str, Any]:
            lifecycle = action.finish(status, error=error)
            if self.runtime_store is not None:
                self.runtime_store.append_event(
                    "tool_action_finished",
                    lifecycle,
                )
            return {
                **metadata,
                **lifecycle,
            }

        if self.runtime_store is not None:
            recovered = self.runtime_store.reuse_tool_result(
                stage=self.stage_name,
                tool_name=tool_name,
                tool_args=cache_args,
            )
            if recovered is not None:
                recovered_result = str(recovered["result"])
                try:
                    recovered_payload, succeeded = parse_tool_result(recovered_result)
                except ToolResultContractError as exc:
                    self.runtime_store.append_event(
                        "tool_result_recovery_invalid",
                        {
                            "stage": self.stage_name,
                            "tool_name": tool_name,
                            "source_attempt_id": recovered.get("source_attempt_id"),
                            "error": str(exc),
                        },
                    )
                    recovered = None
                if recovered is not None:
                    if succeeded and self.source_access_policy.active:
                        sanitized, filtered_count = (
                            self.source_access_policy.sanitize_payload(
                                recovered_payload
                            )
                        )
                        if sanitized is None:
                            succeeded = False
                            recovered_result = json.dumps(
                                {
                                    "status": "error",
                                    "error": (
                                        "Recovered tool result blocked by the "
                                        "active source access policy."
                                    ),
                                },
                                ensure_ascii=False,
                            )
                        else:
                            if filtered_count:
                                sanitized["policy_filtered_count"] = (
                                    int(sanitized.get("policy_filtered_count", 0) or 0)
                                    + filtered_count
                                )
                            recovered_result, succeeded = serialize_tool_result(
                                sanitized
                            )
                    return recovered_result, finish_metadata(
                        "completed" if succeeded else "failed",
                        {
                            "cache_hit": True,
                            "recovery_cache_hit": True,
                            "recovery_source_attempt_id": recovered.get(
                                "source_attempt_id"
                            ),
                            "recovery_source_memory_id": recovered.get(
                                "source_memory_id"
                            ),
                            "tool_success": succeeded,
                            "observed_at": datetime.now(timezone.utc).isoformat(),
                            "duration_ms": round(
                                (time.perf_counter() - started) * 1000,
                                2,
                            ),
                            "serialized_size": len(recovered_result),
                        },
                        error=(
                            "" if succeeded else "recovered tool result was rejected"
                        ),
                    )

        if self.tool_cache and tool_name in self.cacheable_tools:
            cached = self.tool_cache.get(tool_name, cache_args)
            if cached is not None:
                try:
                    cached_result, succeeded = parse_tool_result(cached)
                except ToolResultContractError:
                    cached_result = None
                    succeeded = False
                if cached_result is not None:
                    if succeeded and self.source_access_policy.active:
                        sanitized, filtered_count = (
                            self.source_access_policy.sanitize_payload(cached_result)
                        )
                        if sanitized is None:
                            succeeded = False
                            cached = json.dumps(
                                {
                                    "status": "error",
                                    "error": (
                                        "Cached tool result blocked by the "
                                        "active source access policy."
                                    ),
                                },
                                ensure_ascii=False,
                            )
                        else:
                            if filtered_count:
                                sanitized["policy_filtered_count"] = (
                                    int(sanitized.get("policy_filtered_count", 0) or 0)
                                    + filtered_count
                                )
                            cached, succeeded = serialize_tool_result(sanitized)
                    return cached, finish_metadata(
                        "completed" if succeeded else "failed",
                        {
                            "cache_hit": True,
                            "tool_success": succeeded,
                            "observed_at": datetime.now(timezone.utc).isoformat(),
                            "duration_ms": round(
                                (time.perf_counter() - started) * 1000,
                                2,
                            ),
                            "serialized_size": len(cached),
                        },
                        error=("" if succeeded else "cached tool result was rejected"),
                    )

        try:
            if hasattr(tool, "call_async"):
                result = await run_tool_with_timeout(
                    tool.call_async(tool_args),
                    timeout_seconds=self.tool_timeout_seconds,
                )
            else:
                result = await run_tool_with_timeout(
                    asyncio.to_thread(tool.call, tool_args),
                    timeout_seconds=self.tool_timeout_seconds,
                )
        except asyncio.TimeoutError:
            serialized = json.dumps(
                {
                    "status": "error",
                    "error": (
                        f"ToolActionTimeout: {tool_name} exceeded "
                        f"{self.tool_timeout_seconds:.1f}s"
                    ),
                },
                ensure_ascii=False,
            )
            return serialized, finish_metadata(
                "timed_out",
                {
                    "cache_hit": False,
                    "tool_success": False,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "serialized_size": len(serialized),
                    "tool_exception": "ToolActionTimeout",
                    "tool_timeout_seconds": self.tool_timeout_seconds,
                    "tool_llm_api_calls": 0,
                    "tool_tokens": self._normalize_tool_tokens(None),
                },
                error=f"ToolActionTimeout: {tool_name}",
            )
        except Exception as exc:
            serialized = json.dumps(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )
            runtime_metrics = exception_runtime_metrics(exc)
            return serialized, finish_metadata(
                "failed",
                {
                    "cache_hit": False,
                    "tool_success": False,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "serialized_size": len(serialized),
                    "tool_exception": type(exc).__name__,
                    "tool_llm_api_calls": int(
                        runtime_metrics.get("llm_api_calls", 0) or 0
                    ),
                    "tool_tokens": self._normalize_tool_tokens(
                        runtime_metrics.get("tokens")
                    ),
                },
                error=f"{type(exc).__name__}: {exc}",
            )

        if not isinstance(result, dict):
            runtime_metrics: Dict[str, Any] = {}
            tool_subcalls: List[Dict[str, Any]] = []
        else:
            raw_subcalls = result.get("subcalls", [])
            tool_subcalls = (
                [dict(item) for item in raw_subcalls if isinstance(item, dict)]
                if isinstance(raw_subcalls, list)
                else []
            )
            runtime_metrics = take_runtime_metrics(result)
        try:
            serialized, succeeded = serialize_tool_result(result)
        except ToolResultContractError as exc:
            serialized = json.dumps(
                {"status": "error", "error": f"ToolResultContractError: {exc}"},
                ensure_ascii=False,
            )
            return serialized, finish_metadata(
                "failed",
                {
                    "cache_hit": False,
                    "tool_success": False,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "serialized_size": len(serialized),
                    "tool_exception": "ToolResultContractError",
                    "tool_llm_api_calls": int(
                        runtime_metrics.get("llm_api_calls", 0) or 0
                    ),
                    "tool_tokens": self._normalize_tool_tokens(
                        runtime_metrics.get("tokens")
                    ),
                    "tool_subcalls": tool_subcalls,
                },
                error=str(exc),
            )
        raw_serialized = serialized
        raw_tool_result_artifact: Optional[Dict[str, Any]] = None
        if succeeded and self.runtime_store is not None:
            raw_tool_result_artifact = self.runtime_store.artifacts.put_text(
                raw_serialized,
                media_type="application/json; charset=utf-8",
                suffix=".json",
                metadata={
                    "kind": "raw_tool_result",
                    "stage": self.stage_name,
                    "tool_name": tool_name,
                },
            )
        if succeeded:
            parsed_result, _ = parse_tool_result(serialized)
            serialized, succeeded = serialize_tool_result(
                self._strip_model_media(parsed_result)
            )
        if succeeded and self.source_access_policy.active:
            parsed_result, _ = parse_tool_result(serialized)
            sanitized, filtered_count = self.source_access_policy.sanitize_payload(
                parsed_result
            )
            if sanitized is None:
                serialized = json.dumps(
                    {
                        "status": "error",
                        "error": "Tool result blocked by the active source access policy.",
                    },
                    ensure_ascii=False,
                )
                succeeded = False
            else:
                if filtered_count:
                    sanitized["policy_filtered_count"] = (
                        int(sanitized.get("policy_filtered_count", 0) or 0)
                        + filtered_count
                    )
                serialized, succeeded = serialize_tool_result(sanitized)
        if succeeded and self.tool_cache and tool_name in self.cacheable_tools:
            self.tool_cache.put(tool_name, cache_args, serialized)
        return serialized, finish_metadata(
            "completed" if succeeded else "failed",
            {
                "cache_hit": False,
                "tool_success": succeeded,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "serialized_size": len(serialized),
                "tool_llm_api_calls": int(runtime_metrics.get("llm_api_calls", 0) or 0),
                "tool_tokens": self._normalize_tool_tokens(
                    runtime_metrics.get("tokens")
                ),
                "tool_subcalls": tool_subcalls,
                **(
                    {"raw_tool_result_artifact": raw_tool_result_artifact}
                    if raw_tool_result_artifact is not None
                    else {}
                ),
            },
            error="" if succeeded else "tool returned an error result",
        )

    @staticmethod
    def _normalize_tool_tokens(value: Any) -> Dict[str, int]:
        value = value if isinstance(value, dict) else {}
        return {
            name: int(value.get(name, 0) or 0)
            for name in ("prompt", "completion", "thought")
        }

    def _build_cache_args(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        args = dict(tool_args)
        # RuntimeToolAdapter hides image_input from its public parameters and
        # injects it only when the delegate runs.  The cache key must still
        # include that private image identity before a lookup or singleflight.
        if (
            tool_name
            in {
                "perceive_scene",
                "ocr_with_position",
                "reverse_image_search",
                "focused_visual_inspection",
                "crop_and_inspect",
                "count_objects",
                "check_consistency",
                "compare_with_reference",
                "analyze_visual_anomalies",
            }
            and self.image_path
        ):
            args["__image_input__"] = self.image_path
        if tool_name in {"visit", "crop_and_search"}:
            args["__web_evidence_contract__"] = WEB_EVIDENCE_CONTRACT_VERSION
        return args

    def _build_tool_response_message(
        self,
        result: str,
        *,
        function_call_id: str,
        native_chat: bool = False,
    ) -> str:
        payload = {
            "function_call_id": function_call_id,
            "result": self._model_visible_tool_result(result),
        }
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=None if native_chat else 2,
            separators=(",", ":") if native_chat else None,
        )
        if native_chat:
            return text
        return f"<tool_response>\n{text}\n</tool_response>"

    def _archive_tool_step(self, step: StageStep, *, action_index: int) -> None:
        if self.runtime_store is None or step.action_type != "tool_call":
            return
        descriptor = self.runtime_store.archive_tool_result(
            stage=self.stage_name,
            action_index=action_index,
            tool_name=step.tool_name,
            tool_args=step.tool_args,
            tool_result=step.tool_result,
            metadata={
                "context_request_id": self._last_context_request_id,
                "cache_hit": bool(step.metadata.get("cache_hit", False)),
                "tool_success": bool(step.metadata.get("tool_success", False)),
                "function_call_id": step.metadata.get("function_call_id"),
                "raw_tool_result_artifact": step.metadata.get(
                    "raw_tool_result_artifact"
                ),
            },
        )
        step.metadata["tool_result_artifact"] = descriptor

    def _model_visible_tool_result(
        self,
        result: str,
    ) -> Any:
        """Return the complete textual observation for the next policy turn.

        Provider media blobs and raw page HTML are transport artifacts rather
        than observations. Everything else is retained exactly as returned by
        the canonical tool result: this method intentionally has no character
        budget and no list/field reduction.
        """

        try:
            data = json.loads(result)
        except Exception:
            data = None

        if isinstance(data, dict) and str(data.get("status", "")).lower() in {
            "error",
            "failed",
            "failure",
        }:
            return data
        if data is None and str(result).strip().lower().startswith("error"):
            return {"status": "error", "error": str(result).strip()}

        if isinstance(data, (dict, list)):
            return self._strip_model_media(data)
        return str(result)

    @staticmethod
    def _strip_model_media(value: Any) -> Any:
        """Remove binary/provider payloads without changing textual observations."""

        omitted_keys = {
            "data_url",
            "image_input",
            "base64",
            "content_bytes",
            "raw_html",
            "html",
        }
        if isinstance(value, dict):
            return {
                str(key): StageToolMixin._strip_model_media(child)
                for key, child in value.items()
                if str(key).casefold() not in omitted_keys
            }
        if isinstance(value, list):
            return [StageToolMixin._strip_model_media(item) for item in value]
        if isinstance(value, tuple):
            return [StageToolMixin._strip_model_media(item) for item in value]
        return value

    def _has_duplicate_tool_call(
        self, steps: List[StageStep], tool_name: str, tool_args: Dict[str, Any]
    ) -> bool:
        for step in [
            *self.prior_steps,
            *steps,
        ]:
            if step.action_type != "tool_call":
                continue
            if routes_semantically_equivalent(
                step.tool_name,
                self._normalize_tool_args(step.tool_args),
                tool_name,
                self._normalize_tool_args(tool_args),
            ):
                return True
        return False

    def _available_tool_objects(
        self,
        *,
        steps: Optional[List[StageStep]] = None,
    ) -> List[BaseTool]:
        """Return tools that still have runtime budget for this stage.

        A tool budget is a hard runtime capability boundary, not merely a
        post-hoc validator. Keeping this calculation here makes every provider
        path use the same source of truth when constructing its tool schema.
        """

        current_steps = list(steps or [])
        return [
            tool
            for tool in self.tools_list
            if not self._tool_budget_reached(current_steps, tool.name)
        ]

    def _available_tool_names(
        self,
        *,
        steps: Optional[List[StageStep]] = None,
    ) -> List[str]:
        return [
            tool.name
            for tool in self._available_tool_objects(steps=steps)
            if str(tool.name).strip()
        ]

    def _tool_budget_reached(self, steps: List[StageStep], tool_name: str) -> bool:
        limit = self.tool_call_limits.get(tool_name)
        if limit is None:
            return False
        used = sum(
            1
            for step in [*self.prior_steps, *steps]
            if step.action_type == "tool_call" and step.tool_name == tool_name
        )
        return used >= limit

    @staticmethod
    def _normalize_tool_args(tool_args: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        for key, value in sorted(tool_args.items()):
            if key.startswith("__") or key == "question_id":
                continue
            if isinstance(value, list):
                normalized[key] = value
            elif isinstance(value, str):
                normalized[key] = value.strip()
            else:
                normalized[key] = value
        return normalized

    @staticmethod
    def _execution_tool_args(tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Remove runtime-owned routing fields before invoking a delegate."""

        return {
            key: value
            for key, value in tool_args.items()
            if not key.startswith("__") and key != "question_id"
        }

    def _unknown_tool_message(self, tool_name: str) -> str:
        available = ", ".join(self.tools.keys()) if self.tools else "(none)"
        return f"Tool '{tool_name}' is not available in this stage. Available tools: {available}"

    def _tool_unavailable_message(
        self,
        tool_name: str,
        available_tool_names: Sequence[str],
    ) -> str:
        available = (
            ", ".join(
                dict.fromkeys(
                    str(name).strip()
                    for name in available_tool_names
                    if str(name).strip()
                )
            )
            or "(none)"
        )
        return (
            f"Tool '{tool_name}' is not currently executable: it is exhausted "
            "or no longer belongs to the active runtime route. Do not call it "
            f"again. Available tools: {available}."
        )

    def _duplicate_tool_message(
        self,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None,
    ) -> str:
        rejected_call = json.dumps(
            {
                "tool": tool_name,
                "arguments": self._normalize_tool_args(tool_args or {}),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return (
            "The runtime rejected this duplicate call: "
            f"{rejected_call}. Do not repeat it in this correction chain. "
            "Use a different available tool or materially different arguments."
        )

    def _tool_budget_message(
        self,
        tool_name: str,
        *,
        steps: Optional[List[StageStep]] = None,
    ) -> str:
        available = self._available_tool_names(steps=steps)
        suffix = (
            " Available tools: " + ", ".join(available) + "."
            if available
            else " No executable tools remain; return the required JSON object."
        )
        return (
            f"Tool budget for '{tool_name}' is exhausted. Do not call it again; "
            "use another available tool or finalize the output." + suffix
        )
