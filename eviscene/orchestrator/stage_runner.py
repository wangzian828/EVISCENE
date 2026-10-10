# -*- coding: utf-8 -*-
"""Single-path stage runner for multi-round ReAct execution."""

from __future__ import annotations

from eviscene.environment import get_env

import asyncio
import json
import re
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type
from urllib.parse import urlparse

from pydantic import BaseModel

from eviscene.integrations.gemini import (
    extract_function_calls,
    extract_text,
    missing_required_paths,
    validate_interaction_response,
)
from eviscene.orchestrator.evidence_policy import query_policy_violation
from eviscene.orchestrator.llm_backend import (
    LLMBackend,
    LLMResponse,
    extract_policy_token_capture,
)
from eviscene.orchestrator.runtime_events import CaseRuntimeStore
from eviscene.orchestrator.tool_cache import (
    ToolResultCache,
)
from eviscene.orchestrator.tool_result import parse_tool_result
from eviscene.orchestrator.source_access import SourceAccessPolicy
from eviscene.orchestrator.source_provenance import canonicalize_url
from eviscene.tools.base import BaseTool
from eviscene.tools.vision_utils import controlled_image_to_data_url


from .stage_types import (
    InteractionSession as InteractionSession,
    StageStep as StageStep,
)
from .stage_native import NativeStageMixin
from .stage_schemas import StageSchemaMixin
from .stage_tools import StageToolMixin


def _bounded_timeout(
    value: Optional[float],
    *,
    env_name: str,
    default: float,
) -> float:
    """Resolve an action deadline without allowing a zero/negative timeout."""

    if value is None:
        raw = get_env(env_name, "").strip()
        try:
            value = float(raw) if raw else default
        except ValueError:
            value = default
    return max(5.0, float(value))


class StageRunner(NativeStageMixin, StageToolMixin, StageSchemaMixin):
    """A compact multi-round agent loop for one stage."""

    def __init__(
        self,
        llm: LLMBackend,
        system_prompt: str,
        tools: List[BaseTool],
        output_schema: Optional[Type[BaseModel]] = None,
        max_rounds: int = 5,
        image_path: str = "",
        stage_name: str = "",
        recent_rounds_to_keep: int = 3,
        tool_cache: Optional[ToolResultCache] = None,
        cacheable_tools: Optional[List[str]] = None,
        tool_call_limits: Optional[Dict[str, int]] = None,
        should_stop: Optional[Callable[[List[StageStep]], bool]] = None,
        output_validator: Optional[
            Callable[[BaseModel, List[StageStep]], Tuple[bool, str]]
        ] = None,
        min_tool_calls: int = 0,
        attach_image: bool = True,
        prior_steps: Optional[List[StageStep]] = None,
        max_output_tokens: Optional[int] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        final_output_max_tokens: Optional[int] = None,
        final_output_generation_config: Optional[Dict[str, Any]] = None,
        visual_call_validator: Optional[Callable[[str, Dict[str, Any]], str]] = None,
        source_access_policy: Optional[SourceAccessPolicy] = None,
        max_protocol_corrections: int = 4,
        max_tool_calls_per_turn: Optional[int] = None,
        force_tool_each_round: bool = False,
        stop_output_factory: Optional[Callable[[], BaseModel]] = None,
        protocol_exhaustion_boundary: bool = False,
        request_timeout_seconds: Optional[float] = None,
        tool_timeout_seconds: Optional[float] = None,
        tool_argument_constraints: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        interaction_session: Optional[InteractionSession] = None,
        runtime_store: Optional[CaseRuntimeStore] = None,
        prompt_version: str = "",
        native_history: Optional[List[Dict[str, Any]]] = None,
        native_system_instruction: Optional[str] = None,
        native_history_includes_pending_user: bool = False,
        question_claims: Optional[Dict[str, str]] = None,
        capture_policy_tokens: Optional[bool] = None,
        policy_topk: Optional[int] = None,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in tools}
        self.tools_list = tools
        self.output_schema = output_schema
        self.max_rounds = max_rounds
        self.image_path = image_path
        self.stage_name = stage_name
        self.recent_rounds_to_keep = max(1, recent_rounds_to_keep)
        self.tool_cache = tool_cache
        self.cacheable_tools = set(cacheable_tools or [])
        self.tool_call_limits = dict(tool_call_limits or {})
        self.should_stop = should_stop
        self.output_validator = output_validator
        self.min_tool_calls = max(0, int(min_tool_calls))
        self.attach_image = attach_image
        self.prior_steps = list(prior_steps or [])
        self.max_output_tokens = (
            max(1, int(max_output_tokens)) if max_output_tokens is not None else None
        )
        self.generation_config = dict(generation_config or {})
        self.final_output_max_tokens = (
            max(1, int(final_output_max_tokens))
            if final_output_max_tokens is not None
            else self.max_output_tokens
        )
        self.final_output_generation_config = {
            **self.generation_config,
            **dict(final_output_generation_config or {}),
        }
        self.llm_api_calls = 0
        self.visual_call_validator = visual_call_validator
        self.source_access_policy = source_access_policy or SourceAccessPolicy()
        self.max_protocol_corrections = max(0, int(max_protocol_corrections))
        self.max_tool_calls_per_turn = (
            max(1, int(max_tool_calls_per_turn))
            if max_tool_calls_per_turn is not None
            else None
        )
        self.force_tool_each_round = bool(force_tool_each_round)
        self.stop_output_factory = stop_output_factory
        self.protocol_exhaustion_boundary = bool(protocol_exhaustion_boundary)
        self.tool_argument_constraints = deepcopy(tool_argument_constraints or {})
        self.interaction_session = interaction_session
        self.runtime_store = runtime_store
        self.prompt_version = prompt_version or f"{stage_name or 'stage'}-v1"
        self.native_history = deepcopy(native_history or [])
        self.native_system_instruction = native_system_instruction
        self.native_history_includes_pending_user = bool(
            native_history_includes_pending_user
        )
        self.question_claims = dict(question_claims or {})
        self.capture_policy_tokens = capture_policy_tokens
        self.policy_topk = policy_topk
        self.last_native_history: List[Dict[str, Any]] = []
        self._last_context_request_id = ""
        self._last_interaction_lifecycle_kind = ""
        self._last_image_view: Dict[str, Any] = {}
        self._session_chat_history_start = 0
        self.request_timeout_seconds = _bounded_timeout(
            request_timeout_seconds,
            env_name="AGENT_STAGE_REQUEST_TIMEOUT_SECONDS",
            default=120.0,
        )
        self.tool_timeout_seconds = _bounded_timeout(
            tool_timeout_seconds,
            env_name="AGENT_TOOL_ACTION_TIMEOUT_SECONDS",
            default=150.0,
        )

    async def run(
        self, input_context: str
    ) -> Tuple[Optional[BaseModel], List[StageStep]]:
        """Run the ReAct loop."""
        if (
            str(getattr(self.llm, "provider", "")).lower() == "gemini"
            and str(getattr(self.llm, "wire_api", "")).lower() != "interactions"
        ):
            raise RuntimeError("Gemini stages require wire_api='interactions'.")
        if self._uses_native_interactions():
            try:
                return await self._run_native_interactions(input_context)
            except RuntimeError as exc:
                partial_steps = list(getattr(exc, "stage_steps", []) or [])
                boundary = self._protocol_exhaustion_stage_boundary(partial_steps)
                if boundary is not None:
                    return boundary
                raise
        if self._uses_native_structured_output():
            return await self._run_native_structured_output(input_context)

        steps: List[StageStep] = []
        native_chat = self._uses_native_chat_completions()
        system_msg = {
            "role": "system",
            "content": (
                self.native_system_instruction
                if native_chat and self.native_system_instruction is not None
                else self._build_native_chat_system_content()
                if native_chat
                else self._build_system_content()
            ),
        }
        user_msg = self._build_user_message(input_context)
        if native_chat and self.native_history:
            history: List[Dict[str, Any]] = [
                system_msg,
                *deepcopy(self.native_history),
            ]
            if not self.native_history_includes_pending_user:
                history.append(user_msg)
            self._session_chat_history_start = len(self.native_history)
        elif native_chat and self.interaction_session is not None:
            shared_history = deepcopy(self.interaction_session.chat_history)
            history: List[Dict[str, Any]] = [
                system_msg,
                *shared_history,
                user_msg,
            ]
            self._session_chat_history_start = len(shared_history)
        else:
            history = [system_msg, user_msg]
        evidence_so_far: List[str] = []

        # Chat Completions has no provider-side Interaction lifecycle to
        # distinguish a rejected protocol turn from an accepted ReAct action.
        # Keep the same bounded correction semantics as native Interactions:
        # correction-only requests do not consume the action budget.
        action_turns = 0
        correction_turns = 0
        request_index = 0
        correction_only_turns = native_chat and bool(
            self._build_native_tool_schemas(steps=steps)
        )
        next_lifecycle_kind = (
            "tool_roundtrip" if correction_only_turns else "standalone_request"
        )
        next_parent_context_request_id = ""
        next_generation_config: Optional[Dict[str, Any]] = None

        def direct_schema_generation_config() -> Optional[Dict[str, Any]]:
            """Disable hidden reasoning only for a rejected schema retry.

            A Qwen Chat Completions response may contain a candidate solely in
            its reasoning field.  Valid candidates still pass the normal schema
            validator.  When that candidate is invalid, repeating the original
            thinking request can replay the same stale schema indefinitely.
            Keep the first request unchanged, but make its correction a direct
            structured-output request.  This changes decoding only; it neither
            accepts nor translates an invalid response.
            """

            if not (
                native_chat
                and not self.tools_list
                and self.output_schema is not None
                and bool(self.generation_config.get("enable_thinking"))
            ):
                return None
            config = dict(self.generation_config)
            config["enable_thinking"] = False
            config.pop("thinking_token_budget", None)
            return config

        def needs_direct_schema_correction(affected: StageStep) -> bool:
            return bool(
                affected.metadata.get("response_content_chars", 0) == 0
                and affected.metadata.get("response_reasoning_chars", 0) > 0
                and direct_schema_generation_config() is not None
            )

        def request_chat_protocol_correction(
            affected: StageStep,
            reason: str,
            *,
            direct_schema_output: bool = False,
        ) -> bool:
            nonlocal correction_turns
            nonlocal next_lifecycle_kind
            nonlocal next_parent_context_request_id
            nonlocal next_generation_config
            if direct_schema_output:
                next_generation_config = direct_schema_generation_config()
            if not correction_only_turns:
                next_lifecycle_kind = "protocol_correction"
                next_parent_context_request_id = str(
                    affected.metadata.get("context_request_id", "")
                ).strip()
                return True
            affected.metadata["react_action_turn"] = action_turns
            affected.metadata["chat_request_index"] = request_index
            if correction_turns >= self.max_protocol_corrections:
                affected.metadata["protocol_corrections_used"] = correction_turns
                affected.metadata["correction_budget_exhausted"] = True
                affected.metadata["termination_reason"] = (
                    "protocol_correction_budget_exhausted"
                )
                affected.metadata.setdefault("rejection_reason", reason)
                return False
            correction_turns += 1
            affected.metadata["protocol_corrections_used"] = correction_turns
            next_lifecycle_kind = "protocol_correction"
            next_parent_context_request_id = str(
                affected.metadata.get("context_request_id", "")
            ).strip()
            return True

        while (
            action_turns if correction_only_turns else request_index
        ) < self.max_rounds:
            request_index += 1
            round_num = request_index
            active_tool_schemas = self._build_native_tool_schemas(steps=steps)
            active_tool_names = {
                str(item.get("name", "")).strip()
                for item in active_tool_schemas
                if str(item.get("name", "")).strip()
            }
            if native_chat:
                system_msg["content"] = self._build_native_chat_system_content(
                    available_tool_names=active_tool_names,
                )
            else:
                system_msg["content"] = self._build_system_content(
                    steps=steps,
                )
            messages = self._build_round_messages(
                system_msg,
                user_msg,
                history,
                evidence_so_far,
            )
            completed_tool_calls = sum(
                1 for item in steps if item.action_type == "tool_call"
            )
            response, llm_metadata = await self._call_llm(
                messages,
                require_tool=(
                    native_chat
                    and bool(active_tool_schemas)
                    and (
                        self.force_tool_each_round
                        or completed_tool_calls < self.min_tool_calls
                    )
                ),
                lifecycle_kind=next_lifecycle_kind,
                parent_context_request_id=next_parent_context_request_id,
                generation_config=next_generation_config,
                active_steps=steps,
            )
            next_lifecycle_kind = (
                "tool_roundtrip"
                if native_chat and bool(active_tool_schemas)
                else "standalone_request"
            )
            next_parent_context_request_id = ""
            next_generation_config = None
            step = StageStep(
                round=round_num,
                stage_name=self.stage_name,
                tokens={
                    "prompt": response.prompt_tokens,
                    "completion": response.completion_tokens,
                },
                metadata={
                    "stage": self.stage_name,
                    **llm_metadata,
                    "policy_input": self._policy_input_snapshot(
                        system_instruction=system_msg["content"],
                        input_payload=messages[1:],
                        tools=active_tool_schemas if native_chat else [],
                        response_format=(
                            self._openai_response_format()
                            if native_chat and not active_tool_schemas
                            else None
                        ),
                    ),
                },
            )

            content = (response.text or "").strip()
            history_content = content
            if (
                int(llm_metadata.get("response_content_chars", 0) or 0) == 0
                and int(llm_metadata.get("response_reasoning_chars", 0) or 0) > 0
            ):
                # A reasoning-only Qwen response is an internal diagnostic
                # candidate, not assistant-visible conversation history.
                # Keep it archived in the runtime ledger, but do not replay
                # truncated hidden reasoning into a direct-schema correction.
                history_content = ""
            if not content:
                step.action_type = "format_error"
                step.thought = "(empty response)"
                steps.append(step)
                history.append({"role": "assistant", "content": ""})
                history.append(
                    {
                        "role": "user",
                        "content": "Response was empty. Produce one valid tool call or final output.",
                    }
                )
                if request_chat_protocol_correction(
                    step,
                    "the model returned an empty response",
                ):
                    continue
                break

            step.thought = self._extract_think(content)
            native_assistant = (
                self._openai_assistant_message(response.raw) if native_chat else None
            )
            native_call = (
                self._openai_single_tool_call(response.raw) if native_chat else None
            )

            if "<tool_call>" in content and "</tool_call>" in content:
                tool_name, tool_args = self._parse_tool_call(content)
                if tool_name not in self.tools:
                    step.action_type = "format_error"
                    step.metadata["error_class"] = "protocol_error"
                    step.metadata["invalid_tool_name"] = tool_name
                    steps.append(step)
                    history.append({"role": "assistant", "content": content})
                    history.append(
                        {
                            "role": "user",
                            "content": self._unknown_tool_message(tool_name),
                        }
                    )
                    if request_chat_protocol_correction(
                        step,
                        f"unknown tool {tool_name!r}",
                    ):
                        continue
                    break
                if tool_name not in active_tool_names:
                    step.action_type = "format_error"
                    step.metadata["error_class"] = "protocol_error"
                    step.metadata["tool_unavailable"] = True
                    unavailable_message = self._tool_unavailable_message(
                        tool_name,
                        active_tool_names,
                    )
                    step.metadata["rejection_reason"] = unavailable_message
                    steps.append(step)
                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": unavailable_message})
                    if request_chat_protocol_correction(
                        step,
                        unavailable_message,
                    ):
                        continue
                    break

                step.action_type = "tool_call"
                step.tool_name = tool_name
                step.tool_args = dict(tool_args)
                step.metadata["policy_action"] = {
                    "type": "tool_call",
                    "name": tool_name,
                    "arguments": deepcopy(step.tool_args),
                }
                if native_call is not None:
                    step.metadata["native_chat_completions"] = True
                    step.metadata["function_call_id"] = str(
                        native_call.get("id", "")
                    ).strip()
                if self._has_duplicate_tool_call(steps, tool_name, step.tool_args):
                    duplicate_message = self._duplicate_tool_message(
                        tool_name,
                        step.tool_args,
                    )
                    step.action_type = "format_error"
                    step.metadata["error_class"] = "protocol_error"
                    step.metadata["duplicate_tool_call"] = True
                    step.metadata["rejection_reason"] = duplicate_message
                    step.tool_result = json.dumps(
                        {"status": "error", "error": duplicate_message},
                        ensure_ascii=False,
                    )
                    steps.append(step)
                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": duplicate_message})
                    if request_chat_protocol_correction(
                        step,
                        duplicate_message,
                    ):
                        continue
                    break
                if self._tool_budget_reached(steps, tool_name):
                    step.action_type = "format_error"
                    step.metadata["error_class"] = "protocol_error"
                    step.metadata["tool_budget_reached"] = True
                    steps.append(step)
                    history.append({"role": "assistant", "content": content})
                    history.append(
                        {
                            "role": "user",
                            "content": self._tool_budget_message(
                                tool_name,
                                steps=steps,
                            ),
                        }
                    )
                    if request_chat_protocol_correction(
                        step,
                        f"{tool_name} budget reached",
                    ):
                        continue
                    break
                filtered_query_count = self._sanitize_search_queries(
                    tool_name,
                    step.tool_args,
                )
                if filtered_query_count:
                    step.metadata["policy_filtered_query_count"] = filtered_query_count
                search_policy_error = self._search_policy_error(
                    tool_name,
                    step.tool_args,
                    rejected_query_count=filtered_query_count,
                )
                if search_policy_error:
                    self._record_search_query_rejection(
                        step,
                        tool_name,
                        search_policy_error,
                    )
                    step.tool_result = json.dumps(
                        {"status": "error", "error": search_policy_error},
                        ensure_ascii=False,
                    )
                    steps.append(step)
                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": search_policy_error})
                    if request_chat_protocol_correction(step, search_policy_error):
                        continue
                    break
                if self.visual_call_validator is not None:
                    visual_error = self.visual_call_validator(tool_name, step.tool_args)
                    if visual_error:
                        step.action_type = "format_error"
                        step.metadata["error_class"] = "protocol_error"
                        step.metadata["invalid_visual_question"] = True
                        step.tool_result = json.dumps(
                            {"status": "error", "error": visual_error},
                            ensure_ascii=False,
                        )
                        steps.append(step)
                        history.append({"role": "assistant", "content": content})
                        history.append({"role": "user", "content": visual_error})
                        if request_chat_protocol_correction(step, visual_error):
                            continue
                        break
                serialized, tool_metadata = await self._execute_tool(
                    tool_name,
                    self._execution_tool_args(step.tool_args),
                )
                step.tool_result = serialized
                step.metadata.update(tool_metadata)
                self._archive_tool_step(
                    step,
                    action_index=len(self.prior_steps) + len(steps) + 1,
                )
                step.metadata.setdefault(
                    "function_call_id",
                    (
                        f"legacy-{self.stage_name or 'stage'}-{round_num}-"
                        f"{len(self.prior_steps) + len(steps) + 1}"
                    ),
                )
                steps.append(step)
                evidence_so_far.append(
                    self._summarize_tool_result(tool_name, step.tool_args, serialized)
                )
                tool_response = self._build_tool_response_message(
                    serialized,
                    function_call_id=str(step.metadata["function_call_id"]),
                    native_chat=(
                        native_assistant is not None and native_call is not None
                    ),
                )
                if native_assistant is not None and native_call is not None:
                    history.append(native_assistant)
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(step.metadata["function_call_id"]),
                            "name": tool_name,
                            "content": tool_response,
                        }
                    )
                else:
                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": tool_response})

                action_turns += 1
                step.metadata["react_action_turn"] = action_turns
                step.metadata["protocol_corrections_used"] = correction_turns
                step.metadata["chat_request_index"] = request_index
                if self.should_stop and self.should_stop(steps):
                    if self.stop_output_factory is not None:
                        parsed = self.stop_output_factory()
                        steps.append(
                            StageStep(
                                round=round_num + 1,
                                stage_name=self.stage_name,
                                action_type="output",
                                output=parsed.model_dump(),
                                metadata={
                                    "stage": self.stage_name,
                                    "deterministic_segment_boundary": True,
                                },
                            )
                        )
                        self._commit_native_chat_history(history)
                        return parsed, steps
                    break
                next_lifecycle_kind = "tool_roundtrip"
                next_parent_context_request_id = str(
                    step.metadata.get("context_request_id", "")
                ).strip()
                continue

            output_json = self._extract_output(content)
            format_repair = ""
            if output_json is None:
                output_json, format_repair = self._try_parse_bare_json_with_repair(
                    content
                )
            if output_json is not None:
                step.action_type = "output"
                step.output = output_json
                step.metadata["policy_action"] = deepcopy(output_json)
                if format_repair:
                    step.metadata["format_repair"] = format_repair
                steps.append(step)
                parsed, schema_error = self._validate_output_with_error(output_json)
                if parsed is not None:
                    accepted, reason = self._accept_output(parsed, steps)
                    if accepted:
                        history.append(
                            native_assistant
                            if native_assistant is not None
                            else {"role": "assistant", "content": content}
                        )
                        self._commit_native_chat_history(history)
                        return parsed, steps
                    step.action_type = "output_rejected"
                    step.metadata["rejection_reason"] = reason
                    history.append({"role": "assistant", "content": content})
                    history.append(
                        {
                            "role": "user",
                            "content": self._structured_output_correction_prompt(
                                reason
                            ),
                        }
                    )
                    if request_chat_protocol_correction(step, reason):
                        continue
                    break
                step.action_type = "output_rejected"
                rejection_reason = (
                    schema_error or "output schema was invalid or incomplete"
                )
                step.metadata["rejection_reason"] = rejection_reason
                self._attach_invalid_response_preview(step, content)
                history.append({"role": "assistant", "content": history_content})
                history.append(
                    {
                        "role": "user",
                        "content": (
                            "Output schema was invalid: "
                            + rejection_reason
                            + " Return one corrected complete JSON object."
                        ),
                    }
                )
                if request_chat_protocol_correction(
                    step,
                    rejection_reason,
                    direct_schema_output=needs_direct_schema_correction(step),
                ):
                    continue
                break

            step.action_type = "format_error"
            self._attach_invalid_response_preview(step, content)
            steps.append(step)
            history.append({"role": "assistant", "content": history_content})
            history.append(
                {
                    "role": "user",
                    "content": (
                        "Use exactly one native function call or one JSON object."
                        if native_chat
                        else "Use exactly one <tool_call> or one <output> block."
                    ),
                }
            )
            if request_chat_protocol_correction(
                step,
                "the response was neither a function call nor valid JSON",
                direct_schema_output=needs_direct_schema_correction(step),
            ):
                continue
            break

        boundary = self._protocol_exhaustion_stage_boundary(steps)
        if boundary is not None:
            self._commit_native_chat_history(history)
            return boundary

        last_rejection_reason = ""
        for prior_step in reversed(steps):
            candidate_reason = str(
                (prior_step.metadata or {}).get("rejection_reason", "")
            ).strip()
            if candidate_reason:
                last_rejection_reason = candidate_reason
                break
        last_request_step = next(
            (
                item
                for item in reversed(steps)
                if str(item.metadata.get("context_request_id", "")).strip()
            ),
            None,
        )
        forced_is_correction = bool(
            last_request_step is not None
            and (
                last_request_step.action_type in {"format_error", "output_rejected"}
                or last_request_step.metadata.get("rejection_reason")
            )
        )
        forced_lifecycle_kind = (
            "protocol_correction"
            if forced_is_correction
            else "tool_roundtrip"
            if native_chat and bool(self.tools_list)
            else "standalone_request"
        )
        forced_parent_context_request_id = (
            str(last_request_step.metadata.get("context_request_id", "")).strip()
            if last_request_step is not None
            and forced_lifecycle_kind != "standalone_request"
            else ""
        )
        forced, forced_meta = await self._force_output(
            system_msg,
            user_msg,
            history,
            evidence_so_far,
            last_rejection_reason=last_rejection_reason,
            lifecycle_kind=forced_lifecycle_kind,
            parent_context_request_id=forced_parent_context_request_id,
        )
        if forced is not None:
            accepted, reason = self._accept_output(forced, steps, final_attempt=True)
            if not accepted:
                forced_meta["rejection_reason"] = reason
                forced_meta["policy_action"] = forced.model_dump()
                forced = None
        rejected_output = forced_meta.get("policy_action")
        final_step = StageStep(
            round=len(steps) + 1,
            stage_name=self.stage_name,
            action_type=(
                "output"
                if forced is not None
                else "output_rejected"
                if isinstance(rejected_output, dict)
                else "format_error"
            ),
            output=(
                forced.model_dump()
                if forced is not None
                else rejected_output
                if isinstance(rejected_output, dict)
                else None
            ),
            metadata={"stage": self.stage_name, **forced_meta},
        )
        steps.append(final_step)
        if forced is None and self.protocol_exhaustion_boundary:
            # A semantic checkpoint may opt into a deterministic no-op boundary
            # after its last schema/validator retry. Do not issue a third provider
            # request or turn an invalid checkpoint into a factual decision.
            final_step.metadata["correction_budget_exhausted"] = True
            final_step.metadata["termination_reason"] = (
                "protocol_correction_budget_exhausted"
            )
            boundary = self._protocol_exhaustion_stage_boundary(steps)
            if boundary is not None:
                self._commit_native_chat_history(history)
                return boundary
        self._commit_native_chat_history(history)
        return forced, steps

    def _protocol_exhaustion_stage_boundary(
        self,
        steps: List[StageStep],
    ) -> Optional[Tuple[BaseModel, List[StageStep]]]:
        """Close one failed correction chain without inventing a tool action.

        This is opt-in because a generic structured-output stage should still fail
        closed.  A bounded ReAct orchestrator may instead return to its semantic
        checkpoint after the model repeatedly fails to select a new executable
        route.  The boundary records every rejected provider request it resolves;
        it does not count as an action or claim that a retrieval route succeeded.
        """

        if not self.protocol_exhaustion_boundary or self.stop_output_factory is None:
            return None
        exhausted = [
            step
            for step in steps
            if bool(step.metadata.get("correction_budget_exhausted"))
        ]
        if not exhausted:
            return None
        resolved_request_ids = list(
            dict.fromkeys(
                str(step.metadata.get("context_request_id", "")).strip()
                for step in steps
                if step.action_type
                in {
                    "format_error",
                    "output_rejected",
                    "policy_replan",
                }
                and str(step.metadata.get("context_request_id", "")).strip()
            )
        )
        parsed = self.stop_output_factory()
        boundary_step = StageStep(
            round=len(steps) + 1,
            stage_name=self.stage_name,
            action_type="output",
            output=parsed.model_dump(),
            metadata={
                "stage": self.stage_name,
                "deterministic_segment_boundary": True,
                "protocol_correction_exhaustion_boundary": True,
                "resolved_rejection_request_ids": resolved_request_ids,
                "protocol_corrections_used": max(
                    int(step.metadata.get("protocol_corrections_used", 0) or 0)
                    for step in exhausted
                ),
                "termination_reason": "protocol_correction_budget_exhausted",
            },
        )
        steps.append(boundary_step)
        return parsed, steps

    def _uses_native_interactions(self) -> bool:
        """Use Gemini's native function protocol when this stage has tools."""
        return bool(
            self.tools_list
            and str(getattr(self.llm, "provider", "")).lower() == "gemini"
            and str(getattr(self.llm, "wire_api", "")).lower() == "interactions"
            and callable(getattr(self.llm, "create_interaction", None))
        )

    def _uses_native_structured_output(self) -> bool:
        return bool(
            not self.tools_list
            and self.output_schema is not None
            and str(getattr(self.llm, "provider", "")).lower() == "gemini"
            and str(getattr(self.llm, "wire_api", "")).lower() == "interactions"
            and callable(getattr(self.llm, "create_interaction", None))
        )

    @staticmethod
    def _attach_partial_steps(exc: Exception, steps: List[StageStep]) -> None:
        if steps and not hasattr(exc, "stage_steps"):
            setattr(exc, "stage_steps", list(steps))

    def _build_native_system_content(
        self,
        *,
        available_tool_names: Optional[Sequence[str]] = None,
    ) -> str:
        prompt = re.sub(
            r"Return exactly one JSON object inside <output>\.\.\.</output>",
            "Return exactly one JSON object",
            self.system_prompt,
        )
        if available_tool_names is None:
            available_tool_names = self._available_tool_names()
        available = list(
            dict.fromkeys(
                str(name).strip() for name in available_tool_names if str(name).strip()
            )
        )
        availability_instruction = (
            "- The runtime tool list is authoritative. Invoke only a function "
            "currently present in that list; a tool omitted from the list is "
            "unavailable and must not be called.\n"
            f"- Currently executable tools: {', '.join(available)}.\n"
            if available
            else (
                "- No executable tool remains in this segment. Do not call a "
                "function; return the required JSON object.\n"
            )
        )
        return (
            prompt
            + "\n\nNative Gemini Interactions protocol:\n"
            + "- Invoke tools through native function calls. Never write <tool_call> markup.\n"
            + availability_instruction
            + (
                "- Invoke at most one function in each action turn.\n"
                if self.max_tool_calls_per_turn == 1
                else "- You may invoke multiple independent functions in one turn; every call will be executed and returned.\n"
            )
            + (
                f"- Before final output, this segment requires at least "
                f"{self.min_tool_calls} executable function call(s).\n"
                if self.min_tool_calls and available
                else ""
            )
            + (
                "- Every action turn in this segment must invoke exactly one "
                "function. Final JSON is requested separately after the action "
                "budget.\n"
                if self.force_tool_each_round and available
                else ""
            )
            + "- When the investigation is complete, return exactly one JSON object. "
            + "Do not wrap it in markdown.\n"
            + "- Tool failures are observations to react to, not successful evidence."
        )

    def _build_native_chat_system_content(
        self,
        *,
        available_tool_names: Optional[Sequence[str]] = None,
    ) -> str:
        prompt = re.sub(
            r"Return exactly one JSON object inside <output>\.\.\.</output>",
            "Return exactly one JSON object",
            self.system_prompt,
        )
        if available_tool_names is None:
            available_tool_names = self._available_tool_names()
        available = list(
            dict.fromkeys(
                str(name).strip() for name in available_tool_names if str(name).strip()
            )
        )
        availability_instruction = (
            "\n\nRuntime tool availability:\n"
            "- Invoke only a tool currently supplied by the runtime. A tool "
            "omitted from the current request is unavailable; do not call it.\n"
            f"- Currently executable tools: {', '.join(available)}.\n"
            if available
            else (
                "\n\nRuntime tool availability:\n"
                "- No executable tool remains. Return the required JSON object "
                "without a tool call.\n"
            )
        )
        qwen_schema_hint = ""
        if (
            str(getattr(self.llm, "provider", "")).lower() == "qwen_local"
            and self.output_schema is not None
        ):
            # The official Transformers OpenAI server currently ignores
            # response_format. Keep the application validator strict, but
            # place a compact structural copy in the prompt so local Qwen has
            # the same schema contract as Gemini's guided output.
            schema = self._lmdeploy_response_schema(self._normalized_output_schema())
            qwen_schema_hint = (
                "\n\nQwen local structured-output compatibility:\n"
                "The server may ignore response_format. Return only one JSON "
                "object and follow this schema exactly. Use the literal enum "
                "values and JSON array types shown below; do not invent "
                "alternate labels, numeric salience scores, or scalar values "
                "where an array is shown.\n"
                + json.dumps(
                    schema,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        if not available:
            return (
                prompt
                + qwen_schema_hint
                + availability_instruction
                + "\n\nReturn one JSON object matching the response schema."
            )
        return (
            prompt
            + qwen_schema_hint
            + availability_instruction
            + "\n\nUse native function calls for tools. Return one JSON object "
            + "when finished; tool failures are not evidence."
        )

    @staticmethod
    def _openai_assistant_message(
        payload: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return None
        result = deepcopy(message)
        result["role"] = "assistant"
        return result

    @classmethod
    def _openai_single_tool_call(
        cls,
        payload: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        message = cls._openai_assistant_message(payload)
        if message is None:
            return None
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            return None
        return calls[0] if isinstance(calls[0], dict) else None

    def _normalize_runtime_tool_args(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        """Canonicalize harmless model variants of runtime-owned URLs.

        Dynamic visit candidates are runtime-owned, but models may return an
        equivalent spelling such as a trailing slash. Match only against the
        existing candidate set; never invent or broaden a URL.
        """

        normalized = dict(tool_args)
        if tool_name != "visit":
            return normalized, []
        allowed_values = self.tool_argument_constraints.get("visit", {}).get("url", [])
        if not allowed_values or "url" not in normalized:
            return normalized, []
        raw_value = normalized.get("url")
        is_list = isinstance(raw_value, list)
        values = raw_value if is_list else [raw_value]
        canonical_allowed = {
            canonicalize_url(str(candidate)): str(candidate).strip()
            for candidate in allowed_values
            if canonicalize_url(str(candidate))
        }
        repairs: List[Dict[str, str]] = []
        output: List[Any] = []
        for value in values:
            original = str(value).strip()
            canonical = canonicalize_url(original)
            replacement = canonical_allowed.get(canonical, "")
            if replacement and replacement != original:
                output.append(replacement)
                repairs.append(
                    {
                        "field": "url",
                        "from": original,
                        "to": replacement,
                        "reason": "canonical_url_equivalence",
                    }
                )
            else:
                output.append(value)
        normalized["url"] = output if is_list else output[0]
        return normalized, repairs

    @staticmethod
    def _usage_tokens(usage: Dict[str, Any]) -> Dict[str, int]:
        return {
            "prompt": int(
                usage.get("total_input_tokens", usage.get("input_tokens", 0)) or 0
            ),
            "completion": int(
                usage.get("total_output_tokens", usage.get("output_tokens", 0)) or 0
            ),
            "thought": int(
                usage.get("total_thought_tokens", usage.get("thought_tokens", 0)) or 0
            ),
        }

    def _build_native_input(self, input_context: Any) -> Any:
        # With a chained Gemini Interaction, the provider already retains the
        # original image in the episode history. Re-upload it only for the
        # root request; otherwise every ReAct turn would duplicate the same
        # image in provider-side history.
        if self._session_previous_interaction_id():
            return input_context
        if not isinstance(input_context, str):
            return self._append_native_image(input_context)
        if not (self.attach_image and self.image_path):
            return input_context
        return [
            {"type": "text", "text": input_context},
            self._native_image_item(),
        ]

    def _native_image_item(self) -> Dict[str, Any]:
        """Build one compressed image content item for the current request."""

        image_url, view = controlled_image_to_data_url(self.image_path)
        self._record_image_view(view, purpose=self.stage_name or "stage")
        if not (image_url.startswith("data:") and ";base64," in image_url):
            return {"type": "image", "uri": image_url}
        header, data = image_url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] or "image/jpeg"
        return {"type": "image", "mime_type": mime_type, "data": data}

    def _append_native_image(
        self,
        input_payload: Any,
        *,
        include_image: bool = True,
    ) -> Any:
        """Attach one compressed original image to a follow-up request.

        A follow-up may already contain a ``user_input`` step carrying visual
        candidates or inspection artifacts.  Interactions accepts the step
        sequence, but the provider rejects the same request when the image is
        split across multiple ``user_input`` steps.  Merge all existing
        ``user_input`` content into the first such step and append the
        original image there when this is an independent request.  A chained
        Interaction already has the original image in provider history, so
        callers can disable the append and keep only the new observation.
        The image is never added to accumulated text history or persisted in a
        ``StageStep``.
        """

        if not include_image or not (self.attach_image and self.image_path):
            return input_payload
        image_item = self._native_image_item()
        if isinstance(input_payload, list):
            payload = deepcopy(input_payload)
            if self._native_payload_contains_image(payload, image_item):
                return payload

            step_types = {
                "function_result",
                "function_call",
                "user_input",
                "message",
                "model_output",
                "thought",
            }
            has_steps = any(
                isinstance(item, dict)
                and str(item.get("type", "")).strip().lower().replace("-", "_")
                in step_types
                for item in payload
            )
            if not has_steps:
                payload.append(image_item)
                return payload

            merged: List[Dict[str, Any]] = []
            merged_user_input: Optional[Dict[str, Any]] = None
            for item in payload:
                if (
                    isinstance(item, dict)
                    and str(item.get("type", "")).strip().lower().replace("-", "_")
                    == "user_input"
                ):
                    if merged_user_input is None:
                        merged_user_input = {
                            "type": "user_input",
                            "content": [],
                        }
                        merged.append(merged_user_input)
                    merged_user_input["content"].extend(
                        self._native_user_input_content(item)
                    )
                    continue
                merged.append(item)

            if merged_user_input is None:
                merged.append(
                    {
                        "type": "user_input",
                        "content": [image_item],
                    }
                )
            else:
                merged_user_input["content"].append(image_item)
            return merged
        if input_payload in (None, ""):
            return [{"type": "user_input", "content": [image_item]}]
        return [
            {
                "type": "user_input",
                "content": [
                    {"type": "text", "text": str(input_payload)},
                    image_item,
                ],
            }
        ]

    @staticmethod
    def _native_user_input_content(
        item: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        content = item.get("content")
        if isinstance(content, list):
            return [deepcopy(value) for value in content if isinstance(value, dict)]
        if isinstance(content, dict):
            return [deepcopy(content)]
        if content in (None, ""):
            return []
        return [{"type": "text", "text": str(content)}]

    @staticmethod
    def _native_payload_contains_image(
        payload: List[Any],
        image_item: Dict[str, Any],
    ) -> bool:
        target_uri = image_item.get("uri")
        target_data = image_item.get("data")
        for item in payload:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            values = content if isinstance(content, list) else [content]
            for value in values:
                if not isinstance(value, dict):
                    continue
                if value.get("type") == "image" and (
                    (target_uri and value.get("uri") == target_uri)
                    or (target_data and value.get("data") == target_data)
                ):
                    return True
            result = item.get("result")
            if isinstance(result, list):
                for value in result:
                    if not isinstance(value, dict):
                        continue
                    if value.get("type") == "image" and (
                        (target_uri and value.get("uri") == target_uri)
                        or (target_data and value.get("data") == target_data)
                    ):
                        return True
        return False

    def _session_previous_interaction_id(self) -> Optional[str]:
        if self.interaction_session is None:
            return None
        return self.interaction_session.previous_interaction_id

    def _build_session_input(self, input_context: str) -> Any:
        current_input = self._build_native_input(input_context)
        if (
            self.interaction_session is None
            or not self.interaction_session.pending_input
        ):
            return current_input
        pending = deepcopy(self.interaction_session.pending_input)
        if isinstance(current_input, list):
            user_content = current_input
        else:
            user_content = [{"type": "text", "text": str(current_input)}]
        payload = [
            *pending,
            {"type": "user_input", "content": user_content},
        ]
        # A session handoff may already contain candidate/inspection images in
        # one user_input step. Gemini Interactions expects the follow-up
        # content to be presented as one coherent user step, so merge the
        # pending observation and the current compact context before sending.
        return self._merge_native_user_input_steps(payload)

    @classmethod
    def _merge_native_user_input_steps(
        cls,
        payload: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        merged_user_input: Optional[Dict[str, Any]] = None
        for item in payload:
            if (
                isinstance(item, dict)
                and str(item.get("type", "")).strip().lower().replace("-", "_")
                == "user_input"
            ):
                if merged_user_input is None:
                    merged_user_input = {
                        "type": "user_input",
                        "content": [],
                    }
                    merged.append(merged_user_input)
                merged_user_input["content"].extend(
                    cls._native_user_input_content(item)
                )
            else:
                merged.append(item)
        return merged

    def _advance_interaction_session(self, interaction_id: str) -> None:
        if self.interaction_session is None:
            return
        self.interaction_session.previous_interaction_id = interaction_id
        self.interaction_session.pending_input = []

    def _set_session_pending_input(
        self,
        pending_input: List[Dict[str, Any]],
    ) -> None:
        if self.interaction_session is None:
            return
        self.interaction_session.pending_input = deepcopy(pending_input)

    @staticmethod
    def _coerce_native_arguments(arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, dict):
            return dict(arguments)
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {}
            return dict(parsed) if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _extract_native_function_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        return extract_function_calls(payload)

    @staticmethod
    def _extract_native_text(payload: Dict[str, Any]) -> str:
        return extract_text(payload)

    @staticmethod
    def _extract_native_thought(payload: Dict[str, Any]) -> str:
        chunks: List[str] = []
        for step in payload.get("steps", []) or []:
            if not isinstance(step, dict) or step.get("type") != "thought":
                continue
            direct_text = step.get("text")
            if isinstance(direct_text, str) and direct_text.strip():
                chunks.append(direct_text)
            for field_name in ("summary", "content"):
                field = step.get(field_name)
                if isinstance(field, str) and field.strip():
                    chunks.append(field)
                    continue
                if isinstance(field, dict):
                    field = [field]
                for item in field or []:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        if item["text"].strip():
                            chunks.append(item["text"])
        return "\n".join(chunks)

    def _build_native_function_result(
        self,
        *,
        call_id: str,
        tool_name: str,
        result: str,
    ) -> Dict[str, Any]:
        observed_result = self._model_visible_tool_result(result)
        content: Dict[str, Any] = {
            "function_call_id": call_id,
            "result": observed_result,
        }
        text = json.dumps(content, ensure_ascii=False, default=str)
        return {
            "type": "function_result",
            "name": tool_name,
            "call_id": call_id,
            "result": [{"type": "text", "text": text}],
            **({"is_error": True} if self._tool_result_is_error(result) else {}),
        }

    @staticmethod
    def _tool_result_is_error(result: str) -> bool:
        try:
            _, succeeded = parse_tool_result(result)
        except Exception:
            return True
        return not succeeded

    async def _visual_reinjection_items(
        self,
        tool_name: str,
        *,
        result: str,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if (
            tool_name in {"focused_visual_inspection", "crop_and_inspect"}
            and self.runtime_store is not None
        ):
            artifacts = self._visual_view_artifacts(result)
            for artifact in artifacts:
                descriptor = artifact.get("artifact")
                if not isinstance(descriptor, dict):
                    continue
                try:
                    payload = self.runtime_store.artifacts.read_bytes(descriptor)
                    from eviscene.tools.vision_utils import (
                        controlled_image_bytes_to_data_url,
                    )

                    data_url, _ = controlled_image_bytes_to_data_url(
                        payload,
                        source_kind="runtime_view_artifact",
                        source_media_type=str(
                            descriptor.get("media_type", "image/jpeg")
                        ),
                    )
                    header, encoded = data_url.split(",", 1)
                    mime_type = (
                        header[5:].split(";", 1)[0].strip().lower() or "image/jpeg"
                    )
                except Exception:
                    continue
                items.append(
                    {
                        "type": "image",
                        "mime_type": mime_type,
                        "data": encoded,
                    }
                )

        if tool_name in {
            "reverse_image_search",
            "crop_and_search",
            "text_image_search",
        }:
            candidate_urls = self._reference_image_candidate_urls(result)[:3]
            candidate_items = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        self._native_candidate_image_item,
                        image_url,
                    )
                    for image_url in candidate_urls
                )
            )
            items.extend(item for item in candidate_items if isinstance(item, dict))
        return items

    @staticmethod
    def _native_candidate_image_item(
        image_url: str,
    ) -> Optional[Dict[str, Any]]:
        """Download one provider URL and build a bounded native image item.

        Gemini Interactions accepts inline image bytes reliably across the
        provider URL formats returned by visual search.  The returned URLs
        are public candidates, not Gemini file resources, so passing them as
        ``image.uri`` can be rejected as an invalid request.  Validate and
        compress the downloaded bytes before putting them in the next
        multimodal turn.  A bad candidate is omitted; the textual search
        result remains available to the model.
        """

        from eviscene.tools.vision_utils import controlled_image_to_data_url

        try:
            data_url, _ = controlled_image_to_data_url(str(image_url).strip())
            header, encoded = data_url.split(",", 1)
            source_mime = header[5:].split(";", 1)[0].strip().lower()
            if not source_mime.startswith("image/"):
                return None
        except Exception:
            return None

        return {
            "type": "image",
            "mime_type": source_mime,
            "data": encoded,
        }

    @staticmethod
    def _reference_image_candidate_urls(result: str) -> List[str]:
        """Return only provider image URLs suitable for the next model turn."""

        try:
            data = json.loads(result)
        except Exception:
            return []
        if not isinstance(data, dict):
            return []
        values: List[Any] = []
        for key in ("reference_image_candidates", "reference_image_urls"):
            raw = data.get(key)
            if isinstance(raw, list):
                values.extend(raw)
            elif isinstance(raw, str):
                values.append(raw)
        direct = data.get("reference_image_url")
        if isinstance(direct, str):
            values.append(direct)
        urls: List[str] = []
        seen: set[str] = set()
        for value in values:
            url = str(value or "").strip()
            try:
                parsed = urlparse(url)
            except Exception:
                continue
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or url in seen
            ):
                continue
            seen.add(url)
            urls.append(url)
        return urls

    @staticmethod
    def _visual_view_artifacts(result: str) -> List[Dict[str, Any]]:
        try:
            parsed, succeeded = parse_tool_result(result)
        except Exception:
            return []
        if not succeeded:
            return []
        artifacts = parsed.get("view_artifacts") or []
        if not isinstance(artifacts, list):
            return []
        return [dict(item) for item in artifacts if isinstance(item, dict)]

    async def _force_native_output(
        self,
        *,
        previous_interaction_id: Optional[str],
        pending_input: Any,
        steps: List[StageStep],
        evidence_so_far: List[str],
    ) -> Tuple[Optional[BaseModel], List[StageStep]]:
        if not previous_interaction_id:
            return None, steps
        directive = (
            "No more tool turns remain. Use the collected evidence and return exactly one "
            "final JSON object now. Be explicit about uncertainty."
        )
        if evidence_so_far:
            directive += "\nCollected evidence:\n" + "\n".join(evidence_so_far[-12:])
        if isinstance(pending_input, list):
            forced_input = list(pending_input)
        else:
            forced_input = (
                f"{pending_input}\n\n{directive}" if pending_input else directive
            )

        system_instruction = (
            self._build_native_system_content(available_tool_names=[])
            + "\n\n"
            + directive
        )
        response_format = self._native_response_format()
        request_input: Any = forced_input
        request_parent = previous_interaction_id

        for correction_index in range(2):
            started = time.perf_counter()
            self.llm_api_calls += 1
            wire_request_input = self._append_native_image(
                request_input,
                include_image=not bool(request_parent),
            )
            try:
                payload = await self._create_interaction(
                    input_payload=wire_request_input,
                    system_instruction=system_instruction,
                    tools=[],
                    previous_interaction_id=request_parent,
                    response_format=response_format,
                    store=True,
                    max_tokens=self.final_output_max_tokens,
                    generation_config=self.final_output_generation_config,
                )
                interaction_id, interaction_status = validate_interaction_response(
                    payload
                )
            except Exception as exc:
                self._attach_partial_steps(exc, steps)
                raise
            usage = (
                payload.get("usage", {})
                if isinstance(payload.get("usage"), dict)
                else {}
            )
            metadata = {
                "stage": self.stage_name,
                "native_interactions": True,
                "forced_output": True,
                "forced_output_correction": correction_index > 0,
                "previous_interaction_id": request_parent,
                "interaction_id": interaction_id,
                "interaction_status": interaction_status,
                "llm_duration_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
                "max_output_tokens": self.final_output_max_tokens,
                "thinking_level": self.final_output_generation_config.get(
                    "thinking_level"
                ),
                "policy_input": self._policy_input_snapshot(
                    system_instruction=system_instruction,
                    input_payload=wire_request_input,
                    tools=[],
                    response_format=response_format,
                ),
                "context_request_id": self._last_context_request_id,
                "interaction_lifecycle_kind": self._last_interaction_lifecycle_kind,
            }
            tokens = self._usage_tokens(usage)
            if self._extract_native_function_calls(payload):
                function_calls = self._extract_native_function_calls(payload)
                reason = "model requested another function after the tool budget ended"
                metadata["rejection_reason"] = reason
                metadata["policy_action"] = {
                    "type": (
                        "parallel_tool_calls"
                        if len(function_calls) > 1
                        else "tool_call"
                    ),
                    "calls": deepcopy(function_calls),
                }
                output_json = None
                parsed = None
            else:
                content = self._extract_native_text(payload)
                output_json = self._extract_output(
                    content
                ) or self._try_parse_bare_json(content)
                if output_json is not None:
                    metadata["policy_action"] = deepcopy(output_json)
                parsed = (
                    self._validate_output(output_json)
                    if output_json is not None
                    else None
                )
                if parsed is not None:
                    accepted, reason = self._accept_output(
                        parsed,
                        steps,
                        final_attempt=True,
                    )
                    if accepted:
                        steps.append(
                            StageStep(
                                round=len(steps) + 1,
                                stage_name=self.stage_name,
                                action_type="output",
                                output=parsed.model_dump(),
                                tokens=tokens,
                                metadata=metadata,
                            )
                        )
                        return parsed, steps
                    metadata["rejection_reason"] = reason
                else:
                    reason = (
                        "output schema was invalid or incomplete"
                        if output_json is not None
                        else "model returned no structured output"
                    )
                    metadata["rejection_reason"] = reason

            steps.append(
                StageStep(
                    round=len(steps) + 1,
                    stage_name=self.stage_name,
                    action_type=(
                        "output_rejected" if output_json is not None else "format_error"
                    ),
                    output=output_json,
                    tokens=tokens,
                    metadata=metadata,
                )
            )
            if correction_index == 0:
                request_parent = interaction_id
                request_input = (
                    f"Output rejected: {reason}. Return a corrected JSON object. "
                    "Use only fields allowed by the response schema and do not "
                    "invent runtime ids or state transitions."
                )
        return None, steps

    @staticmethod
    def _policy_input_snapshot(
        *,
        system_instruction: str,
        input_payload: Any,
        tools: List[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Persist model-visible structure without embedding binary media payloads."""

        return {
            "system_instruction": system_instruction,
            "input_payload": StageRunner._snapshot_input_payload(input_payload),
            "tools": deepcopy(tools),
            "response_format": deepcopy(response_format),
        }

    @staticmethod
    def _snapshot_input_payload(value: Any) -> Any:
        """Replace image bytes/URIs with a stable runtime-image reference."""

        if isinstance(value, list):
            return [StageRunner._snapshot_input_payload(item) for item in value]
        if not isinstance(value, dict):
            return deepcopy(value)
        media_type = str(value.get("type", "")).strip().lower()
        if media_type in {"image", "image_url"}:
            snapshot: Dict[str, Any] = {
                "type": media_type,
                "runtime_image": True,
            }
            mime_type = str(value.get("mime_type", "")).strip()
            if mime_type:
                snapshot["mime_type"] = mime_type
            return snapshot
        return {
            key: StageRunner._snapshot_input_payload(item)
            for key, item in value.items()
        }

    def _build_system_content(
        self,
        *,
        steps: Optional[List[StageStep]] = None,
    ) -> str:
        available_tools = self._available_tool_objects(steps=steps)
        if not available_tools:
            return self.system_prompt + self._output_format_instructions()

        lines = [self.system_prompt, "", "Available tools:"]
        lines.append(
            "Only the tools listed below are executable in this request; "
            "omitted tools are unavailable."
        )
        for tool in available_tools:
            required = tool.parameters.get("required", [])
            props = list((tool.parameters.get("properties", {}) or {}).keys())
            line = f"- {tool.name}"
            if required:
                line += f" | required: {', '.join(required)}"
            if props:
                line += f" | args: {', '.join(props[:5])}"
            lines.append(line)
        lines.append("")
        lines.append(self._output_format_instructions())
        return "\n".join(lines)

    def _accept_output(
        self,
        parsed: BaseModel,
        steps: List[StageStep],
        *,
        final_attempt: bool = False,
    ) -> Tuple[bool, str]:
        tool_calls = sum(1 for step in steps if step.action_type == "tool_call")
        if tool_calls < self.min_tool_calls:
            return (
                False,
                f"at least {self.min_tool_calls} tool calls are required; only {tool_calls} completed",
            )
        if self.output_validator:
            accepted, reason = self.output_validator(parsed, steps)
            if not accepted:
                suffix = (
                    " No more tool turns remain."
                    if final_attempt and self._available_tool_names(steps=steps)
                    else " This is the final correction attempt for this stage; a "
                    "non-terminal output remains valid."
                    if final_attempt
                    else ""
                )
                return False, reason + suffix
        return True, ""

    def _search_policy_error(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        *,
        rejected_query_count: int = 0,
    ) -> str:
        if not self.source_access_policy.active or tool_name != "text_search":
            return ""
        queries = tool_args.get("queries", [])
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list):
            return ""
        usable = [str(query).strip() for query in queries if str(query).strip()]
        if not usable:
            if rejected_query_count > 0:
                return (
                    "Search policy removed every query because it targeted a "
                    "ready-made fact-check verdict or an excluded source. "
                    "Reformulate with claim terms, an original statement, an "
                    "official record, a primary source, or independent reporting."
                )
            return "text_search requires exactly one non-empty query."
        if usable and not any(
            query_policy_violation(
                query,
                source_access_policy=self.source_access_policy,
            )
            for query in usable
        ):
            return ""
        return (
            "Search policy removed every query because it targeted a ready-made "
            "fact-check verdict or an excluded source. Reformulate with claim terms, "
            "an original statement, an official record, a primary source, or "
            "independent reporting."
        )

    @staticmethod
    def _is_empty_search_query_error(tool_name: str, message: str) -> bool:
        return (
            tool_name == "text_search"
            and str(message).strip()
            == "text_search requires exactly one non-empty query."
        )

    def _record_search_query_rejection(
        self,
        step: StageStep,
        tool_name: str,
        message: str,
    ) -> None:
        if self._is_empty_search_query_error(tool_name, message):
            step.action_type = "format_error"
            step.metadata["error_class"] = "tool_argument_error"
            step.metadata["search_query_format_error"] = True
        else:
            # Policy-filtered queries require replanning, not schema repair.
            # Count this as a protocol correction, never an executed tool action.
            step.action_type = "policy_replan"
            step.metadata["error_class"] = "policy_replan"
            step.metadata["search_policy_rejection"] = True
            step.metadata["search_policy_replan_required"] = True
            step.metadata["rejection_reason"] = message

    def _sanitize_search_queries(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> int:
        """Drop prohibited retrieval strings without discarding usable sibling queries."""

        if not self.source_access_policy.active or tool_name != "text_search":
            return 0
        raw_queries = tool_args.get("queries", [])
        queries = [raw_queries] if isinstance(raw_queries, str) else raw_queries
        if not isinstance(queries, list):
            return 0
        allowed: List[str] = []
        rejected: List[str] = []
        for raw_query in queries:
            query = str(raw_query).strip()
            if not query:
                continue
            if query_policy_violation(
                query,
                source_access_policy=self.source_access_policy,
            ):
                rejected.append(query)
            else:
                allowed.append(query)
        if rejected:
            tool_args["queries"] = allowed
        return len(rejected)

    @staticmethod
    def _output_format_instructions() -> str:
        return (
            "Each round must produce exactly one action:\n"
            '1. <think>reasoning</think><tool_call>{"name":"tool_name","arguments":{...}}</tool_call>\n'
            "2. <think>reasoning</think><output>{...}</output>"
        )

    def _build_user_message(self, input_context: str) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = []
        if self.attach_image and self.image_path:
            try:
                image_url, view = controlled_image_to_data_url(self.image_path)
                self._record_image_view(view, purpose=self.stage_name or "stage")
                parts.append({"type": "image_url", "image_url": {"url": image_url}})
            except Exception as exc:
                raise RuntimeError(
                    f"{self.stage_name or 'stage'} image preparation failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        parts.append({"type": "text", "text": input_context})
        if len(parts) == 1 and parts[0]["type"] == "text":
            return {"role": "user", "content": parts[0]["text"]}
        return {"role": "user", "content": parts}

    def _record_image_view(self, view: Dict[str, Any], *, purpose: str) -> None:
        fingerprint = json.dumps(view, sort_keys=True, default=str)
        if fingerprint == self._last_image_view.get("fingerprint"):
            return
        self._last_image_view = {"fingerprint": fingerprint, **dict(view)}
        if self.runtime_store is None:
            return
        self.runtime_store.append_event(
            "image_view",
            {
                "stage": self.stage_name,
                "purpose": purpose,
                "image_id": "input-image",
                "crop": None,
                "before_understanding_version": None,
                "after_understanding_version": None,
                "decision_impact": "pending",
                **dict(view),
            },
        )

    def _uses_native_chat_completions(self) -> bool:
        return bool(
            str(getattr(self.llm, "provider", "")).lower() in {"qwen_local", "lmdeploy"}
            and str(getattr(self.llm, "wire_api", "")).lower() == "chat_completions"
        )

    def _policy_token_capture_enabled(self) -> bool:
        if self.capture_policy_tokens is not None:
            return bool(self.capture_policy_tokens)
        return get_env("EVISCENE_CAPTURE_POLICY_TOKENS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _policy_token_capture_topk(self) -> int:
        if self.policy_topk is not None:
            value = int(self.policy_topk)
            if value < 1 or value > 100:
                raise ValueError("policy_topk must be between 1 and 100")
            return value
        raw = get_env("EVISCENE_POLICY_TOPK", "20").strip()
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError("EVISCENE_POLICY_TOPK must be an integer") from exc
        if value < 1 or value > 100:
            raise ValueError("EVISCENE_POLICY_TOPK must be between 1 and 100")
        return value

    def _build_round_messages(
        self,
        system_msg: Dict[str, Any],
        user_msg: Dict[str, Any],
        history: List[Dict[str, Any]],
        evidence_so_far: List[str],
    ) -> List[Dict[str, Any]]:
        if self._uses_native_chat_completions():
            return [system_msg, *deepcopy(history[1:])]

        messages = [system_msg, user_msg]
        if evidence_so_far:
            summary = "\n".join(
                f"{idx + 1}. {item}" for idx, item in enumerate(evidence_so_far[-8:])
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"<think>Reviewing collected evidence.</think>\n## Evidence so far\n{summary}",
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "Continue from the current strongest next step.",
                }
            )

        recent = history[2:]
        if len(recent) > self.recent_rounds_to_keep * 2:
            recent = recent[-(self.recent_rounds_to_keep * 2) :]
        messages.extend(recent)
        return messages

    def _commit_native_chat_history(
        self,
        history: List[Dict[str, Any]],
    ) -> None:
        if not self._uses_native_chat_completions() or self.interaction_session is None:
            if self._uses_native_chat_completions():
                self.last_native_history = deepcopy(history[1:])
            return
        start = 1 + max(0, self._session_chat_history_start)
        self.interaction_session.chat_history.extend(deepcopy(history[start:]))
        self.last_native_history = deepcopy(history[1:])

    async def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        *,
        require_tool: bool = False,
        max_tokens: Optional[int] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        lifecycle_kind: str = "",
        parent_context_request_id: str = "",
        suppress_tools: bool = False,
        active_steps: Optional[List[StageStep]] = None,
    ) -> Tuple[LLMResponse, Dict[str, Any]]:
        started = time.perf_counter()
        self.llm_api_calls += 1
        request_kwargs: Dict[str, Any] = {}
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.max_output_tokens
        )
        effective_generation_config = (
            generation_config
            if generation_config is not None
            else self.generation_config
        )
        if effective_max_tokens is not None:
            request_kwargs["max_tokens"] = effective_max_tokens
        if effective_generation_config:
            request_kwargs["generation_config"] = dict(effective_generation_config)
        response_format: Optional[Dict[str, Any]] = None
        if self._uses_native_chat_completions():
            active_tool_schemas = self._build_native_tool_schemas(
                steps=active_steps,
            )
            if active_tool_schemas and not suppress_tools:
                request_kwargs["tools"] = active_tool_schemas
                request_kwargs["tool_choice"] = "required" if require_tool else "auto"
            elif self.output_schema is not None:
                response_format = self._openai_response_format()
                request_kwargs["response_format"] = response_format
            if self._policy_token_capture_enabled():
                request_kwargs["capture_policy_tokens"] = True
                request_kwargs["policy_topk"] = self._policy_token_capture_topk()
        request_id = ""
        effective_lifecycle_kind = lifecycle_kind.strip() or (
            "tool_roundtrip"
            if self._uses_native_chat_completions()
            and bool(request_kwargs.get("tools"))
            else "standalone_request"
        )
        effective_parent_request_id = parent_context_request_id.strip() or None
        if (
            effective_lifecycle_kind == "protocol_correction"
            and effective_parent_request_id is None
            and self.runtime_store is not None
        ):
            raise RuntimeError("protocol_correction requires a parent context request")
        if self.runtime_store is not None:
            request_id = self.runtime_store.context_ledger.begin_request(
                stage=self.stage_name,
                lifecycle_kind=effective_lifecycle_kind,
                system_instruction="",
                input_payload=messages,
                tools=request_kwargs.get("tools"),
                response_format=response_format,
                generation_config=request_kwargs.get("generation_config"),
                max_output_tokens=effective_max_tokens,
                parent_request_id=effective_parent_request_id,
                model=str(getattr(self.llm, "model_name", "")),
                prompt_version=self.prompt_version,
            )
        try:
            response = await asyncio.wait_for(
                self.llm.get_response(messages, **request_kwargs),
                timeout=self.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            if request_id:
                self.runtime_store.context_ledger.complete_request(
                    request_id,
                    status="error",
                    error="request timeout",
                )
            raise TimeoutError(
                f"{self.stage_name or 'stage'} model request exceeded "
                f"{self.request_timeout_seconds:.1f}s"
            ) from exc
        except Exception as exc:
            if request_id:
                self.runtime_store.context_ledger.complete_request(
                    request_id,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        raw = response.raw if isinstance(response.raw, dict) else {}
        choices = raw.get("choices") if isinstance(raw.get("choices"), list) else []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = (
            choice.get("message") if isinstance(choice.get("message"), dict) else {}
        )
        reasoning_field = ""
        reasoning_text = ""
        for candidate_field in ("reasoning", "reasoning_content"):
            candidate = message.get(candidate_field)
            if isinstance(candidate, str) and candidate.strip():
                reasoning_field = candidate_field
                reasoning_text = candidate
                break
        response_content_chars = len(str(message.get("content") or ""))
        response_reasoning_chars = len(reasoning_text)
        reasoning_artifact: Optional[Dict[str, Any]] = None
        if self.runtime_store is not None and reasoning_text:
            reasoning_artifact = self.runtime_store.artifacts.put_text(
                reasoning_text,
                media_type="text/plain; charset=utf-8",
                suffix=".txt",
                metadata={
                    "kind": "model_reasoning",
                    "stage": self.stage_name,
                    "request_id": request_id,
                    "provider_field": reasoning_field,
                    "excluded_from_model_context": True,
                },
            )
            self.runtime_store.append_event(
                "model_reasoning_archived",
                {
                    "stage": self.stage_name,
                    "request_id": request_id,
                    "provider_field": reasoning_field,
                    "char_count": response_reasoning_chars,
                    "artifact": reasoning_artifact,
                    "excluded_from_model_context": True,
                },
            )
        if request_id:
            usage = (
                raw.get("usage", {})
                if isinstance(raw.get("usage"), dict)
                else {
                    "input_tokens": response.prompt_tokens,
                    "output_tokens": response.completion_tokens,
                }
            )
            self.runtime_store.context_ledger.complete_request(
                request_id,
                usage=usage,
                status="completed",
                response_metadata={
                    "response_content_chars": response_content_chars,
                    "response_reasoning_chars": response_reasoning_chars,
                    "reasoning_artifact": reasoning_artifact,
                },
            )
        self._last_context_request_id = request_id
        metadata = {
            "llm_duration_ms": duration_ms,
            "context_request_id": request_id,
            "parent_context_request_id": effective_parent_request_id,
            "interaction_lifecycle_kind": effective_lifecycle_kind,
            "native_chat_completions": self._uses_native_chat_completions(),
            "finish_reason": str(choice.get("finish_reason", "")),
            "response_content_chars": response_content_chars,
            "response_reasoning_chars": response_reasoning_chars,
            "reasoning_artifact": reasoning_artifact,
        }
        if request_kwargs.get("capture_policy_tokens"):
            metadata["policy_token_capture"] = extract_policy_token_capture(
                raw,
                topk=int(request_kwargs["policy_topk"]),
            )
        return response, metadata

    async def _create_interaction(self, **kwargs: Any) -> Dict[str, Any]:
        """Apply one wall-clock deadline to every native Gemini request."""

        explicit_lifecycle = str(kwargs.pop("lifecycle_kind", "")).strip()
        request_id = ""
        tools = kwargs.get("tools") or []
        previous_interaction_id = kwargs.get("previous_interaction_id")
        lifecycle_kind = explicit_lifecycle or (
            "tool_roundtrip"
            if tools
            else "protocol_correction"
            if previous_interaction_id
            else "standalone_request"
        )
        if previous_interaction_id and lifecycle_kind == "standalone_request":
            raise RuntimeError(
                "standalone_request cannot inherit a previous interaction"
            )
        self._last_interaction_lifecycle_kind = lifecycle_kind
        if self.runtime_store is not None:
            request_id = self.runtime_store.context_ledger.begin_request(
                stage=self.stage_name,
                lifecycle_kind=lifecycle_kind,
                system_instruction=kwargs.get("system_instruction"),
                input_payload=kwargs.get("input_payload"),
                tools=tools,
                response_format=kwargs.get("response_format"),
                generation_config=kwargs.get("generation_config"),
                max_output_tokens=kwargs.get("max_tokens"),
                previous_interaction_id=previous_interaction_id,
                model=str(getattr(self.llm, "model_name", "")),
                prompt_version=self.prompt_version,
            )
        try:
            payload = await asyncio.wait_for(
                self.llm.create_interaction(**kwargs),
                timeout=self.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            if request_id:
                self.runtime_store.context_ledger.complete_request(
                    request_id,
                    status="error",
                    error="request timeout",
                )
            raise TimeoutError(
                f"{self.stage_name or 'stage'} Gemini request exceeded "
                f"{self.request_timeout_seconds:.1f}s"
            ) from exc
        except Exception as exc:
            if request_id:
                self.runtime_store.context_ledger.complete_request(
                    request_id,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise
        if request_id:
            self.runtime_store.context_ledger.complete_request(
                request_id,
                usage=(
                    payload.get("usage", {})
                    if isinstance(payload.get("usage"), dict)
                    else {}
                ),
                interaction_id=str(payload.get("id", "")).strip() or None,
                status=str(payload.get("status", "completed")),
                response_metadata={
                    "retry_metadata": payload.get("__retry_metadata__"),
                },
            )
        self._last_context_request_id = request_id
        return payload

    @staticmethod
    def _extract_think(content: str) -> str:
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_output(content: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"<output>(.*?)</output>", content, re.DOTALL)
        if not match:
            return None
        block = StageRunner._strip_markdown_fence(match.group(1).strip())
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _parse_tool_call(content: str) -> Tuple[str, Dict[str, Any]]:
        match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        if not match:
            return "", {}
        block = StageRunner._strip_markdown_fence(match.group(1).strip())
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            return "", {}
        return str(parsed.get("name", "")).strip(), parsed.get("arguments", {}) or {}

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _validate_output_with_error(
        self,
        output_json: Dict[str, Any],
    ) -> Tuple[Optional[BaseModel], str]:
        if self.output_schema is None:
            return None, "no output schema is configured"
        if not isinstance(output_json, dict):
            return None, "output must be a JSON object"
        normalizer = getattr(self.output_schema, "normalize_legacy_input", None)
        if callable(normalizer):
            output_json = normalizer(output_json)
        missing = missing_required_paths(
            output_json,
            self._normalized_output_schema(),
        )
        if missing:
            return None, "missing required fields: " + ", ".join(missing[:12])
        try:
            return self.output_schema.model_validate(output_json), ""
        except Exception as exc:
            compact = " ".join(str(exc).split())
            return None, compact[:1200]

    def _validate_output(self, output_json: Dict[str, Any]) -> Optional[BaseModel]:
        parsed, _reason = self._validate_output_with_error(output_json)
        return parsed

    @staticmethod
    def _try_parse_bare_json(content: str) -> Optional[Dict[str, Any]]:
        parsed, _repair = StageRunner._try_parse_bare_json_with_repair(content)
        return parsed

    @staticmethod
    def _try_parse_bare_json_with_repair(
        content: str,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        text = StageRunner._strip_markdown_fence(text)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return (parsed, "") if isinstance(parsed, dict) else (None, "")
            except json.JSONDecodeError:
                pass

        # LMDeploy guided decoding can omit the fixed opening token while
        # returning the rest of a complete top-level object. Repair only that
        # syntax defect; never infer fields, values or semantic content.
        if start == -1 and text.startswith('"') and text.endswith("}"):
            try:
                parsed = json.loads("{" + text)
                if isinstance(parsed, dict):
                    return parsed, "prepended_missing_top_level_open_brace"
            except json.JSONDecodeError:
                pass
        return None, ""

    @staticmethod
    def _attach_invalid_response_preview(
        step: StageStep,
        content: str,
    ) -> None:
        """Persist a bounded model-response preview for protocol diagnosis."""

        text = str(content or "")
        step.metadata["invalid_response"] = {
            "content_chars": len(text),
            "head": text[:600],
            "tail": text[-600:] if len(text) > 600 else "",
        }

    def _summarize_tool_result(
        self, tool_name: str, tool_args: Dict[str, Any], result: str
    ) -> str:
        try:
            data = json.loads(result)
        except Exception:
            data = None
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]

        if tool_name == "text_search":
            query = tool_args.get("queries", tool_args.get("query", ""))
            if isinstance(query, list):
                query = query[0] if query else ""
            if isinstance(data, dict):
                responses = data.get("queries", [])
                if (
                    isinstance(responses, list)
                    and responses
                    and isinstance(responses[0], dict)
                ):
                    data = responses[0]
                summary = str(
                    data.get("summary", "") or data.get("evidence", "")
                ).strip()
                if summary:
                    return f"[text_search] {query}: {summary[:120]}"
                results = data.get("results", [])
                if isinstance(results, list) and results:
                    first = results[0]
                    return f"[text_search] {query}: {str(first.get('title', '') or first.get('snippet', ''))[:120]}"
            return f"[text_search] {query}: no useful result"

        if tool_name == "reverse_image_search" and isinstance(data, dict):
            urls = data.get("candidate_page_urls", []) or []
            return f"[reverse_image_search] candidates={len(urls)} ref={str(data.get('reference_image_url', ''))[:80]}"

        if tool_name == "visit" and isinstance(data, dict):
            return f"[visit] {str(data.get('summary', '') or data.get('evidence', ''))[:120]}"

        if tool_name == "compare_with_reference":
            return f"[compare_with_reference] {str(result)[:120]}"

        if tool_name == "check_consistency" and isinstance(data, dict):
            return f"[check_consistency] consistent={data.get('consistent', True)} {str(data.get('details', ''))[:100]}"

        if tool_name == "analyze_visual_anomalies":
            return f"[analyze_visual_anomalies] {str(result)[:120]}"

        if tool_name == "ocr_with_position" and isinstance(data, dict):
            return f"[ocr_with_position] regions={data.get('total_regions', 0)} text={str(data.get('full_text', ''))[:100]}"

        if tool_name == "perceive_scene" and isinstance(data, dict):
            return (
                f"[perceive_scene] scene={str(data.get('scene_description', ''))[:100]}"
            )

        if tool_name == "crop_and_inspect" and isinstance(data, dict):
            observations = data.get("observations")
            if not isinstance(observations, list):
                observations = data.get("findings", [])
            first_observation = (
                observations[0]
                if isinstance(observations, list) and observations
                else ""
            )
            return f"[crop_and_inspect] {str(first_observation or data.get('description', ''))[:120]}"

        if tool_name == "crop_and_search" and isinstance(data, dict):
            return f"[crop_and_search] {str(data.get('summary', '') or data.get('evidence', ''))[:120]}"

        if tool_name == "count_objects" and isinstance(data, dict):
            return f"[count_objects] count={data.get('count', '?')} target={tool_args.get('target_object', '')}"

        if tool_name == "current_time" and isinstance(data, dict):
            return f"[current_time] {data.get('current_date', '')} {data.get('timezone', '')}".strip()

        return f"[{tool_name}] {str(result)[:120]}"

    async def _force_output(
        self,
        system_msg: Dict[str, Any],
        user_msg: Dict[str, Any],
        history: List[Dict[str, Any]],
        evidence_so_far: List[str],
        *,
        last_rejection_reason: str = "",
        lifecycle_kind: str = "standalone_request",
        parent_context_request_id: str = "",
    ) -> Tuple[Optional[BaseModel], Dict[str, Any]]:
        evidence_block = ""
        if evidence_so_far:
            evidence_block = "\n".join(
                f"{idx + 1}. {item}" for idx, item in enumerate(evidence_so_far[-12:])
            )
        if self.tools_list:
            prompt = (
                "You have no more tool turns. Produce one final <output> JSON now.\n"
                "Use the collected evidence. Be explicit about uncertainty.\n"
            )
        else:
            prompt = (
                "This is the last validation attempt for this structured output. "
                "Return one corrected JSON object now; this does not require a "
                "terminal verdict or the end of an investigation. Preserve "
                "uncertainty and choose the non-terminal option when the evidence "
                "does not justify a final verdict.\n"
            )
        if last_rejection_reason:
            prompt += (
                "Runtime validator feedback from the previous attempt:\n"
                f"{last_rejection_reason}\n"
            )
        if evidence_block:
            prompt += f"\nCollected evidence:\n{evidence_block}\n"

        messages = [system_msg, user_msg]
        recent = history[2:]
        if len(recent) > 8:
            recent = recent[-8:]
        messages.extend(recent)
        messages.append({"role": "user", "content": prompt})

        response, call_metadata = await self._call_llm(
            messages,
            max_tokens=self.final_output_max_tokens,
            generation_config=self.final_output_generation_config,
            lifecycle_kind=lifecycle_kind,
            parent_context_request_id=parent_context_request_id,
            suppress_tools=True,
        )
        metadata = {
            "forced_output": True,
            **call_metadata,
            "policy_input": self._policy_input_snapshot(
                system_instruction=system_msg["content"],
                input_payload=messages[1:],
                tools=[],
                response_format=(
                    self._openai_response_format()
                    if self._uses_native_chat_completions()
                    else None
                ),
            ),
        }
        if response.text:
            output_json = self._extract_output(response.text)
            format_repair = ""
            if output_json is None:
                output_json, format_repair = self._try_parse_bare_json_with_repair(
                    response.text
                )
            if format_repair:
                metadata["format_repair"] = format_repair
            if output_json is not None:
                parsed, schema_error = self._validate_output_with_error(output_json)
                if parsed is not None:
                    metadata["policy_action"] = parsed.model_dump()
                    return parsed, metadata
                metadata["policy_action"] = deepcopy(output_json)
                metadata["rejection_reason"] = (
                    schema_error or "output schema was invalid or incomplete"
                )
                metadata["error_class"] = "schema_error"
            preview_step = StageStep(metadata=metadata)
            self._attach_invalid_response_preview(preview_step, response.text)
        return None, metadata

    def _structured_output_correction_prompt(self, reason: str) -> str:
        """Return a bounded semantic retry instruction for a JSON stage.

        Guided decoding guarantees syntax, not that a proposed state update is
        admissible. Feed the exact validator feedback back to the model without
        silently rewriting its state transition or forcing a terminal answer.
        """

        message = (
            "The previous JSON object was rejected by the runtime validator. "
            "Return one corrected complete JSON object. Change only the fields "
            "needed to satisfy this feedback; do not invent IDs or state. "
        )
        if "Search policy removed every query" in reason:
            message += (
                "This search was not executed and produced no Evidence. Rewrite the "
                "query once using neutral entity/event/relation/source terms, without "
                "fact-check, fake, hoax, altered, debunk, or a preselected outlet. "
                "Do not repeat the blocked query. If no neutral reformulation is "
                "available, return a non-terminal continue state instead of emitting "
                "another empty or policy-blocked search. "
            )
        return message + f"Runtime validator feedback: {reason}"
