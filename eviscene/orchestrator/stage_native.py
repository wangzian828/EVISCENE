"""Stage native for the unified ReAct runtime."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel
from eviscene.integrations.gemini import validate_interaction_response
from .stage_types import StageStep


class NativeStageMixin:
    """Internal execution component; StageRunner owns shared case state."""

    async def _run_native_structured_output(
        self,
        input_context: str,
    ) -> Tuple[Optional[BaseModel], List[StageStep]]:
        """Run a no-tool stage with Interactions JSON schema output."""
        steps: List[StageStep] = []
        previous_interaction_id = self._session_previous_interaction_id()
        next_input: Any = self._build_session_input(input_context)

        for round_num in range(1, self.max_rounds + 2):
            request_previous_interaction_id = previous_interaction_id
            request_input = (
                next_input
                if round_num == 1
                else self._append_native_image(
                    next_input,
                    include_image=not bool(request_previous_interaction_id),
                )
            )
            started = time.perf_counter()
            self.llm_api_calls += 1
            system_instruction = self.system_prompt
            response_format = self._native_response_format()
            payload = await self._create_interaction(
                input_payload=request_input,
                system_instruction=system_instruction,
                previous_interaction_id=request_previous_interaction_id,
                response_format=response_format,
                store=True,
                max_tokens=self.max_output_tokens,
                generation_config=self.generation_config,
            )
            interaction_id, status = validate_interaction_response(payload)
            self._advance_interaction_session(interaction_id)
            if self._extract_native_function_calls(payload):
                raise RuntimeError(
                    "No-tool stage received an unexpected function_call response."
                )
            if status != "completed":
                raise RuntimeError(
                    f"No-tool stage requires status=completed, received status={status}."
                )

            usage = (
                payload.get("usage", {})
                if isinstance(payload.get("usage"), dict)
                else {}
            )
            content = self._extract_native_text(payload).strip()
            output_json = self._extract_output(content) or self._try_parse_bare_json(
                content
            )
            step = StageStep(
                round=round_num,
                stage_name=self.stage_name,
                thought=self._extract_native_thought(payload),
                action_type="output" if output_json is not None else "format_error",
                output=output_json,
                tokens=self._usage_tokens(usage),
                metadata={
                    "stage": self.stage_name,
                    "native_interactions": True,
                    "structured_output": True,
                    "previous_interaction_id": request_previous_interaction_id,
                    "interaction_id": interaction_id,
                    "interaction_status": status,
                    "llm_duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "policy_input": self._policy_input_snapshot(
                        system_instruction=system_instruction,
                        input_payload=request_input,
                        tools=[],
                        response_format=response_format,
                    ),
                    "policy_action": deepcopy(output_json),
                    "context_request_id": self._last_context_request_id,
                    "interaction_lifecycle_kind": self._last_interaction_lifecycle_kind,
                },
            )
            steps.append(step)
            parsed: Optional[BaseModel] = None
            schema_error = ""
            if output_json is not None:
                parsed, schema_error = self._validate_output_with_error(output_json)
            if parsed is not None:
                accepted, reason = self._accept_output(parsed, steps)
                if accepted:
                    return parsed, steps
                step.action_type = "output_rejected"
                step.metadata["rejection_reason"] = reason
                next_input = self._structured_output_correction_prompt(reason)
            else:
                if output_json is not None:
                    step.action_type = "output_rejected"
                    rejection_reason = (
                        schema_error or "output schema was invalid or incomplete"
                    )
                    step.metadata["rejection_reason"] = rejection_reason
                    next_input = self._structured_output_correction_prompt(
                        rejection_reason
                    )
                else:
                    next_input = (
                        "Return one JSON object that exactly matches the required "
                        "schema."
                    )
            previous_interaction_id = interaction_id
        if steps and self.protocol_exhaustion_boundary:
            steps[-1].metadata["correction_budget_exhausted"] = True
            steps[-1].metadata["termination_reason"] = (
                "protocol_correction_budget_exhausted"
            )
            boundary = self._protocol_exhaustion_stage_boundary(steps)
            if boundary is not None:
                return boundary
        return None, steps

    async def _run_native_interactions(
        self,
        input_context: str,
    ) -> Tuple[Optional[BaseModel], List[StageStep]]:
        """Run a ReAct loop using Gemini Interactions native function calls."""
        steps: List[StageStep] = []
        evidence_so_far: List[str] = []
        previous_interaction_id = self._session_previous_interaction_id()
        next_input: Any = self._build_session_input(input_context)
        system_suffix = ""

        action_turns = 0
        correction_turns = 0
        request_index = 0

        def request_protocol_correction(
            affected_steps: List[StageStep],
            reason: str,
        ) -> None:
            """Reserve one correction-only request or fail without forcing output."""

            nonlocal correction_turns
            if correction_turns >= self.max_protocol_corrections:
                for affected in affected_steps:
                    affected.metadata["react_action_turn"] = action_turns
                    affected.metadata["protocol_corrections_used"] = correction_turns
                    affected.metadata["interaction_request_index"] = request_index
                    affected.metadata["correction_budget_exhausted"] = True
                    affected.metadata["termination_reason"] = (
                        "protocol_correction_budget_exhausted"
                    )
                exc = RuntimeError(
                    f"{self.stage_name or 'stage'} exhausted its protocol correction "
                    f"budget ({self.max_protocol_corrections}): {reason}"
                )
                self._attach_partial_steps(exc, steps)
                raise exc
            correction_turns += 1
            for affected in affected_steps:
                affected.metadata["react_action_turn"] = action_turns
                affected.metadata["protocol_corrections_used"] = correction_turns
                affected.metadata["interaction_request_index"] = request_index

        while action_turns < self.max_rounds:
            request_index += 1
            request_previous_interaction_id = previous_interaction_id
            request_input = (
                next_input
                if request_index == 1
                else self._append_native_image(
                    next_input,
                    include_image=not bool(request_previous_interaction_id),
                )
            )
            started = time.perf_counter()
            self.llm_api_calls += 1
            native_tools = self._build_native_tool_schemas(steps=steps)
            active_tool_names = {
                str(item.get("name", "")).strip()
                for item in native_tools
                if str(item.get("name", "")).strip()
            }
            system_instruction = (
                self._build_native_system_content(
                    available_tool_names=active_tool_names,
                )
                + system_suffix
            )
            # Interactions cannot combine function tools with response_format.
            # Validate tool turns locally; use the schema for forced no-tool output.
            response_format = None
            request_generation_config = dict(self.generation_config)
            if native_tools and (
                self.force_tool_each_round or action_turns < self.min_tool_calls
            ):
                request_generation_config["tool_choice"] = "any"
            elif not native_tools:
                response_format = self._native_response_format()
            try:
                payload = await self._create_interaction(
                    input_payload=request_input,
                    system_instruction=system_instruction,
                    tools=native_tools,
                    previous_interaction_id=request_previous_interaction_id,
                    response_format=response_format,
                    store=True,
                    max_tokens=self.max_output_tokens,
                    generation_config=request_generation_config,
                )
            except Exception as exc:
                self._attach_partial_steps(exc, steps)
                raise
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            try:
                interaction_id, interaction_status = validate_interaction_response(
                    payload
                )
            except Exception as exc:
                self._attach_partial_steps(exc, steps)
                raise
            self._advance_interaction_session(interaction_id)

            usage = (
                payload.get("usage", {})
                if isinstance(payload.get("usage"), dict)
                else {}
            )
            tokens = self._usage_tokens(usage)
            common_metadata = {
                "stage": self.stage_name,
                "native_interactions": True,
                "previous_interaction_id": request_previous_interaction_id,
                "interaction_id": interaction_id,
                "interaction_status": interaction_status,
                "llm_duration_ms": duration_ms,
                "policy_input": self._policy_input_snapshot(
                    system_instruction=system_instruction,
                    input_payload=request_input,
                    tools=native_tools,
                    response_format=response_format,
                ),
                "context_request_id": self._last_context_request_id,
                "interaction_lifecycle_kind": self._last_interaction_lifecycle_kind,
            }
            thought = self._extract_native_thought(payload)
            function_calls = self._extract_native_function_calls(payload)

            if function_calls:
                if (
                    self.max_tool_calls_per_turn is not None
                    and len(function_calls) > self.max_tool_calls_per_turn
                ):
                    step = StageStep(
                        round=request_index,
                        stage_name=self.stage_name,
                        thought=thought,
                        action_type="format_error",
                        tokens=tokens,
                        metadata={
                            **common_metadata,
                            "error_class": "protocol_error",
                            "parallel_tool_calls_rejected": len(function_calls),
                            "policy_action": {
                                "type": "parallel_tool_calls",
                                "calls": deepcopy(function_calls),
                            },
                        },
                    )
                    steps.append(step)
                    previous_interaction_id = interaction_id
                    request_protocol_correction(
                        [step],
                        "only one tool call is allowed per image-only action turn",
                    )
                    next_input = (
                        "Choose exactly one current task and invoke exactly one "
                        "function for the next action."
                    )
                    system_suffix = ""
                    continue
                missing_call_ids = [
                    call
                    for call in function_calls
                    if not str(call.get("id", "")).strip()
                ]
                if missing_call_ids:
                    raise RuntimeError(
                        "Gemini Interactions returned a function_call without a call id."
                    )

                function_results: List[Dict[str, Any]] = []
                follow_up_visual_items: List[Dict[str, Any]] = []
                response_steps: List[StageStep] = []
                for call_index, call in enumerate(function_calls):
                    call_id = str(call.get("id", "")).strip()
                    tool_name = str(call.get("name", "")).strip()
                    tool_args = self._coerce_native_arguments(call.get("arguments", {}))
                    native_args, runtime_repairs = self._normalize_runtime_tool_args(
                        tool_name,
                        dict(tool_args),
                    )
                    prepared_args = dict(native_args)
                    step = StageStep(
                        round=request_index,
                        stage_name=self.stage_name,
                        thought=thought if call_index == 0 else "",
                        tool_name=tool_name,
                        tool_args=dict(prepared_args),
                        tokens=(
                            tokens
                            if call_index == 0
                            else {"prompt": 0, "completion": 0, "thought": 0}
                        ),
                        metadata={
                            **common_metadata,
                            "function_call_id": call_id,
                            "function_call_index": call_index,
                            "function_call_count": len(function_calls),
                            "policy_action": {
                                "type": "tool_call",
                                "name": tool_name,
                                "arguments": deepcopy(tool_args),
                            },
                        },
                    )
                    if call_index > 0:
                        step.metadata.pop("llm_duration_ms", None)
                    if runtime_repairs:
                        step.metadata["runtime_argument_repairs"] = runtime_repairs

                    error_message = ""
                    if tool_name not in self.tools:
                        error_message = self._unknown_tool_message(tool_name)
                        step.metadata["invalid_tool_name"] = tool_name
                        step.metadata["error_class"] = "protocol_error"
                    elif tool_name not in active_tool_names:
                        error_message = self._tool_unavailable_message(
                            tool_name,
                            active_tool_names,
                        )
                        step.metadata["tool_unavailable"] = True
                        step.metadata["error_class"] = "protocol_error"
                    else:
                        schema_error = self._validate_native_tool_args(
                            tool_name,
                            native_args,
                            steps=steps,
                        )
                        if schema_error:
                            error_message = schema_error
                            step.metadata["invalid_tool_arguments"] = True
                            step.metadata["error_class"] = "protocol_error"
                    if not error_message and self._has_duplicate_tool_call(
                        steps, tool_name, prepared_args
                    ):
                        error_message = self._duplicate_tool_message(
                            tool_name,
                            prepared_args,
                        )
                        step.metadata["duplicate_tool_call"] = True
                        step.metadata["error_class"] = "protocol_error"
                    elif not error_message and self._tool_budget_reached(
                        steps, tool_name
                    ):
                        error_message = self._tool_budget_message(tool_name)
                        step.metadata["tool_budget_reached"] = True
                        step.metadata["error_class"] = "protocol_error"
                    if not error_message and self.visual_call_validator is not None:
                        visual_error = self.visual_call_validator(
                            tool_name, prepared_args
                        )
                        if visual_error:
                            error_message = visual_error
                            step.metadata["invalid_visual_question"] = True
                            step.metadata["error_class"] = "protocol_error"

                    if error_message:
                        step.action_type = "format_error"
                        step.metadata.setdefault(
                            "rejection_reason",
                            error_message,
                        )
                        step.tool_result = json.dumps(
                            {"status": "error", "error": error_message},
                            ensure_ascii=False,
                        )
                    else:
                        step.tool_args = prepared_args
                        filtered_query_count = self._sanitize_search_queries(
                            tool_name,
                            prepared_args,
                        )
                        if filtered_query_count:
                            step.metadata["policy_filtered_query_count"] = (
                                filtered_query_count
                            )
                        search_policy_error = self._search_policy_error(
                            tool_name,
                            prepared_args,
                            rejected_query_count=filtered_query_count,
                        )
                        if search_policy_error:
                            self._record_search_query_rejection(
                                step,
                                tool_name,
                                search_policy_error,
                            )
                            step.tool_result = json.dumps(
                                {
                                    "status": "error",
                                    "error": search_policy_error,
                                },
                                ensure_ascii=False,
                            )
                        else:
                            step.action_type = "tool_call"
                            serialized, tool_metadata = await self._execute_tool(
                                tool_name,
                                self._execution_tool_args(prepared_args),
                            )
                            step.tool_result = serialized
                            step.metadata.update(tool_metadata)
                            self._archive_tool_step(
                                step,
                                action_index=(len(self.prior_steps) + len(steps) + 1),
                            )
                            evidence_so_far.append(
                                self._summarize_tool_result(
                                    tool_name,
                                    prepared_args,
                                    serialized,
                                )
                            )

                    steps.append(step)
                    response_steps.append(step)
                    function_results.append(
                        self._build_native_function_result(
                            call_id=call_id,
                            tool_name=tool_name,
                            result=step.tool_result,
                        )
                    )
                    follow_up_visual_items.extend(
                        await self._visual_reinjection_items(
                            tool_name,
                            result=step.tool_result,
                        )
                    )

                if any(item.action_type == "tool_call" for item in response_steps):
                    action_turns += 1
                else:
                    request_protocol_correction(
                        response_steps,
                        "the model returned no executable function call",
                    )
                for item in response_steps:
                    item.metadata["react_action_turn"] = action_turns
                    item.metadata["protocol_corrections_used"] = correction_turns
                    item.metadata["interaction_request_index"] = request_index
                previous_interaction_id = interaction_id
                next_input = list(function_results)
                if follow_up_visual_items:
                    # Interactions Step items and Content items cannot be
                    # mixed inside one function_result.  Keep the canonical
                    # function result text-only and place candidate/inspection
                    # images in a separate user_input step for the next turn.
                    next_input.append(
                        {
                            "type": "user_input",
                            "content": follow_up_visual_items,
                        }
                    )
                system_suffix = ""
                if self.should_stop and self.should_stop(steps):
                    self._set_session_pending_input(next_input)
                    if self.stop_output_factory is not None:
                        parsed = self.stop_output_factory()
                        steps.append(
                            StageStep(
                                round=request_index + 1,
                                stage_name=self.stage_name,
                                action_type="output",
                                output=parsed.model_dump(),
                                metadata={
                                    "stage": self.stage_name,
                                    "deterministic_segment_boundary": True,
                                    "previous_interaction_id": (
                                        previous_interaction_id
                                    ),
                                },
                            )
                        )
                        return parsed, steps
                    return await self._force_native_output(
                        previous_interaction_id=previous_interaction_id,
                        pending_input=next_input,
                        steps=steps,
                        evidence_so_far=evidence_so_far,
                    )
                continue

            content = self._extract_native_text(payload).strip()
            step = StageStep(
                round=request_index,
                stage_name=self.stage_name,
                thought=thought,
                tokens=tokens,
                metadata=common_metadata,
            )
            if not content:
                step.action_type = "format_error"
                step.thought = step.thought or "(empty native response)"
                steps.append(step)
                previous_interaction_id = interaction_id
                request_protocol_correction(
                    [step], "the model returned an empty response"
                )
                next_input = (
                    "Do not return final output yet. Invoke exactly one available "
                    "function for an active task."
                    if (
                        native_tools
                        and (
                            self.force_tool_each_round
                            or action_turns < self.min_tool_calls
                        )
                    )
                    else (
                        "Return one valid final JSON object."
                        if not native_tools
                        else "Return one valid final JSON object, or call one "
                        "available function."
                    )
                )
                system_suffix = ""
                continue

            output_json = self._extract_output(content) or self._try_parse_bare_json(
                content
            )
            if output_json is not None:
                step.action_type = "output"
                step.output = output_json
                step.metadata["policy_action"] = deepcopy(output_json)
                steps.append(step)
                parsed = self._validate_output(output_json)
                if parsed is not None:
                    accepted, reason = self._accept_output(parsed, steps)
                    if accepted:
                        return parsed, steps
                    step.action_type = "output_rejected"
                    step.metadata["rejection_reason"] = reason
                    previous_interaction_id = interaction_id
                    request_protocol_correction([step], reason)
                    next_input = f"Output rejected: {reason} Continue investigating with one function call."
                    system_suffix = ""
                    continue
                step.action_type = "output_rejected"
                step.metadata["rejection_reason"] = (
                    "output schema was invalid or incomplete"
                )
                next_input = (
                    "The output schema was invalid, and this segment has not yet "
                    "completed its required tool action. Do not return output. "
                    "Invoke exactly one available function for an active task."
                    if (
                        native_tools
                        and (
                            self.force_tool_each_round
                            or action_turns < self.min_tool_calls
                        )
                    )
                    else (
                        "The output schema was invalid. Return one valid JSON object."
                    )
                )
                previous_interaction_id = interaction_id
                request_protocol_correction(
                    [step],
                    "the output schema was invalid or incomplete",
                )
                system_suffix = ""
                continue

            step.action_type = "format_error"
            step.metadata["native_text_preview"] = content[:500]
            steps.append(step)
            previous_interaction_id = interaction_id
            request_protocol_correction(
                [step],
                "the model returned neither a function call nor valid JSON",
            )
            next_input = (
                "Do not return final output yet. Invoke exactly one available "
                "function for an active task."
                if (
                    native_tools
                    and (
                        self.force_tool_each_round or action_turns < self.min_tool_calls
                    )
                )
                else (
                    "Return exactly one valid final JSON object."
                    if not native_tools
                    else "Use a native function call, or return exactly one valid "
                    "final JSON object."
                )
            )
            system_suffix = ""

        return await self._force_native_output(
            previous_interaction_id=previous_interaction_id,
            pending_input=next_input,
            steps=steps,
            evidence_so_far=evidence_so_far,
        )
