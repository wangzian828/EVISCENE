"""Pure canonical-trace to policy-example exporter."""

from __future__ import annotations

import json
from typing import Any, Iterable, List, Mapping, Protocol, Sequence

from eviscene.trajectory.schema import (
    ActionOnlyTrajectoryExample,
    PolicyExample,
    TrajectorySFTExample,
)
from eviscene.orchestrator.react_runtime import (
    REACT_RUNTIME_SCHEMA_VERSION,
    is_unified_react_runtime_budget_action,
)
from eviscene.orchestrator.tool_result import parse_tool_result
from eviscene.trajectory.media_projection import (
    TrajectoryMediaProjection,
    image_markers,
    project_trajectory_media,
)


FORBIDDEN_PRIVATE_KEYS = frozenset(
    {
        "gold",
        "evaluation_gold",
        "factual_status",
        "acceptable_evidence",
        "expected_status",
        "ground_truth",
        "teacher_score",
        "process_metrics",
        "source_access_policy",
        "excluded_domains",
        "excluded_urls",
    }
)


class TokenizerAdapter(Protocol):
    tokenizer_id: str

    def encode(self, text: str) -> Sequence[int]:
        """Encode one canonical JSON string without adding implicit tokens."""


class Utf8ByteTokenizer:
    """Stable default adapter; model-specific tokenizers remain injectable."""

    tokenizer_id = "utf8-byte-v1"

    def encode(self, text: str) -> Sequence[int]:
        return list(text.encode("utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> List[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _assert_no_private_data(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            if key.casefold() in FORBIDDEN_PRIVATE_KEYS:
                raise ValueError(
                    f"private/evaluator field is forbidden in policy input: {child_path}"
                )
            _assert_no_private_data(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_private_data(child, f"{path}[{index}]")


def _observation_refs(policy_input: Mapping[str, Any]) -> List[str]:
    refs: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key in (
                "call_id",
                "function_call_id",
                "evidence_id",
                "finding_id",
                "task_id",
                "fact_id",
                "interaction_id",
            ):
                rendered = str(value.get(key, "") or "").strip()
                if rendered:
                    refs.append(rendered)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(policy_input.get("input_payload"))
    return list(dict.fromkeys(refs))[:64]


def _example_type(stage: str) -> str | None:
    if stage == "unified_react":
        return "react"
    if stage == "unified_judgment":
        return "judgment"
    return None


def _strip_gemini_wire_instructions(value: str) -> str:
    marker = "Native Gemini Interactions protocol:"
    if marker in value:
        value = value.split(marker, 1)[0]
    return value.rstrip()


def _tool_schema_key(tool: Mapping[str, Any]) -> str:
    return canonical_json(tool)


def _normalize_tool_schema(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        if isinstance(item.get("function"), Mapping):
            function = dict(item["function"])
            name = str(function.get("name", "")).strip()
            if not name:
                continue
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(function.get("description", "")),
                        "parameters": function.get("parameters", {}),
                    },
                }
            )
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(item.get("description", "")),
                    "parameters": item.get("parameters", {}),
                },
            }
        )
    return result


def _json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return value


def _qwen_think_block(thought: str) -> str:
    value = thought.strip()
    if value.startswith("<think>") and value.endswith("</think>"):
        return value
    return f"<think>\n{value}\n</think>"


def _qwen_answer_block(answer: str) -> str:
    value = answer.strip()
    if value.startswith("<answer>") and value.endswith("</answer>"):
        return value
    return f"<answer>\n{value}\n</answer>"


def _render_tool_response_content(
    value: str,
    *,
    observation_id: str = "",
    tool_name: str = "",
    tool_success: bool | None = None,
) -> str:
    """Render a result plus the model-visible locator needed by final judgment.

    The opaque provider call ID is the runtime observation ID.  Keeping it only
    in trace metadata makes a later ``verdict_observation_ids`` target causally
    impossible, so accepted exports expose a compact locator next to the public
    result.  No provider transport payload or evaluator data is included.
    """

    raw = value.strip()
    if raw.startswith("<tool_response>") and raw.endswith("</tool_response>"):
        raw = raw[len("<tool_response>") : -len("</tool_response>")].strip()
    try:
        payload = json.loads(raw)
    except Exception:
        payload = raw
    else:
        if isinstance(payload, Mapping) and "result" in payload:
            payload = payload["result"]
    resolved_id = str(observation_id).strip()
    if resolved_id:
        return canonical_json(
            {
                "observation_locator": {
                    "observation_id": resolved_id,
                    "tool_name": str(tool_name).strip(),
                    "tool_success": tool_success is True,
                },
                "result": payload,
            }
        )
    if isinstance(payload, Mapping) and "result" in payload:
        payload = payload["result"]
    if isinstance(payload, str):
        return payload.strip()
    return canonical_json(payload)


def _qwen_tool_call_json(
    *,
    name: str,
    arguments: Mapping[str, Any],
) -> str:
    """Match the observed ms-swift Agent row: name + JSON argument string."""

    return json.dumps(
        {
            "name": name,
            "arguments": canonical_json(arguments),
        },
        ensure_ascii=False,
    )


def _initial_observation_packet(trace: Mapping[str, Any]) -> str:
    state = _mapping(trace.get("state"))
    runtime_case = _mapping(state.get("runtime_case"))
    payload = {
        "case_id": str(runtime_case.get("case_id") or trace.get("image_id") or ""),
        "image_sha256": str(runtime_case.get("image_sha256", "")),
        "input_mode": str(trace.get("input_mode") or state.get("input_mode") or ""),
        "initial_observations": [],
    }
    return (
        "<image>\n"
        "Image factual verification episode. Use the supplied image, "
        "observations, and subsequent tool responses as context. Generate only "
        "the next policy message when it is your turn.\n\n" + canonical_json(payload)
    )


def _stage_user_packet(
    *,
    example_type: str,
    policy_input: Mapping[str, Any],
) -> str:
    input_payload = policy_input.get("input_payload", "")
    payload: dict[str, Any] = {
        "stage": example_type,
        "stage_instruction": _strip_gemini_wire_instructions(
            str(policy_input.get("system_instruction", ""))
        ),
        "input_payload": input_payload,
    }
    response_format = policy_input.get("response_format")
    if isinstance(response_format, Mapping) and example_type != "react":
        payload["required_structured_output_contract"] = dict(response_format)
    tools = _normalize_tool_schema(policy_input.get("tools"))
    if tools and example_type == "react":
        payload["authorized_tool_names"] = [
            str(tool["function"]["name"]) for tool in tools
        ]
    return canonical_json(payload)


def _stage_control_packet(
    *,
    example_type: str,
    policy_input: Mapping[str, Any],
) -> str:
    """Render a small transition control packet for a continued episode.

    The first stage needs its complete input projection because it introduces
    the case.  After that, the full event history already contains the prior
    assistant actions and raw tool observations. Replaying each stage's
    cumulative provider history would duplicate observations. Keep only the
    current stage contract and active tool boundary.
    """

    payload: dict[str, Any] = {
        "stage": example_type,
        "output_mode": (
            "native_tool_call" if example_type == "react" else "structured_json"
        ),
    }
    instruction = _strip_gemini_wire_instructions(
        str(policy_input.get("system_instruction", ""))
    )
    if instruction:
        payload["stage_instruction"] = instruction
    tools = _normalize_tool_schema(policy_input.get("tools"))
    if example_type == "react":
        payload["authorized_tool_names"] = [
            str(tool["function"]["name"]) for tool in tools
        ]
    return canonical_json(payload)


def _trajectory_candidate_steps(
    trace: Mapping[str, Any],
) -> list[tuple[int, Mapping[str, Any], str]]:
    state = _mapping(trace.get("state"))
    candidates: list[tuple[int, Mapping[str, Any], str]] = []
    rejected_judgment_inputs: dict[str, Mapping[str, Any]] = {}
    for index, step in enumerate(_rows(state.get("all_steps"))):
        stage = str(step.get("stage", "")).strip()
        action_type = str(step.get("action_type", "")).strip()
        metadata = _mapping(step.get("metadata"))
        if stage == "unified_judgment" and action_type in {
            "format_error",
            "output_rejected",
        }:
            request_id = str(metadata.get("context_request_id", "")).strip()
            policy_input = metadata.get("policy_input")
            if request_id and isinstance(policy_input, Mapping):
                rejected_judgment_inputs[request_id] = policy_input
            continue
        if (
            stage == "unified_judgment"
            and action_type == "output"
            and metadata.get("forced_output") is True
            and str(metadata.get("interaction_lifecycle_kind", ""))
            == "protocol_correction"
            and isinstance(step.get("output"), Mapping)
            and all(
                key in step["output"]
                and _mapping(trace.get("judgment")).get(key) == step["output"][key]
                for key in (
                    "verdict",
                    "confidence",
                    "verdict_observation_ids",
                    "overall_assessment",
                    "fact_check_report",
                )
            )
        ):
            parent_id = str(metadata.get("parent_context_request_id", "")).strip()
            inherited_input = rejected_judgment_inputs.get(parent_id)
            if inherited_input is not None:
                recovered_step = dict(step)
                recovered_metadata = dict(metadata)
                recovered_metadata["policy_input"] = dict(inherited_input)
                recovered_metadata["policy_action"] = dict(step["output"])
                recovered_metadata[
                    "sft_policy_input_provenance"
                ] = "rejected_parent_request"
                recovered_step["metadata"] = recovered_metadata
                candidates.append((index, recovered_step, "judgment"))
                continue
        if action_type in {
            "format_error",
            "output_rejected",
            "policy_replan",
        }:
            continue
        example_type = _example_type(stage)
        if example_type is None:
            continue
        if not isinstance(metadata.get("policy_input"), Mapping):
            continue
        if not isinstance(metadata.get("policy_action"), Mapping):
            continue
        candidates.append((index, step, example_type))
    return candidates


def trajectory_policy_step_ids(
    trace: Mapping[str, Any],
    *,
    episode_id: str | None = None,
) -> list[str]:
    """Return stable internal step IDs without exporting step-level SFT rows.

    RL reward ledgers still need to identify the policy turns in an episode.
    Those IDs are runtime bookkeeping only; they are deliberately not written
    to the SFT dataset as one-row-per-step examples.
    """

    resolved_episode_id = str(
        episode_id
        or trace.get("image_id")
        or _mapping(trace.get("state")).get("image_id")
        or ""
    ).strip()
    if not resolved_episode_id:
        return []
    result: list[str] = []
    for index, step, example_type in _trajectory_candidate_steps(trace):
        metadata = _mapping(step.get("metadata"))
        interaction_id = str(metadata.get("interaction_id", "")).strip()
        result.append(
            f"{resolved_episode_id}:{example_type}:{interaction_id}"
            if interaction_id
            else f"{resolved_episode_id}:{example_type}:{index + 1}"
        )
    return result


def export_trajectory_sft_example(
    trace: Mapping[str, Any],
    *,
    tokenizer: TokenizerAdapter | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    require_provider_thought: bool = True,
    media_projection: TrajectoryMediaProjection | None = None,
) -> TrajectorySFTExample | ActionOnlyTrajectoryExample:
    """Export one complete accepted episode as one prefix-preserving SFT row.

    The runtime may use a provider-side cumulative InteractionSession, but the
    exported conversation is rebuilt once from the chronological canonical
    steps. Each tool result is therefore represented exactly once as the
    observation before the next policy target; provider request snapshots are
    never copied into the conversation as additional history.
    """

    state = _mapping(trace.get("state"))
    if str(trace.get("input_mode") or state.get("input_mode") or "") != ("image_only"):
        raise ValueError("trajectory SFT exporter accepts image-only traces only")
    policy_version = str(
        trace.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    )
    if policy_version != "unified-react-v1":
        raise ValueError(
            "trajectory SFT exporter received an unsupported decision policy"
        )
    _unified_react_quality_gate(
        trace,
        state,
        require_provider_thought=require_provider_thought,
    )

    tokenizer = tokenizer or Utf8ByteTokenizer()
    source_metadata = source_metadata or {}
    episode_id = str(trace.get("image_id") or state.get("image_id") or "").strip()
    runtime_case = _mapping(state.get("runtime_case"))
    case_id = str(runtime_case.get("case_id") or episode_id).strip()
    image_path = str(
        runtime_case.get("image_path")
        or trace.get("image_path")
        or state.get("image_path")
        or ""
    ).strip()
    if not episode_id or not case_id or not image_path:
        raise ValueError("canonical trace requires episode ID, case ID, and image path")

    # This is the provider-neutral form consumed by the Qwen/ms-swift
    # adapter.  It intentionally mirrors the verified external sample:
    # tool calls and tool results are separate message roles, while the
    # assistant thought remains its own supervised assistant message.
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are the Image Factual Verifier policy model. Investigate "
                "the supplied image, use the available tools when useful, and "
                "produce the next tool call or final answer. Preserve the "
                "visible reasoning format used by the Qwen Agent template."
            ),
        },
        {"role": "user", "content": _initial_observation_packet(trace)},
    ]
    tool_schemas: dict[str, dict[str, Any]] = {}
    tool_call_count = 0
    candidates = _trajectory_candidate_steps(trace)
    if media_projection is None:
        media_projection = project_trajectory_media(
            trace,
            candidate_steps=[step for _, step, _ in candidates],
            fallback_image_path=image_path,
        )
    if not media_projection.initial_images:
        raise ValueError("trajectory SFT export cannot recover the initial image")
    pending_tool_response: str | None = None
    pending_tool_step_index: int | None = None
    previous_example_type = ""
    for position, (_, step, example_type) in enumerate(candidates):
        metadata = _mapping(step.get("metadata"))
        policy_input = dict(_mapping(metadata.get("policy_input")))
        policy_action = dict(_mapping(metadata.get("policy_action")))
        _assert_no_private_data(policy_input)
        _assert_no_private_data(policy_action)
        for tool_schema in _normalize_tool_schema(policy_input.get("tools")):
            tool_schemas.setdefault(_tool_schema_key(tool_schema), tool_schema)
        initial_stage_packet = _stage_user_packet(
            example_type=example_type,
            policy_input=policy_input,
        )
        stage_control = _stage_control_packet(
            example_type=example_type,
            policy_input=policy_input,
        )
        stage_changed = position > 0 and example_type != previous_example_type
        if position == 0:
            messages[1]["content"] += "\n\n" + initial_stage_packet
        if pending_tool_response is not None:
            projected_images = media_projection.images_after_step.get(
                int(pending_tool_step_index or 0),
                [],
            )
            if projected_images:
                pending_tool_response += "\n\n" + image_markers(len(projected_images))
            messages.append(
                {
                    "role": "tool_response",
                    "content": pending_tool_response,
                }
            )
            pending_tool_response = None
            pending_tool_step_index = None
        if stage_changed:
            # Qwen's Agent template does not accept a new user turn after the
            # initial prompt.  Keep the stage boundary visible by attaching
            # its compact control to the immediately preceding observation.
            if messages and messages[-1].get("role") == "tool_response":
                messages[-1]["content"] += (
                    "\n\n<stage_control>" + stage_control + "</stage_control>"
                )
            else:
                messages[1]["content"] += "\n\n" + stage_control
        thought = str(step.get("thought", "") or "").strip()
        if thought:
            _assert_no_private_data(thought)
        if example_type == "react":
            if (
                policy_version == "unified-react-v1"
                and require_provider_thought
                and not thought
            ):
                raise ValueError(
                    "unified-react reasoning SFT requires provider-visible "
                    "thought for every ReAct action"
                )
            if str(policy_action.get("type", "")) != "tool_call":
                raise ValueError("react trajectory action must be a tool call")
            # Supervise the executed, normalized action, not the pre-execution
            # proposal that may differ from the following observation.
            if not isinstance(step.get("tool_args"), Mapping):
                raise ValueError(
                    "react trajectory action lacks recorded executed tool arguments"
                )
            executed_arguments = dict(step["tool_args"])
            executed_tool_name = str(step.get("tool_name", "")).strip()
            proposed_tool_name = str(policy_action.get("name", "")).strip()
            if not executed_tool_name:
                raise ValueError(
                    "react trajectory action lacks recorded executed tool name"
                )
            if proposed_tool_name and proposed_tool_name != executed_tool_name:
                raise ValueError(
                    "react trajectory proposal and executed tool names disagree"
                )
            tool_call = {
                "name": executed_tool_name,
                "arguments": executed_arguments,
            }
            if not tool_call["name"]:
                raise ValueError("react trajectory action lacks tool name")
            if not thought and require_provider_thought:
                raise ValueError(
                    "unified-react reasoning SFT requires provider-visible "
                    "thought for every ReAct action"
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": _qwen_think_block(thought)
                    if thought
                    else "<think>\n\n</think>",
                }
            )
            messages.append(
                {
                    "role": "tool_call",
                    "content": _qwen_tool_call_json(
                        name=tool_call["name"],
                        arguments=tool_call["arguments"],
                    ),
                }
            )
            tool_call_count += 1
            tool_result = str(step.get("tool_result", "") or "").strip()
            if not tool_result:
                raise ValueError("react trajectory action lacks a recorded tool result")
            observation_id = str(metadata.get("function_call_id", "") or "").strip()
            if not observation_id:
                raise ValueError(
                    "react trajectory action lacks a model-visible observation ID"
                )
            try:
                _, parsed_tool_success = parse_tool_result(tool_result)
            except Exception:
                parsed_tool_success = None
            pending_tool_response = _render_tool_response_content(
                tool_result,
                observation_id=observation_id,
                tool_name=tool_call["name"],
                tool_success=(
                    metadata.get("tool_success")
                    if isinstance(metadata.get("tool_success"), bool)
                    else parsed_tool_success
                ),
            )
            pending_tool_step_index = position
        else:
            answer = canonical_json(policy_action)
            assistant_content = (
                _qwen_think_block(thought) + "\n\n" if thought else ""
            ) + _qwen_answer_block(answer)
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                }
            )
        previous_example_type = example_type

    if pending_tool_response is not None:
        raise ValueError(
            "trajectory ended after a tool call without a next policy turn"
        )

    if len(messages) <= 2:
        raise ValueError("trajectory SFT export found no supervised policy turns")
    tools = canonical_json(list(tool_schemas.values())) if tool_schemas else ""
    token_payload = {"messages": messages}
    if tools:
        token_payload["tools"] = tools
    token_count = len(tokenizer.encode(canonical_json(token_payload)))
    example_payload = {
        "trajectory_version": (
            "ifv-trajectory-action-only-v1"
            if policy_version == "unified-react-v1" and not require_provider_thought
            else "ifv-trajectory-sft-v3"
        ),
        "episode_id": episode_id,
        "case_id": case_id,
        "source_run_id": str(source_metadata.get("source_run_id", "")),
        "runtime_commit": str(source_metadata.get("runtime_commit", "")),
        "release_id": str(source_metadata.get("release_id", "")),
        "runtime_contract_version": str(
            source_metadata.get("runtime_contract_version", "")
        ),
        "process_reference_protocol_version": str(
            source_metadata.get("process_reference_protocol_version", "")
        ),
        "messages": messages,
        "images": media_projection.images,
        "tools": tools,
        "token_count_estimate": token_count,
        "message_count": len(messages),
        "tool_call_count": tool_call_count,
    }
    if policy_version == "unified-react-v1" and not require_provider_thought:
        return ActionOnlyTrajectoryExample(**example_payload)
    return TrajectorySFTExample(**example_payload)


def export_trajectory_action_only_example(
    trace: Mapping[str, Any],
    *,
    tokenizer: TokenizerAdapter | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    media_projection: TrajectoryMediaProjection | None = None,
) -> ActionOnlyTrajectoryExample:
    """Export a unified trace with executable actions but no thought targets."""

    exported = export_trajectory_sft_example(
        trace,
        tokenizer=tokenizer,
        source_metadata=source_metadata,
        require_provider_thought=False,
        media_projection=media_projection,
    )
    if not isinstance(exported, ActionOnlyTrajectoryExample):
        raise ValueError("action-only export is reserved for unified-react-v1 traces")
    return exported


def _unified_react_quality_gate(
    trace: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    require_provider_thought: bool = True,
) -> None:
    """Reject mixed or non-auditable unified-ReAct episodes before SFT export."""

    if str(trace.get("termination", "")) != "success":
        raise ValueError("unified-react policy export requires a successful trace")
    investigation = _mapping(state.get("investigation_state"))
    if str(investigation.get("schema_version", "")).strip() != (
        REACT_RUNTIME_SCHEMA_VERSION
    ):
        raise ValueError(
            "unified-react policy export requires the raw-history runtime schema"
        )
    steps = _rows(state.get("all_steps"))
    invalid_stages = {
        str(step.get("stage", "")).strip()
        for step in steps
        if str(step.get("stage", "")).strip()
        not in {"unified_react", "unified_judgment"}
    }
    if invalid_stages:
        raise ValueError(
            "unified-react trace contains unsupported policy stages: "
            + ", ".join(sorted(invalid_stages))
        )
    actions = [
        step
        for step in steps
        if str(step.get("stage", "")) == "unified_react"
        and str(step.get("action_type", "")) == "tool_call"
        and is_unified_react_runtime_budget_action(step)
    ]
    if len(actions) < 1:
        raise ValueError("unified-react trace requires at least one ReAct action")
    if any(not isinstance(step.get("tool_args"), Mapping) for step in actions):
        raise ValueError(
            "unified-react trace requires recorded executed tool arguments per action"
        )
    if any(
        not str(_mapping(step.get("metadata")).get("function_call_id", "")).strip()
        for step in actions
    ):
        raise ValueError(
            "unified-react trace requires a model-visible observation ID per action"
        )
    if any(not str(step.get("tool_result", "")).strip() for step in actions):
        raise ValueError(
            "unified-react trace requires a recorded tool result per action"
        )
    if require_provider_thought and any(
        not str(step.get("thought", "") or "").strip() for step in actions
    ):
        raise ValueError(
            "unified-react trace lacks provider-visible thought; route to action-only/RL"
        )
    if not any(
        str(step.get("stage", "")).strip() == "unified_judgment"
        and str(step.get("action_type", "")).strip() == "output"
        for step in steps
    ):
        raise ValueError("current unified-react trace requires a final judgment output")


def unified_react_training_buckets(
    trace: Mapping[str, Any],
) -> list[str]:
    """Return non-overlapping SFT buckets plus the RL-candidate membership.

    The structural audit is shared.  A trace with missing readable provider
    thought remains an executable action-only/RL candidate, but it must never
    enter the reasoning SFT corpus by accident.
    """

    state = _mapping(trace.get("state"))
    policy_version = str(
        trace.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    )
    if policy_version != "unified-react-v1":
        raise ValueError("unified training buckets accept unified-react-v1 traces only")
    _unified_react_quality_gate(
        trace,
        state,
        require_provider_thought=False,
    )
    actions = [
        step
        for step in _rows(state.get("all_steps"))
        if str(step.get("stage", "")) == "unified_react"
        and str(step.get("action_type", "")) == "tool_call"
    ]
    reasoning_ready = all(
        str(step.get("thought", "") or "").strip() for step in actions
    )
    return [
        "reasoning_sft" if reasoning_ready else "action_only",
        "rl_candidate",
    ]


def export_policy_examples(
    trace: Mapping[str, Any],
    *,
    tokenizer: TokenizerAdapter | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> List[PolicyExample]:
    """Export actual model-visible requests/actions from one canonical trace."""

    state = _mapping(trace.get("state"))
    if str(trace.get("input_mode") or state.get("input_mode") or "") != ("image_only"):
        raise ValueError("policy exporter accepts image-only traces only")
    policy_version = str(
        trace.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    )
    if policy_version != "unified-react-v1":
        raise ValueError("policy exporter received an unsupported decision policy")
    _unified_react_quality_gate(trace, state)

    tokenizer = tokenizer or Utf8ByteTokenizer()
    source_metadata = source_metadata or {}
    episode_id = str(trace.get("image_id") or state.get("image_id") or "").strip()
    if not episode_id:
        raise ValueError("canonical trace requires image_id")

    candidates: List[tuple[int, Mapping[str, Any], str]] = []
    for index, step in enumerate(_rows(state.get("all_steps"))):
        if str(step.get("action_type", "")) in {
            "format_error",
            "output_rejected",
            "policy_replan",
        }:
            continue
        stage = str(step.get("stage", "")).strip()
        example_type = _example_type(stage)
        metadata = _mapping(step.get("metadata"))
        if example_type is None:
            continue
        if not isinstance(metadata.get("policy_input"), Mapping):
            continue
        if not isinstance(metadata.get("policy_action"), Mapping):
            continue
        candidates.append((index, step, example_type))

    trace_failed = str(trace.get("termination", "")).strip() == "error"
    examples: List[PolicyExample] = []
    for position, (index, step, example_type) in enumerate(candidates):
        metadata = _mapping(step.get("metadata"))
        policy_input = dict(_mapping(metadata.get("policy_input")))
        policy_action = dict(_mapping(metadata.get("policy_action")))
        _assert_no_private_data(policy_input)
        _assert_no_private_data(policy_action)
        input_ids = list(tokenizer.encode(canonical_json(policy_input)))
        action_ids = list(tokenizer.encode(canonical_json(policy_action)))
        action_valid = str(step.get("action_type", "")) not in {
            "format_error",
            "output_rejected",
            "policy_replan",
        }
        fatal_boundary = trace_failed and position == len(candidates) - 1
        trainable = action_valid and not fatal_boundary
        terminated = (
            example_type == "judgment"
            or position == len(candidates) - 1
            and str(trace.get("termination", "")) in {"success", "error"}
        )
        interaction_id = str(metadata.get("interaction_id", "")).strip()
        step_id = (
            f"{episode_id}:{example_type}:{interaction_id}"
            if interaction_id
            else f"{episode_id}:{example_type}:{index + 1}"
        )
        examples.append(
            PolicyExample(
                trajectory_version=("ifv-policy-v3"),
                tokenizer_id=tokenizer.tokenizer_id,
                episode_id=episode_id,
                step_id=step_id,
                source_run_id=str(source_metadata.get("source_run_id", "")),
                runtime_commit=str(source_metadata.get("runtime_commit", "")),
                release_id=str(source_metadata.get("release_id", "")),
                runtime_contract_version=str(
                    source_metadata.get("runtime_contract_version", "")
                ),
                process_reference_protocol_version=str(
                    source_metadata.get(
                        "process_reference_protocol_version",
                        "",
                    )
                ),
                example_type=example_type,
                runtime_observation_refs=_observation_refs(policy_input),
                policy_input=policy_input,
                policy_action=policy_action,
                policy_input_token_ids=input_ids,
                policy_action_token_ids=action_ids,
                policy_action_loss_mask=[1 if trainable else 0 for _ in action_ids],
                action_valid=action_valid,
                terminated=terminated,
                fatal_boundary=fatal_boundary,
            )
        )
    return examples


def examples_as_jsonl(examples: Iterable[PolicyExample]) -> str:
    lines = [item.model_dump_json(exclude_none=False) for item in examples]
    return "\n".join(lines) + ("\n" if lines else "")
