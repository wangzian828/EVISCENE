from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .contracts import assert_model_visible
from .io import (
    load_json,
    load_jsonl,
    require_new_or_empty,
    sha256_file,
    write_json,
    write_jsonl,
)
from .policy_contract import live_runtime_tool_contract, project_converted_policy_row


SUPPORTED_DATASET_VERSIONS = frozenset(
    {
        "ifv-trajectory-sft-dataset-v2",
        "ifv-trajectory-sft-dataset-v3",
    }
)
LEGACY_STEP_DATASET_VERSION = "ifv-policy-dataset-v2"
OUTPUT_VERSION = "ifv-ms-swift-qwen-agent-v4"
SPLITS = ("train", "validation", "test")
TARGET_ROLES = frozenset(
    {"system", "user", "assistant", "tool_call", "tool_response"}
)


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _extract_think_block(content: str) -> str:
    match = re.search(r"<think>.*?</think>", content, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()
    return content.strip()


def _tool_call_from_legacy_assistant(content: str) -> dict[str, Any] | None:
    match = re.search(
        r"<function=([^>\s]+)>(.*?)</function>",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    arguments: dict[str, Any] = {}
    for parameter in re.finditer(
        r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
        match.group(2),
        re.IGNORECASE | re.DOTALL,
    ):
        name = parameter.group(1).strip()
        raw_value = parameter.group(2).strip()
        try:
            arguments[name] = json.loads(raw_value)
        except json.JSONDecodeError:
            arguments[name] = raw_value
    return {
        "name": match.group(1).strip(),
        "arguments": json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def _normalize_tool_call_content(content: str) -> str:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, Mapping):
        name = str(payload.get("name", "")).strip()
        arguments = payload.get("arguments", "{}")
        if name:
            return json.dumps(
                {
                    "name": name,
                    "arguments": _json_text(arguments),
                },
                ensure_ascii=False,
            )
    legacy = _tool_call_from_legacy_assistant(content)
    if legacy is None:
        raise ValueError("tool_call content is not a Qwen tool-call object")
    return json.dumps(legacy, ensure_ascii=False)


def _render_tool_response(content: str) -> str:
    """Drop the old transport envelope but keep the actual result as text."""

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content.strip()
    if (
        isinstance(payload, Mapping)
        and "observation_locator" in payload
        and "result" in payload
    ):
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if isinstance(payload, Mapping) and "result" in payload:
        payload = payload["result"]
    if isinstance(payload, str):
        return payload.strip()
    return json.dumps(payload, ensure_ascii=False)


def _to_qwen_agent_messages(messages: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Convert the old provider-neutral row to the observed ms-swift format.

    The Qwen template accepts a single initial user message followed by
    assistant/tool_call/tool_response turns.  Historical exporter rows used
    ``role=tool`` plus additional ``role=user`` stage-control messages; those
    user messages are folded into the immediately preceding tool response so
    the model still sees them without violating the Qwen turn grammar.
    """

    output: list[dict[str, str]] = []
    initial_user_index: int | None = None
    last_tool_response_index: int | None = None
    pending_legacy_tool_call = False

    for source_index, source in enumerate(messages):
        role = str(source.get("role", "")).strip()
        content = source.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{source_index}].content must be non-empty")

        if role == "system":
            if output and output[0]["role"] == "system":
                output[0]["content"] += "\n" + content
            else:
                output.insert(0, {"role": "system", "content": content})
            continue

        if role == "user":
            if initial_user_index is None:
                output.append({"role": "user", "content": content})
                initial_user_index = len(output) - 1
            elif last_tool_response_index is not None:
                output[last_tool_response_index]["content"] += (
                    "\n\n" + content
                )
            else:
                output[initial_user_index]["content"] += "\n\n" + content
            continue

        if role == "assistant":
            tool_call_match = re.search(
                r"<tool_call>.*?</tool_call>",
                content,
                re.IGNORECASE | re.DOTALL,
            )
            if tool_call_match:
                thought = _extract_think_block(
                    content[: tool_call_match.start()]
                )
                if not thought:
                    raise ValueError(
                        "assistant tool-call turn has no preserved <think> block"
                    )
                output.append({"role": "assistant", "content": thought})
                output.append(
                    {
                        "role": "tool_call",
                        "content": _normalize_tool_call_content(content),
                    }
                )
                pending_legacy_tool_call = True
                last_tool_response_index = None
            else:
                output.append(
                    {
                        "role": "assistant",
                        "content": content.strip(),
                    }
                )
                pending_legacy_tool_call = False
                last_tool_response_index = None
            continue

        if role == "tool_call":
            output.append(
                {
                    "role": "tool_call",
                    "content": _normalize_tool_call_content(content),
                }
            )
            pending_legacy_tool_call = True
            last_tool_response_index = None
            continue

        if role == "tool":
            if not pending_legacy_tool_call:
                raise ValueError(
                    f"messages[{source_index}] has a tool result without a call"
                )
            output.append(
                {
                    "role": "tool_response",
                    "content": _render_tool_response(content),
                }
            )
            pending_legacy_tool_call = False
            last_tool_response_index = len(output) - 1
            continue

        if role == "tool_response":
            if not pending_legacy_tool_call:
                raise ValueError(
                    f"messages[{source_index}] has a tool response without a call"
                )
            output.append(
                {
                    "role": "tool_response",
                    "content": content.strip(),
                }
            )
            pending_legacy_tool_call = False
            last_tool_response_index = len(output) - 1
            continue

        raise ValueError(f"messages[{source_index}] has unsupported role {role!r}")

    if pending_legacy_tool_call:
        raise ValueError("trajectory ends with a tool call without a response")
    return output


def _validate_qwen_agent_messages(
    messages: list[Mapping[str, Any]],
    *,
    image_count: int,
) -> None:
    if len(messages) < 3:
        raise ValueError("Qwen Agent row requires at least three messages")
    if messages[0].get("role") != "system":
        raise ValueError("Qwen Agent row must start with one system message")
    if messages[1].get("role") != "user":
        raise ValueError("Qwen Agent row must have one initial user message")
    if image_count and "<image>" not in str(messages[1].get("content", "")):
        raise ValueError("Qwen Agent image row requires <image> in initial user")
    marker_count = sum(
        str(message.get("content", "")).count("<image>")
        for message in messages
    )
    if marker_count != image_count:
        raise ValueError(
            "Qwen Agent <image> marker count must match images length"
        )
    if messages[-1].get("role") != "assistant":
        raise ValueError("Qwen Agent row must end with an assistant target")
    if any(str(message.get("role")) == "user" for message in messages[2:]):
        raise ValueError("Qwen Agent row cannot contain later user messages")
    for index, message in enumerate(messages):
        role = str(message.get("role", ""))
        if role not in TARGET_ROLES:
            raise ValueError(f"messages[{index}] has unsupported role {role!r}")
        allowed_keys = {"role", "content", "loss", "loss_scale"}
        unexpected = set(message) - allowed_keys
        if unexpected:
            raise ValueError(
                f"messages[{index}] has unsupported keys: {sorted(unexpected)}"
            )
        if "loss" in message and not isinstance(message["loss"], bool):
            raise ValueError(f"messages[{index}].loss must be boolean")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}].content must be non-empty")
    index = 2
    while index < len(messages):
        role = str(messages[index]["role"])
        if role == "assistant":
            if index == len(messages) - 1:
                break
            if str(messages[index + 1]["role"]) != "tool_call":
                raise ValueError(
                    "every non-final assistant target must be followed by tool_call"
                )
            index += 1
            continue
        if role == "tool_call":
            try:
                call = json.loads(str(messages[index]["content"]))
            except json.JSONDecodeError as exc:
                raise ValueError("tool_call content must be valid JSON") from exc
            if not isinstance(call, Mapping) or not str(
                call.get("name", "")
            ).strip():
                raise ValueError("tool_call must contain a name")
            if not isinstance(call.get("arguments"), str):
                raise ValueError("tool_call arguments must be a JSON string")
            if index + 1 >= len(messages) or str(
                messages[index + 1]["role"]
            ) != "tool_response":
                raise ValueError("every tool_call must be followed by tool_response")
            # ms-swift Agent rows may emit another tool call immediately after
            # a response. This represents one assistant action batch and is
            # present in the real reference sample.
            index += 2
            continue
        if role == "tool_response":
            raise ValueError("tool_response must follow tool_call")
        raise ValueError(f"messages[{index}] has an invalid turn role {role!r}")


def _convert_policy_row_with_stats(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if row.get("dataset_version") == LEGACY_STEP_DATASET_VERSION:
        raise ValueError(
            "step-level ifv-policy-dataset-v2 rows are no longer accepted "
            "by the default SFT converter; use full trajectory_sft rows."
        )
    source_messages = row.get("messages")
    if not isinstance(source_messages, list) or len(source_messages) < 2:
        raise ValueError("trajectory SFT row requires messages")
    # Validate the source representation before normalizing it.  Otherwise a
    # private field attached to a legacy message could be silently discarded
    # while converting only ``role`` and ``content``.
    assert_model_visible(
        {"messages": source_messages, "tools": row.get("tools", "")},
        location="trajectory_sft_source_row",
    )
    for index, message in enumerate(source_messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{index}] must be an object")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}].content must be non-empty")
    messages = _to_qwen_agent_messages(source_messages)
    images = row.get("images", [])
    if images and (
        not isinstance(images, list)
        or not all(isinstance(item, str) and item.strip() for item in images)
    ):
        raise ValueError("trajectory SFT row images must be a non-empty list")
    _validate_qwen_agent_messages(messages, image_count=len(images))
    tools = row.get("tools", "")
    if tools is None:
        tools = ""
    if not isinstance(tools, str):
        raise ValueError("trajectory SFT row tools must be a JSON string")
    parsed_tools: Any = []
    if tools.strip():
        try:
            parsed_tools = json.loads(tools)
        except json.JSONDecodeError as exc:
            raise ValueError("trajectory SFT row tools must be valid JSON") from exc
    # Dataset metadata such as teacher_score and split assignment is used for
    # auditing only.  Only the actual model-visible conversation and tool
    # schemas belong under the private-field check.
    assert_model_visible(
        {"messages": messages, "tools": parsed_tools},
        location="trajectory_sft_row",
    )
    output: dict[str, Any] = {}
    if isinstance(tools, str) and tools.strip():
        output["tools"] = tools
    output["messages"] = [dict(message) for message in messages]
    output["images"] = []
    if images:
        first_user = messages[1]
        if first_user is None or "<image>" not in str(
            first_user.get("content", "")
        ):
            raise ValueError(
                "trajectory SFT row images require <image> in the first user "
                "message"
            )
        output["images"] = list(images)
    output, projection_stats = project_converted_policy_row(output)
    _validate_qwen_agent_messages(
        output["messages"],
        image_count=len(output.get("images") or []),
    )
    return output, projection_stats


def convert_policy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return _convert_policy_row_with_stats(row)[0]


def _empty_projection_totals() -> dict[str, Any]:
    return {
        "rows": 0,
        "tool_calls": Counter(),
        "masked_unrepairable_tool_calls": Counter(),
        "removed_argument_fields": Counter(),
        "normalized_argument_fields": Counter(),
        "tool_responses": Counter(),
        "tool_schemas": Counter(),
        "final_observation_ids": Counter(),
    }


def _accumulate_projection(
    totals: dict[str, Any],
    stats: Mapping[str, Any],
) -> None:
    totals["rows"] += 1
    for section in (
        "tool_calls",
        "masked_unrepairable_tool_calls",
        "removed_argument_fields",
        "normalized_argument_fields",
        "tool_responses",
        "tool_schemas",
        "final_observation_ids",
    ):
        for key, value in (stats.get(section) or {}).items():
            if isinstance(value, bool):
                totals[section][str(key)] += int(value)
            elif isinstance(value, (int, float)):
                totals[section][str(key)] += value


def _render_projection_totals(totals: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rows": int(totals["rows"]),
        **{
            section: dict(sorted(totals[section].items()))
            for section in (
                "tool_calls",
                "masked_unrepairable_tool_calls",
                "removed_argument_fields",
                "normalized_argument_fields",
                "tool_responses",
                "tool_schemas",
                "final_observation_ids",
            )
        },
    }


def convert_policy_dataset(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_manifest = load_json(input_dir / "manifest.json")
    source_version = source_manifest.get("dataset_version")
    if source_version == LEGACY_STEP_DATASET_VERSION:
        raise ValueError(
            "refusing legacy step-level ifv-policy-dataset-v2; export "
            "ifv-trajectory-sft-dataset-v2 instead"
        )
    if source_version not in SUPPORTED_DATASET_VERSIONS:
        raise ValueError(
            "training adapter accepts only "
            f"{sorted(SUPPORTED_DATASET_VERSIONS)}, got "
            f"{source_version!r}"
        )
    require_new_or_empty(output_dir)

    artifacts: dict[str, dict[str, Any]] = {}
    index_rows: list[dict[str, Any]] = []
    projection_totals = _empty_projection_totals()
    row_id = 0
    for split in SPLITS:
        converted_rows: list[dict[str, Any]] = []
        for source_index, row in enumerate(load_jsonl(input_dir / f"{split}.jsonl")):
            converted, projection_stats = _convert_policy_row_with_stats(row)
            _accumulate_projection(projection_totals, projection_stats)
            converted_rows.append(converted)
            index_rows.append(
                {
                    "row_id": row_id,
                    "split": split,
                    "source_index": source_index,
                    "episode_id": row.get("episode_id"),
                    "case_id": row.get("case_id"),
                    "token_count_estimate": row.get("token_count_estimate"),
                    "tool_call_count": row.get("tool_call_count"),
                    "source_run_id": row.get("source_run_id"),
                    "runtime_commit": row.get("runtime_commit"),
                }
            )
            row_id += 1
        output_path = output_dir / f"{split}.jsonl"
        write_jsonl(output_path, converted_rows)
        artifacts[split] = {
            "path": output_path.name,
            "rows": len(converted_rows),
            "sha256": sha256_file(output_path),
        }

    index_path = output_dir / "index.jsonl"
    write_jsonl(index_path, index_rows)
    artifacts["index"] = {
        "path": index_path.name,
        "rows": len(index_rows),
        "sha256": sha256_file(index_path),
    }
    declared_counts = source_manifest.get("example_counts")
    if isinstance(declared_counts, Mapping) and all(
        isinstance(declared_counts.get(split), int) for split in SPLITS
    ):
        source_rows = sum(int(declared_counts[split]) for split in SPLITS)
        if source_rows != row_id:
            raise ValueError(
                "source manifest example counts do not match exported JSONL rows: "
                f"declared={source_rows}, observed={row_id}"
            )
    else:
        source_rows = row_id
    manifest = {
        "schema_version": "ifv-ms-swift-dataset-manifest-v1",
        "dataset_version": OUTPUT_VERSION,
        "framework": {"name": "ms-swift", "version": "4.4.2"},
        "format_contract": {
            "name": "ms-swift-qwen-agent",
            "version": "v4",
            "message_roles": [
                "system",
                "user",
                "assistant",
                "tool_call",
                "tool_response",
            ],
            "preserves_native_think": True,
            "assistant_loss_metadata": (
                "message.loss=false masks only historical actions that the live "
                "runtime rejects; all complete trajectories are retained"
            ),
        },
        "source": {
            "dataset_version": source_manifest["dataset_version"],
            "manifest_sha256": sha256_file(input_dir / "manifest.json"),
        },
        "example_count": row_id,
        "trajectory_format": "one_episode_per_row",
        "row_retention": {
            "source_rows": source_rows,
            "output_rows": row_id,
            "dropped_rows": 0,
        },
        "contract_projection": _render_projection_totals(projection_totals),
        "live_runtime_contract": live_runtime_tool_contract()[1],
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def repair_derived_policy_dataset(
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Repair a v2/v3 ms-swift package without dropping complete trajectories."""

    source_manifest = load_json(input_dir / "manifest.json")
    source_version = str(source_manifest.get("dataset_version", ""))
    if source_version not in {
        "ifv-ms-swift-qwen-agent-v2",
        "ifv-ms-swift-qwen-agent-v3",
        "ifv-ms-swift-qwen-agent-v4",
    }:
        raise ValueError(
            "repair-policy-contract requires an ms-swift Qwen Agent v2/v3/v4 dataset"
        )
    require_new_or_empty(output_dir)
    source_index_path = input_dir / "index.jsonl"
    source_index = load_jsonl(source_index_path) if source_index_path.is_file() else []
    source_index_by_split: dict[tuple[str, int], Mapping[str, Any]] = {}
    for item in source_index:
        if not isinstance(item, Mapping):
            continue
        source_index_by_split[
            (str(item.get("split", "")), int(item.get("source_index", -1)))
        ] = item

    artifacts: dict[str, dict[str, Any]] = {}
    output_index: list[dict[str, Any]] = []
    projection_totals = _empty_projection_totals()
    row_id = 0
    for split in SPLITS:
        source_artifact = (source_manifest.get("artifacts") or {}).get(split, {})
        source_path = input_dir / str(source_artifact.get("path", f"{split}.jsonl"))
        rows = load_jsonl(source_path) if source_path.is_file() else []
        projected_rows: list[dict[str, Any]] = []
        for source_index_value, row in enumerate(rows):
            projected, projection_stats = project_converted_policy_row(row)
            _validate_qwen_agent_messages(
                projected["messages"],
                image_count=len(projected.get("images") or []),
            )
            projected_rows.append(projected)
            _accumulate_projection(projection_totals, projection_stats)
            old_index = dict(
                source_index_by_split.get((split, source_index_value), {})
            )
            old_index.update(
                {
                    "row_id": row_id,
                    "split": split,
                    "source_index": source_index_value,
                }
            )
            output_index.append(old_index)
            row_id += 1
        output_path = output_dir / f"{split}.jsonl"
        write_jsonl(output_path, projected_rows)
        artifacts[split] = {
            "path": output_path.name,
            "rows": len(projected_rows),
            "sha256": sha256_file(output_path),
        }

    index_path = output_dir / "index.jsonl"
    write_jsonl(index_path, output_index)
    artifacts["index"] = {
        "path": index_path.name,
        "rows": len(output_index),
        "sha256": sha256_file(index_path),
    }
    manifest = {
        "schema_version": "ifv-ms-swift-dataset-manifest-v1",
        "dataset_version": OUTPUT_VERSION,
        "framework": source_manifest.get(
            "framework", {"name": "ms-swift", "version": "4.4.2"}
        ),
        "format_contract": {
            "name": "ms-swift-qwen-agent",
            "version": "v4",
            "message_roles": [
                "system",
                "user",
                "assistant",
                "tool_call",
                "tool_response",
            ],
            "preserves_native_think": True,
            "assistant_loss_metadata": (
                "message.loss=false masks only historical actions that the live "
                "runtime rejects; all complete trajectories are retained"
            ),
        },
        "source": {
            "dataset_version": source_version,
            "manifest_sha256": sha256_file(input_dir / "manifest.json"),
        },
        "example_count": row_id,
        "trajectory_format": "one_episode_per_row",
        "row_retention": {
            "source_rows": sum(
                int(((source_manifest.get("artifacts") or {}).get(split) or {}).get("rows", 0))
                for split in SPLITS
            ),
            "output_rows": row_id,
            "dropped_rows": 0,
        },
        "contract_projection": _render_projection_totals(projection_totals),
        "live_runtime_contract": live_runtime_tool_contract()[1],
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
