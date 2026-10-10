"""Versioned schemas for training examples derived from canonical traces."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PolicyExample(StrictModel):
    trajectory_version: Literal["ifv-policy-v3"] = "ifv-policy-v3"
    tokenizer_id: str = Field(min_length=1, max_length=200)
    episode_id: str = Field(min_length=1, max_length=200)
    step_id: str = Field(min_length=1, max_length=300)
    source_run_id: str = Field(default="", max_length=300)
    runtime_commit: str = Field(default="", max_length=100)
    release_id: str = Field(default="", max_length=300)
    runtime_contract_version: str = Field(default="", max_length=200)
    process_reference_protocol_version: str = Field(
        default="",
        max_length=200,
    )
    example_type: Literal["react", "judgment"]
    runtime_observation_refs: List[str] = Field(default_factory=list, max_length=64)
    policy_input: Dict[str, Any]
    policy_action: Dict[str, Any]
    policy_input_token_ids: List[int] = Field(default_factory=list)
    policy_action_token_ids: List[int] = Field(default_factory=list)
    policy_action_loss_mask: List[int] = Field(default_factory=list)
    action_valid: bool
    terminated: bool
    fatal_boundary: bool = False

    @model_validator(mode="after")
    def validate_token_alignment(self) -> "PolicyExample":
        if len(self.policy_action_token_ids) != len(
            self.policy_action_loss_mask
        ):
            raise ValueError(
                "policy_action_token_ids and policy_action_loss_mask must align"
            )
        if any(value not in {0, 1} for value in self.policy_action_loss_mask):
            raise ValueError("policy_action_loss_mask values must be 0 or 1")
        return self


class PerceptionExample(StrictModel):
    trajectory_version: Literal["ifv-perception-v1"] = "ifv-perception-v1"
    episode_id: str = Field(min_length=1, max_length=200)
    source_run_id: str = Field(default="", max_length=300)
    runtime_commit: str = Field(default="", max_length=100)
    release_id: str = Field(default="", max_length=300)
    runtime_contract_version: str = Field(default="", max_length=200)
    image_path: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    instruction: str = Field(min_length=1)
    perception_report: Dict[str, Any]


class DatasetPerceptionExample(PerceptionExample):
    dataset_version: Literal["ifv-policy-dataset-v2"] = (
        "ifv-policy-dataset-v2"
    )
    split: Literal["train", "validation", "test"]
    split_group_id: str = Field(min_length=1, max_length=100)


class TrajectorySFTExample(StrictModel):
    """One complete accepted episode in the Qwen/ms-swift Agent format."""

    trajectory_version: Literal["ifv-trajectory-sft-v3"] = (
        "ifv-trajectory-sft-v3"
    )
    episode_id: str = Field(min_length=1, max_length=200)
    case_id: str = Field(min_length=1, max_length=200)
    source_run_id: str = Field(default="", max_length=300)
    runtime_commit: str = Field(default="", max_length=100)
    release_id: str = Field(default="", max_length=300)
    runtime_contract_version: str = Field(default="", max_length=200)
    process_reference_protocol_version: str = Field(
        default="",
        max_length=200,
    )
    messages: List[Dict[str, Any]] = Field(min_length=2)
    images: List[str] = Field(default_factory=list)
    tools: str = ""
    token_count_estimate: int = Field(ge=1)
    message_count: int = Field(ge=2)
    tool_call_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_messages(self) -> "TrajectorySFTExample":
        if self.message_count != len(self.messages):
            raise ValueError("message_count must match messages length")
        allowed_roles = {
            "system",
            "user",
            "assistant",
            "tool_call",
            "tool_response",
        }
        if len(self.messages) < 3:
            raise ValueError("trajectory requires system, user, and assistant")
        if self.messages[0].get("role") != "system":
            raise ValueError("trajectory must start with a system message")
        if self.messages[1].get("role") != "user":
            raise ValueError("trajectory must have one initial user message")
        marker_count = sum(
            str(message.get("content", "")).count("<image>")
            for message in self.messages
        )
        if self.images and "<image>" not in str(
            self.messages[1].get("content", "")
        ):
            raise ValueError(
                "trajectory images require an <image> marker in the initial user"
            )
        if marker_count != len(self.images):
            raise ValueError(
                "trajectory <image> marker count must match images length"
            )
        if self.messages[-1].get("role") != "assistant":
            raise ValueError("trajectory must end with an assistant target")
        for index, message in enumerate(self.messages):
            if not isinstance(message, dict):
                raise ValueError(f"messages[{index}] must be an object")
            role = str(message.get("role", ""))
            if role not in allowed_roles:
                raise ValueError(f"messages[{index}] has unsupported role")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"messages[{index}].content must be non-empty")
            if set(message) != {"role", "content"}:
                raise ValueError(
                    f"messages[{index}] must contain only role and content"
                )
        if any(
            str(message.get("role")) == "user"
            for message in self.messages[2:]
        ):
            raise ValueError("trajectory cannot contain later user messages")
        index = 2
        while index < len(self.messages):
            role = self.messages[index].get("role")
            if role == "assistant":
                if index == len(self.messages) - 1:
                    break
                if self.messages[index + 1].get("role") != "tool_call":
                    raise ValueError(
                        "every non-final assistant target must be followed by tool_call"
                    )
                index += 1
                continue
            if role == "tool_call":
                if index + 1 >= len(self.messages) or self.messages[
                    index + 1
                ].get("role") != "tool_response":
                    raise ValueError(
                        "every tool_call must be followed by tool_response"
                    )
                index += 2
                continue
            if role == "tool_response":
                raise ValueError("tool_response must follow tool_call")
            raise ValueError(
                f"messages[{index}] has an invalid turn role {role!r}"
            )
        return self


class ActionOnlyTrajectoryExample(TrajectorySFTExample):
    """A complete executable episode with tool actions but no thought targets.

    This is intentionally a separate artifact from reasoning SFT.  It may be
    used for action imitation or RL initialization, never silently mixed into
    the `<think> + tool_call` reasoning corpus.
    """

    trajectory_version: Literal["ifv-trajectory-action-only-v1"] = (
        "ifv-trajectory-action-only-v1"
    )


class DatasetTrajectorySFTExample(TrajectorySFTExample):
    dataset_version: Literal["ifv-trajectory-sft-dataset-v3"] = (
        "ifv-trajectory-sft-dataset-v3"
    )
    split: Literal["train", "validation", "test"]
    split_group_id: str = Field(min_length=1, max_length=100)
    source_family_keys: List[str] = Field(default_factory=list, max_length=100)
    teacher_score: float = 0.0
