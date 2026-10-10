from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .contracts import assert_model_visible
from .io import load_json, load_jsonl, sha256_file
from .policy import _validate_qwen_agent_messages
from .policy_contract import audit_policy_contract_rows


GEMINI_WIRE_PROTOCOL_MARKER = "Native Gemini Interactions protocol:"


def _valid_image_reference(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    if value.startswith("data:image/") and "," in value:
        return True
    return Path(value).is_file()


def _audit_message(message: Any, location: str) -> None:
    if not isinstance(message, Mapping):
        raise ValueError(f"{location} must be an object")
    role = str(message.get("role", ""))
    if role not in {
        "system",
        "user",
        "assistant",
        "tool_call",
        "tool_response",
    }:
        raise ValueError(f"{location} has unsupported role {role!r}")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{location}.content must be a non-empty string")
    allowed_keys = {"role", "content", "loss", "loss_scale"}
    unexpected = set(message) - allowed_keys
    if unexpected:
        raise ValueError(f"{location} has unsupported keys: {sorted(unexpected)}")
    if "loss" in message and not isinstance(message["loss"], bool):
        raise ValueError(f"{location}.loss must be boolean")


def audit_derived_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest = load_json(dataset_dir / "manifest.json")
    errors: list[str] = []
    roles: Counter[str] = Counter()
    rows_seen = 0
    contract_rows: list[Mapping[str, Any]] = []
    primary_artifacts = {"train", "validation", "test"}
    for artifact_name, artifact in manifest.get("artifacts", {}).items():
        if artifact_name not in primary_artifacts:
            continue
        path = dataset_dir / str(artifact.get("path", ""))
        if not path.is_file():
            errors.append(f"missing artifact: {path}")
            continue
        if sha256_file(path) != artifact.get("sha256"):
            errors.append(f"sha256 mismatch: {path.name}")
        rows = load_jsonl(path)
        if len(rows) != artifact.get("rows"):
            errors.append(f"row count mismatch: {path.name}")
        for row_index, row in enumerate(rows):
            contract_rows.append(row)
            location = f"{path.name}[{row_index}]"
            try:
                assert_model_visible(row, location=location)
                messages = row.get("messages")
                if not isinstance(messages, list) or len(messages) < 2:
                    raise ValueError(f"{location}.messages is invalid")
                _validate_qwen_agent_messages(
                    messages,
                    image_count=len(row.get("images") or []),
                )
                for message_index, message in enumerate(messages):
                    _audit_message(message, f"{location}.messages[{message_index}]")
                    roles[str(message["role"])] += 1
                images = row.get("images")
                if images is not None:
                    if not isinstance(images, list):
                        raise ValueError(f"{location}.images must be a list")
                    if images:
                        if messages[1].get("role") != "user" or "<image>" not in str(
                            messages[1].get("content", "")
                        ):
                            raise ValueError(
                                f"{location} image row lacks <image> placeholder"
                            )
                        for image in images:
                            if not _valid_image_reference(image):
                                raise FileNotFoundError(
                                    f"{location} image is unavailable: {image}"
                                )
                if GEMINI_WIRE_PROTOCOL_MARKER in json.dumps(
                    row,
                    ensure_ascii=False,
                ):
                    raise ValueError(
                        f"{location} contains provider wire instructions"
                    )
            except Exception as exc:
                errors.append(str(exc))
            rows_seen += 1
    is_policy_dataset = str(manifest.get("dataset_version", "")).startswith(
        "ifv-ms-swift-qwen-agent-"
    )
    if is_policy_dataset:
        causal_contract = audit_policy_contract_rows(contract_rows)
        for blocker, count in causal_contract["production_blockers"].items():
            if count:
                errors.append(f"policy causal contract blocker: {blocker}={count}")
    else:
        causal_contract = {
            "schema_version": "ifv-policy-causal-contract-audit-v2",
            "passed": True,
            "applicable": False,
            "reason": "dataset is not a Qwen Agent policy dataset",
            "production_blockers": {},
        }
    return {
        "schema_version": "ifv-ms-swift-dataset-audit-v2",
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "row_count": rows_seen,
        "role_counts": dict(sorted(roles.items())),
        "causal_contract": causal_contract,
    }
