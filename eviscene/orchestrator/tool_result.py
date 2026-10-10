"""Strict result contract shared by every active tool execution path."""
from __future__ import annotations

import json
from typing import Any, Dict, Tuple


SUCCESS_STATUS = "success"
ERROR_STATUS = "error"


class ToolResultContractError(RuntimeError):
    """Raised when a tool does not return the required JSON object contract."""


def validate_tool_result(result: Any) -> Tuple[Dict[str, Any], bool]:
    """Validate a tool result and report whether it represents success."""
    if not isinstance(result, dict):
        raise ToolResultContractError(
            f"tool returned {type(result).__name__}; expected an object with status='success' or status='error'"
        )

    status = str(result.get("status", "")).strip().lower()
    if status not in {SUCCESS_STATUS, ERROR_STATUS}:
        raise ToolResultContractError(
            "tool result is missing a valid status; expected exactly 'success' or 'error'"
        )
    if status == ERROR_STATUS and not str(result.get("error", "")).strip():
        raise ToolResultContractError("tool error result is missing a non-empty error message")
    return result, status == SUCCESS_STATUS


def parse_tool_result(serialized: str) -> Tuple[Dict[str, Any], bool]:
    """Parse and validate a serialized tool result."""
    try:
        parsed = json.loads(str(serialized))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ToolResultContractError("tool result is not a valid JSON object") from exc
    return validate_tool_result(parsed)


def serialize_tool_result(result: Any) -> Tuple[str, bool]:
    """Validate and serialize a tool result without inventing success semantics."""
    parsed, succeeded = validate_tool_result(result)
    return json.dumps(parsed, ensure_ascii=False, indent=2), succeeded
