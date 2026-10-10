"""Private runtime accounting for Gemini calls made inside tools."""
from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping


RUNTIME_METRICS_KEY = "__runtime_metrics__"
EXCEPTION_METRICS_ATTR = "_gemini_runtime_metrics"


def require_low_thinking(value: str, *, env_name: str) -> str:
    """Require the low-thinking setting used by active tool calls."""

    level = str(value or "").strip().lower()
    if level != "low":
        raise ValueError(f"{env_name} must be 'low' for the active agent.")
    return level


def interaction_runtime_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    usage = payload.get("usage")
    usage = dict(usage) if isinstance(usage, Mapping) else {}
    return {
        "llm_api_calls": 1,
        "tokens": {
            "prompt": int(
                usage.get("total_input_tokens", usage.get("input_tokens", 0)) or 0
            ),
            "completion": int(
                usage.get("total_output_tokens", usage.get("output_tokens", 0)) or 0
            ),
            "thought": int(
                usage.get("total_thought_tokens", usage.get("thought_tokens", 0)) or 0
            ),
        },
    }


def attach_runtime_metrics(exc: Exception, metrics: Mapping[str, Any] | None) -> Exception:
    if isinstance(metrics, Mapping):
        setattr(exc, EXCEPTION_METRICS_ATTR, dict(metrics))
    return exc


def exception_runtime_metrics(exc: BaseException) -> Dict[str, Any]:
    metrics = getattr(exc, EXCEPTION_METRICS_ATTR, {})
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def add_runtime_metrics(
    target: MutableMapping[str, Any],
    metrics: Mapping[str, Any] | None,
) -> None:
    if not isinstance(metrics, Mapping):
        return
    existing = target.get(RUNTIME_METRICS_KEY)
    if not isinstance(existing, Mapping):
        existing = {}
    existing_tokens = existing.get("tokens")
    existing_tokens = dict(existing_tokens) if isinstance(existing_tokens, Mapping) else {}
    incoming_tokens = metrics.get("tokens")
    incoming_tokens = dict(incoming_tokens) if isinstance(incoming_tokens, Mapping) else {}
    target[RUNTIME_METRICS_KEY] = {
        "llm_api_calls": int(existing.get("llm_api_calls", 0) or 0)
        + int(metrics.get("llm_api_calls", 0) or 0),
        "tokens": {
            name: int(existing_tokens.get(name, 0) or 0)
            + int(incoming_tokens.get(name, 0) or 0)
            for name in ("prompt", "completion", "thought")
        },
    }


def take_runtime_metrics(value: Any) -> Dict[str, Any]:
    if isinstance(value, MutableMapping):
        direct = value.pop(RUNTIME_METRICS_KEY, {})
        child_total: Dict[str, Any] = {}
        for child in value.values():
            add_runtime_metrics(child_total, take_runtime_metrics(child))
        if isinstance(direct, Mapping) and (
            int(direct.get("llm_api_calls", 0) or 0)
            or isinstance(direct.get("tokens"), Mapping)
        ):
            return dict(direct)
        return dict(child_total.get(RUNTIME_METRICS_KEY, {}))
    if isinstance(value, list):
        total: Dict[str, Any] = {}
        for child in value:
            add_runtime_metrics(total, take_runtime_metrics(child))
        return dict(total.get(RUNTIME_METRICS_KEY, {}))
    return {}
