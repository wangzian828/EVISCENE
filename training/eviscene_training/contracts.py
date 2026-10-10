from __future__ import annotations

from typing import Any, Mapping


FORBIDDEN_MODEL_VISIBLE_KEYS = frozenset(
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

def private_paths(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            if key.casefold() in FORBIDDEN_MODEL_VISIBLE_KEYS:
                found.append(child_path)
            found.extend(private_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(private_paths(child, f"{path}[{index}]"))
    return found


def assert_model_visible(value: Any, *, location: str) -> None:
    leaks = private_paths(value)
    if leaks:
        raise ValueError(
            f"evaluator-private fields are forbidden at {location}: "
            + ", ".join(leaks)
        )
