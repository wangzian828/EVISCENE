"""Semantic route canonicalization shared by runtime and trajectory scoring."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from eviscene.orchestrator.source_provenance import canonicalize_url


_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "where",
    "which",
    "who",
    "with",
}


def route_signature(tool_name: str, tool_args: Mapping[str, Any]) -> dict[str, Any]:
    tool = str(tool_name or "").strip()
    args = dict(tool_args or {})
    task_id = str(
        args.get("__question_id")
        or args.get("question_id")
        or args.get("task_id")
        or ""
    ).strip()
    signature: dict[str, Any] = {"tool": tool, "task_id": task_id}
    if tool == "text_search":
        queries = args.get("queries", [])
        if isinstance(queries, str):
            queries = [queries]
        signature["queries"] = sorted(
            {
                " ".join(_semantic_tokens(str(query)))
                for query in queries or []
                if _semantic_tokens(str(query))
            }
        )
        signature["goal"] = " ".join(
            _semantic_tokens(
                str(args.get("retrieval_goal") or args.get("goal", ""))
            )
        )
    elif tool == "visit":
        urls = args.get("url", [])
        if isinstance(urls, str):
            urls = [urls]
        signature["urls"] = sorted(
            {
                canonicalize_url(str(url))
                for url in urls or []
                if canonicalize_url(str(url))
            }
        )
        signature["claim_id"] = str(
            args.get("__claim_id") or args.get("claim_id", "")
        ).strip()
        signature["goal"] = " ".join(
            _semantic_tokens(
                str(args.get("retrieval_goal") or args.get("goal", ""))
            )
        )
    elif tool == "compare_with_reference":
        signature["reference_url"] = canonicalize_url(
            str(args.get("reference_url", ""))
        )
    elif tool in {
        "crop_and_search",
        "crop_and_inspect",
        "focused_visual_inspection",
        "count_objects",
        "ocr_with_position",
    }:
        if tool == "crop_and_search":
            signature["image_target"] = _image_target_identity(args)
        if tool == "focused_visual_inspection":
            signature["scope"] = str(args.get("scope", "")).strip().lower()
            signature["regions"] = [
                _normalized_bbox(item)
                for item in args.get("anchor_regions", []) or []
            ]
        else:
            signature["bbox"] = _normalized_bbox(args.get("bbox"))
        signature["focus"] = " ".join(
            _semantic_tokens(
                str(
                    args.get("goal")
                    or args.get("focus_question")
                    or args.get("question")
                    or args.get("expected_property")
                    or args.get("target_object")
                    or ""
                )
            )
        )
    elif tool == "reverse_image_search":
        signature["image_input"] = str(args.get("image_input", "")).strip()
        signature["image_target"] = _image_target_identity(args)
        signature["branch"] = str(args.get("branch", "lens")).strip().lower()
    elif tool in {"check_consistency", "analyze_visual_anomalies"}:
        signature["focus"] = " ".join(
            _semantic_tokens(
                str(args.get("focus") or args.get("question") or "")
            )
        )
    else:
        signature["args"] = {
            key: value
            for key, value in sorted(args.items())
            if key not in {
                "image_input",
                "__claim_text",
                "__evidence_goal",
                "question_id",
            }
            and not key.startswith("__")
        }
    return signature


def routes_semantically_equivalent(
    left_tool: str,
    left_args: Mapping[str, Any],
    right_tool: str,
    right_args: Mapping[str, Any],
) -> bool:
    left = route_signature(left_tool, left_args)
    right = route_signature(right_tool, right_args)
    if left["tool"] != right["tool"]:
        return False
    if (
        left["tool"]
        not in {
            "compare_with_reference",
            "current_time",
            "text_search",
            "reverse_image_search",
        }
        and
        left.get("task_id")
        and right.get("task_id")
        and left["task_id"] != right["task_id"]
    ):
        return False
    tool = left["tool"]
    if tool == "text_search":
        queries_match = _query_sets_equivalent(
            left.get("queries", []),
            right.get("queries", []),
        )
        left_goal = str(left.get("goal", ""))
        right_goal = str(right.get("goal", ""))
        goals_match = (
            not left_goal
            or not right_goal
            or _text_similarity(left_goal, right_goal) >= 0.5
        )
        return queries_match and goals_match
    if tool == "visit":
        left_urls = set(left.get("urls", []))
        right_urls = set(right.get("urls", []))
        if not left_urls or not right_urls or left_urls != right_urls:
            return False
        left_task = str(left.get("task_id", ""))
        right_task = str(right.get("task_id", ""))
        if left_task and right_task and left_task != right_task:
            return _text_similarity(
                str(left.get("goal", "")),
                str(right.get("goal", "")),
            ) >= 0.75
        return True
    if tool == "compare_with_reference":
        return bool(left.get("reference_url")) and (
            left.get("reference_url") == right.get("reference_url")
        )
    if tool in {
        "crop_and_search",
        "crop_and_inspect",
        "focused_visual_inspection",
        "count_objects",
        "ocr_with_position",
    }:
        if tool == "focused_visual_inspection":
            left_regions = left.get("regions", [])
            right_regions = right.get("regions", [])
            regions_match = (
                left_regions == right_regions
                or (
                    not left_regions
                    and not right_regions
                )
            )
            return (
                left.get("scope") == right.get("scope")
                and regions_match
                and _text_similarity(
                    str(left.get("focus", "")),
                    str(right.get("focus", "")),
                )
                >= 0.65
            )
        return (
            _bbox_overlap(left.get("bbox"), right.get("bbox")) >= 0.9
            and _text_similarity(
                str(left.get("focus", "")),
                str(right.get("focus", "")),
            )
            >= 0.75
            and (
                tool != "crop_and_search"
                or (
                    left.get("image_target")
                    and left.get("image_target") == right.get("image_target")
                )
            )
        )
    if tool == "reverse_image_search":
        return bool(left.get("image_target")) and (
            left.get("image_target") == right.get("image_target")
            and left.get("branch", "lens") == right.get("branch", "lens")
        )
    if tool in {
        "current_time",
        "check_consistency",
        "analyze_visual_anomalies",
    }:
        return (
            _text_similarity(
                str(left.get("focus", "")),
                str(right.get("focus", "")),
            )
            >= 0.75
        )
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(
        right,
        sort_keys=True,
        default=str,
    )


def semantic_duplicate_count(
    routes: Sequence[tuple[str, Mapping[str, Any]]],
) -> int:
    accepted: list[tuple[str, Mapping[str, Any]]] = []
    duplicates = 0
    for tool_name, tool_args in routes:
        if any(
            routes_semantically_equivalent(
                prior_tool,
                prior_args,
                tool_name,
                tool_args,
            )
            for prior_tool, prior_args in accepted
        ):
            duplicates += 1
        else:
            accepted.append((tool_name, tool_args))
    return duplicates


def _semantic_tokens(value: str) -> tuple[str, ...]:
    tokens = re.findall(r"[\w]+", str(value or "").casefold(), flags=re.UNICODE)
    normalized = {
        _light_stem(token)
        for token in tokens
        if len(token) > 1 and token not in _STOPWORDS
    }
    return tuple(sorted(normalized))


def _light_stem(token: str) -> str:
    for suffix in ("ing", "ers", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]
    return token


def _text_similarity(left: str, right: str) -> float:
    left_tokens = set(_semantic_tokens(left))
    right_tokens = set(_semantic_tokens(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _query_sets_equivalent(left: Sequence[str], right: Sequence[str]) -> bool:
    if not left or not right:
        return False
    return all(
        max(_text_similarity(query, candidate) for candidate in right) >= 0.8
        for query in left
    ) and all(
        max(_text_similarity(query, candidate) for candidate in left) >= 0.8
        for query in right
    )


def _normalized_bbox(value: Any) -> tuple[float, float, float, float] | tuple[()]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return ()
    try:
        return tuple(round(float(item), 3) for item in value)
    except (TypeError, ValueError):
        return ()


def _bbox_overlap(left: Any, right: Any) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0,
        min(ly2, ry2) - max(ly1, ry1),
    )
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _image_target_identity(tool_args: Mapping[str, Any]) -> str:
    """Return the concrete image target, independent of task or branch."""

    args = dict(tool_args or {})
    return str(
        args.get("image_target_id")
        or args.get("image_input")
        or ""
    ).strip()
