"""Strictly audit current raw-history image-only traces."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eviscene.orchestrator.source_access import (  # noqa: E402
    SourceAccessPolicy,
    url_variants,
)
from eviscene.orchestrator.source_provenance import domain_matches  # noqa: E402
from eviscene.orchestrator.react_runtime import (  # noqa: E402
    REACT_RUNTIME_SCHEMA_VERSION,
    REACT_RUNTIME_TOOLS,
    is_unified_react_runtime_budget_action,
)
from eviscene.orchestrator.tool_result import parse_tool_result  # noqa: E402


HARD = "hard"
SCHEDULER = "scheduler"
PROTOCOL = "protocol"
ROUTE_CONTROL = "route_control"
CORRECTION = "correction"
BOUNDED_FALLBACK = "bounded_fallback"
FORMAT = "format"
REJECTION_ACTIONS = frozenset({"format_error", "output_rejected", "policy_replan"})
UNIFIED_REACT_POLICY_VERSION = "unified-react-v1"
@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    category: str = HARD
    location: str = ""


@dataclass
class TraceReport:
    path: str
    image_id: str = ""
    termination: str = ""
    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    def failures(self, *, strict_scheduler: bool) -> list[Issue]:
        return [
            issue
            for issue in self.issues
            if issue.category == HARD
            or (strict_scheduler and issue.category in {SCHEDULER, PROTOCOL})
        ]

    def warnings(self, *, strict_scheduler: bool) -> list[Issue]:
        failed = set(self.failures(strict_scheduler=strict_scheduler))
        return [issue for issue in self.issues if issue not in failed]

    def to_dict(self, *, strict_scheduler: bool) -> dict[str, Any]:
        failures = self.failures(strict_scheduler=strict_scheduler)
        warnings = self.warnings(strict_scheduler=strict_scheduler)
        return {
            "path": self.path,
            "image_id": self.image_id,
            "termination": self.termination,
            "passed": not failures,
            "failure_count": len(failures),
            "warning_count": len(warnings),
            "stats": self.stats,
            "issues": [
                {
                    **asdict(issue),
                    "severity": "error" if issue in failures else "warning",
                }
                for issue in self.issues
            ],
        }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _state(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    state = trace.get("state")
    return state if isinstance(state, Mapping) else trace


def _issue(
    report: TraceReport,
    code: str,
    message: str,
    *,
    category: str = HARD,
    location: str = "",
) -> None:
    report.issues.append(
        Issue(code=code, message=message, category=category, location=location)
    )


def _step_label(index: int, step: Mapping[str, Any]) -> str:
    call_id = str(_mapping(step.get("metadata")).get("function_call_id", "")).strip()
    suffix = f",call={call_id}" if call_id else ""
    return f"state.all_steps[{index}]({step.get('stage', '')}:{step.get('round', '')}{suffix})"


def _audit_termination(
    trace: Mapping[str, Any], state: Mapping[str, Any], report: TraceReport
) -> None:
    outer = str(trace.get("termination", "")).strip()
    inner = str(state.get("termination", "")).strip()
    report.termination = outer or inner
    if outer != "success" or inner != "success":
        _issue(
            report,
            "TERMINATION_NOT_SUCCESS",
            f"termination must be success at trace and state levels (trace={outer!r}, state={inner!r})",
            location="termination",
        )


def _numeric_token(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return numeric if math.isfinite(numeric) else None


def _audit_thought_tokens(
    trace: Mapping[str, Any], state: Mapping[str, Any], steps: Sequence[Mapping[str, Any]], report: TraceReport
) -> None:
    locations: list[tuple[str, Any, bool]] = []
    for prefix, usage in (
        ("token_usage", trace.get("token_usage")),
        ("state.token_usage", state.get("token_usage")),
    ):
        row = _mapping(usage)
        if "thought" in row:
            locations.append((f"{prefix}.thought", row.get("thought"), True))
        if "thought_tokens" in row:
            locations.append(
                (f"{prefix}.thought_tokens", row.get("thought_tokens"), True)
            )
        if "total_thought_tokens" in row:
            locations.append(
                (
                    f"{prefix}.total_thought_tokens",
                    row.get("total_thought_tokens"),
                    True,
                )
            )

    native_steps = 0
    for index, step in enumerate(steps):
        metadata = _mapping(step.get("metadata"))
        tokens = _mapping(step.get("tokens"))
        if metadata.get("native_interactions"):
            native_steps += 1
            if not any(
                key in tokens
                for key in ("thought", "thought_tokens", "total_thought_tokens")
            ):
                _issue(
                    report,
                    "THOUGHT_TOKENS_UNRECORDED",
                    "Gemini Interactions step does not record a thought-token field",
                    location=_step_label(index, step),
                )
        for key in ("thought", "thought_tokens", "total_thought_tokens"):
            if key in tokens:
                locations.append(
                    (
                        f"state.all_steps[{index}].tokens.{key}",
                        tokens[key],
                        True,
                    )
                )

    recorded_paths = {path for path, _, _ in locations}
    for key, raw_value, path in _iter_named_values(trace):
        token_key = key.casefold()
        parent_segments = {
            segment.casefold()
            for segment in re.split(r"[.\[\]<>]+", path.rsplit(".", 1)[0])
            if segment
        }
        is_token_field = token_key in {"thought_tokens", "total_thought_tokens"}
        is_usage_total = (
            token_key == "thought"
            and bool(parent_segments & {"usage", "tokens", "token_usage"})
        )
        if (is_token_field or is_usage_total) and path not in recorded_paths:
            locations.append((path, raw_value, True))
            recorded_paths.add(path)

    report.stats["native_interaction_steps"] = native_steps
    report.stats["thought_token_fields"] = len(locations)
    for path, value, nonzero_allowed in locations:
        numeric = _numeric_token(value)
        if numeric is None:
            _issue(
                report,
                "THOUGHT_TOKENS_INVALID",
                f"thought-token value must be numeric, got {value!r}",
                location=path,
            )



def _fact_check_domain(
    value: str,
    *,
    source_access_policy: SourceAccessPolicy | None = None,
) -> str:
    if source_access_policy is not None:
        for variant in url_variants(value):
            hostname = (urlsplit(variant).hostname or "").lower().rstrip(".")
            for domain in sorted(
                source_access_policy.excluded_domains,
                key=len,
                reverse=True,
            ):
                if domain_matches(hostname, domain):
                    return domain
        return ""
    return ""


def _fact_check_query_domain(
    value: str,
    *,
    source_access_policy: SourceAccessPolicy | None = None,
) -> str:
    text = unquote(str(value or ""))
    candidates = re.findall(
        r"(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s\"'<>]*)?",
        text,
        flags=re.IGNORECASE,
    )
    for candidate in candidates:
        if domain := _fact_check_domain(
            candidate.rstrip(".,;:!?)]}"),
            source_access_policy=source_access_policy,
        ):
            return domain
    return ""


def _fact_check_embedded_url(
    value: str,
    *,
    source_access_policy: SourceAccessPolicy | None = None,
) -> str:
    urls = re.findall(
        r"https?://[^\s\"'<>]+",
        str(value or ""),
        flags=re.IGNORECASE,
    )
    for url in urls:
        if domain := _fact_check_domain(
            url.rstrip(".,;:!?)]}"),
            source_access_policy=source_access_policy,
        ):
            return domain
    return ""


def _fact_check_url_query(
    value: str,
    *,
    source_access_policy: SourceAccessPolicy | None = None,
) -> str:
    for variant in url_variants(value):
        try:
            query = " ".join(
                f"{key} {item}" for key, item in parse_qsl(urlsplit(variant).query)
            )
        except ValueError:
            continue
        if domain := _fact_check_query_reference(
            query,
            source_access_policy=source_access_policy,
        ):
            return f"known fact-check domain {domain!r} in URL query"
    return ""


def _fact_check_query_reference(
    value: str,
    *,
    source_access_policy: SourceAccessPolicy | None = None,
) -> str:
    if source_access_policy is None:
        return ""
    return (
        source_access_policy.blocked_query_reference(value)
        or _fact_check_query_domain(
            value,
            source_access_policy=source_access_policy,
        )
    )


def _iter_named_values(
    value: Any,
    path: str = "",
    inherited_key: str = "",
) -> Iterable[tuple[str, str, str]]:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            if isinstance(child, (str, int, float)) and not isinstance(child, bool):
                if key == "tool_result" and isinstance(child, str):
                    try:
                        parsed = json.loads(child)
                    except (TypeError, json.JSONDecodeError):
                        parsed = None
                    if isinstance(parsed, (Mapping, list)):
                        yield from _iter_named_values(
                            parsed,
                            f"{child_path}<json>",
                        )
                        continue
                yield key, str(child), child_path
            else:
                yield from _iter_named_values(child, child_path, key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            if isinstance(child, (str, int, float)) and not isinstance(child, bool):
                yield inherited_key, str(child), child_path
            else:
                yield from _iter_named_values(child, child_path, inherited_key)


def _looks_like_url_key(key: str) -> bool:
    lowered = key.casefold()
    return lowered in {
        "source",
        "hostname",
        "registered_domain",
        "domain",
        "link",
        "href",
        "reference_image_candidates",
    } or "url" in lowered


def _looks_like_query_key(key: str) -> bool:
    lowered = key.casefold()
    return lowered in {"query", "queries", "suggested_query", "suggested_queries", "crop_query", "vlm_query"}


def _policy_rejected_step_for_path(
    trace: Mapping[str, Any],
    path: str,
) -> Mapping[str, Any] | None:
    """Return the step when a forbidden proposal was blocked before execution."""

    match = re.search(r"(?:^|\.)all_steps\[(\d+)\](?:\.|$)", path)
    if match is None:
        return None
    steps = _rows(_state(trace).get("all_steps"))
    index = int(match.group(1))
    if index >= len(steps):
        return None
    step = steps[index]
    metadata = _mapping(step.get("metadata"))
    if (
        str(step.get("action_type", ""))
        in REJECTION_ACTIONS
        or metadata.get("search_policy_rejection") is True
        or int(metadata.get("policy_filtered_query_count", 0) or 0) > 0
    ):
        return step
    return None


def _audit_leaks(
    trace: Mapping[str, Any],
    report: TraceReport,
    *,
    enforce_source_access_policy: bool,
    source_access_policy: SourceAccessPolicy | None = None,
) -> None:
    seen: set[tuple[str, str]] = set()
    url_count = 0
    query_count = 0
    rejected_query_count = 0
    for key, raw_value, path in _iter_named_values(trace):
        query_key = _looks_like_query_key(key)
        if enforce_source_access_policy and _looks_like_url_key(key):
            domain = _fact_check_domain(
                raw_value,
                source_access_policy=source_access_policy,
            ) or _fact_check_query_domain(
                raw_value,
                source_access_policy=source_access_policy,
            )
        elif enforce_source_access_policy and not query_key:
            domain = _fact_check_embedded_url(
                raw_value,
                source_access_policy=source_access_policy,
            )
        else:
            domain = ""
        if domain:
            signature = ("url", path)
            if signature not in seen:
                seen.add(signature)
                url_count += 1
                _issue(
                    report,
                    "FACT_CHECK_URL_LEAK",
                    f"known fact-check domain {domain!r} appears in URL-bearing trace data",
                    location=path,
                )
        if enforce_source_access_policy and _looks_like_url_key(key) and (
            url_query_leak := _fact_check_url_query(
                raw_value,
                source_access_policy=source_access_policy,
            )
        ):
            signature = ("query", path)
            if signature not in seen:
                seen.add(signature)
                if _policy_rejected_step_for_path(trace, path) is not None:
                    rejected_query_count += 1
                    _issue(
                        report,
                        "FACT_CHECK_QUERY_REJECTED",
                        f"{url_query_leak} was rejected before execution: {raw_value!r}",
                        category=CORRECTION,
                        location=path,
                    )
                else:
                    query_count += 1
                    _issue(
                        report,
                        "FACT_CHECK_QUERY_LEAK",
                        f"{url_query_leak} leaked into the trace: {raw_value!r}",
                        location=path,
                    )
        if query_key:
            query_domain = (
                _fact_check_query_reference(
                    raw_value,
                    source_access_policy=source_access_policy,
                )
                if enforce_source_access_policy
                else ""
            )
            if query_domain:
                signature = ("query", path)
                if signature not in seen:
                    seen.add(signature)
                    detail = f"excluded source {query_domain!r} is named in query"
                    if _policy_rejected_step_for_path(trace, path) is not None:
                        rejected_query_count += 1
                        _issue(
                            report,
                            "FACT_CHECK_QUERY_REJECTED",
                            f"{detail} was rejected before execution: {raw_value!r}",
                            category=CORRECTION,
                            location=path,
                        )
                    else:
                        query_count += 1
                        _issue(
                            report,
                            "FACT_CHECK_QUERY_LEAK",
                            f"{detail} leaked into the trace: {raw_value!r}",
                            location=path,
                        )
    report.stats["fact_check_url_leaks"] = url_count
    report.stats["fact_check_query_leaks"] = query_count
    report.stats["fact_check_query_rejections"] = rejected_query_count




def _rejection_category(step: Mapping[str, Any]) -> str:
    metadata = _mapping(step.get("metadata"))
    if metadata.get("search_query_format_error"):
        return FORMAT
    if metadata.get("search_policy_rejection"):
        return CORRECTION
    if _empty_text_search_query(step):
        return FORMAT
    if metadata.get("unbalanced_priority_coverage") or metadata.get("tool_budget_reached"):
        return SCHEDULER
    reason = str(metadata.get("rejection_reason", "")).casefold()
    if not reason:
        try:
            reason = str(
                _mapping(json.loads(str(step.get("tool_result", "")))).get("error", "")
            ).casefold()
        except (TypeError, json.JSONDecodeError):
            reason = ""
    if "source policy" in reason or "excluded source reference" in reason:
        return CORRECTION
    scheduler_markers = (
        "coverage requires",
        "required investigation questions",
        "tool budget",
        "budget ended",
        "before resampling",
        "untouched p1",
        "untouched p2",
        "at least 1 tool call",
    )
    if any(marker in reason for marker in scheduler_markers):
        return SCHEDULER
    return PROTOCOL


def _empty_text_search_query(step: Mapping[str, Any]) -> bool:
    if str(step.get("tool_name", "")).strip() != "text_search":
        return False
    tool_args = _mapping(step.get("tool_args"))
    queries = tool_args.get("queries", tool_args.get("query", []))
    if isinstance(queries, str):
        return not queries.strip()
    if not isinstance(queries, list):
        return True
    return not any(str(query).strip() for query in queries)


def _audit_rejections(
    steps: Sequence[Mapping[str, Any]], report: TraceReport
) -> None:
    scheduler_count = 0
    protocol_count = 0
    route_control_count = 0
    corrected_count = 0
    bounded_fallback_count = 0
    format_count = 0

    def rejected(candidate: Mapping[str, Any]) -> bool:
        candidate_metadata = _mapping(candidate.get("metadata"))
        return (
            str(candidate.get("action_type", "")) in REJECTION_ACTIONS
            or bool(candidate_metadata.get("rejection_reason"))
        )

    def attempt_id(candidate: Mapping[str, Any]) -> str:
        candidate_metadata = _mapping(candidate.get("metadata"))
        if candidate_metadata.get("native_chat_completions"):
            return str(
                candidate_metadata.get("context_request_id", "")
                or candidate_metadata.get("interaction_id", "")
            ).strip()
        return str(
            candidate_metadata.get("interaction_id", "")
            or candidate_metadata.get("context_request_id", "")
        ).strip()

    def correction_parent_id(candidate: Mapping[str, Any]) -> str:
        candidate_metadata = _mapping(candidate.get("metadata"))
        if candidate_metadata.get("native_chat_completions"):
            return str(
                candidate_metadata.get("parent_context_request_id", "")
                or candidate_metadata.get("previous_interaction_id", "")
            ).strip()
        return str(
            candidate_metadata.get("previous_interaction_id", "")
            or candidate_metadata.get("parent_context_request_id", "")
        ).strip()

    def eventually_corrected(
        rejected_index: int,
        rejected_step: Mapping[str, Any],
    ) -> bool:
        root_id = attempt_id(rejected_step)
        if not root_id:
            return False
        reachable_rejected_ids = {root_id}
        stage = str(rejected_step.get("stage", ""))
        for candidate in steps[rejected_index + 1 :]:
            if str(candidate.get("stage", "")) != stage:
                continue
            candidate_metadata = _mapping(candidate.get("metadata"))
            lifecycle = str(
                candidate_metadata.get("interaction_lifecycle_kind", "")
            ).strip()
            is_unified_outer_correction = (
                stage == "unified_react"
                and lifecycle == "tool_roundtrip"
                and str(candidate.get("action_type", "")) == "tool_call"
                and bool(
                    _mapping(
                        candidate_metadata.get("unified_react_delta")
                    ).get("state_update", {})
                )
            )
            if lifecycle != "protocol_correction" and not is_unified_outer_correction:
                continue
            if correction_parent_id(candidate) not in reachable_rejected_ids:
                continue
            if not rejected(candidate):
                return True
            candidate_id = attempt_id(candidate)
            if candidate_id:
                reachable_rejected_ids.add(candidate_id)
        return False

    def eventually_reaches_bounded_fallback(
        rejected_index: int,
        rejected_step: Mapping[str, Any],
    ) -> bool:
        root_id = attempt_id(rejected_step)
        if not root_id:
            return False
        stage = str(rejected_step.get("stage", ""))
        for candidate in steps[rejected_index + 1 :]:
            if str(candidate.get("stage", "")) != stage:
                continue
            candidate_metadata = _mapping(candidate.get("metadata"))
            if not candidate_metadata.get(
                "protocol_correction_exhaustion_boundary"
            ):
                continue
            resolved = {
                str(item).strip()
                for item in candidate_metadata.get(
                    "resolved_rejection_request_ids", []
                )
                or []
                if str(item).strip()
            }
            if root_id in resolved:
                return True
        return False

    for index, step in enumerate(steps):
        metadata = _mapping(step.get("metadata"))
        if not rejected(step):
            continue
        corrected = eventually_corrected(index, step)
        bounded_fallback = eventually_reaches_bounded_fallback(index, step)
        reason = str(metadata.get("rejection_reason", "")).strip()
        if not reason:
            try:
                payload = json.loads(str(step.get("tool_result", "")))
                reason = str(_mapping(payload).get("error", "")).strip()
            except (TypeError, json.JSONDecodeError):
                reason = ""
        route_control = bool(metadata.get("duplicate_tool_call")) or (
            bool(metadata.get("invalid_question_id"))
            and "already resolved" in reason.casefold()
        )
        category = (
            CORRECTION
            if corrected
            else BOUNDED_FALLBACK
            if bounded_fallback
            else ROUTE_CONTROL
            if route_control
            else _rejection_category(step)
        )
        if category == CORRECTION:
            corrected_count += 1
        elif category == BOUNDED_FALLBACK:
            bounded_fallback_count += 1
        elif category == ROUTE_CONTROL:
            route_control_count += 1
        elif category == FORMAT:
            format_count += 1
        elif category == SCHEDULER:
            scheduler_count += 1
        else:
            protocol_count += 1
        _issue(
            report,
            (
                "PROTOCOL_CORRECTION"
                if category == CORRECTION
                else "PROTOCOL_EXHAUSTION_BOUNDARY"
                if category == BOUNDED_FALLBACK
                else "ROUTE_CONTROL_REJECTION"
                if category == ROUTE_CONTROL
                else "TOOL_ARGUMENT_FORMAT_ERROR"
                if category == FORMAT
                else "SCHEDULER_REJECTION"
                if category == SCHEDULER
                else "PROTOCOL_REJECTION"
            ),
            reason or f"{step.get('action_type', 'rejected')} step",
            category=category,
            location=_step_label(index, step),
        )
    report.stats["scheduler_rejections"] = scheduler_count
    report.stats["protocol_rejections"] = protocol_count
    report.stats["route_control_rejections"] = route_control_count
    report.stats["successful_protocol_corrections"] = corrected_count
    report.stats["bounded_protocol_fallbacks"] = bounded_fallback_count
    report.stats["tool_argument_format_errors"] = format_count




def _audit_unified_react_trace(
    trace: Mapping[str, Any],
    state: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    report: TraceReport,
) -> None:
    """Audit the one supported raw-history ReAct contract."""

    if str(
        trace.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    ).strip() != UNIFIED_REACT_POLICY_VERSION:
        _issue(
            report,
            "REACT_RUNTIME_POLICY_MISMATCH",
            "current trace must use decision_policy_version=unified-react-v1",
            location="decision_policy_version",
        )
        return
    investigation = _mapping(state.get("investigation_state"))
    if str(investigation.get("schema_version", "")).strip() != (
        REACT_RUNTIME_SCHEMA_VERSION
    ):
        _issue(
            report,
            "REACT_RUNTIME_SCHEMA_INVALID",
            "current trace must use the raw-history runtime schema",
            location="state.investigation_state.schema_version",
        )
        return
    _audit_current_react_runtime_trace(trace, state, steps, report)

def _audit_current_react_runtime_trace(
    trace: Mapping[str, Any],
    state: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    report: TraceReport,
) -> None:
    """Audit the active raw-history runtime without reducer semantics."""

    investigation = _mapping(state.get("investigation_state"))
    allowed_state_fields = {
        "schema_version",
        "case_id",
        "image_sha256",
        "objective",
        "action_count",
        "stop_reason",
        "finish_rationale",
    }
    unexpected_state_fields = sorted(set(investigation) - allowed_state_fields)
    if unexpected_state_fields:
        _issue(
            report,
            "REACT_RUNTIME_STATE_FIELDS_INVALID",
            "raw-history ReAct state contains non-mechanical fields: "
            + ", ".join(unexpected_state_fields),
            location="state.investigation_state",
        )

    actions = [
        (index, step)
        for index, step in enumerate(steps)
        if str(step.get("stage", "")).strip() == "unified_react"
        and str(step.get("action_type", "")).strip() == "tool_call"
    ]
    if not actions:
        _issue(
            report,
            "REACT_RUNTIME_ACTIONS_MISSING",
            "current ReAct trace requires at least one accepted tool action",
            location="state.all_steps",
        )

    accepted_tool_names = set(REACT_RUNTIME_TOOLS) | {"finish_investigation"}
    call_ids: set[str] = set()
    observation_rows: list[dict[str, str]] = []
    successful_observation_ids: list[str] = []
    successful_results = 0
    error_results = 0
    malformed_results = 0
    native_interactions: list[tuple[int, Mapping[str, Any]]] = []
    for index, step in actions:
        location = _step_label(index, step)
        tool_name = str(step.get("tool_name", "")).strip()
        metadata = _mapping(step.get("metadata"))
        call_id = str(metadata.get("function_call_id", "")).strip()
        if tool_name not in accepted_tool_names:
            _issue(
                report,
                "REACT_RUNTIME_TOOL_UNKNOWN",
                f"unknown current ReAct tool {tool_name!r}",
                location=location,
            )
        if not call_id:
            _issue(
                report,
                "REACT_RUNTIME_CALL_ID_MISSING",
                "accepted current ReAct action lacks function_call_id",
                location=location,
            )
        elif call_id in call_ids:
            _issue(
                report,
                "REACT_RUNTIME_CALL_ID_DUPLICATE",
                f"duplicate current ReAct function_call_id {call_id!r}",
                location=location,
            )
        call_ids.add(call_id)

        policy_action = _mapping(metadata.get("policy_action"))
        if (
            str(policy_action.get("type", "")).strip() != "tool_call"
            or str(policy_action.get("name", "")).strip() != tool_name
        ):
            _issue(
                report,
                "REACT_RUNTIME_POLICY_ACTION_MISMATCH",
                "policy_action must record the same current ReAct tool call",
                location=location,
            )

        try:
            _payload, succeeded = parse_tool_result(str(step.get("tool_result", "")))
            result_status = "success" if succeeded else "error"
        except Exception as exc:
            result_status = "malformed"
            _issue(
                report,
                "REACT_RUNTIME_TOOL_RESULT_INVALID",
                f"cannot parse tool result: {exc}",
                location=location,
            )
        if result_status == "success":
            successful_results += 1
        elif result_status == "error":
            error_results += 1
        else:
            malformed_results += 1
        if tool_name == "finish_investigation" or not call_id:
            continue
        observation_rows.append(
            {
                "observation_id": call_id,
                "function_call_id": call_id,
                "tool_name": tool_name,
                "status": result_status,
            }
        )
        if result_status == "success":
            successful_observation_ids.append(call_id)

    # Include protocol-correction requests as well as accepted tool actions.
    # They are real provider interactions and must participate in the same
    # parent chain; otherwise a correction can make the next accepted action
    # look like a broken root.
    native_interactions = [
        (index, _mapping(step.get("metadata")))
        for index, step in enumerate(steps)
        if str(step.get("stage", "")).strip() == "unified_react"
        and _mapping(step.get("metadata")).get("native_interactions")
    ]

    budget_actions = [
        step
        for _, step in actions
        if is_unified_react_runtime_budget_action(step)
    ]
    action_count = int(investigation.get("action_count", 0) or 0)
    if action_count != len(budget_actions):
        _issue(
            report,
            "REACT_RUNTIME_ACTION_COUNT_MISMATCH",
            (
                f"action_count={action_count}, but trace records "
                f"{len(budget_actions)} budgeted ReAct actions"
            ),
            location="state.investigation_state.action_count",
        )
    if action_count < 0 or action_count > 24:
        _issue(
            report,
            "REACT_RUNTIME_ACTION_BUDGET_INVALID",
            f"current ReAct action_count must be between 0 and 24, got {action_count}",
            location="state.investigation_state.action_count",
        )

    interaction_rows: dict[str, tuple[int, Mapping[str, Any], Mapping[str, Any]]] = {}
    previous_interaction_id: str | None = None
    for index, metadata in native_interactions:
        interaction_id = str(metadata.get("interaction_id", "")).strip()
        parent_id = str(metadata.get("previous_interaction_id", "") or "").strip()
        lifecycle = str(
            metadata.get("interaction_lifecycle_kind", "")
        ).strip()
        location = _step_label(index, steps[index])
        if not interaction_id:
            _issue(
                report,
                "REACT_RUNTIME_INTERACTION_ID_MISSING",
                "native current ReAct action lacks interaction_id",
                location=location,
            )
        if lifecycle not in {"tool_roundtrip", "protocol_correction"}:
            _issue(
                report,
                "REACT_RUNTIME_INTERACTION_LIFECYCLE_INVALID",
                "native current ReAct interaction has an invalid lifecycle",
                location=location,
            )
        if previous_interaction_id is None:
            if parent_id:
                _issue(
                    report,
                    "REACT_RUNTIME_INTERACTION_ROOT_INVALID",
                    (
                        "the first unified ReAct interaction must have an empty "
                        f"parent, got {parent_id!r}"
                    ),
                    location=location,
                )
        elif parent_id != previous_interaction_id:
            _issue(
                report,
                "REACT_RUNTIME_INTERACTION_CHAIN_BROKEN",
                (
                    "expected previous_interaction_id "
                    f"{previous_interaction_id!r}, got {parent_id!r}"
                ),
                location=location,
            )
        previous_step = (
            interaction_rows.get(parent_id, (None, None, None))[2]
            if parent_id
            else None
        )
        if previous_step is not None and str(
            previous_step.get("action_type", "")
        ).strip() == "tool_call":
            previous_call_id = str(
                _mapping(previous_step.get("metadata")).get(
                    "function_call_id",
                    "",
                )
            ).strip()
            policy_input = _mapping(metadata.get("policy_input"))
            input_payload = policy_input.get("input_payload")
            function_result_ids: list[str] = []

            def collect_function_result_ids(value: Any) -> None:
                if isinstance(value, Mapping):
                    if str(value.get("type", "")).strip() == "function_result":
                        call_id = str(value.get("call_id", "")).strip()
                        if call_id:
                            function_result_ids.append(call_id)
                    for child in value.values():
                        collect_function_result_ids(child)
                elif isinstance(value, list):
                    for child in value:
                        collect_function_result_ids(child)

            collect_function_result_ids(input_payload)
            if previous_call_id and previous_call_id not in function_result_ids:
                _issue(
                    report,
                    "REACT_RUNTIME_TOOL_RESULT_NOT_REINJECTED",
                    (
                        "the request following a completed tool action does not "
                        f"contain its function_result call_id {previous_call_id!r}"
                    ),
                    location=location,
                )
        if interaction_id:
            interaction_rows[interaction_id] = (
                index,
                metadata,
                steps[index],
            )
            previous_interaction_id = interaction_id

    judgment_outputs = [
        (index, step)
        for index, step in enumerate(steps)
        if str(step.get("stage", "")).strip() == "unified_judgment"
        and str(step.get("action_type", "")).strip() == "output"
    ]
    if len(judgment_outputs) != 1:
        _issue(
            report,
            "REACT_RUNTIME_JUDGMENT_COUNT_INVALID",
            f"current ReAct trace must contain exactly one final Judgment, got {len(judgment_outputs)}",
            location="state.all_steps",
        )

    basis = _mapping(trace.get("verdict_basis"))
    if str(basis.get("schema_version", "")).strip() != (
        "ifv-raw-history-judgment-basis-v1"
    ):
        _issue(
            report,
            "REACT_RUNTIME_BASIS_INVALID",
            "raw-history ReAct verdict basis has an invalid schema_version",
            location="verdict_basis.schema_version",
        )
    if str(basis.get("decision_mode", "")).strip() != "bounded_binary_judgment":
        _issue(
            report,
            "REACT_RUNTIME_DECISION_MODE_INVALID",
            "current ReAct verdict basis must use bounded_binary_judgment",
            location="verdict_basis.decision_mode",
        )

    basis_observation_ids = [
        str(item).strip()
        for item in basis.get("observation_ids", []) or []
        if str(item).strip()
    ]
    if basis_observation_ids != successful_observation_ids:
        _issue(
            report,
            "REACT_RUNTIME_BASIS_OBSERVATIONS_MISMATCH",
            "verdict basis observation_ids must equal successful raw observations",
            location="verdict_basis.observation_ids",
        )
    basis_observation_rows = [
        {
            "observation_id": str(item.get("observation_id", "")).strip(),
            "function_call_id": str(item.get("function_call_id", "")).strip(),
            "tool_name": str(item.get("tool_name", "")).strip(),
            "status": str(item.get("status", "")).strip(),
        }
        for item in _rows(basis.get("observations"))
    ]
    if basis_observation_rows != observation_rows:
        _issue(
            report,
            "REACT_RUNTIME_OBSERVATION_LOCATOR_MISMATCH",
            "verdict basis observations must match raw tool calls in order",
            location="verdict_basis.observations",
        )
    if int(basis.get("action_count", 0) or 0) != action_count:
        _issue(
            report,
            "REACT_RUNTIME_BASIS_ACTION_COUNT_MISMATCH",
            "verdict basis action_count must equal investigation action_count",
            location="verdict_basis.action_count",
        )

    judgment = _mapping(trace.get("judgment") or state.get("judgment"))
    verdict = str(trace.get("verdict", "")).strip()
    if verdict not in {"real", "fake"}:
        _issue(
            report,
            "REACT_RUNTIME_VERDICT_INVALID",
            f"current ReAct verdict must be real or fake, got {verdict!r}",
            location="verdict",
        )
    if str(judgment.get("policy_rule_id", "")).strip() != (
        UNIFIED_REACT_POLICY_VERSION
    ):
        _issue(
            report,
            "REACT_RUNTIME_JUDGMENT_POLICY_INVALID",
            "current ReAct Judgment must use unified-react-v1",
            location="judgment.policy_rule_id",
        )
    if str(judgment.get("verdict", "")).strip() != verdict:
        _issue(
            report,
            "REACT_RUNTIME_JUDGMENT_VERDICT_MISMATCH",
            "trace verdict and current ReAct Judgment verdict must match",
            location="judgment.verdict",
        )
    judgment_selected_observations = [
        str(item).strip()
        for item in judgment.get("selected_observation_ids", []) or []
        if str(item).strip()
    ]
    if judgment_selected_observations != basis_observation_ids:
        _issue(
            report,
            "REACT_RUNTIME_JUDGMENT_BASIS_MISMATCH",
            (
                "Judgment selected_observation_ids must equal "
                "verdict_basis.observation_ids"
            ),
            location="judgment.selected_observation_ids",
        )
    verdict_observation_ids = [
        str(item).strip()
        for item in judgment.get("verdict_observation_ids", []) or []
        if str(item).strip()
    ]
    if len(verdict_observation_ids) != len(set(verdict_observation_ids)):
        _issue(
            report,
            "REACT_RUNTIME_VERDICT_OBSERVATIONS_DUPLICATE",
            "Judgment verdict_observation_ids must be unique",
            location="judgment.verdict_observation_ids",
        )
    unknown_verdict_observations = sorted(
        set(verdict_observation_ids) - set(successful_observation_ids)
    )
    if unknown_verdict_observations:
        _issue(
            report,
            "REACT_RUNTIME_VERDICT_OBSERVATION_UNKNOWN",
            "Judgment cites unsuccessful or unknown observations: "
            + ", ".join(unknown_verdict_observations),
            location="judgment.verdict_observation_ids",
        )
    if not isinstance(judgment.get("fact_check_report"), Mapping):
        _issue(
            report,
            "REACT_RUNTIME_REPORT_MISSING",
            "current ReAct Judgment must contain fact_check_report",
            location="judgment.fact_check_report",
        )

    report.stats.update(
        {
            "unified_react_actions": action_count,
            "react_runtime_observations": len(observation_rows),
            "react_runtime_successful_observations": len(
                successful_observation_ids
            ),
            "react_runtime_success_results": successful_results,
            "react_runtime_error_results": error_results,
            "react_runtime_malformed_results": malformed_results,
        }
    )


def audit_trace(
    path: Path,
    *,
    enforce_source_access_policy: bool = True,
    source_access_policy: SourceAccessPolicy | None = None,
    payload: Mapping[str, Any] | None = None,
) -> TraceReport:
    report = TraceReport(path=str(path))
    if payload is None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            _issue(report, "TRACE_JSON_INVALID", f"cannot read trace JSON: {exc}")
            return report
    if not isinstance(payload, Mapping):
        _issue(report, "TRACE_ROOT_INVALID", "trace JSON root must be an object")
        return report

    state = _state(payload)
    report.image_id = str(payload.get("image_id", state.get("image_id", path.stem)))
    steps = _rows(state.get("all_steps"))
    is_image_only = str(
        payload.get("input_mode") or state.get("input_mode") or ""
    ) == "image_only"
    report.stats.update(
        {
            "steps": len(steps),
        }
    )

    if "state" not in payload or not isinstance(payload.get("state"), Mapping):
        _issue(report, "CANONICAL_STATE_MISSING", "canonical trace must contain a state object")
    if not isinstance(state.get("all_steps"), list):
        _issue(report, "STEPS_MISSING", "canonical state must contain all_steps")
    if not is_image_only:
        _issue(
            report,
            "UNSUPPORTED_TRACE_MODE",
            "v3 strict audit accepts input_mode=image_only only",
            location="input_mode",
        )
        return report

    _audit_termination(payload, state, report)
    _audit_thought_tokens(payload, state, steps, report)
    policy_version = str(
        payload.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    )
    if policy_version == UNIFIED_REACT_POLICY_VERSION:
        _audit_unified_react_trace(payload, state, steps, report)
    else:
        _issue(
            report,
            "UNSUPPORTED_POLICY_VERSION",
            (
                "current strict audit accepts only "
                f"{UNIFIED_REACT_POLICY_VERSION}; legacy policy traces are archived"
            ),
            location="decision_policy_version",
        )
    _audit_leaks(
        payload,
        report,
        enforce_source_access_policy=enforce_source_access_policy,
        source_access_policy=source_access_policy,
    )
    _audit_rejections(steps, report)
    return report


def discover_trace_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [
            input_path
        ] if input_path.suffix.casefold() == ".json" and not _looks_like_benchmark_artifact(
            input_path
        ) else []
    if not input_path.is_dir():
        return []
    if input_path.name.casefold() == "traces":
        return sorted(
            path
            for path in input_path.glob("*.json")
            if path.is_file() and not _looks_like_benchmark_artifact(path)
        )
    traces_dirs = sorted(
        path for path in input_path.rglob("traces") if path.is_dir()
    )
    if traces_dirs:
        return sorted(
            {
                trace
                for directory in traces_dirs
                for trace in directory.glob("*.json")
                if trace.is_file() and not _looks_like_benchmark_artifact(trace)
            }
        )
    return sorted(
        path
        for path in input_path.glob("*.json")
        if path.is_file() and not _looks_like_benchmark_artifact(path)
    )


def _looks_like_benchmark_artifact(path: Path) -> bool:
    stem = path.stem.casefold()
    if stem in {"run_manifest", "summary", "predictions", "manifest"}:
        return True
    return bool(re.search(r"(?:^|[-_.])(benchmark|gold)(?:$|[-_.])", stem))


def _print_human(reports: Sequence[TraceReport], *, strict_scheduler: bool) -> None:
    for report in reports:
        failures = report.failures(strict_scheduler=strict_scheduler)
        warnings = report.warnings(strict_scheduler=strict_scheduler)
        status = "PASS" if not failures else "FAIL"
        print(
            f"[{status}] {report.path} "
            f"(termination={report.termination or 'unknown'}, failures={len(failures)}, warnings={len(warnings)})"
        )
        print(
            "  "
            f"steps={report.stats.get('steps', 0)} "
            f"claims={report.stats.get('claims', 0)} "
            f"evidence={report.stats.get('evidence', 0)} "
            f"scheduler_rejections={report.stats.get('scheduler_rejections', 0)} "
            f"protocol_rejections={report.stats.get('protocol_rejections', 0)} "
            f"route_control_rejections={report.stats.get('route_control_rejections', 0)}"
        )
        failure_set = set(failures)
        for issue in report.issues:
            severity = "ERROR" if issue in failure_set else "WARN"
            where = f" at {issue.location}" if issue.location else ""
            print(f"  {severity} {issue.code}{where}: {issue.message}")

    failed = sum(
        bool(report.failures(strict_scheduler=strict_scheduler)) for report in reports
    )
    warnings = sum(
        len(report.warnings(strict_scheduler=strict_scheduler)) for report in reports
    )
    scheduler_rejections = sum(
        report.stats.get("scheduler_rejections", 0) for report in reports
    )
    protocol_rejections = sum(
        report.stats.get("protocol_rejections", 0) for report in reports
    )
    route_control_rejections = sum(
        report.stats.get("route_control_rejections", 0)
        for report in reports
    )
    print(
        f"Audited {len(reports)} trace(s): {len(reports) - failed} passed, "
        f"{failed} failed, {warnings} warning(s), "
        f"{scheduler_rejections} scheduler rejection(s), "
        f"{protocol_rejections} protocol rejection(s), "
        f"{route_control_rejections} route-control rejection(s)."
    )


def _json_summary(
    reports: Sequence[TraceReport], *, strict_scheduler: bool
) -> dict[str, Any]:
    failed = sum(
        bool(report.failures(strict_scheduler=strict_scheduler)) for report in reports
    )
    return {
        "passed": failed == 0,
        "strict_scheduler": strict_scheduler,
        "trace_count": len(reports),
        "passed_count": len(reports) - failed,
        "failed_count": failed,
        "warning_count": sum(
            len(report.warnings(strict_scheduler=strict_scheduler)) for report in reports
        ),
        "scheduler_rejections": sum(
            report.stats.get("scheduler_rejections", 0) for report in reports
        ),
        "protocol_rejections": sum(
            report.stats.get("protocol_rejections", 0) for report in reports
        ),
        "route_control_rejections": sum(
            report.stats.get("route_control_rejections", 0)
            for report in reports
        ),
        "traces": [
            report.to_dict(strict_scheduler=strict_scheduler) for report in reports
        ],
    }


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Strictly audit canonical v3 image-only traces."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="A canonical trace JSON file or a directory containing traces.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit one machine-readable JSON summary.",
    )
    parser.add_argument(
        "--strict-scheduler",
        action="store_true",
        help="Treat runtime action/output rejections as failures instead of warnings.",
    )
    parser.add_argument(
        "--source-access-policy",
        type=Path,
        help=(
            "Optional active source-access policy. Only sources listed in this "
            "policy are checked for query or URL leaks."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = discover_trace_files(args.input_path)
    if not files:
        message = f"No trace JSON files found at: {args.input_path}"
        if args.json_output:
            print(
                json.dumps(
                    {
                        "passed": False,
                        "strict_scheduler": args.strict_scheduler,
                        "trace_count": 0,
                        "passed_count": 0,
                        "failed_count": 0,
                        "warning_count": 0,
                        "error": message,
                        "traces": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(message, file=sys.stderr)
        return 1

    source_access_policy = (
        SourceAccessPolicy.load(args.source_access_policy)
        if args.source_access_policy is not None
        else None
    )
    reports = [
        audit_trace(path, source_access_policy=source_access_policy)
        for path in files
    ]
    if args.json_output:
        print(
            json.dumps(
                _json_summary(reports, strict_scheduler=args.strict_scheduler),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        _print_human(reports, strict_scheduler=args.strict_scheduler)
    return int(
        any(
            report.failures(strict_scheduler=args.strict_scheduler)
            for report in reports
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
