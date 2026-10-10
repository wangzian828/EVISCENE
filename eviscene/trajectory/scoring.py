"""Deterministic process metrics and componentized teacher scoring."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from eviscene.orchestrator.evidence_semantics import same_capture_can_support_visual_claim
from eviscene.orchestrator.tool_result import parse_tool_result
from eviscene.orchestrator.react_runtime import (
    is_unified_react_runtime_budget_action,
)


FACT_MATCH_THRESHOLD = 0.35

LABEL_TO_VERDICT = {
    "supported": "real",
    "refuted": "fake",
    "unverifiable": "unverifiable",
}
def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> List[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _tokens(value: Any) -> set[str]:
    result: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", str(value or "").casefold()):
        if token in {
            "the",
            "a",
            "an",
            "is",
            "are",
            "at",
            "in",
            "of",
            "to",
            "ve",
            "vf",
            "rf",
        }:
            continue
        if re.fullmatch(r"[0-9a-f]{12,}", token):
            continue
        normalized = _light_fact_stem(token)
        if normalized:
            result.add(normalized)
    return result


def _light_fact_stem(token: str) -> str:
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            stem = token[: -len(suffix)]
            return stem + "y" if suffix == "ies" else stem
    return token


def _token_f1(left: Any, right: Any) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    unmatched_right = set(right_tokens)
    overlap = 0
    for left_token in sorted(left_tokens):
        matched = next(
            (
                right_token
                for right_token in sorted(unmatched_right)
                if _fact_tokens_equivalent(left_token, right_token)
            ),
            None,
        )
        if matched is not None:
            overlap += 1
            unmatched_right.remove(matched)
    if not overlap:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def _fact_tokens_equivalent(left: str, right: str) -> bool:
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return (
        len(shorter) >= 6
        and len(longer) - len(shorter) <= 2
        and longer.startswith(shorter)
    )


def _fact_similarity(
    runtime_fact: Mapping[str, Any],
    gold_fact: Mapping[str, Any],
    *,
    recovered_source_context: str = "",
) -> float:
    statement = runtime_fact.get("statement")
    score = max(
        _token_f1(statement, gold_fact.get("statement")),
        _token_f1(
            " ".join(
                item
                for item in (
                    str(statement or "").strip(),
                    str(recovered_source_context or "").strip(),
                )
                if item
            ),
            gold_fact.get("statement"),
        ),
    )
    if str(runtime_fact.get("kind", "")) == str(gold_fact.get("kind", "")):
        score = min(1.0, score + 0.1)
    return score


def _match_gold_facts(
    runtime_facts: Sequence[Mapping[str, Any]],
    gold_facts: Sequence[Mapping[str, Any]],
    *,
    recovered_source_context_by_fact: Mapping[str, str] | None = None,
) -> List[Dict[str, Any]]:
    context_by_fact = recovered_source_context_by_fact or {}
    available = {
        str(item.get("fact_id", "")).strip(): item
        for item in runtime_facts
        if str(item.get("fact_id", "")).strip()
    }
    matches: List[Dict[str, Any]] = []
    for gold in gold_facts:
        ranked = sorted(
            (
                (
                    _fact_similarity(
                        runtime,
                        gold,
                        recovered_source_context=context_by_fact.get(
                            fact_id,
                            "",
                        ),
                    ),
                    fact_id,
                    runtime,
                )
                for fact_id, runtime in available.items()
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if not ranked or ranked[0][0] < FACT_MATCH_THRESHOLD:
            matches.append(
                {
                    "gold_fact_id": str(gold.get("fact_id", "")),
                    "runtime_fact_id": None,
                    "similarity": round(ranked[0][0], 6) if ranked else 0.0,
                    "expected_status": str(gold.get("expected_status", "")),
                    "runtime_status": None,
                    "status_match": False,
                }
            )
            continue
        similarity, fact_id, runtime = ranked[0]
        available.pop(fact_id, None)
        expected_status = str(gold.get("expected_status", ""))
        runtime_status = str(runtime.get("status", ""))
        matches.append(
            {
                "gold_fact_id": str(gold.get("fact_id", "")),
                "runtime_fact_id": fact_id,
                "similarity": round(similarity, 6),
                "expected_status": expected_status,
                "runtime_status": runtime_status,
                "status_match": runtime_status == expected_status,
            }
        )
    return matches


def _basis_same_capture_source_context(
    investigation: Mapping[str, Any],
    basis: Mapping[str, Any],
) -> Dict[str, str]:
    """Recover source-page context only from selected same-capture Evidence."""

    basis_evidence_ids = {
        str(item) for item in basis.get("evidence_ids", []) or []
    }
    evidence = {
        str(item.get("evidence_id", "")): item
        for item in _rows(investigation.get("evidence"))
        if str(item.get("evidence_id", ""))
    }
    discoveries = _rows(investigation.get("discoveries"))
    context_by_fact: Dict[str, List[str]] = {}
    for evidence_id in basis_evidence_ids:
        item = evidence.get(evidence_id)
        if item is None or not (
            str(item.get("claim_binding", "")) == "same_capture"
            and item.get("same_capture_or_near_duplicate") is True
            and item.get("likely_different_original_capture") is not True
        ):
            continue
        reference_url = _canonical_url(item.get("source_url"))
        task_id = str(item.get("task_id", ""))
        fact_ids = {
            str(fact_id) for fact_id in item.get("fact_ids", []) or []
        }
        if not reference_url or not task_id or not fact_ids:
            continue
        for discovery in discoveries:
            if (
                str(discovery.get("task_id", "")) != task_id
                or _canonical_url(discovery.get("reference_image_url"))
                != reference_url
                or not fact_ids
                & {
                    str(fact_id)
                    for fact_id in discovery.get("fact_ids", []) or []
                }
            ):
                continue
            source_context = " ".join(
                str(discovery.get(key, "") or "").strip()
                for key in ("title", "candidate_url", "snippet")
                if str(discovery.get(key, "") or "").strip()
            )
            if not source_context:
                continue
            for fact_id in fact_ids:
                context_by_fact.setdefault(fact_id, []).append(
                    source_context
                )
    return {
        fact_id: " ".join(dict.fromkeys(contexts))
        for fact_id, contexts in context_by_fact.items()
    }


def _canonical_url(value: Any) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        return ""
    try:
        parsed = urlsplit(rendered)
    except ValueError:
        return rendered.casefold().rstrip("/")
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            path,
            parsed.query,
            "",
        )
    )


def _family_key(value: Any) -> str:
    rendered = str(value or "").casefold().strip()
    rendered = rendered.removeprefix("domain:")
    return rendered.split(".", 1)[0]


def _evidence_matches_reference(
    evidence: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> bool:
    evidence_text = " ".join(str(evidence.get("exact_text", "")).split())
    reference_text = " ".join(str(reference.get("exact_span", "")).split())
    text_match = (
        evidence_text == reference_text
        or reference_text in evidence_text
        or evidence_text in reference_text
    ) if evidence_text and reference_text else False
    return all(
        (
            _canonical_url(evidence.get("source_url"))
            == _canonical_url(reference.get("canonical_url")),
            text_match,
            str(evidence.get("stance", "")) == str(reference.get("stance", "")),
            str(evidence.get("artifact_sha256", "")).casefold()
            == str(reference.get("artifact_sha256", "")).casefold(),
            _family_key(evidence.get("source_family"))
            == _family_key(reference.get("source_family")),
        )
    )


def _successful_call_ids(
    steps: Sequence[Mapping[str, Any]],
) -> set[str]:
    result: set[str] = set()
    for step in steps:
        if str(step.get("action_type", "")) != "tool_call":
            continue
        call_id = str(
            _mapping(step.get("metadata")).get("function_call_id", "")
        ).strip()
        if not call_id:
            continue
        try:
            _, succeeded = parse_tool_result(str(step.get("tool_result", "")))
        except Exception:
            succeeded = False
        if succeeded:
            result.add(call_id)
    return result


def _valid_findings(
    findings: Sequence[Mapping[str, Any]],
    tasks: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    successful_calls: set[str],
) -> set[str]:
    valid: set[str] = set()
    for finding in findings:
        finding_id = str(finding.get("finding_id", "")).strip()
        task_id = str(finding.get("task_id", "")).strip()
        task = tasks.get(task_id)
        if not finding_id or task is None:
            continue
        fact_ids = {str(item) for item in finding.get("fact_ids", []) or []}
        if not fact_ids or not fact_ids <= {
            str(item) for item in task.get("fact_ids", []) or []
        }:
            continue
        owned_evidence = [
            evidence.get(str(evidence_id))
            for evidence_id in finding.get("evidence_ids", []) or []
        ]
        if not owned_evidence or any(item is None for item in owned_evidence):
            continue
        if any(
            str(item.get("task_id", "")).strip() != task_id
            or str(item.get("function_call_id", "")).strip()
            not in successful_calls
            for item in owned_evidence
            if item is not None
        ):
            continue
        valid.add(finding_id)
    return valid


def _first_error(
    trace: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
) -> Dict[str, Any] | None:
    for index, step in enumerate(steps):
        if str(step.get("action_type", "")) in {
            "format_error",
            "output_rejected",
        }:
            return {
                "step_index": index,
                "stage": step.get("stage"),
                "tool_name": step.get("tool_name"),
                "error": str(
                    _mapping(step.get("metadata")).get(
                        "rejection_reason",
                        step.get("tool_result", ""),
                    )
                )[:1000],
            }
        if str(step.get("action_type", "")) == "tool_call":
            try:
                payload, succeeded = parse_tool_result(
                    str(step.get("tool_result", ""))
                )
            except Exception as exc:
                return {
                    "step_index": index,
                    "stage": step.get("stage"),
                    "tool_name": step.get("tool_name"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if not succeeded:
                return {
                    "step_index": index,
                    "stage": step.get("stage"),
                    "tool_name": step.get("tool_name"),
                    "error": str(payload.get("error", ""))[:1000],
                }
    if str(trace.get("termination", "")) == "error":
        return {
            "step_index": None,
            "stage": None,
            "tool_name": None,
            "error": str(trace.get("error", ""))[:1000],
        }
    return None


def _has_actual_visual_bridge(
    fact: Mapping[str, Any],
    related_evidence: Sequence[Mapping[str, Any]],
    *,
    semantic_decision: Mapping[str, Any] | None = None,
) -> bool:
    status = str(fact.get("status", ""))
    bindings = {
        str(item.get("claim_binding", ""))
        for item in related_evidence
        if not item.get("risk_flags")
    }
    if status == "supported" and same_capture_can_support_visual_claim(fact):
        return bool(bindings & {"pixel_observation", "same_capture"})
    if status == "refuted":
        decision_output = _mapping(
            _mapping(semantic_decision).get("output")
        )
        selected_ids = {
            str(item)
            for item in decision_output.get("selected_evidence_ids", []) or []
        }
        if (
            decision_output.get("assessment") == "refuted"
            and decision_output.get("binding_requirement") == "text_sufficient"
            and any(
                str(item.get("evidence_id", "")) in selected_ids
                and str(item.get("claim_binding", "")) == "source_assertion"
                and str(item.get("quality", "")) in {"strong", "moderate"}
                and not item.get("risk_flags")
                for item in related_evidence
            )
        ):
            return True
        return any(
            str(item.get("claim_binding", ""))
            in {"pixel_observation", "same_capture", "source_assertion"}
            and str(item.get("directness", "")) == "direct"
            and str(item.get("quality", "")) in {"strong", "moderate"}
            and not item.get("risk_flags")
            for item in related_evidence
        )
    return bool(bindings & {"pixel_observation", "same_capture"})


def _raw_observation_rows(
    steps: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    max_no_match_streak = 0
    current_no_match_streak = 0
    for index, step in enumerate(steps):
        if not is_unified_react_runtime_budget_action(step):
            continue
        metadata = _mapping(step.get("metadata"))
        call_id = str(metadata.get("function_call_id", "")).strip()
        raw_result = str(step.get("tool_result", ""))
        tool_name = str(step.get("tool_name", "")).strip()
        status = "malformed"
        succeeded = False
        payload: Mapping[str, Any] = {}
        try:
            parsed, succeeded = parse_tool_result(raw_result)
            payload = parsed
            status = "success" if succeeded else "error"
        except Exception:
            pass
        result_count = 0
        for key in (
            "results",
            "lens_results",
            "semantic_results",
            "reference_image_candidates",
            "candidate_page_urls",
            "evidence_records",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                result_count += len(value)
        for query_row in payload.get("queries", []) or []:
            if isinstance(query_row, Mapping):
                result_count += len(query_row.get("results", []) or [])
        observation_status = str(
            payload.get("observation_status", "")
        ).strip().casefold()
        is_search = tool_name in {
            "text_search",
            "text_image_search",
            "reverse_image_search",
        }
        no_match = bool(
            succeeded
            and is_search
            and (
                observation_status
                in {"empty_results", "no_match", "no_results"}
                or result_count == 0
            )
        )
        if no_match:
            current_no_match_streak += 1
            max_no_match_streak = max(max_no_match_streak, current_no_match_streak)
        else:
            current_no_match_streak = 0
        rows.append(
            {
                "step_index": index,
                "observation_id": call_id,
                "tool_name": tool_name,
                "status": status,
                "successful": succeeded,
                "result_count": result_count,
                "no_match": no_match,
            }
        )
    return rows, max_no_match_streak


def _score_raw_history_trace(
    trace: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    score_metadata: Mapping[str, Any] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    state = _mapping(trace.get("state"))
    steps = _rows(state.get("all_steps"))
    observations, max_no_match_streak = _raw_observation_rows(steps)
    successful_ids = [
        str(item["observation_id"])
        for item in observations
        if item["successful"] and item["observation_id"]
    ]
    malformed_count = sum(item["status"] == "malformed" for item in observations)
    error_count = sum(item["status"] == "error" for item in observations)
    action_count = len(observations)
    expected_verdict = LABEL_TO_VERDICT.get(
        str(gold.get("factual_status", "")),
        "unverifiable",
    )
    runtime_verdict = str(trace.get("verdict", "")).strip()
    engineering_error = bool(
        str(trace.get("termination", "")).strip() != "success"
        or str(state.get("termination", "")).strip() not in {"", "success"}
        or malformed_count
        or runtime_verdict not in {"real", "fake"}
    )
    result_correct = bool(
        not engineering_error
        and expected_verdict in {"real", "fake"}
        and runtime_verdict == expected_verdict
    )
    basis = _mapping(trace.get("verdict_basis"))
    judgment = _mapping(trace.get("judgment") or state.get("judgment"))
    basis_ids = [
        str(item).strip()
        for item in basis.get("observation_ids", []) or []
        if str(item).strip()
    ]
    cited_ids = [
        str(item).strip()
        for item in judgment.get("verdict_observation_ids", []) or []
        if str(item).strip()
    ]
    citation_valid = bool(
        len(cited_ids) == len(set(cited_ids))
        and set(cited_ids) <= set(successful_ids)
    )
    basis_aligned = basis_ids == successful_ids
    judgment_outputs = [
        index
        for index, step in enumerate(steps)
        if str(step.get("stage", "")).strip() == "unified_judgment"
        and str(step.get("action_type", "")).strip() == "output"
    ]
    terminal_index = judgment_outputs[-1] if judgment_outputs else len(steps)
    post_verdict_action_count = sum(
        is_unified_react_runtime_budget_action(step)
        for step in steps[terminal_index + 1 :]
    )
    low_value_action_count = sum(
        item["tool_name"] == "current_time" for item in observations
    )
    first_error = _first_error(trace, steps)
    exclusion_reasons: list[str] = []
    if not result_correct:
        exclusion_reasons.append("incorrect_result")
    if engineering_error:
        exclusion_reasons.append("engineering_error")
    if not basis_aligned:
        exclusion_reasons.append("verdict_basis_misaligned")
    if not citation_valid:
        exclusion_reasons.append("verdict_observation_citation_invalid")
    if post_verdict_action_count:
        exclusion_reasons.append("actions_after_judgment")
    training_eligible = not exclusion_reasons
    process_metrics = {
        "schema_version": "ifv-process-metrics-raw-history-v1",
        "case_id": str(gold.get("case_id") or trace.get("image_id") or ""),
        "score_metadata": dict(score_metadata or {}),
        "engineering_error": engineering_error,
        "expected_verdict": expected_verdict,
        "runtime_verdict": runtime_verdict,
        "result_correct": result_correct,
        "tool_actions": action_count,
        "raw_observations": observations,
        "successful_observation_count": len(successful_ids),
        "error_observation_count": error_count,
        "malformed_observation_count": malformed_count,
        "successful_empty_search_count": sum(item["no_match"] for item in observations),
        "max_successful_no_match_streak": max_no_match_streak,
        "verdict_basis_alignment": 1.0 if basis_aligned else 0.0,
        "verdict_observation_citation_validity": 1.0 if citation_valid else 0.0,
        "post_determination_action_count": post_verdict_action_count,
        "low_value_action_count": low_value_action_count,
        "low_value_action_rate": round(
            low_value_action_count / action_count if action_count else 0.0,
            6,
        ),
        "stop_quality": 1.0 if judgment_outputs and not post_verdict_action_count else 0.0,
        "premature_finish": not bool(judgment_outputs),
        "duplicate_action_count": 0,
        "duplicate_action_rate": 0.0,
        "llm_api_calls": int(trace.get("llm_api_calls", 0) or 0),
        "token_usage": dict(_mapping(trace.get("token_usage"))),
        "latency_seconds": float(trace.get("time_taken", 0.0) or 0.0),
        "first_error": first_error,
        "training_eligible": training_eligible,
        "training_exclusion_reasons": exclusion_reasons,
    }
    components = {
        "result_reward": 1.0 if result_correct else 0.0,
        "raw_history_grounding_reward": 1.0 if citation_valid else 0.0,
        "stop_quality_reward": process_metrics["stop_quality"],
        "protocol_penalty": -1.0 if malformed_count else 0.0,
        "normalized_cost_penalty": round(-0.25 * min(1.0, action_count / 24.0), 6),
    }
    teacher_score = {
        "schema_version": "ifv-trajectory-score-raw-history-v1",
        "case_id": process_metrics["case_id"],
        "score_metadata": dict(score_metadata or {}),
        "components": components,
        "total": round(sum(components.values()), 6),
        "training_eligible": training_eligible,
        "training_exclusion_reasons": exclusion_reasons,
        "diagnostics": {
            "successful_observation_ids": successful_ids,
            "basis_observation_ids": basis_ids,
            "verdict_observation_ids": cited_ids,
            "max_successful_no_match_streak": max_no_match_streak,
            "error_observation_count": error_count,
            "malformed_observation_count": malformed_count,
            "first_error": first_error,
        },
    }
    return process_metrics, teacher_score


def score_process_trace(
    trace: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    score_metadata: Mapping[str, Any] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Score one current raw-history trace against private references."""

    state = _mapping(trace.get("state"))
    policy_version = str(
        trace.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    )
    if policy_version != "unified-react-v1":
        raise ValueError(
            "score_process_trace accepts only unified-react-v1 raw-history traces"
        )
    if str(state.get("investigation_state", {}).get("schema_version", "")) != (
        "ifv-unified-react-raw-history-v1"
    ):
        raise ValueError(
            "score_process_trace requires the raw-history runtime schema"
        )
    return _score_raw_history_trace(
        trace,
        gold,
        score_metadata=score_metadata,
    )
