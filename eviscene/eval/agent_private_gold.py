"""Read-only projections used to audit Agent fact-check reports.

The rollout never receives private construction gold.  This module only projects
an already-persisted trace into the evidence-bounded packet seen by a post-hoc
private-gold judge.
"""

from __future__ import annotations

from typing import Any, Mapping

from eviscene.orchestrator.react_runtime import REACT_RUNTIME_SCHEMA_VERSION
from eviscene.orchestrator.tool_result import parse_tool_result


AGENT_PRIVATE_GOLD_PROJECTION_SCHEMA_VERSION = (
    "ifv-agent-private-gold-candidate-v3"
)

_BINARY_OR_TRANSPORT_FIELDS = {
    "base64",
    "content_bytes",
    "data_url",
    "html",
    "image",
    "image_input",
    "raw_html",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _text(value: Any, *, limit: int | None = None) -> str:
    text = str(value or "").strip()
    return text[:limit] if limit is not None else text


def _ids(value: Any, *, limit: int | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _semantic_value(value: Any) -> Any:
    """Keep complete judge-relevant JSON while dropping non-text payloads.

    Agent traces can contain repeated policy inputs, embedded images, and raw HTML
    that are neither model outputs nor evidence text.  Everything else is kept
    without truncating lists, mappings, nested values, or strings.
    """

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.casefold() in _BINARY_OR_TRANSPORT_FIELDS:
                continue
            if (
                key.casefold() == "image_url"
                and isinstance(child, str)
                and child.lstrip().casefold().startswith("data:")
            ):
                continue
            result[key] = _semantic_value(child)
        return result
    if isinstance(value, list):
        return [_semantic_value(item) for item in value]
    if isinstance(value, str):
        return value.strip()
    return value


def _successful_evidence(row: Mapping[str, Any]) -> bool:
    if "successful_call" in row:
        return bool(row.get("successful_call"))
    if "tool_success" in row:
        return bool(row.get("tool_success"))
    status = _text(row.get("status") or row.get("tool_status"), limit=100).lower()
    return status not in {"error", "failed", "failure"}


def _evidence_projection(
    row: Mapping[str, Any],
    *,
    selected_evidence_ids: set[str],
) -> dict[str, Any]:
    """Retain the legacy structured evidence view without clipping its text."""

    evidence_id = _text(row.get("evidence_id"), limit=100)
    projected = _semantic_value(row)
    assert isinstance(projected, dict)
    projected.update(
        {
            "evidence_id": evidence_id,
            "selected_by_verdict_basis": evidence_id in selected_evidence_ids,
            "successful_call": _successful_evidence(row),
        }
    )
    return projected


def _raw_action_history(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project every declared ReAct action and its complete semantic result.

    Hidden reasoning artifacts and repeated policy inputs are deliberately not
    loaded.  Direct-QA judging likewise receives the declared answer rather than
    native chain-of-thought.  Tool calls, arguments, terminal actions, and tool
    observations are declared Agent outputs and are all retained here.
    """

    history: list[dict[str, Any]] = []
    for index, step in enumerate(_rows(state.get("all_steps"))):
        if (
            _text(step.get("stage"), limit=80) != "unified_react"
            or _text(step.get("action_type"), limit=80) != "tool_call"
        ):
            continue
        metadata = _mapping(step.get("metadata"))
        call_id = _text(metadata.get("function_call_id"), limit=160)
        tool_name = _text(step.get("tool_name"), limit=120)
        raw_result = str(step.get("tool_result", ""))
        try:
            payload, succeeded = parse_tool_result(raw_result)
            observation = _semantic_value(payload)
            status = "success" if succeeded else "error"
        except Exception:
            observation = raw_result
            status = "malformed"
            succeeded = False
        arguments = {
            str(key): _semantic_value(value)
            for key, value in _mapping(step.get("tool_args")).items()
            if str(key).casefold() not in _BINARY_OR_TRANSPORT_FIELDS
        }
        history.append(
            {
                "step_index": index,
                "turn": len(history) + 1,
                "observation_id": call_id,
                "tool": tool_name,
                "is_terminal_action": tool_name == "finish_investigation",
                "arguments": arguments,
                "observation": observation,
                "status": status,
                "tool_success": bool(succeeded),
            }
        )
    return history


def build_agent_private_gold_candidate(
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the complete semantic Agent output packet seen by the judge."""

    state = _mapping(trace.get("state"))
    investigation = _mapping(state.get("investigation_state"))
    is_unified_react_runtime = (
        str(investigation.get("schema_version", "")).strip()
        == REACT_RUNTIME_SCHEMA_VERSION
    )
    judgment = _mapping(
        trace.get("judgment")
        or state.get("judgment")
        or investigation.get("discrepancy_judgment")
    )
    basis = _mapping(
        trace.get("verdict_basis")
        or investigation.get("discrepancy_verdict_basis")
    )
    selected_evidence_ids = set(
        _ids(basis.get("evidence_ids") or basis.get("observation_ids"))
    )
    selected_finding_ids = set(_ids(basis.get("finding_ids")))
    selected_claim_ids = set(_ids(basis.get("claim_ids")))
    evidence = [
        _evidence_projection(item, selected_evidence_ids=selected_evidence_ids)
        for item in _rows(investigation.get("evidence"))
        if _text(item.get("evidence_id"), limit=100)
        and _successful_evidence(item)
    ]
    findings = [
        {
            **_semantic_value(item),
            "selected_by_verdict_basis": _text(
                item.get("finding_id"), limit=100
            )
            in selected_finding_ids,
        }
        for item in _rows(investigation.get("findings"))
        if _text(item.get("finding_id"), limit=100)
    ]
    target_facts = [
        {
            **_semantic_value(item),
            "selected_by_verdict_basis": _text(
                item.get("claim_id"), limit=100
            )
            in selected_claim_ids,
        }
        for item in _rows(investigation.get("target_facts"))
        if _text(item.get("claim_id"), limit=100)
    ]
    citations = [
        _semantic_value(item)
        for item in _rows(judgment.get("evidence_citations"))
        if _text(item.get("evidence_id"), limit=100)
    ]
    raw_history = _raw_action_history(state) if is_unified_react_runtime else []
    result = {
        "projection_schema_version": (
            AGENT_PRIVATE_GOLD_PROJECTION_SCHEMA_VERSION
        ),
        "candidate_kind": "agent_trace",
        "runtime_mode": (
            "image_grounded_react" if is_unified_react_runtime else "legacy_graph"
        ),
        "recorded_verdict": _text(
            judgment.get("verdict") or trace.get("verdict"),
            limit=100,
        ),
        "overall_assessment": _text(
            judgment.get("overall_assessment")
            or trace.get("overall_assessment")
        ),
        "fact_check_report": _semantic_value(
            judgment.get("fact_check_report") or trace.get("fact_check_report")
        ),
        "termination": _text(trace.get("termination"), limit=100),
        "verdict_basis": {
            "decision_mode": _text(basis.get("decision_mode"), limit=100),
            "objective": _text(
                basis.get("objective") or basis.get("verdict_target")
            ),
            "verdict_target": _text(basis.get("verdict_target")),
            "claim_ids": _ids(basis.get("claim_ids")),
            "discrepancy_ids": _ids(basis.get("discrepancy_ids")),
            "finding_ids": _ids(basis.get("finding_ids")),
            "evidence_ids": _ids(basis.get("evidence_ids")),
            "observation_ids": _ids(basis.get("observation_ids")),
            "unresolved_gaps": _ids(
                basis.get("unresolved_gaps")
                or basis.get("open_questions")
            ),
        },
        "complete_verdict_basis": _semantic_value(basis),
        "terminal_judgment": _semantic_value(judgment),
        "investigation_state": _semantic_value(investigation),
        "final_visual_audit": _semantic_value(trace.get("final_visual_audit")),
        "runtime_evidence_citations": citations,
        "selected_evidence": [
            item for item in evidence if item["selected_by_verdict_basis"]
        ],
        "successful_evidence": evidence,
        "selected_findings": [
            item for item in findings if item["selected_by_verdict_basis"]
        ],
        "selected_target_facts": [
            item for item in target_facts if item["selected_by_verdict_basis"]
        ],
        "rejected_policy_outputs": _semantic_value(
            _rows(state.get("rejection_history"))
        ),
        "runtime_objective": _text(investigation.get("objective")),
        "raw_observations": raw_history,
        "raw_observation_ids": [
            item["observation_id"]
            for item in raw_history
            if item["observation_id"]
        ],
        "successful_observation_ids": [
            item["observation_id"]
            for item in raw_history
            if item["tool_success"] and item["observation_id"]
        ],
        "unsuccessful_observation_ids": [
            item["observation_id"]
            for item in raw_history
            if not item["tool_success"] and item["observation_id"]
        ],
        "action_count": int(investigation.get("action_count", 0) or 0),
        "stop_reason": _text(investigation.get("stop_reason"), limit=100),
        "finish_rationale": _text(investigation.get("finish_rationale")),
    }
    return result


def agent_candidate_answer(packet: Mapping[str, Any]) -> dict[str, str]:
    """Project the Agent terminal packet into the common candidate-answer shape."""

    basis = _mapping(packet.get("verdict_basis"))
    report = _mapping(packet.get("fact_check_report"))
    target_facts = _rows(packet.get("selected_target_facts"))
    findings = _rows(packet.get("selected_findings"))
    report_findings = report.get("key_findings")
    if not isinstance(report_findings, list):
        report_findings = []
    core_fact = _text(report.get("claim_under_review"))
    if not core_fact:
        core_fact = _text(packet.get("overall_assessment"))
    if not core_fact:
        core_fact = _text(basis.get("objective") or basis.get("verdict_target"))
    if not core_fact and target_facts:
        core_fact = _text(target_facts[0].get("statement"))
    reason_parts = [
        _text(report.get("verdict_summary")),
        _text(report.get("evidence_summary")),
        *[
            _text(item)
            for item in report_findings
            if isinstance(item, str)
        ],
        *[_text(item.get("summary")) for item in findings],
        _text(packet.get("overall_assessment")),
    ]
    reason = "\n\n".join(dict.fromkeys(part for part in reason_parts if part))
    return {
        "core_fact": core_fact,
        "verdict": _text(packet.get("recorded_verdict"), limit=100),
        "reason": reason,
    }
