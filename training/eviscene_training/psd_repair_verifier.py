"""Strict, gold-gated verification for EviScene PSD repair continuations."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import _repo_import  # noqa: F401
from scripts.audit_real_trace import audit_trace
from eviscene.trajectory.scoring import score_process_trace
from eviscene.orchestrator.stage_runner import StageStep
from eviscene.orchestrator.react_runtime import (
    REACT_RUNTIME_SCHEMA_VERSION,
    UNIFIED_REACT_RUNTIME_POLICY_VERSION,
    UnifiedReactState,
    compile_react_judgment_basis,
    record_react_action,
)

from .psd_repair import (
    PSD_LOCAL_VERIFICATION_SCHEMA_VERSION,
    VerificationResult,
    _sha,
    _text,
    verify_repair,
)


_PRIVATE_KEYS = frozenset(
    {
        "gold",
        "evaluation_gold",
        "factual_status",
        "expected_verdict",
        "ground_truth",
        "private_gold",
        "private_target",
    }
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _assert_no_private_fields(value: Any, *, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            child_path = f"{path}.{name}" if path else name
            if name.casefold() in _PRIVATE_KEYS:
                raise ValueError(f"private field in student trace: {child_path}")
            _assert_no_private_fields(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_private_fields(child, path=f"{path}[{index}]")


def _int_ids(value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        return []
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            return []
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            return []
    return result


def _capture_binding(value: Any) -> tuple[str, str]:
    capture = _mapping(value)
    if _text(capture.get("status")) != "complete":
        return "", ""
    prompt_ids = _int_ids(capture.get("prompt_token_ids"))
    completion_ids = _int_ids(capture.get("completion_token_ids"))
    if not prompt_ids or not completion_ids:
        return "", ""
    return _sha(prompt_ids), _sha(completion_ids)


def _teacher_step_binding(steps: Sequence[StageStep]) -> tuple[str, str]:
    for step in steps:
        prompt_sha256, completion_sha256 = _capture_binding(
            _mapping(step.metadata).get("policy_token_capture")
        )
        if prompt_sha256 and completion_sha256:
            return prompt_sha256, completion_sha256
    return "", ""


def _trace_contains_token_binding(
    trace: Mapping[str, Any],
    *,
    teacher_prompt_sha256: str,
    teacher_completion_sha256: str,
) -> bool:
    for step in _rows(_mapping(trace.get("state")).get("all_steps")):
        prompt_sha256, completion_sha256 = _capture_binding(
            _mapping(step.get("metadata")).get("policy_token_capture")
        )
        if (
            prompt_sha256 == teacher_prompt_sha256
            and completion_sha256 == teacher_completion_sha256
        ):
            return True
    return False


def _step_row(step: StageStep, *, stage: str, role: str) -> dict[str, Any]:
    metadata = copy.deepcopy(step.metadata or {})
    metadata["psd_role"] = role
    metadata["psd_suffix_step"] = True
    row: dict[str, Any] = {
        "round": int(step.round or 0),
        "stage": stage,
        "action_type": str(step.action_type or ""),
        "tool_name": str(step.tool_name or ""),
        "tool_args": copy.deepcopy(step.tool_args or {}),
        "tool_result": str(step.tool_result or ""),
        "tokens": copy.deepcopy(step.tokens or {}),
        "metadata": metadata,
    }
    if step.thought:
        row["thought"] = step.thought
    if step.output is not None:
        row["output"] = copy.deepcopy(step.output)
    return row


def merge_suffix_into_trace(
    base_trace: Mapping[str, Any],
    suffix_steps: Sequence[StageStep],
    *,
    role: str,
    source_stage: str = "unified_react",
) -> dict[str, Any]:
    """Append a bounded suffix without inventing terminal episode state."""

    if role not in {"teacher", "student"}:
        raise ValueError("PSD repair trace role must be teacher or student")
    trace = copy.deepcopy(dict(base_trace))
    state = trace.get("state")
    if not isinstance(state, dict):
        raise ValueError("PSD repair trace requires a state object")
    all_steps = state.get("all_steps")
    if not isinstance(all_steps, list):
        raise ValueError("PSD repair trace requires state.all_steps")
    if not suffix_steps:
        raise ValueError("PSD repair suffix must contain at least one step")
    _assert_no_private_fields(trace if role == "student" else {})
    all_steps.extend(
        _step_row(step, stage=source_stage, role=role)
        for step in suffix_steps
    )
    tool_count = sum(1 for step in suffix_steps if step.action_type == "tool_call")
    state["total_tool_calls"] = int(state.get("total_tool_calls", 0) or 0) + tool_count
    state["llm_api_calls"] = int(state.get("llm_api_calls", 0) or 0) + sum(
        1 for step in suffix_steps if step.metadata.get("llm_duration_ms") is not None
    )
    trace["psd_repair"] = {
        "role": role,
        "suffix_step_count": len(suffix_steps),
        "terminal_state_replayed": False,
    }
    return trace


def _runtime_totals(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    token_usage = {"prompt": 0, "completion": 0, "thought": 0}
    total_tool_subcalls = 0
    tool_subcalls_by_kind: dict[str, int] = {}
    llm_api_calls = 0
    for step in steps:
        tokens = _mapping(step.get("tokens"))
        metadata = _mapping(step.get("metadata"))
        tool_tokens = _mapping(metadata.get("tool_tokens"))
        for name in token_usage:
            token_usage[name] += int(tokens.get(name, 0) or 0)
            token_usage[name] += int(tool_tokens.get(name, 0) or 0)
        if metadata.get("llm_duration_ms") is not None:
            llm_api_calls += 1
        llm_api_calls += int(metadata.get("tool_llm_api_calls", 0) or 0)
        for raw_subcall in metadata.get("tool_subcalls", []) or []:
            if not isinstance(raw_subcall, Mapping):
                continue
            count = max(0, int(raw_subcall.get("request_count", 1) or 0))
            kind = _text(raw_subcall.get("kind")) or "unknown"
            total_tool_subcalls += count
            tool_subcalls_by_kind[kind] = (
                tool_subcalls_by_kind.get(kind, 0) + count
            )
    return {
        "token_usage": token_usage,
        "total_tool_calls": sum(
            _text(step.get("action_type")) == "tool_call" for step in steps
        ),
        "total_tool_subcalls": total_tool_subcalls,
        "tool_subcalls_by_kind": tool_subcalls_by_kind,
        "llm_api_calls": llm_api_calls,
    }


def build_complete_hinted_episode_trace(
    base_trace: Mapping[str, Any],
    *,
    failure_site: Any,
    teacher_steps: Sequence[StageStep],
    stop_reason: str,
) -> dict[str, Any]:
    """Replace the failed decision suffix with one terminal hinted episode.

    ``teacher_steps`` must contain the replacement ReAct actions followed by
    exactly one real ``RawHistoryJudgmentOutput`` step.  The old failed action
    and every downstream source step are discarded; otherwise a verifier could
    accidentally approve a trace containing both the failed and repaired
    branches.
    """

    source_index = getattr(failure_site, "source_step_index", None)
    if not isinstance(source_index, int) or isinstance(source_index, bool):
        raise ValueError("complete hinted episode requires a source step index")
    trace = copy.deepcopy(dict(base_trace))
    state = trace.get("state")
    if not isinstance(state, dict):
        raise ValueError("PSD repair trace requires a state object")
    source_steps = state.get("all_steps")
    if not isinstance(source_steps, list) or not (0 <= source_index < len(source_steps)):
        raise ValueError("PSD repair source step index is out of range")
    source_step = source_steps[source_index]
    if not isinstance(source_step, Mapping):
        raise ValueError("PSD repair source step is malformed")
    source_metadata = _mapping(source_step.get("metadata"))
    if _sha(source_metadata.get("policy_action")) != _sha(
        getattr(failure_site, "policy_action", {})
    ):
        raise ValueError("PSD repair source policy action changed")

    judgment_steps = [
        step
        for step in teacher_steps
        if step.stage_name == "psd_teacher_judgment"
        and step.action_type == "output"
        and isinstance(step.output, Mapping)
    ]
    if len(judgment_steps) != 1:
        raise ValueError("complete hinted episode requires one teacher judgment")
    canonical_suffix = [
        _step_row(step, stage="unified_react", role="teacher")
        for step in teacher_steps
        if step.stage_name != "psd_teacher_judgment"
        and not _mapping(step.metadata).get("deterministic_segment_boundary")
    ]
    # Keep actual rejected/format-correction judgment decisions as well as the
    # accepted report. The reviewer must not see an artificially clean suffix.
    canonical_judgments = [_step_row(step, stage="unified_judgment", role="teacher")
        for step in teacher_steps if step.stage_name == "psd_teacher_judgment"
        and not _mapping(step.metadata).get("deterministic_segment_boundary")]
    all_steps: list[dict[str, Any]] = [
        copy.deepcopy(dict(step))
        for step in source_steps[:source_index]
        if isinstance(step, Mapping)
    ]
    all_steps.extend(canonical_suffix)

    runtime_case = _mapping(state.get("runtime_case"))
    investigation_source = _mapping(state.get("investigation_state"))
    case_id = _text(runtime_case.get("case_id") or investigation_source.get("case_id"))
    image_sha256 = _text(
        runtime_case.get("image_sha256") or investigation_source.get("image_sha256")
    )
    if not case_id or len(image_sha256) != 64:
        raise ValueError("PSD repair trace lacks a valid runtime case binding")
    investigation = UnifiedReactState(
        schema_version=REACT_RUNTIME_SCHEMA_VERSION,
        case_id=case_id,
        image_sha256=image_sha256,
        objective=_text(investigation_source.get("objective"))
        or UnifiedReactState.model_fields["objective"].default,
    )
    for step in all_steps:
        if (
            _text(step.get("stage")) == "unified_react"
            and _text(step.get("action_type")) == "tool_call"
        ):
            metadata = _mapping(step.get("metadata"))
            record_react_action(
                investigation,
                tool_name=_text(step.get("tool_name")),
                tool_args=_mapping(step.get("tool_args")),
                call_id=_text(metadata.get("function_call_id")) or "missing",
            )
    if not investigation.stop_reason:
        investigation.stop_reason = _text(stop_reason) or "psd_suffix_complete"

    basis = compile_react_judgment_basis(investigation, all_steps)
    raw_judgment = dict(judgment_steps[0].output or {})
    verdict_observation_ids = [
        _text(item)
        for item in raw_judgment.get("verdict_observation_ids", []) or []
        if _text(item)
    ]
    unknown = sorted(
        set(verdict_observation_ids) - set(basis.get("observation_ids", []))
    )
    if unknown:
        raise ValueError(
            "teacher judgment cites unknown observations: " + ",".join(unknown)
        )
    judgment = {
        "verdict": _text(raw_judgment.get("verdict")),
        "confidence": raw_judgment.get("confidence"),
        "policy_rule_id": UNIFIED_REACT_RUNTIME_POLICY_VERSION,
        "selected_observation_ids": list(basis.get("observation_ids", [])),
        "verdict_observation_ids": verdict_observation_ids,
        "overall_assessment": _text(raw_judgment.get("overall_assessment")),
        "fact_check_report": copy.deepcopy(raw_judgment.get("fact_check_report")),
        "evidence_citations": [],
    }
    if judgment["verdict"] not in {"real", "fake"}:
        raise ValueError("teacher judgment verdict is invalid")
    all_steps.extend(canonical_judgments)
    totals = _runtime_totals(all_steps)
    state.update(
        {
            "investigation_state": investigation.model_dump(mode="json"),
            "judgment": judgment,
            "all_steps": all_steps,
            "total_tool_calls": totals["total_tool_calls"],
            "total_tool_subcalls": totals["total_tool_subcalls"],
            "tool_subcalls_by_kind": totals["tool_subcalls_by_kind"],
            "llm_api_calls": totals["llm_api_calls"],
            "token_usage": totals["token_usage"],
            "termination": "success",
            "errors": [],
        }
    )
    trace.update(
        {
            "judgment": judgment,
            "verdict": judgment["verdict"],
            "confidence": judgment["confidence"],
            "overall_assessment": judgment["overall_assessment"],
            "fact_check_report": copy.deepcopy(judgment["fact_check_report"]),
            "evidence_citations": [],
            "stop_reason": investigation.stop_reason,
            "action_count": investigation.action_count,
            "verdict_basis": basis,
            "state": state,
            "termination": "success",
            "token_usage": totals["token_usage"],
            "total_tool_calls": totals["total_tool_calls"],
            "total_tool_subcalls": totals["total_tool_subcalls"],
            "tool_subcalls_by_kind": totals["tool_subcalls_by_kind"],
            "llm_api_calls": totals["llm_api_calls"],
            "error": None,
            "psd_repair": {
                "role": "teacher",
                "repair_step_id": _text(getattr(failure_site, "step_id", "")),
                "source_step_index": source_index,
                "terminal_state_replayed": True,
            },
        }
    )
    _assert_no_private_fields(trace)
    return trace


def _write_trace(trace: Mapping[str, Any]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="eviscene-psd-repair-verify-",
        delete=False,
        encoding="utf-8",
    )
    path = Path(handle.name)
    with handle:
        json.dump(trace, handle, ensure_ascii=False)
    return path


def validate_local_verification(
    value: Mapping[str, Any] | None,
    *,
    repair_step_id: str,
    source_trace_sha256: str,
    hint_sha256: str,
    teacher_prompt_sha256: str = "",
    teacher_completion_sha256: str = "",
) -> dict[str, Any]:
    """Validate a task-verifier artifact; model activity is never a proxy."""

    artifact = _mapping(value)
    errors: list[str] = []
    if artifact.get("schema_version") != PSD_LOCAL_VERIFICATION_SCHEMA_VERSION:
        errors.append("local_verifier_schema_invalid")
    if _text(artifact.get("repair_step_id")) != _text(repair_step_id):
        errors.append("local_verifier_repair_step_mismatch")
    if _text(artifact.get("source_trace_sha256")) != _text(
        source_trace_sha256
    ):
        errors.append("local_verifier_source_trace_mismatch")
    if _text(artifact.get("hint_sha256")) != _text(hint_sha256):
        errors.append("local_verifier_hint_mismatch")
    artifact_prompt_sha256 = _text(artifact.get("teacher_prompt_sha256"))
    artifact_completion_sha256 = _text(
        artifact.get("teacher_completion_sha256")
    )
    if not artifact_prompt_sha256 or not artifact_completion_sha256:
        errors.append("local_verifier_teacher_token_binding_missing")
    if teacher_prompt_sha256 and artifact_prompt_sha256 != teacher_prompt_sha256:
        errors.append("local_verifier_teacher_prompt_mismatch")
    if (
        teacher_completion_sha256
        and artifact_completion_sha256 != teacher_completion_sha256
    ):
        errors.append("local_verifier_teacher_completion_mismatch")
    verifier = _mapping(artifact.get("verifier"))
    if _text(verifier.get("kind")) != "task":
        errors.append("local_verifier_kind_not_task")
    if not _text(verifier.get("name")) or not _text(verifier.get("version")):
        errors.append("local_verifier_identity_missing")
    passed = artifact.get("passed")
    if not isinstance(passed, bool):
        errors.append("local_verifier_pass_not_boolean")
    checks = _rows(artifact.get("checks"))
    if not checks:
        errors.append("local_verifier_checks_missing")
    elif any(not isinstance(row.get("passed"), bool) for row in checks):
        errors.append("local_verifier_check_result_invalid")
    elif passed is True and any(row.get("passed") is not True for row in checks):
        errors.append("local_verifier_checks_disagree")
    evidence = artifact.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append("local_verifier_evidence_missing")
    return {
        "valid": not errors,
        "passed": bool(passed is True and not errors),
        "errors": errors,
        "verifier": {
            "kind": _text(verifier.get("kind")),
            "name": _text(verifier.get("name")),
            "version": _text(verifier.get("version")),
        },
        "repair_step_id": _text(artifact.get("repair_step_id")),
        "teacher_prompt_sha256": artifact_prompt_sha256,
        "teacher_completion_sha256": artifact_completion_sha256,
        "artifact_sha256": _sha(artifact) if artifact else "",
    }


def verify_source_rollout_failure(
    trace: Mapping[str, Any],
    *,
    gold: Mapping[str, Any],
    score_metadata: Mapping[str, Any] | None = None,
    source_task_review: Mapping[str, Any] | None = None,
    source_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove that the original no-hint on-policy rollout actually failed."""

    _assert_no_private_fields(trace)
    reasons: list[str] = []
    result_correct: bool | None = None
    expected_verdict = ""
    semantic_failure = False
    structural_failure = False
    try:
        metrics, _score = score_process_trace(
            trace,
            gold,
            score_metadata=score_metadata,
        )
        raw_result = metrics.get("result_correct")
        result_correct = raw_result if isinstance(raw_result, bool) else None
        expected_verdict = _text(metrics.get("expected_verdict"))
        if source_task_review is not None:
            from .psd_source_review import validate_source_review
            semantic_failure = validate_source_review(source_task_review, trace=trace, gold=gold) == "fail"
        if source_audit is not None:
            if (source_audit.get("source_trace_canonical_sha256") != _sha(trace)
                    or type(source_audit.get("passed")) is not bool
                    or not isinstance(source_audit.get("failures"), list)):
                raise ValueError("source audit trace binding invalid")
            structural_failure = source_audit["passed"] is False and bool(source_audit["failures"])
            if source_audit["passed"] == bool(source_audit["failures"]):
                raise ValueError("source audit result disagrees with failures")
        if result_correct is not False and not semantic_failure and not structural_failure:
            reasons.append("source_rollout_not_explicitly_failed")
    except Exception as exc:
        reasons.append(f"source_verification_error:{type(exc).__name__}:{exc}")
    return {
        "passed": (result_correct is False or semantic_failure or structural_failure) and not reasons,
        "semantic_failure": semantic_failure,
        "structural_failure": structural_failure,
        "result_correct": result_correct,
        "recorded_verdict": _text(trace.get("verdict")),
        "expected_verdict": expected_verdict,
        "reasons": reasons,
    }


def prepare_causal_episode_static(
    hinted_trace: Mapping[str, Any],
    *,
    source_trace: Mapping[str, Any],
    gold: Mapping[str, Any],
    source_trace_sha256: str,
    score_metadata: Mapping[str, Any] | None = None,
    source_access_policy: Any = None,
    source_task_review: Mapping[str, Any] | None = None,
    source_audit: Mapping[str, Any] | None = None,
    source_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute checks shared by every local target in one teacher episode.

    A slate can contain several corrected positions, but all of them bind to
    the same canonical episode.  Keep the strict gate intact while avoiding a
    full recursive audit and private-gold score for every position.
    """

    _assert_no_private_fields(hinted_trace)
    static: dict[str, Any] = {
        "source_result": dict(source_result or verify_source_rollout_failure(
            source_trace,
            gold=gold,
            score_metadata=score_metadata,
            source_task_review=source_task_review,
            source_audit=source_audit,
        )),
        "reasons": [],
        "strict_pass": False,
        "metrics": {},
        "expected_verdict": "",
        "recorded_verdict": _text(hinted_trace.get("verdict")),
        "has_report": False,
        "has_terminal_output": False,
        "source_trace_sha256": source_trace_sha256,
        "episode_sha256": _sha(hinted_trace),
    }
    path = _write_trace(hinted_trace)
    try:
        # Keep the legacy audit hook contract when no source policy is
        # supplied; the expensive file/recursive audit still runs only once
        # for the whole episode.
        report = (audit_trace(path, source_access_policy=source_access_policy)
                  if source_access_policy is not None else audit_trace(path))
        failures = report.failures(strict_scheduler=True)
        static["reasons"].extend(f"audit:{item.code}" for item in failures)
        static["strict_pass"] = not failures
        metrics, _score = score_process_trace(
            hinted_trace,
            gold,
            score_metadata=score_metadata,
        )
        static["metrics"] = metrics
        static["expected_verdict"] = _text(metrics.get("expected_verdict"))
        judgment = _mapping(
            hinted_trace.get("judgment")
            or _mapping(hinted_trace.get("state")).get("judgment")
        )
        static["has_report"] = isinstance(judgment.get("fact_check_report"), Mapping)
        static["has_terminal_output"] = any(
            str(step.get("stage", "")).strip() == "unified_judgment"
            and str(step.get("action_type", "")).strip() == "output"
            for step in _rows(_mapping(hinted_trace.get("state")).get("all_steps"))
        )
        if metrics.get("result_correct") is not True:
            static["reasons"].append("private_gold_verdict_mismatch")
        if metrics.get("engineering_error") is True:
            static["reasons"].append("engineering_error")
        if not static["has_terminal_output"]:
            static["reasons"].append("terminal_judgment_missing")
        if not static["has_report"]:
            static["reasons"].append("fact_check_report_missing")
    except Exception as exc:
        static["reasons"].append(f"verification_error:{type(exc).__name__}:{exc}")
    finally:
        path.unlink(missing_ok=True)
    return static


def verify_causal_episode(
    hinted_trace: Mapping[str, Any],
    *,
    source_trace: Mapping[str, Any],
    source_trace_sha256: str,
    gold: Mapping[str, Any],
    local_verification: Mapping[str, Any] | None,
    repair_step_id: str,
    hint_sha256: str,
    teacher_prompt_sha256: str = "",
    teacher_completion_sha256: str = "",
    downstream_patch_count: int = 0,
    score_metadata: Mapping[str, Any] | None = None,
    source_access_policy: Any = None,
    source_task_review: Mapping[str, Any] | None = None,
    source_audit: Mapping[str, Any] | None = None,
    cached_source_result: Mapping[str, Any] | None = None,
    cached_episode_static: Mapping[str, Any] | None = None,
) -> VerificationResult:
    """Verify one hinted frozen-policy continuation against its failed source.

    A bounded suffix or unhinted student retry is deliberately insufficient.
    The caller must supply the canonical complete episode generated by the
    frozen round-start policy with the hint, plus a real local task-verifier
    artifact bound to the selected repair step.
    """

    if cached_episode_static is None:
        static = prepare_causal_episode_static(
            hinted_trace,
            source_trace=source_trace,
            gold=gold,
            source_trace_sha256=source_trace_sha256,
            score_metadata=score_metadata,
            source_access_policy=source_access_policy,
            source_task_review=source_task_review,
            source_audit=source_audit,
            source_result=cached_source_result,
        )
    else:
        # This cache is created immediately before the per-target loop by the
        # slate assembler and is never persisted or accepted from callers.
        static = dict(cached_episode_static)
    source_result = dict(static["source_result"])
    local_result = validate_local_verification(
        local_verification,
        repair_step_id=repair_step_id,
        source_trace_sha256=source_trace_sha256,
        hint_sha256=hint_sha256,
        teacher_prompt_sha256=teacher_prompt_sha256,
        teacher_completion_sha256=teacher_completion_sha256,
    )
    reasons: list[str] = list(source_result["reasons"])
    reasons.extend(f"local:{item}" for item in local_result["errors"])
    if local_result["valid"] and not local_result["passed"]:
        reasons.append("hinted_local_verifier_failed")
    if _mapping(_mapping(local_verification).get("verifier")).get("name") == "gemini-psd-repair":
        artifact = _mapping(local_verification)
        if artifact.get("teacher_episode_canonical_sha256") != _sha(hinted_trace):
            local_result["passed"] = False
            reasons.append("local:teacher_episode_changed_after_review")
        if artifact.get("private_reference_sha256") != _sha(gold):
            local_result["passed"] = False
            reasons.append("local:private_reference_changed_after_review")
    recorded_verdict = _text(hinted_trace.get("verdict"))
    expected_verdict = ""
    strict_pass = False
    full_pass = False
    token_binding_pass = bool(
        teacher_prompt_sha256
        and teacher_completion_sha256
        and _trace_contains_token_binding(
            hinted_trace,
            teacher_prompt_sha256=teacher_prompt_sha256,
            teacher_completion_sha256=teacher_completion_sha256,
        )
    )
    if not teacher_prompt_sha256 or not teacher_completion_sha256:
        reasons.append("teacher_token_binding_missing")
    elif not token_binding_pass:
        reasons.append("hinted_episode_teacher_token_binding_mismatch")
    reasons.extend(static["reasons"])
    strict_pass = bool(static["strict_pass"])
    metrics = static["metrics"]
    expected_verdict = _text(static["expected_verdict"])
    recorded_verdict = _text(static["recorded_verdict"])
    has_report = bool(static["has_report"])
    has_terminal_output = bool(static["has_terminal_output"])
    full_pass = bool(
        strict_pass
        and metrics.get("result_correct") is True
        and has_report
        and has_terminal_output
        and recorded_verdict in {"real", "fake"}
        and token_binding_pass
    )
    return verify_repair(
        source_rollout_failed=source_result["passed"],
        hinted_local_pass=local_result["passed"],
        hinted_recorded_verdict=recorded_verdict,
        expected_verdict=expected_verdict,
        hinted_strict_trace_audit_pass=strict_pass,
        hinted_episode_pass=full_pass,
        downstream_patch_count=downstream_patch_count,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def verify_continuation_pair(
    *,
    base_trace: Mapping[str, Any],
    source_trace_sha256: str,
    teacher_steps: Sequence[StageStep],
    student_steps: Sequence[StageStep],
    hinted_teacher_episode_trace: Mapping[str, Any] | None,
    unhinted_student_episode_trace: Mapping[str, Any] | None,
    gold: Mapping[str, Any],
    local_verification: Mapping[str, Any] | None,
    repair_step_id: str,
    hint_sha256: str,
    source_access_policy: Any = None,
    source_task_review: Mapping[str, Any] | None = None,
    source_audit: Mapping[str, Any] | None = None,
) -> tuple[VerificationResult, dict[str, Any]]:
    """Verify the hinted teacher; keep the unhinted student as diagnostics."""

    teacher_trace = merge_suffix_into_trace(
        base_trace,
        teacher_steps,
        role="teacher",
    )
    student_suffix_trace = (
        merge_suffix_into_trace(
            base_trace,
            student_steps,
            role="student",
        )
        if student_steps
        else None
    )
    teacher_prompt_sha256, teacher_completion_sha256 = _teacher_step_binding(
        teacher_steps
    )
    if hinted_teacher_episode_trace is None:
        source_result = verify_source_rollout_failure(base_trace, gold=gold,
            source_task_review=source_task_review, source_audit=source_audit)
        local_result = validate_local_verification(
            local_verification,
            repair_step_id=repair_step_id,
            source_trace_sha256=source_trace_sha256,
            hint_sha256=hint_sha256,
            teacher_prompt_sha256=teacher_prompt_sha256,
            teacher_completion_sha256=teacher_completion_sha256,
        )
        reasons = [*source_result["reasons"]]
        reasons.extend(f"local:{item}" for item in local_result["errors"])
        reasons.append("full_hinted_teacher_episode_trace_required")
        result = verify_repair(
            source_rollout_failed=source_result["passed"],
            hinted_local_pass=local_result["passed"],
            hinted_recorded_verdict="",
            expected_verdict="",
            hinted_strict_trace_audit_pass=False,
            hinted_episode_pass=False,
            reasons=reasons,
        )
    else:
        result = verify_causal_episode(
            hinted_teacher_episode_trace,
            source_trace=base_trace,
            source_trace_sha256=source_trace_sha256,
            gold=gold,
            local_verification=local_verification,
            repair_step_id=repair_step_id,
            hint_sha256=hint_sha256,
            teacher_prompt_sha256=teacher_prompt_sha256,
            teacher_completion_sha256=teacher_completion_sha256,
            source_access_policy=source_access_policy,
            source_task_review=source_task_review,
            source_audit=source_audit,
        )
    student_diagnostic = None
    if unhinted_student_episode_trace is not None:
        student_diagnostic = verify_source_rollout_failure(
            unhinted_student_episode_trace,
            gold=gold,
        )
    return result, {
        "teacher_suffix_trace": teacher_trace,
        "student_suffix_trace": student_suffix_trace,
        "hinted_teacher_episode_verified": hinted_teacher_episode_trace is not None,
        "teacher_prompt_sha256": teacher_prompt_sha256,
        "teacher_completion_sha256": teacher_completion_sha256,
        "unhinted_student_diagnostic": student_diagnostic,
        "unhinted_student_affects_acceptance": False,
    }
