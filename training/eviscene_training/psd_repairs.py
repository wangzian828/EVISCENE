"""Assemble verifier-gated PSD repairs and base-pass preservation rows.

This module joins public repair candidates with privileged repair attempts.
Only a fully verified repair is allowed through, and the emitted student
prefix comes from the original no-hint capture or an explicitly verified,
corrected hint-free slate prefix. These lineages must never be conflated.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .io import (
    canonical_json,
    iter_jsonl,
    load_jsonl,
    require_new_or_empty,
    sha256_file,
    write_json,
    write_jsonl,
)
from .psd import (TRAINABLE_HINT_LEVELS, audit_hint, summarize_round_model_roles,
                  validate_topk_by_position)


PSD_REPAIR_SCHEMA_VERSION = "ifv-psd-repair-v1"
PSD_PRESERVATION_SCHEMA_VERSION = "ifv-psd-preservation-v1"
PSD_REPAIR_ASSEMBLY_MANIFEST_SCHEMA_VERSION = "ifv-psd-repair-assembly-manifest-v1"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _required_int_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{field}[{index}] must be an integer")
        try:
            result.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}[{index}] must be an integer") from exc
    return result


def _positive_weight(value: Any, *, field: str) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return weight


def _sha_ids(value: list[int]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _source_trace_sha256(row: Mapping[str, Any]) -> str:
    return _text(
        row.get("source_trace_sha256")
        or _mapping(row.get("source")).get("source_trace_sha256")
    )


def _source_fields(candidate: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(candidate.get("source"))
    return {
        "source_run_id": _text(
            candidate.get("source_run_id") or source.get("source_run_id")
        ),
        "runtime_commit": _text(
            candidate.get("runtime_commit") or source.get("runtime_commit")
        ),
        "source_trace_sha256": _source_trace_sha256(candidate),
        **({"source_task_review": source["source_task_review"]} if source.get("source_task_review") else {}),
        "psd_round_index": candidate.get("psd_round_index")
        or source.get("psd_round_index"),
        "psd_rollout_gate_sha256": _text(
            candidate.get("psd_rollout_gate_sha256")
            or source.get("psd_rollout_gate_sha256")
        ),
    }


def _candidate_step(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(candidate.get("repair_site"))


def _candidate_student_prompt_ids(candidate: Mapping[str, Any]) -> list[int]:
    if candidate.get("student_prefix_capture"):
        from .psd_slate import validate_prefix_lineage
        return _required_int_list(validate_prefix_lineage(candidate), field="corrected student prefix")
    capture = _mapping(_candidate_step(candidate).get("rollout_token_capture"))
    if _text(capture.get("status")) != "complete":
        raise ValueError("candidate rollout_token_capture is not complete")
    return _required_int_list(
        capture.get("prompt_token_ids"),
        field="candidate.rollout_token_capture.prompt_token_ids",
    )


def _attempt_token_ids(
    attempt: Mapping[str, Any],
) -> tuple[list[int], list[int]]:
    capture = _mapping(
        attempt.get("repair_rollout_token_capture")
        or attempt.get("rollout_token_capture")
    )
    if capture:
        if _text(capture.get("status")) != "complete":
            raise ValueError("repair rollout token capture is not complete")
        teacher_prompt_ids = _required_int_list(
            capture.get("prompt_token_ids"),
            field="repair_rollout_token_capture.prompt_token_ids",
        )
        completion_ids = _required_int_list(
            capture.get("completion_token_ids"),
            field="repair_rollout_token_capture.completion_token_ids",
        )
        return teacher_prompt_ids, completion_ids
    return (
        _required_int_list(
            attempt.get("teacher_prompt_ids"),
            field="teacher_prompt_ids",
        ),
        _required_int_list(
            attempt.get("completion_ids"),
            field="completion_ids",
        ),
    )


def _captured_topk(
    capture: Mapping[str, Any],
    *,
    completion_ids: list[int],
) -> list[Any] | None:
    from .psd_capture_semantics import RAW_POLICY_LOGPROBS
    if capture.get("teacher_logprob_semantics") != RAW_POLICY_LOGPROBS:
        # Preserve the original tokens/episode. Unknown or grammar-masked
        # distributions need exact forced-token scoring, never relabeling.
        return None
    value = capture.get("completion_topk_by_position")
    if value in (None, []):
        return None
    topk = capture.get("topk")
    if isinstance(topk, bool):
        raise ValueError("captured_topk_invalid")
    try:
        topk_value = int(topk)
    except (TypeError, ValueError) as exc:
        raise ValueError("captured_topk_invalid") from exc
    if topk_value != 20:
        raise ValueError(f"captured_topk_must_be_20:{topk_value}")
    validate_topk_by_position(completion_ids, value, topk=topk_value)
    return list(value)


def _attempt_rejection(
    *,
    row_index: int,
    attempt: Mapping[str, Any],
    reason: str,
    candidate_id: str = "",
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "candidate_id": candidate_id or _text(attempt.get("candidate_id")),
        "attempt_id": _text(attempt.get("attempt_id")),
        "reason": reason,
    }


def _validate_attempt(
    *,
    attempt: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    if _text(attempt.get("schema_version")) != "ifv-psd-repair-attempt-v1":
        raise ValueError("repair_attempt_schema_invalid")
    candidate_id = _text(candidate.get("candidate_id"))
    attempt_id = _text(attempt.get("attempt_id"))
    if not attempt_id:
        raise ValueError("attempt_id_missing")
    if _text(attempt.get("candidate_id")) != candidate_id:
        raise ValueError("candidate_id_mismatch")

    case_id = _text(candidate.get("case_id"))
    episode_id = _text(candidate.get("episode_id"))
    repair_site = _candidate_step(candidate)
    step_id = _text(repair_site.get("step_id"))
    source_trace_sha256 = _source_trace_sha256(candidate)
    if not case_id or not episode_id or not step_id or not source_trace_sha256:
        raise ValueError("candidate_identity_incomplete")
    source_fields = _source_fields(candidate)
    round_index = source_fields.get("psd_round_index")
    if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 1:
        raise ValueError("candidate_psd_round_invalid")
    rollout_gate_sha256 = _text(source_fields.get("psd_rollout_gate_sha256")).casefold()
    if len(rollout_gate_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in rollout_gate_sha256
    ):
        raise ValueError("candidate_psd_rollout_gate_invalid")
    if _text(attempt.get("case_id")) != case_id:
        raise ValueError("case_id_mismatch")
    if _text(attempt.get("episode_id")) != episode_id:
        raise ValueError("episode_id_mismatch")
    if _text(
        attempt.get("repair_step_id")
        or _mapping(attempt.get("repair_site")).get("step_id")
    ) != step_id:
        raise ValueError("repair_step_id_mismatch")
    if _source_trace_sha256(attempt) != source_trace_sha256:
        raise ValueError("source_trace_sha256_mismatch")

    if attempt.get("accepted") is not True:
        raise ValueError("attempt_not_accepted")
    if candidate.get("student_prefix_capture") or attempt.get("student_prefix_capture"):
        from .psd_slate import validate_prefix_lineage
        validate_prefix_lineage(candidate, attempt)
    repair_tier = _text(attempt.get("repair_tier") or attempt.get("tier"))
    if repair_tier != "causal_episode_pass":
        raise ValueError("repair_tier_not_causal_episode_pass")
    verification = _mapping(attempt.get("verification"))
    if verification.get("source_rollout_failed") is not True:
        raise ValueError("verification_source_rollout_failure_required")
    if verification.get("hinted_local_pass") is not True:
        raise ValueError("verification_hinted_local_pass_required")
    if verification.get("hinted_episode_pass") is not True:
        raise ValueError("verification_hinted_episode_pass_required")
    if verification.get("hinted_strict_trace_audit_pass") is not True:
        raise ValueError("verification_hinted_strict_trace_audit_pass_required")

    model_roles = _mapping(attempt.get("model_roles"))
    constructor = _mapping(model_roles.get("hint_constructor"))
    teacher = _mapping(model_roles.get("frozen_self_teacher"))
    student = _mapping(model_roles.get("trainable_student"))
    if constructor.get("supplies_training_distribution") is not False:
        raise ValueError("hint_constructor_distribution_must_not_train_student")
    if teacher.get("supplies_training_distribution") is not True:
        raise ValueError("frozen_self_teacher_distribution_required")
    if teacher.get("sees_hint") is not True:
        raise ValueError("frozen_self_teacher_must_see_hint")
    if student.get("sees_hint") is not False:
        raise ValueError("trainable_student_must_not_see_hint")
    if _text(teacher.get("round_start_checkpoint")) != _text(
        student.get("initial_checkpoint")
    ):
        raise ValueError("self_teacher_student_checkpoint_mismatch")
    if _text(teacher.get("model")) != _text(student.get("model")):
        raise ValueError("self_teacher_student_policy_mismatch")
    if _text(teacher.get("provider")).casefold() != _text(
        student.get("provider")
    ).casefold():
        raise ValueError("self_teacher_student_provider_mismatch")
    teacher_manifest_sha256 = _text(
        teacher.get("checkpoint_manifest_sha256")
    ).casefold()
    if (
        len(teacher_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in teacher_manifest_sha256)
        or teacher_manifest_sha256
        != _text(student.get("checkpoint_manifest_sha256")).casefold()
    ):
        raise ValueError("self_teacher_student_checkpoint_manifest_mismatch")

    hint_record = _mapping(attempt.get("hint_record"))
    hint = _text(hint_record.get("text") or attempt.get("hint"))
    raw_level = hint_record.get("level", attempt.get("hint_level"))
    try:
        hint_level = int(raw_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("hint_level_invalid") from exc
    if hint_level not in TRAINABLE_HINT_LEVELS:
        raise ValueError("hint_level_not_trainable")
    hint_audit = audit_hint(attempt, hint=hint, hint_level=hint_level)
    if not hint_audit["passed"]:
        raise ValueError(
            "hint_audit_failed:" + ",".join(hint_audit["errors"])
        )

    local_verification = _mapping(attempt.get("local_verification"))
    if (
        local_verification.get("schema_version")
        != "ifv-psd-local-verification-v1"
    ):
        raise ValueError("local_verification_schema_invalid")
    if local_verification.get("passed") is not True:
        raise ValueError("local_verification_pass_required")
    if _text(local_verification.get("repair_step_id")) != step_id:
        raise ValueError("local_verification_repair_step_mismatch")
    if _text(local_verification.get("source_trace_sha256")) != source_trace_sha256:
        raise ValueError("local_verification_source_trace_mismatch")
    if _text(local_verification.get("hint_sha256")) != hint_audit["hint_sha256"]:
        raise ValueError("local_verification_hint_mismatch")
    local_verifier = _mapping(local_verification.get("verifier"))
    if (
        _text(local_verifier.get("kind")) != "task"
        or not _text(local_verifier.get("name"))
        or not _text(local_verifier.get("version"))
    ):
        raise ValueError("local_task_verifier_identity_invalid")
    local_checks = local_verification.get("checks")
    if not isinstance(local_checks, list) or not local_checks:
        raise ValueError("local_verification_checks_missing")
    if any(
        not isinstance(check, Mapping) or check.get("passed") is not True
        for check in local_checks
    ):
        raise ValueError("local_verification_check_failed")
    local_evidence = local_verification.get("evidence")
    if not isinstance(local_evidence, list) or not local_evidence:
        raise ValueError("local_verification_evidence_missing")

    student_prompt_ids = _candidate_student_prompt_ids(candidate)
    teacher_prompt_ids, completion_ids = _attempt_token_ids(attempt)
    if teacher_prompt_ids == student_prompt_ids:
        raise ValueError("teacher_prompt_equals_student_prompt")
    if _text(local_verification.get("teacher_prompt_sha256")) != _sha_ids(
        teacher_prompt_ids
    ):
        raise ValueError("local_verification_teacher_prompt_mismatch")
    if _text(local_verification.get("teacher_completion_sha256")) != _sha_ids(
        completion_ids
    ):
        raise ValueError("local_verification_teacher_completion_mismatch")
    capture = _mapping(
        attempt.get("repair_rollout_token_capture")
        or attempt.get("rollout_token_capture")
    )
    captured_topk = (
        _captured_topk(capture, completion_ids=completion_ids)
        if capture
        else None
    )

    return {
        "attempt_id": attempt_id,
        "hint": hint,
        "hint_level": hint_level,
        "hint_audit": hint_audit,
        "student_prompt_ids": student_prompt_ids,
        "teacher_prompt_ids": teacher_prompt_ids,
        **({"psd_media": dict(attempt["psd_media"])} if attempt.get("psd_media") else {}),
        "completion_ids": completion_ids,
        "teacher_topk_by_position": captured_topk,
        "teacher_logprob_semantics": capture.get("teacher_logprob_semantics", ""),
        "row_weight": _positive_weight(
            attempt.get("row_weight", 1.0),
            field="row_weight",
        ),
        "verification": dict(verification),
        "model_roles": dict(model_roles),
        "local_verification": dict(local_verification),
        **({"student_prefix_capture": candidate["student_prefix_capture"]}
           if candidate.get("student_prefix_capture") else {}),
    }


def _repair_row(
    *,
    candidate: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    repair_site = _candidate_step(candidate)
    source = _source_fields(candidate)
    return {
        "schema_version": PSD_REPAIR_SCHEMA_VERSION,
        "candidate_id": _text(candidate.get("candidate_id")),
        "attempt_id": _text(selected.get("attempt_id")),
        "class": "verified_privileged_repair",
        "case_id": _text(candidate.get("case_id")),
        "episode_id": _text(candidate.get("episode_id")),
        "repair_step_id": _text(repair_site.get("step_id")),
        "stage": _text(repair_site.get("stage")),
        "example_type": _text(repair_site.get("example_type")),
        "accepted": True,
        **({"student_prefix_capture": selected["student_prefix_capture"]}
           if selected.get("student_prefix_capture") else {}),
        "repair_tier": "causal_episode_pass",
        "hint": _text(selected.get("hint")),
        "hint_level": int(selected["hint_level"]),
        "hint_audit": dict(_mapping(selected.get("hint_audit"))),
        "student_prompt_ids": list(selected["student_prompt_ids"]),
        "teacher_prompt_ids": list(selected["teacher_prompt_ids"]),
        **({"psd_media": dict(selected["psd_media"])} if selected.get("psd_media") else {}),
        "completion_ids": list(selected["completion_ids"]),
        **(
            {
                "teacher_logprob_semantics": selected.get("teacher_logprob_semantics", ""),
                "teacher_topk_by_position": list(
                    selected["teacher_topk_by_position"]
                )
            }
            if selected.get("teacher_topk_by_position") is not None
            else {}
        ),
        "row_weight": float(selected["row_weight"]),
        "verification": dict(_mapping(selected.get("verification"))),
        "model_roles": dict(_mapping(selected.get("model_roles"))),
        "local_verification": dict(
            _mapping(selected.get("local_verification"))
        ),
        **source,
    }


def _preservation_row(
    candidate: Mapping[str, Any],
    *,
    model_roles: Mapping[str, Any],
    media_dir: Path | None = None,
) -> dict[str, Any]:
    if _text(candidate.get("class")) != "base_pass_preserve":
        raise ValueError("preservation_candidate_class_invalid")
    if candidate.get("verified_full_task") is not True:
        raise ValueError("preservation_candidate_not_verified")
    if candidate.get("strict_trace_audit_pass") is not True:
        raise ValueError("preservation_strict_trace_audit_required")
    raw_steps = candidate.get("preservation_steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("preservation_steps_missing")

    steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_steps):
        step = _mapping(raw_step)
        capture = _mapping(step.get("rollout_token_capture"))
        if _text(capture.get("status")) != "complete":
            raise ValueError(
                f"preservation_step_{index}_token_capture_not_complete"
            )
        prompt_ids = _required_int_list(
            capture.get("prompt_token_ids"),
            field=f"preservation_steps[{index}].prompt_token_ids",
        )
        completion_ids = _required_int_list(
            capture.get("completion_token_ids"),
            field=f"preservation_steps[{index}].completion_token_ids",
        )
        media = {}
        if 248056 in prompt_ids:
            from .psd_media import media_from_archive
            if media_dir is None:
                raise ValueError("visual preservation requires a media output directory")
            media = media_from_archive(
                step, processor_path=model_roles["frozen_self_teacher"]["round_start_checkpoint"],
                output_dir=media_dir, prompt_ids=prompt_ids,
            )
        step_id = _text(step.get("step_id"))
        if not step_id:
            raise ValueError(f"preservation_step_{index}_step_id_missing")
        steps.append(
            {
                "step_id": step_id,
                "stage": _text(step.get("stage")),
                "example_type": _text(step.get("example_type")),
                "student_prompt_ids": prompt_ids,
                **({"psd_media": media} if media else {}),
                "completion_ids": completion_ids,
                **(
                    {"teacher_topk_by_position": captured_topk,
                     "teacher_logprob_semantics": capture.get("teacher_logprob_semantics", "")}
                    if (
                        captured_topk := _captured_topk(
                            capture,
                            completion_ids=completion_ids,
                        )
                    )
                    is not None
                    else {}
                ),
            }
        )

    return {
        "schema_version": PSD_PRESERVATION_SCHEMA_VERSION,
        "candidate_id": _text(candidate.get("candidate_id")),
        "class": "base_pass_preserve",
        "case_id": _text(candidate.get("case_id")),
        "episode_id": _text(candidate.get("episode_id")),
        "verified_full_task": True,
        "strict_trace_audit_pass": True,
        "row_weight": _positive_weight(
            candidate.get("row_weight", 1.0),
            field="row_weight",
        ),
        "model_roles": dict(model_roles),
        "preservation_steps": steps,
        **_source_fields(candidate),
    }


def assemble_psd_repair_package(
    *,
    repair_candidates_path: Path,
    repair_attempts_path: Path,
    preservation_candidates_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Select the weakest verified repair and materialize preservation rows."""

    require_new_or_empty(output_dir)
    repair_candidates = load_jsonl(repair_candidates_path)
    repair_attempts = load_jsonl(repair_attempts_path)

    candidates_by_id: dict[str, Mapping[str, Any]] = {}
    for index, candidate in enumerate(repair_candidates):
        candidate_id = _text(candidate.get("candidate_id"))
        if not candidate_id:
            raise ValueError(f"repair candidate row {index} lacks candidate_id")
        if candidate_id in candidates_by_id:
            raise ValueError(f"duplicate repair candidate_id: {candidate_id}")
        candidates_by_id[candidate_id] = candidate

    attempts_by_candidate: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    attempt_rejections: list[dict[str, Any]] = []
    seen_attempt_ids: set[str] = set()
    for row_index, attempt in enumerate(repair_attempts):
        attempt_id = _text(attempt.get("attempt_id"))
        candidate_id = _text(attempt.get("candidate_id"))
        if not attempt_id:
            attempt_rejections.append(
                _attempt_rejection(
                    row_index=row_index,
                    attempt=attempt,
                    reason="attempt_id_missing",
                )
            )
            continue
        if attempt_id in seen_attempt_ids:
            attempt_rejections.append(
                _attempt_rejection(
                    row_index=row_index,
                    attempt=attempt,
                    reason="duplicate_attempt_id",
                )
            )
            continue
        seen_attempt_ids.add(attempt_id)
        if candidate_id not in candidates_by_id:
            attempt_rejections.append(
                _attempt_rejection(
                    row_index=row_index,
                    attempt=attempt,
                    reason="candidate_not_found",
                )
            )
            continue
        attempts_by_candidate[candidate_id].append((row_index, attempt))

    repairs: list[dict[str, Any]] = []
    unrepaired: list[dict[str, Any]] = []
    for candidate_id, candidate in candidates_by_id.items():
        valid: list[tuple[tuple[int, int, str], int, Mapping[str, Any]]] = []
        candidate_reasons: Counter[str] = Counter()
        for row_index, attempt in attempts_by_candidate.get(candidate_id, []):
            try:
                normalized = _validate_attempt(
                    attempt=attempt,
                    candidate=candidate,
                )
            except ValueError as exc:
                reason = str(exc)
                candidate_reasons[reason] += 1
                attempt_rejections.append(
                    _attempt_rejection(
                        row_index=row_index,
                        attempt=attempt,
                        candidate_id=candidate_id,
                        reason=reason,
                    )
                )
                continue
            rank = (
                int(normalized["hint_level"]),
                len(_text(normalized.get("hint"))),
                _text(normalized.get("attempt_id")),
            )
            valid.append((rank, row_index, normalized))

        if not valid:
            unrepaired.append(
                {
                    "candidate_id": candidate_id,
                    "case_id": _text(candidate.get("case_id")),
                    "episode_id": _text(candidate.get("episode_id")),
                    "repair_step_id": _text(_candidate_step(candidate).get("step_id")),
                    "reason": (
                        "no_attempts"
                        if not attempts_by_candidate.get(candidate_id)
                        else "no_verified_attempt"
                    ),
                    "attempt_reasons": dict(sorted(candidate_reasons.items())),
                }
            )
            continue

        valid.sort(key=lambda item: item[0])
        _, _, selected = valid[0]
        repairs.append(_repair_row(candidate=candidate, selected=selected))
        for _, row_index, normalized in valid[1:]:
            attempt_rejections.append(
                {
                    "row_index": row_index,
                    "candidate_id": candidate_id,
                    "attempt_id": _text(normalized.get("attempt_id")),
                    "reason": "superseded_by_weaker_or_shorter_verified_hint",
                    "selected_attempt_id": _text(selected.get("attempt_id")),
                }
            )

    try:
        round_model_roles = summarize_round_model_roles(
            [row["model_roles"] for row in repairs])
    except ValueError as error:
        raise ValueError(
            "selected repairs do not share one round-start policy") from error
    repair_source_runs = {
        _text(row.get("source_run_id")) for row in repairs
    } - {""}
    if len(repair_source_runs) > 1:
        raise ValueError("selected repairs span multiple source rollout runs")
    repair_rollout_gates = {
        _text(row.get("psd_rollout_gate_sha256")) for row in repairs
    } - {""}
    if len(repair_rollout_gates) != 1 and repairs:
        raise ValueError("selected repairs do not share one PSD rollout gate")
    repair_rounds = {row.get("psd_round_index") for row in repairs}
    if len(repair_rounds) != 1 and repairs:
        raise ValueError("selected repairs do not share one PSD round")

    preservation_candidate_count = 0
    preservation_row_count = 0
    preservation_rejection_count = 0
    seen_preservation_ids: set[str] = set()
    # Use the first verified trace per case in frozen source order.
    # Multiple rollout slots must not multiply preservation weight.
    selected_preservation_cases: set[str] = set()
    preservation_output = output_dir / "preservation.jsonl"
    preservation_rejections_output = output_dir / "preservation_rejections.jsonl"
    with (preservation_output.open("w", encoding="utf-8") as preservation_handle,
          preservation_rejections_output.open("w", encoding="utf-8") as rejection_handle):
        for row_index, candidate in enumerate(iter_jsonl(preservation_candidates_path)):
            preservation_candidate_count += 1
            candidate_id = _text(candidate.get("candidate_id"))
            case_id = _text(candidate.get("case_id"))
            rejection = None
            if not candidate_id:
                rejection = {"row_index": row_index, "reason": "candidate_id_missing"}
            elif candidate_id in seen_preservation_ids:
                rejection = {"row_index": row_index, "candidate_id": candidate_id,
                    "reason": "duplicate_candidate_id"}
            elif case_id in selected_preservation_cases:
                rejection = {"row_index": row_index, "candidate_id": candidate_id,
                    "case_id": case_id,
                    "reason": "superseded_by_first_verified_case_preservation"}
            else:
                seen_preservation_ids.add(candidate_id)
                try:
                    if round_model_roles is None:
                        raise ValueError("round_start_model_roles_missing")
                    candidate_run = _text(candidate.get("source_run_id")
                        or _mapping(candidate.get("source")).get("source_run_id"))
                    if repair_source_runs and candidate_run not in repair_source_runs:
                        raise ValueError("preservation_source_run_mismatch")
                    candidate_source = _source_fields(candidate)
                    if repair_rollout_gates and _text(
                            candidate_source.get("psd_rollout_gate_sha256")) not in repair_rollout_gates:
                        raise ValueError("preservation_rollout_gate_mismatch")
                    if repair_rounds and candidate_source.get("psd_round_index") not in repair_rounds:
                        raise ValueError("preservation_psd_round_mismatch")
                    row = _preservation_row(candidate, model_roles=round_model_roles,
                        media_dir=output_dir / "media")
                    preservation_handle.write(canonical_json(row) + "\n")
                    preservation_row_count += 1
                    selected_preservation_cases.add(case_id)
                except ValueError as exc:
                    rejection = {"row_index": row_index, "candidate_id": candidate_id,
                        "case_id": _text(candidate.get("case_id")),
                        "episode_id": _text(candidate.get("episode_id")), "reason": str(exc)}
            if rejection is not None:
                rejection_handle.write(canonical_json(rejection) + "\n")
                preservation_rejection_count += 1

    write_jsonl(output_dir / "repairs.jsonl", repairs)
    write_jsonl(output_dir / "attempt_rejections.jsonl", attempt_rejections)
    write_jsonl(output_dir / "unrepaired_candidates.jsonl", unrepaired)

    selected_levels = Counter(row["hint_level"] for row in repairs)
    manifest = {
        "schema_version": PSD_REPAIR_ASSEMBLY_MANIFEST_SCHEMA_VERSION,
        "source": {
            "repair_candidates": str(repair_candidates_path),
            "repair_candidates_sha256": sha256_file(repair_candidates_path),
            "repair_attempts": str(repair_attempts_path),
            "repair_attempts_sha256": sha256_file(repair_attempts_path),
            "preservation_candidates": str(preservation_candidates_path),
            "preservation_candidates_sha256": sha256_file(
                preservation_candidates_path
            ),
        },
        "counts": {
            "repair_candidates": len(repair_candidates),
            "repair_attempts": len(repair_attempts),
            "selected_repairs": len(repairs),
            "unrepaired_candidates": len(unrepaired),
            "attempt_rejections": len(attempt_rejections),
            "preservation_candidates": preservation_candidate_count,
            "preservation_rows": preservation_row_count,
            "preservation_rejections": preservation_rejection_count,
            "preservation_unique_cases": len(selected_preservation_cases),
        },
        "preservation_selection": "first_verified_row_per_case_in_frozen_source_order",
        "selected_hint_levels": {
            str(level): count for level, count in sorted(selected_levels.items())
        },
        "artifacts": {
            "repairs": "repairs.jsonl",
            "preservation": "preservation.jsonl",
            "attempt_rejections": "attempt_rejections.jsonl",
            "unrepaired_candidates": "unrepaired_candidates.jsonl",
            "preservation_rejections": "preservation_rejections.jsonl",
        },
        "status": (
            "ready_for_target_build"
            if repairs and preservation_row_count
            else "blocked_missing_source_kind"
            if repairs or preservation_candidate_count
            else "empty"
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
