"""Build public PSD candidate queues from postprocessed Qwen rollouts.

This is deliberately an intermediate layer.  A failed rollout is not itself a
PSD target and this module never claims that its final action is the causal
error.  It records the real model-visible state at a conservative repair
anchor, then leaves causal localization, privileged hinting, teacher decoding,
and verifier-gated repair to later stages.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from .io import (
    canonical_json,
    load_json,
    load_jsonl,
    require_new_or_empty,
    sha256_file,
    write_json,
    write_jsonl,
)
from .psd_round import validate_rollout_gate_for_candidates


PSD_CANDIDATE_SCHEMA_VERSION = "ifv-psd-candidate-v1"
PSD_CANDIDATE_MANIFEST_SCHEMA_VERSION = "ifv-psd-candidate-manifest-v2"

_FORBIDDEN_PRIVATE_KEYS = frozenset(
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
_REJECTED_ACTION_TYPES = frozenset(
    {"format_error", "output_rejected", "policy_replan"}
)
_STAGE_EXAMPLE_TYPES = {
    "unified_react": "react",
    "unified_judgment": "judgment",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[Mapping[str, Any]]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _assert_public(value: Any, *, path: str = "") -> None:
    """Reject evaluator/private state before it becomes a candidate artifact."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            if key.casefold() in _FORBIDDEN_PRIVATE_KEYS:
                raise ValueError(f"private/evaluator field in policy snapshot: {child_path}")
            _assert_public(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_public(child, path=f"{path}[{index}]")


def _int_list(value: Any) -> list[int]:
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


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        return []
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            return []
        try:
            parsed = float(item)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(parsed):
            return []
        result.append(parsed)
    return result


def _policy_token_capture(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only a complete numeric Qwen capture for later PSD token binding."""

    raw = metadata.get("policy_token_capture")
    if not isinstance(raw, Mapping):
        return {
            "status": "missing",
            "missing": ["policy_token_capture"],
        }
    prompt_token_ids = _int_list(raw.get("prompt_token_ids"))
    completion_token_ids = _int_list(raw.get("completion_token_ids"))
    completion_logprobs = _float_list(raw.get("completion_logprobs"))
    topk = raw.get("completion_topk_by_position")
    if not isinstance(topk, list):
        topk = []
    missing: list[str] = []
    if _text(raw.get("status")) != "complete":
        missing.append("capture_status_not_complete")
    if not prompt_token_ids:
        missing.append("prompt_token_ids")
    if not completion_token_ids:
        missing.append("completion_token_ids")
    if not completion_logprobs:
        missing.append("completion_logprobs")
    elif len(completion_logprobs) != len(completion_token_ids):
        missing.append("completion_logprob_length_mismatch")
    if topk and len(topk) != len(completion_token_ids):
        missing.append("completion_topk_length_mismatch")
    if missing:
        return {
            "status": "incomplete",
            "missing": sorted(set(missing)),
        }
    return {
        "status": "complete",
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": completion_token_ids,
        "completion_logprobs": completion_logprobs,
        "completion_topk_by_position": topk,
        "teacher_logprob_semantics": _text(raw.get("teacher_logprob_semantics")),
        "topk": int(raw.get("topk", 0) or 0),
    }


def _stable_step_id(
    *,
    episode_id: str,
    example_type: str,
    source_step_index: int,
    interaction_id: str,
) -> str:
    return (
        f"{episode_id}:{example_type}:{interaction_id}"
        if interaction_id
        else f"{episode_id}:{example_type}:{source_step_index + 1}"
    )


def _policy_steps(
    trace: Mapping[str, Any],
    *,
    episode_id: str,
) -> list[dict[str, Any]]:
    """Project all usable policy turns, including rejected protocol turns."""

    state = _mapping(trace.get("state"))
    projected: list[dict[str, Any]] = []
    for source_step_index, raw_step in enumerate(_rows(state.get("all_steps"))):
        step = _mapping(raw_step)
        example_type = _STAGE_EXAMPLE_TYPES.get(_text(step.get("stage")))
        metadata = _mapping(step.get("metadata"))
        policy_input = metadata.get("policy_input")
        policy_action = metadata.get("policy_action")
        if (
            example_type is None
            or not isinstance(policy_input, Mapping)
            or not isinstance(policy_action, Mapping)
        ):
            continue
        _assert_public(policy_input, path="policy_input")
        _assert_public(policy_action, path="policy_action")
        action_type = _text(step.get("action_type"))
        protocol_rejected = (
            action_type in _REJECTED_ACTION_TYPES
            or _text(metadata.get("error_class")) == "protocol_error"
            or bool(_text(metadata.get("rejection_reason")))
        )
        interaction_id = _text(metadata.get("interaction_id"))
        projected.append(
            {
                "step_id": _stable_step_id(
                    episode_id=episode_id,
                    example_type=example_type,
                    source_step_index=source_step_index,
                    interaction_id=interaction_id,
                ),
                "source_step_index": source_step_index,
                "stage": _text(step.get("stage")),
                "example_type": example_type,
                "interaction_id": interaction_id,
                "context_request_id": _text(metadata.get("context_request_id")),
                "runtime_store_path": _trace_runtime_store_path(trace),
                "action_type": action_type,
                "protocol_rejected": protocol_rejected,
                "model_visible": {"policy_input": dict(policy_input)},
                "observed_policy_action": dict(policy_action),
                "rollout_token_capture": _policy_token_capture(metadata),
            }
        )
    return projected


def _candidate_id(
    *,
    kind: str,
    case_id: str,
    episode_id: str,
    step_id: str,
    source_trace_sha256: str,
) -> str:
    payload = {
        "kind": kind,
        "case_id": case_id,
        "episode_id": episode_id,
        "step_id": step_id,
        "source_trace_sha256": source_trace_sha256,
    }
    return "psd-candidate:" + hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _safe_trace_path(run_dir: Path, recorded: str, episode_id: str) -> Path:
    if recorded:
        candidate = (run_dir / recorded).resolve()
        try:
            candidate.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"trace_path escapes run directory: {recorded}") from exc
        return candidate
    safe_episode_id = "".join(
        character if character.isalnum() or character in "-_."
        else "_"
        for character in episode_id
    )
    candidates = (
        run_dir / "traces" / f"{episode_id}.json",
        run_dir / "traces" / f"{safe_episode_id}.json",
    )
    return next((path for path in candidates if path.is_file()), candidates[-1])


def _trace_case_id(trace: Mapping[str, Any]) -> str:
    state = _mapping(trace.get("state"))
    runtime_case = _mapping(state.get("runtime_case"))
    return _text(
        runtime_case.get("case_id")
        or trace.get("case_id")
        or state.get("case_id")
    )


def _trace_runtime_store_path(trace: Mapping[str, Any]) -> str:
    state = _mapping(trace.get("state"))
    runtime_store = _mapping(state.get("runtime_store"))
    return _text(runtime_store.get("runtime_path"))


def _load_train_case_allowlist(path: Path) -> dict[str, str]:
    """Load an explicit train-case manifest, not a generic all-split manifest."""

    if not path.is_file():
        raise FileNotFoundError(f"train case manifest does not exist: {path}")

    if path.suffix.casefold() == ".jsonl":
        raw_rows: Iterable[Any] = load_jsonl(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            raw_rows = payload
        elif isinstance(payload, Mapping):
            for key in ("cases", "items", "records"):
                if isinstance(payload.get(key), list):
                    raw_rows = payload[key]
                    break
            else:
                raw_rows = [
                    {"case_id": key, "split": value}
                    for key, value in payload.items()
                    if isinstance(key, str)
                ]
        else:
            raise ValueError("train case manifest must be a JSON object/list or JSONL")

    allowlist: dict[str, str] = {}
    for index, raw_row in enumerate(raw_rows):
        if isinstance(raw_row, str):
            case_id, split = _text(raw_row), "train"
        else:
            row = _mapping(raw_row)
            case_id = _text(
                row.get("case_id")
                or row.get("candidate_id")
                or row.get("assignment_id")
                or row.get("image_id")
                or row.get("id")
            )
            raw_split = row.get("split")
            if isinstance(raw_split, Mapping):
                raw_split = raw_split.get("name")
            split = _text(raw_split) or "train"
        if not case_id:
            raise ValueError(f"train case manifest row {index} lacks a case ID")
        if case_id in allowlist and allowlist[case_id] != split:
            raise ValueError(f"conflicting split for train case {case_id}")
        allowlist[case_id] = split
    if not allowlist:
        raise ValueError("train case manifest is empty")
    return allowlist


def _source_metadata(
    run_manifest: Mapping[str, Any],
    *,
    rollout_gate: Mapping[str, Any],
    rollout_gate_path: Path,
) -> dict[str, Any]:
    return {
        "source_run_id": _text(run_manifest.get("run_id")),
        "runtime_commit": _text(run_manifest.get("git_commit")),
        "psd_round_index": rollout_gate.get("round_index"),
        "psd_rollout_gate_sha256": sha256_file(rollout_gate_path),
        "source_access_policy_sha256": (
            sha256_file(Path(run_manifest["source_access_policy"]["path"]))
            if isinstance(run_manifest.get("source_access_policy"), Mapping)
            and run_manifest["source_access_policy"].get("path") else None),
    }


def _step_index_from_location(value: Any) -> int | None:
    match = re.search(r"all_steps\[(\d+)\]", _text(value))
    return int(match.group(1)) if match else None


def _failure_localization(
    *,
    steps: list[dict[str, Any]],
    reward: Mapping[str, Any],
) -> dict[str, Any]:
    """Locate the earliest *observed* failure without pretending to know causality.

    Post-rollout audit reports may carry an exact ``state.all_steps[i]`` path.
    When they do not, protocol rejection is still directly attributable to its
    first rejected step.  A terminal label mismatch is intentionally marked as
    requiring privileged attribution instead of being pinned to the last
    judgment step; the proposer/verifier must decide whether an earlier action
    caused it.
    """

    audit_rows = reward.get("strict_trace_audit_failures", [])
    if isinstance(audit_rows, list):
        for item in audit_rows:
            if not isinstance(item, Mapping):
                continue
            index = _step_index_from_location(
                item.get("location") or item.get("path") or item.get("message")
            )
            if index is None:
                continue
            projected_index = next(
                (
                    projected
                    for projected, step in enumerate(steps)
                    if int(step.get("source_step_index", -1)) == index
                ),
                None,
            )
            if projected_index is None:
                continue
            return {
                "status": "observed_failure",
                "basis": "strict_trace_audit_location",
                "requires_privileged_attribution": False,
                "observed_failure_source_step_indices": [index],
                "observed_failure_step_indices": [projected_index],
                "repair_anchor_step_index": projected_index,
                "repair_anchor_source_step_index": index,
                "repair_anchor_step_id": steps[projected_index]["step_id"],
            }

    for index, step in enumerate(steps):
        if step.get("protocol_rejected"):
            return {
                "status": "observed_failure",
                "basis": "first_protocol_rejection",
                "requires_privileged_attribution": False,
                "observed_failure_source_step_indices": [
                    int(step["source_step_index"])
                ],
                "observed_failure_step_indices": [index],
                "repair_anchor_step_index": index,
                "repair_anchor_source_step_index": int(
                    step["source_step_index"]
                ),
                "repair_anchor_step_id": step["step_id"],
            }

    judgment_indices = [
        index
        for index, step in enumerate(steps)
        if step.get("example_type") == "judgment"
    ]
    anchor_index = judgment_indices[0] if judgment_indices else (len(steps) - 1)
    return {
        "status": "needs_privileged_attribution",
        "basis": "terminal_outcome_or_unlocated_audit_failure",
        "requires_privileged_attribution": True,
        "observed_failure_source_step_indices": [
            int(steps[index]["source_step_index"])
            for index in judgment_indices[:1]
        ],
        "observed_failure_step_indices": judgment_indices[:1],
        "repair_anchor_step_index": anchor_index,
        "repair_anchor_source_step_index": int(
            steps[anchor_index]["source_step_index"]
        ),
        "repair_anchor_step_id": steps[anchor_index]["step_id"],
    }


def _rejection(
    *,
    row_index: int,
    reward: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "case_id": _text(reward.get("case_id")),
        "episode_id": _text(reward.get("episode_id")),
        "reason": reason,
    }


_CANDIDATE_ARTIFACTS = {
    "repair_candidates": "repair_candidates.jsonl",
    "preservation_candidates": "preservation_candidates.jsonl",
    "engineering_requeue": "engineering_requeue.jsonl",
    "token_capture_requeue": "token_capture_requeue.jsonl",
    "source_review_pending": "source_review_pending.jsonl",
    "source_review_abstained": "source_review_abstained.jsonl",
    "rejections": "rejections.jsonl",
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write_json(temporary, value)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _emit_candidate_row(job: Mapping[str, Any], category: str, value: Mapping[str, Any]):
    row_index = int(job["row_index"])
    path = Path(job["rows_dir"]) / category / f"{row_index:06d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"row_index": row_index, "category": category, "path": str(path),
        "candidate_id": _text(value.get("candidate_id")),
        "repair_signal": _text(value.get("repair_signal"))}


def _parallel_candidate_job(job: Mapping[str, Any]):
    reward = job["reward"]
    row_index = int(job["row_index"])
    case_id = _text(reward.get("case_id"))
    episode_id = _text(reward.get("episode_id")) or case_id
    split = job.get("split")

    def reject(reason: str):
        return _emit_candidate_row(job, "rejections",
            _rejection(row_index=row_index, reward=reward, reason=reason))

    if not case_id or not episode_id:
        return reject("missing_case_or_episode_id")
    if split is None:
        return reject("case_not_in_explicit_train_allowlist")
    if str(split).casefold() != "train":
        return reject(f"source_split_forbidden:{split}")
    if bool(reward.get("training_prohibited")):
        return reject("training_prohibited")
    if bool(reward.get("fatal_engineering_error")):
        return _emit_candidate_row(job, "engineering_requeue", {
            "schema_version": PSD_CANDIDATE_SCHEMA_VERSION,
            "case_id": case_id, "episode_id": episode_id,
            "queue_reason": "fatal_engineering_error", "source": job["source"]})

    run_dir = Path(job["run_dir"])
    try:
        trace_path = _safe_trace_path(run_dir, job.get("trace_path", ""), episode_id)
        if not trace_path.is_file():
            raise FileNotFoundError("canonical_trace_missing")
        raw = trace_path.read_bytes()
        trace_sha256 = hashlib.sha256(raw).hexdigest()
        expected_sha256 = _text(job.get("trace_sha256"))
        if expected_sha256 and trace_sha256 != expected_sha256:
            raise ValueError("trace_sha256_changed_after_postprocess")
        trace = json.loads(raw)
        if not isinstance(trace, dict):
            raise ValueError("trace_root_not_object")
        trace_case_id = _trace_case_id(trace)
        if trace_case_id and trace_case_id != case_id:
            raise ValueError(f"trace_case_id_mismatch:{trace_case_id}!={case_id}")
        steps = _policy_steps(trace, episode_id=episode_id)
        if not steps:
            raise ValueError("trace_has_no_usable_policy_steps")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return reject(f"trace_rejected:{exc}")

    strict_audit_pass = bool(reward.get("strict_trace_audit_pass"))
    classification_correct = bool(reward.get("classification_correct"))
    trace_source = {**job["source"],
        "source_trace_path": trace_path.relative_to(run_dir).as_posix(),
        "source_trace_sha256": trace_sha256,
        "source_runtime_store_path": _trace_runtime_store_path(trace),
        **({"source_audit": reward["source_audit"]} if reward.get("source_audit") else {})}
    from .psd_source_review import source_review_reference, validate_source_review
    review_status = "pending"
    try:
        trace_canonical_sha256 = None
        audit_reference = reward.get("source_audit")
        if isinstance(audit_reference, Mapping):
            audit_path = Path(_text(audit_reference.get("path")))
            if (not audit_path.is_file()
                    or sha256_file(audit_path) != _text(audit_reference.get("sha256"))):
                raise ValueError("source audit binding changed")
            source_audit = load_json(audit_path)
            if source_audit.get("source_trace_sha256") != trace_sha256:
                raise ValueError("source audit trace changed")
            trace_canonical_sha256 = _text(source_audit.get("source_trace_canonical_sha256")) or None
        source_review = source_review_reference(reward)
        if source_review is not None:
            review_status = validate_source_review(source_review, trace=trace,
                trace_canonical_sha256=trace_canonical_sha256)
            trace_source["source_task_review"] = reward["source_task_review"]
            if reward.get("source_task_status") != review_status:
                raise ValueError("source task status differs from bound review")
    except (OSError, ValueError, KeyError) as exc:
        return reject(f"source_review_rejected:{type(exc).__name__}")

    if classification_correct and strict_audit_pass and review_status == "pending":
        return _emit_candidate_row(job, "source_review_pending", {"case_id": case_id,
            "episode_id": episode_id, "queue_reason": "source_semantic_review_required",
            "source": trace_source})
    if review_status == "unresolved":
        return _emit_candidate_row(job, "source_review_abstained", {"case_id": case_id,
            "episode_id": episode_id, "queue_reason": "source_semantic_review_abstained",
            "source": trace_source})
    if classification_correct and strict_audit_pass and review_status == "pass":
        incomplete_steps = [step["step_id"] for step in steps
            if step["rollout_token_capture"]["status"] != "complete"]
        if incomplete_steps:
            return _emit_candidate_row(job, "token_capture_requeue", {
                "schema_version": PSD_CANDIDATE_SCHEMA_VERSION, "case_id": case_id,
                "episode_id": episode_id, "queue_reason": "missing_policy_token_capture",
                "incomplete_step_ids": incomplete_steps, "source": trace_source})
        candidate = {"schema_version": PSD_CANDIDATE_SCHEMA_VERSION,
            "candidate_id": _candidate_id(kind="preserve", case_id=case_id,
                episode_id=episode_id, step_id="episode", source_trace_sha256=trace_sha256),
            "class": "base_pass_preserve", "candidate_status": "ready_for_teacher_collection",
            "case_id": case_id, "episode_id": episode_id, "verified_full_task": True,
            "strict_trace_audit_pass": True, "preservation_steps": steps, "source": trace_source}
        return _emit_candidate_row(job, "preservation_candidates", candidate)

    protocol_steps = [step for step in steps if step["protocol_rejected"]]
    if protocol_steps:
        repair_signal, repair_site = "protocol_rejection", protocol_steps[0]
    else:
        localization = _failure_localization(steps=steps, reward=reward)
        repair_signal = ("strict_trace_audit_failure" if not strict_audit_pass
            else "source_semantic_failure" if classification_correct and review_status == "fail"
            else "terminal_outcome_mismatch")
        repair_site = steps[int(localization["repair_anchor_step_index"])]
    if repair_site["rollout_token_capture"]["status"] != "complete":
        return _emit_candidate_row(job, "token_capture_requeue", {
            "schema_version": PSD_CANDIDATE_SCHEMA_VERSION, "case_id": case_id,
            "episode_id": episode_id, "queue_reason": "missing_policy_token_capture",
            "incomplete_step_ids": [repair_site["step_id"]], "source": trace_source})
    failure_localization = ({**_failure_localization(steps=steps, reward=reward),
        "repair_signal": repair_signal} if not protocol_steps else {
        "status": "observed_failure", "basis": "first_protocol_rejection",
        "requires_privileged_attribution": False,
        "observed_failure_source_step_indices": [int(protocol_steps[0]["source_step_index"])],
        "observed_failure_step_indices": [steps.index(protocol_steps[0])],
        "repair_anchor_step_index": steps.index(protocol_steps[0]),
        "repair_anchor_source_step_index": int(protocol_steps[0]["source_step_index"]),
        "repair_anchor_step_id": protocol_steps[0]["step_id"], "repair_signal": repair_signal})
    candidate = {"schema_version": PSD_CANDIDATE_SCHEMA_VERSION,
        "candidate_id": _candidate_id(kind="repair", case_id=case_id,
            episode_id=episode_id, step_id=repair_site["step_id"],
            source_trace_sha256=trace_sha256), "class": "repair_seed",
        "candidate_status": "needs_privileged_localization", "case_id": case_id,
        "episode_id": episode_id, "repair_signal": repair_signal,
        "failure_localization": failure_localization,
        "strict_trace_audit_failure_codes": [_text(value)
            for value in reward.get("strict_trace_audit_failure_codes", []) if _text(value)],
        "repair_site": repair_site, "source": trace_source}
    return _emit_candidate_row(job, "repair_candidates", candidate)


def build_psd_candidate_package_parallel(*, run_dir: Path, train_cases_path: Path,
                                         rollout_gate_path: Path, output_dir: Path,
                                         workers: int = 16) -> dict[str, Any]:
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 32:
        raise ValueError("PSD candidate workers must be in [1, 32]")
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"rollout directory does not exist: {run_dir}")
    rewards_path = run_dir / "post_rollout_rewards.jsonl"
    if not rewards_path.is_file():
        raise FileNotFoundError(f"missing post-rollout rewards: {rewards_path}")
    require_new_or_empty(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = output_dir / ".rows"
    train_cases = _load_train_case_allowlist(train_cases_path)
    rollout_gate = validate_rollout_gate_for_candidates(
        rollout_gate_path=rollout_gate_path, run_dir=run_dir,
        train_cases_path=train_cases_path)
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = load_json(run_manifest_path) if run_manifest_path.is_file() else {}
    groups = {_text(item.get("episode_id")): item
        for item in load_jsonl(run_dir / "rollout_groups.jsonl") if _text(item.get("episode_id"))}
    source = _source_metadata(run_manifest, rollout_gate=rollout_gate,
        rollout_gate_path=rollout_gate_path)
    rewards = load_jsonl(rewards_path)
    jobs = []
    for row_index, reward in enumerate(rewards):
        case_id = _text(reward.get("case_id"))
        episode_id = _text(reward.get("episode_id")) or case_id
        group = groups.get(episode_id, {})
        jobs.append({"row_index": row_index, "reward": reward,
            "split": train_cases.get(case_id), "run_dir": str(run_dir),
            "trace_path": _text(group.get("trace_path")),
            "trace_sha256": _text(group.get("trace_sha256")),
            "source": source, "rows_dir": str(rows_dir)})
    progress = output_dir / "candidate-progress.json"
    _atomic_json(progress, {"status": "running", "completed": 0,
        "total": len(jobs), "workers": workers})
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
    results = []
    try:
        for completed, result in enumerate(executor.map(_parallel_candidate_job, jobs, chunksize=1), 1):
            results.append(result)
            if completed % 10 == 0 or completed == len(jobs):
                _atomic_json(progress, {"status": "running", "completed": completed,
                    "total": len(jobs), "workers": workers})
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    candidate_ids = [row["candidate_id"] for row in results if row["candidate_id"]]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("PSD candidate IDs must be unique")
    counts = Counter(row["category"] for row in results)
    repair_signals = Counter(row["repair_signal"] for row in results
        if row["category"] == "repair_candidates" and row["repair_signal"])
    for category, filename in _CANDIDATE_ARTIFACTS.items():
        with (output_dir / filename).open("wb") as destination:
            for row in sorted((item for item in results if item["category"] == category),
                              key=lambda item: item["row_index"]):
                with Path(row["path"]).open("rb") as source_file:
                    shutil.copyfileobj(source_file, destination, length=1024 * 1024)
    shutil.rmtree(rows_dir)
    manifest = {"schema_version": PSD_CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "source": {"run_dir": str(run_dir), "post_rollout_rewards": str(rewards_path),
            "post_rollout_rewards_sha256": sha256_file(rewards_path),
            "train_cases": str(train_cases_path), "train_cases_sha256": sha256_file(train_cases_path),
            "rollout_gate": str(rollout_gate_path), "rollout_gate_sha256": sha256_file(rollout_gate_path),
            **source},
        "counts": {name: counts.get(name, 0) for name in _CANDIDATE_ARTIFACTS},
        "repair_signals": dict(sorted(repair_signals.items())),
        "artifacts": dict(_CANDIDATE_ARTIFACTS),
        "status": ("ready_for_privileged_localization"
            if counts.get("repair_candidates") or counts.get("preservation_candidates") else "empty")}
    write_json(output_dir / "manifest.json", manifest)
    _atomic_json(progress, {"status": "completed", "completed": len(jobs),
        "total": len(jobs), "workers": workers})
    return manifest


def build_psd_candidate_package(
    *,
    run_dir: Path,
    train_cases_path: Path,
    rollout_gate_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Produce the queues consumed by later PSD localization and repair stages.

    ``train_cases_path`` is intentionally an allowlist.  Merely having a
    non-prohibited rollout row is insufficient to admit a case, so a test case
    can never reach the repair or preservation queue by accident.
    """

    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"rollout directory does not exist: {run_dir}")
    rewards_path = run_dir / "post_rollout_rewards.jsonl"
    if not rewards_path.is_file():
        raise FileNotFoundError(f"missing post-rollout rewards: {rewards_path}")
    require_new_or_empty(output_dir)

    train_cases = _load_train_case_allowlist(train_cases_path)
    rollout_gate = validate_rollout_gate_for_candidates(
        rollout_gate_path=rollout_gate_path,
        run_dir=run_dir,
        train_cases_path=train_cases_path,
    )
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = load_json(run_manifest_path) if run_manifest_path.is_file() else {}
    group_paths = {
        _text(item.get("episode_id")): _text(item.get("trace_path"))
        for item in load_jsonl(run_dir / "rollout_groups.jsonl")
        if _text(item.get("episode_id"))
    }

    repair_candidates: list[dict[str, Any]] = []
    preservation_candidates: list[dict[str, Any]] = []
    engineering_requeue: list[dict[str, Any]] = []
    token_capture_requeue: list[dict[str, Any]] = []
    source_review_pending: list[dict[str, Any]] = []
    source_review_abstained: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    source = _source_metadata(
        run_manifest,
        rollout_gate=rollout_gate,
        rollout_gate_path=rollout_gate_path,
    )

    for row_index, reward in enumerate(load_jsonl(rewards_path)):
        case_id = _text(reward.get("case_id"))
        episode_id = _text(reward.get("episode_id")) or case_id
        split = train_cases.get(case_id)
        if not case_id or not episode_id:
            rejections.append(
                _rejection(
                    row_index=row_index,
                    reward=reward,
                    reason="missing_case_or_episode_id",
                )
            )
            continue
        if split is None:
            rejections.append(
                _rejection(
                    row_index=row_index,
                    reward=reward,
                    reason="case_not_in_explicit_train_allowlist",
                )
            )
            continue
        if split.casefold() != "train":
            rejections.append(
                _rejection(
                    row_index=row_index,
                    reward=reward,
                    reason=f"source_split_forbidden:{split}",
                )
            )
            continue
        if bool(reward.get("training_prohibited")):
            rejections.append(
                _rejection(
                    row_index=row_index,
                    reward=reward,
                    reason="training_prohibited",
                )
            )
            continue
        if bool(reward.get("fatal_engineering_error")):
            engineering_requeue.append(
                {
                    "schema_version": PSD_CANDIDATE_SCHEMA_VERSION,
                    "case_id": case_id,
                    "episode_id": episode_id,
                    "queue_reason": "fatal_engineering_error",
                    "source": source,
                }
            )
            continue

        try:
            trace_path = _safe_trace_path(
                run_dir,
                group_paths.get(episode_id, ""),
                episode_id,
            )
            if not trace_path.is_file():
                raise FileNotFoundError("canonical_trace_missing")
            trace = load_json(trace_path)
            trace_case_id = _trace_case_id(trace)
            if trace_case_id and trace_case_id != case_id:
                raise ValueError(
                    f"trace_case_id_mismatch:{trace_case_id}!={case_id}"
                )
            trace_sha256 = sha256_file(trace_path)
            steps = _policy_steps(trace, episode_id=episode_id)
            if not steps:
                raise ValueError("trace_has_no_usable_policy_steps")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            rejections.append(
                _rejection(
                    row_index=row_index,
                    reward=reward,
                    reason=f"trace_rejected:{exc}",
                )
            )
            continue

        strict_audit_pass = bool(reward.get("strict_trace_audit_pass"))
        classification_correct = bool(reward.get("classification_correct"))
        trace_source = {
            **source,
            "source_trace_path": trace_path.relative_to(run_dir).as_posix(),
            "source_trace_sha256": trace_sha256,
            "source_runtime_store_path": _trace_runtime_store_path(trace),
            **({"source_audit": reward["source_audit"]} if reward.get("source_audit") else {}),
        }
        from .psd_source_review import source_review_reference, validate_source_review
        review_status = "pending"
        try:
            source_review = source_review_reference(reward)
            if source_review is not None:
                review_status = validate_source_review(source_review, trace=trace)
                trace_source["source_task_review"] = reward["source_task_review"]
                if reward.get("source_task_status") != review_status:
                    raise ValueError("source task status differs from bound review")
        except (OSError, ValueError, KeyError) as exc:
            rejections.append(_rejection(row_index=row_index, reward=reward,
                reason=f"source_review_rejected:{type(exc).__name__}"))
            continue
        if classification_correct and strict_audit_pass and review_status == "pending":
            source_review_pending.append({"case_id": case_id, "episode_id": episode_id,
                "queue_reason": "source_semantic_review_required", "source": trace_source})
            continue
        if review_status == "unresolved":
            source_review_abstained.append({"case_id": case_id, "episode_id": episode_id,
                "queue_reason": "source_semantic_review_abstained", "source": trace_source})
            continue
        if classification_correct and strict_audit_pass and review_status == "pass":
            incomplete_steps = [
                step["step_id"]
                for step in steps
                if step["rollout_token_capture"]["status"] != "complete"
            ]
            if incomplete_steps:
                token_capture_requeue.append(
                    {
                        "schema_version": PSD_CANDIDATE_SCHEMA_VERSION,
                        "case_id": case_id,
                        "episode_id": episode_id,
                        "queue_reason": "missing_policy_token_capture",
                        "incomplete_step_ids": incomplete_steps,
                        "source": trace_source,
                    }
                )
                continue
            preservation_candidates.append(
                {
                    "schema_version": PSD_CANDIDATE_SCHEMA_VERSION,
                    "candidate_id": _candidate_id(
                        kind="preserve",
                        case_id=case_id,
                        episode_id=episode_id,
                        step_id="episode",
                        source_trace_sha256=trace_sha256,
                    ),
                    "class": "base_pass_preserve",
                    "candidate_status": "ready_for_teacher_collection",
                    "case_id": case_id,
                    "episode_id": episode_id,
                    "verified_full_task": True,
                    "strict_trace_audit_pass": True,
                    "preservation_steps": steps,
                    "source": trace_source,
                }
            )
            continue

        protocol_steps = [step for step in steps if step["protocol_rejected"]]
        if protocol_steps:
            repair_signal, repair_site = "protocol_rejection", protocol_steps[0]
        else:
            localization = _failure_localization(steps=steps, reward=reward)
            repair_index = int(localization["repair_anchor_step_index"])
            repair_signal = (
                "strict_trace_audit_failure"
                if not strict_audit_pass
                else "source_semantic_failure" if classification_correct and review_status == "fail"
                else "terminal_outcome_mismatch"
            )
            repair_site = steps[repair_index]
        if repair_site["rollout_token_capture"]["status"] != "complete":
            token_capture_requeue.append(
                {
                    "schema_version": PSD_CANDIDATE_SCHEMA_VERSION,
                    "case_id": case_id,
                    "episode_id": episode_id,
                    "queue_reason": "missing_policy_token_capture",
                    "incomplete_step_ids": [repair_site["step_id"]],
                    "source": trace_source,
                }
            )
            continue

        repair_candidates.append(
            {
                "schema_version": PSD_CANDIDATE_SCHEMA_VERSION,
                "candidate_id": _candidate_id(
                    kind="repair",
                    case_id=case_id,
                    episode_id=episode_id,
                    step_id=repair_site["step_id"],
                    source_trace_sha256=trace_sha256,
                ),
                "class": "repair_seed",
                "candidate_status": "needs_privileged_localization",
                "case_id": case_id,
                "episode_id": episode_id,
                "repair_signal": repair_signal,
                "failure_localization": (
                    {
                        **_failure_localization(steps=steps, reward=reward),
                        "repair_signal": repair_signal,
                    }
                    if not protocol_steps
                    else {
                        "status": "observed_failure",
                        "basis": "first_protocol_rejection",
                        "requires_privileged_attribution": False,
                        "observed_failure_source_step_indices": [
                            int(protocol_steps[0]["source_step_index"])
                        ],
                        "observed_failure_step_indices": [
                            steps.index(protocol_steps[0])
                        ],
                        "repair_anchor_step_index": steps.index(protocol_steps[0]),
                        "repair_anchor_source_step_index": int(
                            protocol_steps[0]["source_step_index"]
                        ),
                        "repair_anchor_step_id": protocol_steps[0]["step_id"],
                        "repair_signal": repair_signal,
                    }
                ),
                "strict_trace_audit_failure_codes": [
                    _text(value)
                    for value in reward.get("strict_trace_audit_failure_codes", [])
                    if _text(value)
                ],
                "repair_site": repair_site,
                "source": trace_source,
            }
        )

    candidate_ids = [
        row["candidate_id"]
        for row in repair_candidates + preservation_candidates
    ]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("PSD candidate IDs must be unique")

    write_jsonl(output_dir / "repair_candidates.jsonl", repair_candidates)
    write_jsonl(
        output_dir / "preservation_candidates.jsonl",
        preservation_candidates,
    )
    write_jsonl(output_dir / "engineering_requeue.jsonl", engineering_requeue)
    write_jsonl(output_dir / "token_capture_requeue.jsonl", token_capture_requeue)
    write_jsonl(output_dir / "source_review_pending.jsonl", source_review_pending)
    write_jsonl(output_dir / "source_review_abstained.jsonl", source_review_abstained)
    write_jsonl(output_dir / "rejections.jsonl", rejections)
    manifest = {
        "schema_version": PSD_CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "source": {
            "run_dir": str(run_dir),
            "post_rollout_rewards": str(rewards_path),
            "post_rollout_rewards_sha256": sha256_file(rewards_path),
            "train_cases": str(train_cases_path),
            "train_cases_sha256": sha256_file(train_cases_path),
            "rollout_gate": str(rollout_gate_path),
            "rollout_gate_sha256": sha256_file(rollout_gate_path),
            **source,
        },
        "counts": {
            "repair_candidates": len(repair_candidates),
            "preservation_candidates": len(preservation_candidates),
            "engineering_requeue": len(engineering_requeue),
            "token_capture_requeue": len(token_capture_requeue),
            "source_review_pending": len(source_review_pending),
            "source_review_abstained": len(source_review_abstained),
            "rejections": len(rejections),
        },
        "repair_signals": dict(
            sorted(
                Counter(
                    _text(item["repair_signal"]) for item in repair_candidates
                ).items()
            )
        ),
        "artifacts": {
            "repair_candidates": "repair_candidates.jsonl",
            "preservation_candidates": "preservation_candidates.jsonl",
            "engineering_requeue": "engineering_requeue.jsonl",
            "token_capture_requeue": "token_capture_requeue.jsonl",
            "source_review_pending": "source_review_pending.jsonl",
            "source_review_abstained": "source_review_abstained.jsonl",
            "rejections": "rejections.jsonl",
        },
        "status": (
            "ready_for_privileged_localization"
            if repair_candidates or preservation_candidates
            else "empty"
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
