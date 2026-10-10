"""Finalize generated PSD repairs after their local task verifier completes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from .psd_repair import _sha, _text
from .psd_repair_verifier import verify_causal_episode


PSD_VERIFICATION_BUNDLE_SCHEMA_VERSION = (
    "ifv-psd-repair-verification-bundle-v1"
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bound_artifact(value: Any, *, root: Path) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    path_value = _text(value)
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    return load_json(path)


def _bundle_rows(path: Path) -> dict[int, Mapping[str, Any]]:
    value = load_json(path)
    if value.get("schema_version") != PSD_VERIFICATION_BUNDLE_SCHEMA_VERSION:
        raise ValueError("PSD repair verification bundle schema is invalid")
    rows = value.get("attempts")
    if not isinstance(rows, list):
        raise ValueError("PSD repair verification bundle requires attempts")
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("PSD repair verification attempt must be an object")
        index = row.get("hint_index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("PSD repair verification hint_index is invalid")
        if index in result:
            raise ValueError(f"duplicate PSD repair verification hint_index: {index}")
        result[index] = row
    return result


def _atomic_write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write_jsonl(temporary, rows)
    os.replace(temporary, path)


def finalize_psd_repair_run(
    *,
    run_dir: Path,
    source_trace_path: Path,
    gold_path: Path,
    verification_bundle_path: Path,
    require_all: bool = False,
) -> dict[str, Any]:
    """Verify cached teacher episodes without repeating any model/tool calls."""

    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    attempts_path = run_dir / "repair_attempts.jsonl"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "ifv-psd-repair-driver-result-v1":
        raise ValueError("PSD repair driver manifest schema is invalid")
    source_policy = None
    if manifest.get("source_access_policy"):
        from eviscene.orchestrator.source_access import SourceAccessPolicy
        recorded_policy = manifest["source_access_policy"]
        policy_path = Path(recorded_policy["path"])
        if sha256_file(policy_path) != recorded_policy["sha256"]:
            raise ValueError("PSD source policy changed before finalization")
        source_policy = SourceAccessPolicy.load(policy_path)
    source_trace = load_json(source_trace_path)
    gold = load_json(gold_path)
    from .psd_source_review import source_review_reference
    source_task_review = source_review_reference(manifest)
    source_audit = None
    if manifest.get("source_audit"):
        ref = manifest["source_audit"]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("source audit changed before finalization")
        source_audit = load_json(Path(ref["path"]))
    source_sha256 = sha256_file(source_trace_path)
    attempts = load_jsonl(attempts_path)
    if not attempts:
        raise ValueError("PSD repair run has no attempts")
    bound = _bundle_rows(verification_bundle_path)
    unknown_indices = sorted(set(bound) - set(range(len(attempts))))
    if unknown_indices:
        raise ValueError(
            "verification bundle references unknown hint indices: "
            + ",".join(str(item) for item in unknown_indices)
        )

    finalized: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for index, raw_attempt in enumerate(attempts):
        attempt = dict(raw_attempt)
        if _text(attempt.get("source_trace_sha256")) != source_sha256:
            raise ValueError(f"attempt {index} source trace hash mismatch")
        binding = bound.get(index)
        if binding is None:
            unresolved.append({"hint_index": index, "reason": "verifier_missing"})
            finalized.append(attempt)
            continue
        hint_sha256 = _text(
            _mapping(_mapping(attempt.get("hint_record")).get("audit")).get(
                "hint_sha256"
            )
        )
        if not hint_sha256 or _text(binding.get("hint_sha256")) != hint_sha256:
            raise ValueError(f"attempt {index} hint hash mismatch")
        continuation = _mapping(attempt.get("continuation"))
        episode_reference = _text(
            continuation.get("hinted_teacher_episode_trace")
        )
        episode_path = Path(episode_reference)
        if not episode_path.is_absolute():
            episode_path = run_dir / episode_path
        expected_episode_sha256 = _text(
            continuation.get("hinted_teacher_episode_trace_sha256")
        )
        if (
            not expected_episode_sha256
            or not episode_path.is_file()
            or sha256_file(episode_path) != expected_episode_sha256
        ):
            raise ValueError(f"attempt {index} teacher episode hash mismatch")
        episode = _bound_artifact(
            episode_reference,
            root=run_dir,
        )
        if episode is None:
            raise ValueError(f"attempt {index} teacher episode is missing")
        local_verification = _bound_artifact(
            binding.get("local_verification"),
            root=verification_bundle_path.parent,
        )
        if local_verification is None:
            unresolved.append(
                {"hint_index": index, "reason": "local_verification_missing"}
            )
            finalized.append(attempt)
            continue
        teacher_prompt_ids = attempt.get("teacher_prompt_ids")
        completion_ids = attempt.get("completion_ids")
        if not isinstance(teacher_prompt_ids, list) or not teacher_prompt_ids:
            raise ValueError(f"attempt {index} teacher prompt IDs are missing")
        if not isinstance(completion_ids, list) or not completion_ids:
            raise ValueError(f"attempt {index} completion IDs are missing")
        result = verify_causal_episode(
            episode,
            source_trace=source_trace,
            source_trace_sha256=source_sha256,
            gold=gold,
            local_verification=local_verification,
            repair_step_id=_text(attempt.get("repair_step_id")),
            hint_sha256=hint_sha256,
            teacher_prompt_sha256=_sha(
                [int(item) for item in teacher_prompt_ids]
            ),
            teacher_completion_sha256=_sha(
                [int(item) for item in completion_ids]
            ),
            source_access_policy=source_policy,
            source_task_review=source_task_review,
            source_audit=source_audit,
        )
        attempt.update(
            {
                "repair_tier": result.repair_tier,
                "source_rollout_failed": result.source_rollout_failed,
                "hinted_local_pass": result.hinted_local_pass,
                "hinted_episode_pass": result.hinted_episode_pass,
                "hinted_strict_trace_audit_pass": (
                    result.hinted_strict_trace_audit_pass
                ),
                "verification": {
                    "source_rollout_failed": result.source_rollout_failed,
                    "hinted_local_pass": result.hinted_local_pass,
                    "hinted_episode_pass": result.hinted_episode_pass,
                    "hinted_strict_trace_audit_pass": (
                        result.hinted_strict_trace_audit_pass
                    ),
                    "repair_tier": result.repair_tier,
                    "hinted_recorded_verdict": result.hinted_recorded_verdict,
                    "expected_verdict": result.expected_verdict,
                    "reasons": list(result.reasons),
                },
                "local_verification": dict(local_verification),
                "accepted": result.accepted_for_primary_psd,
            }
        )
        finalized.append(attempt)

    backup_path = run_dir / "repair_attempts.pre-finalize.jsonl"
    if not backup_path.exists():
        write_jsonl(backup_path, attempts)
    _atomic_write_jsonl(attempts_path, finalized)
    accepted = sum(row.get("accepted") is True for row in finalized)
    manifest.update(
        {
            "candidate_count": len(finalized),
            "accepted_count": accepted,
            "pending_hinted_episode_count": len(unresolved),
            "finalization": {
                "mode": "offline_no_provider_replay",
                "source_trace_sha256": source_sha256,
                "verification_bundle_sha256": sha256_file(
                    verification_bundle_path
                ),
                "finalized_attempts": len(finalized) - len(unresolved),
                "unresolved_attempts": len(unresolved),
                "require_all": bool(require_all),
            },
            "artifacts": {
                **dict(_mapping(manifest.get("artifacts"))),
                "repair_attempts": "repair_attempts.jsonl",
                "pre_finalize_backup": "repair_attempts.pre-finalize.jsonl",
            },
        }
    )
    write_json(manifest_path, manifest)
    result = {
        "schema_version": "ifv-psd-repair-finalization-v1",
        "status": (
            "ready_for_assembly"
            if not unresolved
            else "blocked_unresolved_verifiers"
            if require_all
            else "partially_finalized"
        ),
        "attempts": len(finalized),
        "accepted": accepted,
        "rejected": sum(row.get("accepted") is False for row in finalized),
        "unresolved": unresolved,
        "provider_calls": 0,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "finalization.json", result)
    return result
