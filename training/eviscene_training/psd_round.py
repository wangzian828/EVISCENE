from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .io import sha256_file, write_json
from .psd_collection import require_completed_collection


PSD_ROLLOUT_GATE_SCHEMA_VERSION = "ifv-psd-round-rollout-gate-v1"
PSD_ROUND_COMPLETION_SCHEMA_VERSION = "ifv-psd-round-completion-v1"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _check(actual: Any, expected: Any) -> dict[str, Any]:
    return {"passed": actual == expected, "actual": actual, "expected": expected}


def validate_rollout_gate_for_candidates(
    *,
    rollout_gate_path: Path,
    run_dir: Path,
    train_cases_path: Path,
) -> dict[str, Any]:
    gate = _load_object(rollout_gate_path)
    gate_run = _mapping(gate.get("run"))
    gate_cases = _mapping(gate.get("train_cases"))
    run_dir = run_dir.expanduser().resolve()
    checks = {
        "schema": gate.get("schema_version") == PSD_ROLLOUT_GATE_SCHEMA_VERSION,
        "passed": gate.get("passed") is True,
        "run_directory": _text(gate_run.get("directory")) == str(run_dir),
        "run_manifest": _text(gate_run.get("manifest_sha256"))
        == sha256_file(run_dir / "run_manifest.json"),
        "rewards": _text(gate_run.get("post_rollout_rewards_sha256"))
        == sha256_file(run_dir / "post_rollout_rewards.jsonl"),
        "groups": _text(gate_run.get("rollout_groups_sha256"))
        == sha256_file(run_dir / "rollout_groups.jsonl"),
        "train_cases": _text(gate_cases.get("sha256"))
        == sha256_file(train_cases_path),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("PSD rollout gate mismatch: " + ",".join(failed))
    return gate


def verify_psd_round_rollout(
    *,
    round_index: int,
    run_dir: Path,
    train_cases_path: Path,
    serving_profile_path: Path,
    round_start_checkpoint_manifest_path: Path,
    output: Path,
    previous_round_completion_path: Path | None = None,
) -> dict[str, Any]:
    """Attest that a PSD bank is a fresh rollout from this round's policy."""

    if round_index < 1:
        raise ValueError("round_index must be positive")
    run_dir = run_dir.expanduser().resolve()
    run_manifest_path = run_dir / "run_manifest.json"
    rewards_path = run_dir / "post_rollout_rewards.jsonl"
    groups_path = run_dir / "rollout_groups.jsonl"
    run_manifest = _load_object(run_manifest_path)
    serving = _load_object(serving_profile_path)
    checkpoint = _load_object(round_start_checkpoint_manifest_path)
    agent = _mapping(run_manifest.get("agent"))
    benchmark = _mapping(run_manifest.get("benchmark"))
    checkpoint_record = _mapping(checkpoint.get("checkpoint"))
    checkpoint_sha = sha256_file(round_start_checkpoint_manifest_path)
    try:
        completion_check = require_completed_collection(run_dir, run_manifest)
    except (ValueError, OSError, KeyError) as error:
        completion_check = {'passed': False, 'native_status': run_manifest.get('status'),
                            'reason': str(error)}
    checks: dict[str, dict[str, Any]] = {
        "run_completed": completion_check,
        "run_id_present": {"passed": bool(_text(run_manifest.get("run_id")))},
        "run_completion_timestamp_present": {
            "passed": bool(_text(run_manifest.get("completed_at")))
        },
        "training_allowed": _check(benchmark.get("training_prohibited", False), False),
        "rewards_present": {"passed": rewards_path.is_file()},
        "rollout_groups_present": {"passed": groups_path.is_file()},
        "serving_profile_schema": _check(
            serving.get("schema_version"), "ifv-qwen-serving-profile-v1"
        ),
        "checkpoint_manifest_schema": _check(
            checkpoint.get("schema_version"), "ifv-qwen-checkpoint-manifest-v1"
        ),
        "serving_checkpoint_binding": _check(
            _text(serving.get("checkpoint_manifest_sha256")).casefold(),
            checkpoint_sha.casefold(),
        ),
        "serving_checkpoint_path": _check(
            _text(serving.get("model_path")), _text(checkpoint_record.get("path"))
        ),
        "rollout_model": _check(
            _text(agent.get("model")), _text(serving.get("profile_id"))
        ),
        "rollout_base_url": _check(
            _text(agent.get("base_url")).rstrip("/"),
            _text(serving.get("base_url")).rstrip("/"),
        ),
        "rollout_wire_api": _check(
            _text(agent.get("llm_wire_api")), _text(serving.get("wire_api"))
        ),
        "multimodal_policy": _check(serving.get("multimodal"), True),
    }

    previous = None
    if round_index == 1:
        checks["no_previous_round_for_round_one"] = _check(
            previous_round_completion_path, None
        )
    elif previous_round_completion_path is None:
        checks["previous_round_completion_present"] = {
            "passed": False,
            "expected": f"round {round_index - 1} completion",
        }
    else:
        previous = _load_object(previous_round_completion_path)
        previous_output = _mapping(previous.get("output_checkpoint_manifest"))
        previous_rollout = _mapping(previous.get("rollout_gate"))
        checks.update(
            {
                "previous_completion_schema": _check(
                    previous.get("schema_version"),
                    PSD_ROUND_COMPLETION_SCHEMA_VERSION,
                ),
                "previous_completion_passed": _check(previous.get("passed"), True),
                "previous_round_index": _check(
                    previous.get("round_index"), round_index - 1
                ),
                "checkpoint_advances_previous_round": _check(
                    _text(previous_output.get("sha256")).casefold(),
                    checkpoint_sha.casefold(),
                ),
                "fresh_rollout_run_id": {
                    "passed": _text(previous_rollout.get("run_id"))
                    != _text(run_manifest.get("run_id")),
                    "actual": _text(run_manifest.get("run_id")),
                    "not_expected": _text(previous_rollout.get("run_id")),
                },
            }
        )

    passed = all(item.get("passed") is True for item in checks.values())
    result = {
        "schema_version": PSD_ROLLOUT_GATE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "round_index": round_index,
        "passed": passed,
        "checks": checks,
        "run": {
            "directory": str(run_dir),
            "run_id": _text(run_manifest.get("run_id")),
            "manifest": str(run_manifest_path),
            "manifest_sha256": sha256_file(run_manifest_path),
            "post_rollout_rewards": str(rewards_path),
            "post_rollout_rewards_sha256": (
                sha256_file(rewards_path) if rewards_path.is_file() else None
            ),
            "rollout_groups": str(groups_path),
            "rollout_groups_sha256": (
                sha256_file(groups_path) if groups_path.is_file() else None
            ),
        },
        "train_cases": {
            "path": str(train_cases_path.resolve()),
            "sha256": sha256_file(train_cases_path),
        },
        "serving_profile": {
            "path": str(serving_profile_path.resolve()),
            "sha256": sha256_file(serving_profile_path),
            "profile_id": serving.get("profile_id"),
        },
        "round_start_checkpoint_manifest": {
            "path": str(round_start_checkpoint_manifest_path.resolve()),
            "sha256": checkpoint_sha,
            "checkpoint_path": checkpoint_record.get("path"),
            "global_step": checkpoint_record.get("global_step"),
        },
        "previous_round_completion": (
            {
                "path": str(previous_round_completion_path.resolve()),
                "sha256": sha256_file(previous_round_completion_path),
            }
            if previous_round_completion_path is not None
            else None
        ),
    }
    write_json(output, result)
    return result


def complete_psd_round(
    *,
    rollout_gate_path: Path,
    training_profile_path: Path,
    output_checkpoint_manifest_path: Path,
    output: Path,
    initialization_gate_path: Path | None = None,
) -> dict[str, Any]:
    """Close a round only after a gated optimizer run emitted a new checkpoint."""

    rollout = _load_object(rollout_gate_path)
    training = _load_object(training_profile_path)
    checkpoint = _load_object(output_checkpoint_manifest_path)
    start_checkpoint = _mapping(rollout.get("round_start_checkpoint_manifest"))
    start_manifest = _load_object(Path(start_checkpoint["path"]))
    output_checkpoint = _mapping(checkpoint.get("checkpoint"))
    checkpoint_dataset = _mapping(checkpoint.get("training_dataset"))
    checkpoint_sha = sha256_file(output_checkpoint_manifest_path)
    checkpoint_save = _mapping(training.get("checkpoint_save"))
    raw_gate = _mapping(training.get("raw_dataset_gate"))
    input_gate = _mapping(raw_gate.get("verification"))
    input_manifest = _mapping(input_gate.get("manifest"))
    initialization = _load_object(initialization_gate_path) if initialization_gate_path else {}
    saved_paths = {
        _text(item.get("path"))
        for item in checkpoint_save.get("states", [])
        if isinstance(item, Mapping)
    }
    def weights(manifest):
        return sorted((item.get("path"), item.get("sha256"))
            for item in manifest.get("artifacts", [])
            if item.get("scope") == "model"
            and str(item.get("path", "")).endswith((".safetensors", ".bin")))

    def artifacts_intact(manifest):
        artifacts = manifest.get("artifacts", [])
        if not artifacts:
            return False
        record = manifest["checkpoint"]
        for item in artifacts:
            root = Path(record.get("training_state_path") or record["path"]) if item["scope"] == "training_state" else Path(record["path"])
            root = root.resolve()
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != item.get("sha256"):
                return False
        return True

    checks = {
        "rollout_gate_schema": _check(
            rollout.get("schema_version"), PSD_ROLLOUT_GATE_SCHEMA_VERSION
        ),
        "rollout_gate_passed": _check(rollout.get("passed"), True),
        "training_production_gate": _check(
            training.get("passed_production_gate"), True
        ),
        "initialization_gate_passed": _check(initialization.get("passed"), True),
        "optimizer_started_from_rollout_policy": _check(
            initialization.get("checkpoint_manifest_sha256"), start_checkpoint.get("sha256")),
        "output_checkpoint_schema": _check(
            checkpoint.get("schema_version"), "ifv-qwen-checkpoint-manifest-v1"
        ),
        "output_checkpoint_was_saved": {
            "passed": bool(
                {
                    _text(output_checkpoint.get("path")),
                    _text(output_checkpoint.get("training_state_path")),
                }
                & saved_paths
            ),
            "actual": {
                "path": _text(output_checkpoint.get("path")),
                "training_state_path": _text(
                    output_checkpoint.get("training_state_path")
                ),
            },
            "expected_one_of": sorted(saved_paths),
        },
        "training_dataset_manifest_bound": _check(
            _text(checkpoint_dataset.get("manifest_sha256")),
            _text(input_manifest.get("sha256")),
        ),
        "dataset_hash_present": {"passed": bool(_text(input_manifest.get("sha256")))},
        "input_checkpoint_manifest_unchanged": _check(
            sha256_file(Path(start_checkpoint["path"])), start_checkpoint.get("sha256")),
        "optimizer_step_completed": {"passed": type(output_checkpoint.get("global_step")) is int
            and output_checkpoint["global_step"] > 0},
        "resumable_training_state": {"passed": all(output_checkpoint.get(name) is True
            for name in ("optimizer_state_available", "scheduler_state_available", "rng_state_available"))
            and any(item.get("scope") == "training_state" for item in checkpoint.get("artifacts", []))},
        "output_artifacts_intact": {"passed": artifacts_intact(checkpoint)},
        "weight_artifacts_changed": {"passed": bool(weights(checkpoint))
            and bool(weights(start_manifest)) and weights(checkpoint) != weights(start_manifest)},
        "checkpoint_changed": {
            "passed": checkpoint_sha != _text(start_checkpoint.get("sha256")),
            "actual": checkpoint_sha,
            "not_expected": _text(start_checkpoint.get("sha256")),
        },
    }
    passed = all(item.get("passed") is True for item in checks.values())
    result = {
        "schema_version": PSD_ROUND_COMPLETION_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "round_index": rollout.get("round_index"),
        "passed": passed,
        "checks": checks,
        "rollout_gate": {
            "path": str(rollout_gate_path.resolve()),
            "sha256": sha256_file(rollout_gate_path),
            "run_id": _mapping(rollout.get("run")).get("run_id"),
        },
        "training_profile": {
            "path": str(training_profile_path.resolve()),
            "sha256": sha256_file(training_profile_path),
        },
        "input_checkpoint_manifest": dict(start_checkpoint),
        "initialization_gate": {
            "path": str(initialization_gate_path.resolve()) if initialization_gate_path else None,
            "sha256": sha256_file(initialization_gate_path) if initialization_gate_path else None,
        },
        "output_checkpoint_manifest": {
            "path": str(output_checkpoint_manifest_path.resolve()),
            "sha256": checkpoint_sha,
            "checkpoint_path": output_checkpoint.get("path"),
            "global_step": output_checkpoint.get("global_step"),
        },
    }
    write_json(output, result)
    return result
