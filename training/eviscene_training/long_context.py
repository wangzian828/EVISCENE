from __future__ import annotations

import hashlib
import json
import math
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from eviscene.environment import normalize_environment

from .io import sha256_file, write_json


LONG_CONTEXT_BOUNDARIES = (32_768, 65_536, 98_304, 120_000, 131_072)
STAGE_ORDER = ("memory-probe", "canary", "resume")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def parse_env_profile(path: Path) -> dict[str, str]:
    """Parse the repository's literal KEY=VALUE profiles without executing them."""

    values: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key.startswith(("EVISCENE_", "IFV_")) and key != "CELOSS_PARALLEL_SIZE":
            raise ValueError(f"{path}:{line_number}: unsupported profile key {key!r}")
        try:
            tokens = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if len(tokens) > 1:
            raise ValueError(f"{path}:{line_number}: value must be one shell word")
        values[key] = tokens[0] if tokens else ""
    return normalize_environment(values)


def _as_int(values: Mapping[str, str], key: str) -> int | None:
    try:
        return int(values[key])
    except (KeyError, ValueError):
        return None


def _as_bool(values: Mapping[str, str], key: str) -> bool | None:
    value = values.get(key, "").lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _check(actual: Any, expected: Any) -> dict[str, Any]:
    return {"passed": actual == expected, "actual": actual, "expected": expected}


def validate_sp8_128k_profile(
    profile: Mapping[str, str], *, stage: str | None = None
) -> dict[str, Any]:
    profile = normalize_environment(profile)
    checks = {
        "full_parameter_training": _check(profile.get("EVISCENE_TUNER_TYPE"), "full"),
        "deepspeed_disabled": _check(profile.get("EVISCENE_DEEPSPEED", ""), ""),
        "fsdp2_enabled": _check(profile.get("EVISCENE_FSDP"), "fsdp2"),
        "fsdp_version_2": _check(_as_int(profile, "EVISCENE_FSDP_VERSION"), 2),
        "eight_gpus": _check(_as_int(profile, "EVISCENE_EXPECTED_GPU_COUNT"), 8),
        "sequence_parallel_8": _check(
            _as_int(profile, "EVISCENE_SEQUENCE_PARALLEL_SIZE"), 8
        ),
        "context_128k": _check(_as_int(profile, "EVISCENE_MAX_LENGTH"), 131_072),
        "real_120k_boundary": {
            "passed": (_as_int(profile, "EVISCENE_MIN_PROCESSOR_TRAIN_INPUT_TOKENS") or 0)
            >= 120_000,
            "actual": _as_int(profile, "EVISCENE_MIN_PROCESSOR_TRAIN_INPUT_TOKENS"),
            "expected": ">=120000",
        },
        "real_120k_row_count": {
            "passed": (
                _as_int(profile, "EVISCENE_MIN_PROCESSOR_TRAIN_ROWS_AT_OR_ABOVE") or 0
            )
            >= 1,
            "actual": _as_int(
                profile, "EVISCENE_MIN_PROCESSOR_TRAIN_ROWS_AT_OR_ABOVE"
            ),
            "expected": ">=1",
        },
        "batch_size_1": _check(_as_int(profile, "EVISCENE_TRAIN_BATCH_SIZE"), 1),
        "gradient_accumulation_1": _check(
            _as_int(profile, "EVISCENE_GRADIENT_ACCUMULATION_STEPS"), 1
        ),
        "padding_free": _check(_as_bool(profile, "EVISCENE_PADDING_FREE"), True),
        "flash_attention": {
            "passed": profile.get("EVISCENE_ATTN_IMPL", "").lower().startswith("flash"),
            "actual": profile.get("EVISCENE_ATTN_IMPL"),
            "expected": "flash_attn",
        },
        "gradient_checkpointing": _check(
            _as_bool(profile, "EVISCENE_GRADIENT_CHECKPOINTING"), True
        ),
        "vit_gradient_checkpointing": _check(
            _as_bool(profile, "EVISCENE_VIT_GRADIENT_CHECKPOINTING"), True
        ),
        "no_logit_retention": _check(
            _as_bool(profile, "EVISCENE_USE_LOGITS_TO_KEEP"), False
        ),
        "fail_on_truncation": _check(
            profile.get("EVISCENE_TRUNCATION_STRATEGY"), "raise"
        ),
        "environment_preflight": _check(
            _as_bool(profile, "EVISCENE_REQUIRE_TRAINING_ENV_PREFLIGHT"), True
        ),
        "gpu_memory_floor": {
            "passed": (_as_int(profile, "EVISCENE_GPU_MEMORY_TARGET_MIN_MIB") or 0)
            >= 36_000,
            "actual": _as_int(profile, "EVISCENE_GPU_MEMORY_TARGET_MIN_MIB"),
            "expected": ">=36000 MiB",
        },
        "gpu_memory_ceiling": {
            "passed": 36_000
            <= (_as_int(profile, "EVISCENE_GPU_MEMORY_TARGET_MAX_MIB") or 0)
            <= 39_936,
            "actual": _as_int(profile, "EVISCENE_GPU_MEMORY_TARGET_MAX_MIB"),
            "expected": "36000..39936 MiB",
        },
        "gpu_utilization_floor": {
            "passed": (_as_int(profile, "EVISCENE_GPU_UTILIZATION_TARGET_MIN_PERCENT") or 0)
            >= 85,
            "actual": _as_int(profile, "EVISCENE_GPU_UTILIZATION_TARGET_MIN_PERCENT"),
            "expected": ">=85%",
        },
    }
    if stage is not None:
        if stage not in STAGE_ORDER:
            raise ValueError(f"unknown 128K stage: {stage}")
        stage_expected = {
            "memory-probe": (1, "no", "no"),
            "canary": (10, "steps", "steps"),
            "resume": (11, "steps", "steps"),
        }[stage]
        checks.update(
            {
                "stage_max_steps": _check(
                    _as_int(profile, "EVISCENE_MAX_STEPS"), stage_expected[0]
                ),
                "stage_eval_strategy": _check(
                    profile.get("EVISCENE_EVAL_STRATEGY"), stage_expected[1]
                ),
                "stage_save_strategy": _check(
                    profile.get("EVISCENE_SAVE_STRATEGY"), stage_expected[2]
                ),
            }
        )
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }


def _boundary_rows(report: Mapping[str, Any], train_path: str) -> list[dict[str, Any]]:
    by_dataset = report.get("input_token_boundaries_by_dataset")
    if not isinstance(by_dataset, Mapping):
        return []
    rows = by_dataset.get(train_path)
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _empirical_forecast(
    paths: Sequence[Path], *, total_steps: int
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for path in paths:
        profile = _load_object(path)
        launch = profile.get("launch") if isinstance(profile.get("launch"), Mapping) else {}
        parallel = (
            profile.get("parallelism")
            if isinstance(profile.get("parallelism"), Mapping)
            else {}
        )
        step_wall = (
            profile.get("step_wall_seconds")
            if isinstance(profile.get("step_wall_seconds"), Mapping)
            else {}
        )
        seconds = step_wall.get("steady_p90") or step_wall.get("steady_mean")
        compatible = all(
            (
                profile.get("passed_production_gate") is True,
                launch.get("max_length") == 131_072,
                parallel.get("world_size") == 8,
                parallel.get("sequence_parallel_size") == 8,
                isinstance(seconds, (int, float)) and float(seconds) > 0,
            )
        )
        candidates.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "compatible": compatible,
                "steady_seconds_per_step": seconds,
            }
        )
    compatible_rows = [item for item in candidates if item["compatible"]]
    selected = compatible_rows[-1] if compatible_rows else None
    seconds_total = (
        float(selected["steady_seconds_per_step"]) * total_steps if selected else None
    )
    return {
        "sources": candidates,
        "selected": selected,
        "forecast_available": selected is not None,
        "estimated_seconds": seconds_total,
        "estimated_hours": round(seconds_total / 3600.0, 3) if seconds_total else None,
        "note": (
            "Only a passed 128K/world8/SP8 production profile is extrapolated."
        ),
    }


def build_long_context_plan(
    *,
    processor_report_path: Path,
    train_jsonl: Path,
    sft_profile_paths: Sequence[Path],
    output: Path,
    epochs: float = 1.0,
    parameter_count: int = 9_000_000_000,
    empirical_profile_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    if epochs <= 0 or parameter_count <= 0:
        raise ValueError("epochs and parameter_count must be positive")
    report = _load_object(processor_report_path)
    train_resolved = train_jsonl.expanduser().resolve()
    train_key = str(train_resolved)
    distributions = report.get("input_tokens_by_dataset")
    distributions = distributions if isinstance(distributions, Mapping) else {}
    distribution = distributions.get(train_key)
    distribution = dict(distribution) if isinstance(distribution, Mapping) else {}
    boundary_rows = _boundary_rows(report, train_key)
    boundary_by_value = {
        int(item["boundary_tokens"]): item
        for item in boundary_rows
        if isinstance(item.get("boundary_tokens"), int)
    }
    file_records = report.get("dataset_files")
    file_records = file_records if isinstance(file_records, list) else []
    train_record = next(
        (
            dict(item)
            for item in file_records
            if isinstance(item, Mapping) and item.get("path") == train_key
        ),
        None,
    )
    current_sha = sha256_file(train_resolved)
    data_checks = {
        "processor_report_schema": {
            "passed": report.get("schema_version") in {
                "ifv-ms-swift-agent-processor-verification-v3",
                "ifv-ms-swift-agent-processor-verification-v4",
            },
            "actual": report.get("schema_version"),
            "expected": [
                "ifv-ms-swift-agent-processor-verification-v3",
                "ifv-ms-swift-agent-processor-verification-v4",
            ],
        },
        "processor_report_passed": _check(report.get("passed"), True),
        "train_file_covered": {"passed": train_record is not None},
        "train_file_unchanged": {
            "passed": bool(train_record and train_record.get("sha256") == current_sha),
            "actual": current_sha,
            "expected": train_record.get("sha256") if train_record else None,
        },
        "token_distribution_present": {
            "passed": isinstance(distribution.get("count"), int),
        },
        "boundary_distribution_present": {
            "passed": all(value in boundary_by_value for value in LONG_CONTEXT_BOUNDARIES)
        },
        "real_120k_row_present": {
            "passed": int(boundary_by_value.get(120_000, {}).get("rows_at_or_above", 0))
            >= 1,
            "actual": boundary_by_value.get(120_000, {}).get("rows_at_or_above", 0),
            "expected": ">=1",
        },
    }
    profile_reports: list[dict[str, Any]] = []
    for index, path in enumerate(sft_profile_paths):
        stage = STAGE_ORDER[index] if index < len(STAGE_ORDER) else None
        values = parse_env_profile(path)
        validation = validate_sp8_128k_profile(values, stage=stage)
        profile_reports.append(
            {
                "stage": stage,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                **validation,
            }
        )
    train_rows = int(distribution.get("count", 0))
    # SP8 spans a single sample across all ranks: data parallel size is one.
    global_batch = 1
    steps_per_epoch = math.ceil(train_rows / global_batch) if train_rows else 0
    total_steps = math.ceil(steps_per_epoch * epochs)
    global_state_bytes = parameter_count * 12
    per_rank_state_bytes = math.ceil(global_state_bytes / 8)
    memory_ceiling = 38_912 * 1024 * 1024
    result = {
        "schema_version": "ifv-long-context-capacity-plan-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed_static_gate": all(item["passed"] for item in data_checks.values())
        and len(profile_reports) == len(STAGE_ORDER)
        and all(item["passed"] for item in profile_reports),
        "processor_report": {
            "path": str(processor_report_path.resolve()),
            "sha256": sha256_file(processor_report_path),
        },
        "train_dataset": {
            "path": train_key,
            "sha256": current_sha,
            "distribution": distribution,
            "boundaries": [
                boundary_by_value.get(value, {"boundary_tokens": value})
                for value in LONG_CONTEXT_BOUNDARIES
            ],
        },
        "data_checks": data_checks,
        "profiles": profile_reports,
        "parallelism": {
            "world_size": 8,
            "sequence_parallel_size": 8,
            "data_parallel_size": 1,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "unique_samples_per_optimizer_step": global_batch,
        },
        "schedule": {
            "epochs": epochs,
            "train_rows": train_rows,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "optimizer_steps_total": total_steps,
        },
        "memory_budget": {
            "parameter_count_assumption": parameter_count,
            "bytes_per_parameter_training_state_lower_bound": 12,
            "global_sharded_training_state_lower_bound_bytes": global_state_bytes,
            "per_rank_sharded_training_state_lower_bound_mib": round(
                per_rank_state_bytes / (1024 * 1024), 1
            ),
            "configured_per_rank_memory_ceiling_mib": 38_912,
            "remaining_for_activations_collectives_allocator_mib": round(
                (memory_ceiling - per_rank_state_bytes) / (1024 * 1024), 1
            ),
            "fit_proven": False,
            "note": (
                "This is a lower bound; only the real 128K memory probe proves fit. "
                "The configured 36-38 GiB target intentionally uses most of each A100."
            ),
        },
        "throughput_forecast": _empirical_forecast(
            empirical_profile_paths, total_steps=total_steps
        ),
        "execution_order": [
            {
                "stage": "memory-probe",
                "requires": ["static gate", "eight idle A100 GPUs"],
            },
            {"stage": "canary", "requires": ["passed memory-probe gate"]},
            {"stage": "resume", "requires": ["passed canary gate", "checkpoint-10"]},
            {
                "stage": "full-training",
                "requires": ["passed resume gate", "user-approved final dataset"],
            },
        ],
    }
    write_json(output, result)
    return result


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _data_lineage(profile: Mapping[str, Any]) -> dict[str, Any]:
    raw_gate = profile.get("raw_dataset_gate")
    raw_gate = raw_gate if isinstance(raw_gate, Mapping) else {}
    verification = raw_gate.get("verification")
    verification = verification if isinstance(verification, Mapping) else {}
    datasets = verification.get("datasets")
    datasets = datasets if isinstance(datasets, Mapping) else {}
    processor = verification.get("processor_report")
    processor = processor if isinstance(processor, Mapping) else {}
    lineage = {
        "model": verification.get("model"),
        "template_contract": verification.get("template_contract"),
        "train_sha256": (
            datasets.get("train", {}).get("sha256")
            if isinstance(datasets.get("train"), Mapping)
            else None
        ),
        "validation_sha256": (
            datasets.get("validation", {}).get("sha256")
            if isinstance(datasets.get("validation"), Mapping)
            else None
        ),
        "processor_report_sha256": processor.get("sha256"),
    }
    return {**lineage, "fingerprint": _canonical_sha(lineage)}


def validate_128k_stage(
    *,
    stage: str,
    training_profile_path: Path,
    sft_profile_path: Path,
    output: Path,
    prior_gate_path: Path | None = None,
) -> dict[str, Any]:
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown 128K stage: {stage}")
    training = _load_object(training_profile_path)
    sft = parse_env_profile(sft_profile_path)
    config_validation = validate_sp8_128k_profile(sft, stage=stage)
    steps = training.get("steps") if isinstance(training.get("steps"), Mapping) else {}
    process = (
        training.get("process") if isinstance(training.get("process"), Mapping) else {}
    )
    resources = (
        training.get("resources")
        if isinstance(training.get("resources"), Mapping)
        else {}
    )
    raw_gate = (
        training.get("raw_dataset_gate")
        if isinstance(training.get("raw_dataset_gate"), Mapping)
        else {}
    )
    env_gate = (
        training.get("environment_preflight")
        if isinstance(training.get("environment_preflight"), Mapping)
        else {}
    )
    cache_gate = (
        training.get("encoded_processor_cache")
        if isinstance(training.get("encoded_processor_cache"), Mapping)
        else {}
    )
    lineage = _data_lineage(training)
    checks: dict[str, dict[str, Any]] = {
        "static_profile": {"passed": config_validation["passed"]},
        "basic_log_gate": _check(training.get("passed_basic_log_gate"), True),
        "clean_exit": _check(process.get("clean_exit"), True),
        "configured_steps_complete": _check(steps.get("complete"), True),
        "resource_gate_required": _check(resources.get("required"), True),
        "resource_gate_passed": _check(resources.get("passed"), True),
        "raw_dataset_gate_required": _check(raw_gate.get("required"), True),
        "raw_dataset_gate_passed": _check(raw_gate.get("passed"), True),
        "environment_gate_required": _check(env_gate.get("required"), True),
        "environment_gate_passed": _check(env_gate.get("passed"), True),
        "encode_cache_gate_required": _check(cache_gate.get("required"), True),
        "encode_cache_gate_passed": _check(cache_gate.get("passed"), True),
        "complete_data_lineage": {
            "passed": all(
                lineage.get(key)
                for key in (
                    "model",
                    "template_contract",
                    "train_sha256",
                    "validation_sha256",
                    "processor_report_sha256",
                )
            )
        },
    }
    if stage == "memory-probe":
        checks.update(
            {
                "smoke_mode": _check(training.get("run_mode"), "smoke_only"),
                "one_step_observed": _check(steps.get("last"), 1),
            }
        )
    else:
        checks["production_gate"] = _check(
            training.get("passed_production_gate"), True
        )
        checkpoint = training.get("checkpoint_save")
        checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
        checks["full_state_checkpoint"] = _check(checkpoint.get("passed"), True)
        expected_step = 10 if stage == "canary" else 11
        checks["expected_step_observed"] = _check(steps.get("last"), expected_step)
        if stage == "resume":
            resume = (
                training.get("resume")
                if isinstance(training.get("resume"), Mapping)
                else {}
            )
            source = resume.get("source") if isinstance(resume.get("source"), Mapping) else {}
            checks["resume_requested"] = _check(resume.get("requested"), True)
            checks["resume_advanced"] = _check(resume.get("advanced"), True)
            checks["resume_source_step_10"] = _check(source.get("global_step"), 10)

    expected_prior = None if stage == "memory-probe" else STAGE_ORDER[STAGE_ORDER.index(stage) - 1]
    prior_record = None
    if expected_prior is not None:
        if prior_gate_path is None:
            checks["prior_gate_present"] = {"passed": False, "expected": expected_prior}
        else:
            prior_record = _load_object(prior_gate_path)
            checks["prior_gate_passed"] = _check(prior_record.get("passed"), True)
            checks["prior_stage"] = _check(prior_record.get("stage"), expected_prior)
            prior_lineage = prior_record.get("data_lineage")
            prior_lineage = prior_lineage if isinstance(prior_lineage, Mapping) else {}
            checks["same_data_lineage"] = _check(
                lineage["fingerprint"], prior_lineage.get("fingerprint")
            )

    passed = all(item.get("passed") is True for item in checks.values())
    result = {
        "schema_version": "ifv-128k-stage-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "passed": passed,
        "checks": checks,
        "static_profile_validation": config_validation,
        "data_lineage": lineage,
        "training_profile": {
            "path": str(training_profile_path.resolve()),
            "sha256": sha256_file(training_profile_path),
        },
        "sft_profile": {
            "path": str(sft_profile_path.resolve()),
            "sha256": sha256_file(sft_profile_path),
        },
        "prior_gate": (
            {
                "path": str(prior_gate_path.resolve()),
                "sha256": sha256_file(prior_gate_path),
            }
            if prior_gate_path is not None
            else None
        ),
    }
    write_json(output, result)
    return result
