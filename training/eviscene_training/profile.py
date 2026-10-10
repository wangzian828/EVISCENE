from __future__ import annotations

import ast
import json
import re
import shlex
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


ERROR_PATTERNS = {
    "cuda_oom": re.compile(
        r"(torch\.cuda\.OutOfMemoryError|CUDA out of memory|CUDACachingAllocator)",
        re.IGNORECASE,
    ),
    "cpu_oom_or_sigkill": re.compile(
        r"(exitcode\s*[-=]\s*9|SIGKILL|Killed\b|oom-kill|out of memory killer)",
        re.IGNORECASE,
    ),
    "nccl": re.compile(r"\bNCCL\b.*(error|failed|unhandled|timeout)", re.IGNORECASE),
    "nan_or_inf": re.compile(r"\b(nan|inf)\b", re.IGNORECASE),
    "traceback": re.compile(r"^Traceback \(most recent call last\):"),
}
SCHEDULER_ORDER_WARNING = re.compile(
    r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`"
)

METRIC_KEYS = {
    "loss",
    "eval_loss",
    "global_step/max_steps",
    "train_speed(s/it)",
    "memory(GiB)",
    "token_acc",
    "grad_norm",
    "learning_rate",
    "epoch",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "train_tokens_per_second",
    "num_tokens",
    "eval_runtime",
}


def _coerce_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        if re.fullmatch(r"[-+]?\d+", stripped):
            return int(stripped)
        if re.fullmatch(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", stripped):
            return float(stripped)
    except ValueError:
        return value
    return value


def _parse_metric_dict(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return None
    payload = line[start : end + 1]
    try:
        parsed = ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    if not METRIC_KEYS.intersection(str(key) for key in parsed):
        return None
    return {str(key): _coerce_number(value) for key, value in parsed.items()}


def _parse_step(value: Any) -> tuple[int | None, int | None]:
    if not isinstance(value, str) or "/" not in value:
        return None, None
    left, right = value.split("/", 1)
    try:
        return int(left.strip()), int(right.strip())
    except ValueError:
        return None, None


def read_training_metric_rows(train_log: Path) -> list[dict[str, Any]]:
    """Read normalized metric dictionaries from an ms-swift training log."""

    rows: list[dict[str, Any]] = []
    with train_log.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            metric = _parse_metric_dict(line)
            if metric is None:
                continue
            current_step, max_steps = _parse_step(
                metric.get("global_step/max_steps")
            )
            if current_step is not None:
                metric["global_step"] = current_step
            if max_steps is not None:
                metric["max_steps"] = max_steps
            rows.append(metric)
    return rows


def _metric_value(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _elapsed_seconds(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    total = 0.0
    matched = False
    for amount, suffix in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", value):
        matched = True
        factor = {"h": 3600.0, "m": 60.0, "s": 1.0}[suffix]
        total += float(amount) * factor
    return total if matched else None


def _command_value(command: str, name: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    flag = f"--{name}"
    for index, token in enumerate(tokens[:-1]):
        if token == flag:
            return tokens[index + 1]
    return None


def _int_value(value: str | None, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_value(value: str | None, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _checkpoint_step(path: str | None) -> int | None:
    if not path:
        return None
    match = re.search(r"(?:^|[/\\])checkpoint-(\d+)(?:[/\\]?$)", path)
    return int(match.group(1)) if match else None


def _checkpoint_state(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {
            "path": "",
            "exists": False,
            "global_step": None,
            "optimizer_state_available": False,
            "scheduler_state_available": False,
            "rng_state_available": False,
            "file_count": 0,
            "bytes": 0,
        }
    path = Path(path_value)
    exists = path.is_dir()
    files = list(path.rglob("*")) if exists else []
    files = [item for item in files if item.is_file()]
    relative_names = [str(item.relative_to(path)) for item in files]
    global_step = _checkpoint_step(path_value)
    trainer_state = path / "trainer_state.json"
    if trainer_state.is_file():
        try:
            payload = json.loads(trainer_state.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(
                payload.get("global_step"), int
            ):
                global_step = payload["global_step"]
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "path": str(path),
        "exists": exists,
        "global_step": global_step,
        "optimizer_state_available": any(
            "optim" in name.lower()
            or name.startswith("optimizer_")
            for name in relative_names
        ),
        "scheduler_state_available": any(
            Path(name).name == "scheduler.pt" for name in relative_names
        ),
        "rng_state_available": any(
            Path(name).name.startswith("rng_state")
            for name in relative_names
        ),
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def _load_sidecar(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"sidecar must contain a JSON object: {path}")
    return value


def summarize_training_log(
    train_log: Path,
    *,
    steady_window: int = 5,
    experiment_id: str = "",
    profile_id: str = "",
    resource_summary: Path | None = None,
    cache_verification: Path | None = None,
    dataset_verification: Path | None = None,
    environment_preflight: Path | None = None,
    encode_cache_report: Path | None = None,
    scheduler_audit: Path | None = None,
    checkpoint_preflight: Path | None = None,
    checkpoint_io_profile: Path | None = None,
    train_exit_code: int | None = None,
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    errors: dict[str, list[int]] = {name: [] for name in ERROR_PATTERNS}
    line_count = 0
    launch_command = ""
    world_size: int | None = None
    train_dataset_size: int | None = None
    checkpoint_save_paths: list[str] = []
    scheduler_order_warning_lines: list[int] = []
    last_model_checkpoint = ""
    best_model_checkpoint = ""
    end_time_observed = False

    with train_log.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line_count = line_number
            if line.startswith("Executing:") and not launch_command:
                launch_command = line.removeprefix("Executing:").strip()
            elif line.startswith("run sh: `") and not launch_command:
                launch_command = line.removeprefix("run sh: `").rsplit("`", 1)[0]
            if world_size is None:
                match = re.search(r"\bworld_size:\s*(\d+)", line)
                if match:
                    world_size = int(match.group(1))
            if train_dataset_size is None and "train_dataset" in line:
                match = re.search(r"\bsize=(\d+)", line)
                if match:
                    train_dataset_size = int(match.group(1))
            checkpoint_match = re.search(
                r"Saving model checkpoint to\s+(\S+)",
                line,
            )
            if checkpoint_match:
                checkpoint_save_paths.append(checkpoint_match.group(1))
            last_checkpoint_match = re.search(
                r"last_model_checkpoint:\s*(\S+)",
                line,
            )
            if last_checkpoint_match:
                last_model_checkpoint = last_checkpoint_match.group(1)
            best_checkpoint_match = re.search(
                r"best_model_checkpoint:\s*(\S+)",
                line,
            )
            if best_checkpoint_match:
                best_model_checkpoint = best_checkpoint_match.group(1)
            if "End time of running main:" in line:
                end_time_observed = True
            for name, pattern in ERROR_PATTERNS.items():
                if pattern.search(line):
                    errors[name].append(line_number)
            if SCHEDULER_ORDER_WARNING.search(line):
                scheduler_order_warning_lines.append(line_number)
            metric = _parse_metric_dict(line)
            if metric is None:
                continue
            current_step, max_steps = _parse_step(metric.get("global_step/max_steps"))
            if current_step is not None:
                metric["global_step"] = current_step
            if max_steps is not None:
                metric["max_steps"] = max_steps
            metrics.append(metric)

    train_step_metrics = [
        row
        for row in metrics
        if "loss" in row and "eval_loss" not in row and "train_loss" not in row
    ]
    eval_metrics = [row for row in metrics if "eval_loss" in row]
    summary_metrics = [row for row in metrics if "train_runtime" in row]
    speed_source = train_step_metrics if train_step_metrics else metrics
    speed_values = [
        value
        for row in speed_source
        if (value := _metric_value(row, "train_speed(s/it)", "train_runtime_seconds_per_step"))
        is not None
    ]
    memory_values = [
        value for row in metrics if (value := _metric_value(row, "memory(GiB)")) is not None
    ]
    loss_values = [value for row in metrics if (value := _metric_value(row, "loss")) is not None]
    eval_loss_values = [
        value for row in metrics if (value := _metric_value(row, "eval_loss")) is not None
    ]
    epoch_values = [
        value for row in metrics if (value := _metric_value(row, "epoch")) is not None
    ]
    steps = [int(row["global_step"]) for row in metrics if isinstance(row.get("global_step"), int)]
    steady_values = speed_values[-steady_window:] if steady_window > 0 else []
    elapsed_values = [
        value
        for row in train_step_metrics
        if (value := _elapsed_seconds(row.get("elapsed_time"))) is not None
    ]
    step_wall_values: list[float] = []
    for index, cumulative_mean in enumerate(speed_values):
        previous_total = speed_values[index - 1] * index if index else 0.0
        current_total = cumulative_mean * (index + 1)
        step_wall_values.append(
            round(max(0.0, current_total - previous_total), 6)
        )
    steady_step_wall = (
        step_wall_values[-steady_window:] if steady_window > 0 else []
    )
    step_wall_mean = (
        round(mean(step_wall_values), 6) if step_wall_values else None
    )
    steady_step_wall_mean = (
        round(mean(steady_step_wall), 6) if steady_step_wall else None
    )
    steady_step_wall_median = (
        round(median(steady_step_wall), 6) if steady_step_wall else None
    )
    steady_step_wall_p90 = (
        round(float(_quantile(steady_step_wall, 0.90)), 6)
        if steady_step_wall
        else None
    )
    steady_step_wall_max = (
        round(max(steady_step_wall), 6) if steady_step_wall else None
    )
    steady_step_wall_cv = (
        round(pstdev(steady_step_wall) / steady_step_wall_mean, 6)
        if len(steady_step_wall) > 1
        and steady_step_wall_mean is not None
        and steady_step_wall_mean > 0
        else 0.0
        if steady_step_wall
        else None
    )
    stall_threshold = (
        round(steady_step_wall_median * 1.5, 6)
        if steady_step_wall_median is not None
        else None
    )
    steady_stall_count = (
        sum(value > stall_threshold for value in steady_step_wall)
        if stall_threshold is not None
        else 0
    )
    startup_step_count = max(
        0,
        len(step_wall_values) - len(steady_step_wall),
    )

    train_batch_size = _int_value(
        _command_value(launch_command, "per_device_train_batch_size")
    )
    gradient_accumulation = _int_value(
        _command_value(launch_command, "gradient_accumulation_steps")
    )
    sequence_parallel_size = _int_value(
        _command_value(launch_command, "sequence_parallel_size"),
        1,
    )
    data_parallel_size = None
    if (
        world_size is not None
        and sequence_parallel_size is not None
        and sequence_parallel_size > 0
        and world_size % sequence_parallel_size == 0
    ):
        data_parallel_size = world_size // sequence_parallel_size
    unique_samples_per_step = None
    if (
        train_batch_size is not None
        and gradient_accumulation is not None
        and data_parallel_size is not None
    ):
        unique_samples_per_step = (
            train_batch_size * gradient_accumulation * data_parallel_size
        )
    observed_train_steps = len(train_step_metrics)
    observed_unique_samples = (
        observed_train_steps * unique_samples_per_step
        if unique_samples_per_step is not None
        else None
    )
    train_elapsed = sum(step_wall_values) if step_wall_values else (
        elapsed_values[-1] if elapsed_values else None
    )
    unique_samples_per_second = (
        round(observed_unique_samples / train_elapsed, 6)
        if observed_unique_samples is not None and train_elapsed
        else None
    )
    steady_unique_samples_per_second = (
        round(unique_samples_per_step / steady_step_wall_mean, 6)
        if unique_samples_per_step is not None
        and steady_step_wall_mean
        and steady_step_wall_mean > 0
        else None
    )

    max_steps = _int_value(_command_value(launch_command, "max_steps"))
    num_train_epochs = _float_value(
        _command_value(launch_command, "num_train_epochs")
    )
    eval_strategy = _command_value(launch_command, "eval_strategy") or ""
    save_strategy = _command_value(launch_command, "save_strategy") or ""
    smoke_only = eval_strategy.lower() == "no"
    validation_required = bool(eval_strategy and eval_strategy.lower() != "no")
    checkpoint_required = bool(save_strategy and save_strategy.lower() != "no")
    save_only_model = (
        (_command_value(launch_command, "save_only_model") or "false").lower()
        == "true"
    )
    resume_checkpoint = _command_value(
        launch_command,
        "resume_from_checkpoint",
    )
    resume_state = _checkpoint_state(resume_checkpoint)
    unique_checkpoint_paths = list(dict.fromkeys(checkpoint_save_paths))
    if last_model_checkpoint and last_model_checkpoint not in unique_checkpoint_paths:
        unique_checkpoint_paths.append(last_model_checkpoint)
    saved_checkpoint_states = [
        _checkpoint_state(path) for path in unique_checkpoint_paths
    ]
    last_step = max(steps) if steps else None
    last_epoch = max(epoch_values) if epoch_values else None
    epoch_training_complete = (
        num_train_epochs is not None
        and last_step is not None
        and (
            (
                last_epoch is not None
                and last_epoch >= num_train_epochs - 1e-6
            )
            or (end_time_observed and bool(summary_metrics))
        )
    )
    training_steps_complete = (
        last_step is not None
        and max_steps is not None
        and last_step >= max_steps
    ) or epoch_training_complete
    resume_requested = bool(resume_checkpoint)
    resume_advanced = (
        resume_requested
        and isinstance(resume_state.get("global_step"), int)
        and last_step is not None
        and last_step > int(resume_state["global_step"])
    )
    resource_payload = _load_sidecar(resource_summary)
    cache_payload = _load_sidecar(cache_verification)
    dataset_verification_payload = _load_sidecar(dataset_verification)
    environment_preflight_payload = _load_sidecar(environment_preflight)
    encode_cache_payload = _load_sidecar(encode_cache_report)
    scheduler_audit_payload = _load_sidecar(scheduler_audit)
    checkpoint_preflight_payload = _load_sidecar(checkpoint_preflight)
    checkpoint_io_payload = _load_sidecar(checkpoint_io_profile)
    clean_exit = (
        train_exit_code == 0
        if train_exit_code is not None
        else end_time_observed
    )
    validation_passed = bool(eval_metrics)
    checkpoint_training_state_required = checkpoint_required and not save_only_model
    checkpoint_passed = any(
        item.get("exists") is True
        and (
            not checkpoint_training_state_required
            or (
                item.get("optimizer_state_available") is True
                and item.get("scheduler_state_available") is True
                and item.get("rng_state_available") is True
            )
        )
        for item in saved_checkpoint_states
    )
    resource_exit_code = (
        resource_payload.get("exit_code")
        if isinstance(resource_payload, dict)
        else None
    )
    resource_exit_matches = (
        train_exit_code is None
        or resource_exit_code is None
        or resource_exit_code == train_exit_code
    )
    resource_acceptance = (
        resource_payload.get("acceptance")
        if isinstance(resource_payload, dict)
        else None
    )
    resource_acceptance_passed = (
        resource_acceptance.get("passed") is True
        if isinstance(resource_acceptance, dict)
        and resource_acceptance.get("required") is True
        else True
    )
    resource_required = resource_summary is not None
    resource_passed = (
        isinstance(resource_payload, dict)
        and resource_exit_matches
        and resource_acceptance_passed
        if resource_required
        else True
    )
    detected_errors = {name: lines for name, lines in errors.items() if lines}
    cache_required = _command_value(launch_command, "cached_dataset") is not None
    cache_passed = (
        cache_payload.get("passed") is True
        if isinstance(cache_payload, dict)
        else not cache_required
    )
    raw_dataset_required = _command_value(launch_command, "dataset") is not None
    raw_dataset_passed = (
        dataset_verification_payload.get("passed") is True
        if isinstance(dataset_verification_payload, dict)
        else not raw_dataset_required
    )
    environment_preflight_required = environment_preflight is not None
    environment_preflight_passed = (
        environment_preflight_payload.get("passed") is True
        if isinstance(environment_preflight_payload, dict)
        else not environment_preflight_required
    )
    encode_cache_required = encode_cache_report is not None
    encode_cache_passed = (
        encode_cache_payload.get("passed") is True
        if isinstance(encode_cache_payload, dict)
        else not encode_cache_required
    )
    scheduler_audit_required = bool(scheduler_order_warning_lines)
    scheduler_audit_passed = (
        scheduler_audit_payload.get("passed") is True
        and scheduler_audit_payload.get("classification")
        == "wrapper_false_positive"
        if isinstance(scheduler_audit_payload, dict)
        else not scheduler_audit_required
    )
    checkpoint_preflight_required = checkpoint_required
    checkpoint_preflight_passed = (
        checkpoint_preflight_payload.get("passed") is True
        if isinstance(checkpoint_preflight_payload, dict)
        else not checkpoint_preflight_required
    )
    checkpoint_io_required = checkpoint_required
    checkpoint_io_passed = (
        checkpoint_io_payload.get("passed") is True
        if isinstance(checkpoint_io_payload, dict)
        else not checkpoint_io_required
    )
    passed_production_gate = all(
        (
            not detected_errors,
            clean_exit,
            training_steps_complete,
            not smoke_only,
            validation_passed or not validation_required,
            checkpoint_passed or not checkpoint_required,
            resume_advanced or not resume_requested,
            resource_passed,
            cache_passed,
            raw_dataset_passed,
            environment_preflight_passed,
            encode_cache_passed,
            scheduler_audit_passed,
            checkpoint_preflight_passed,
            checkpoint_io_passed,
        )
    )
    train_runtime = (
        _metric_value(summary_metrics[-1], "train_runtime")
        if summary_metrics
        else None
    )
    eval_runtime = (
        _metric_value(eval_metrics[-1], "eval_runtime") if eval_metrics else None
    )
    trainer_throughput = {
        "train_samples_per_second": (
            _metric_value(summary_metrics[-1], "train_samples_per_second")
            if summary_metrics
            else None
        ),
        "train_steps_per_second": (
            _metric_value(summary_metrics[-1], "train_steps_per_second")
            if summary_metrics
            else None
        ),
        "train_tokens_per_second": (
            _metric_value(summary_metrics[-1], "train_tokens_per_second")
            if summary_metrics
            else None
        ),
        "num_tokens": (
            _metric_value(summary_metrics[-1], "num_tokens")
            if summary_metrics
            else None
        ),
    }
    post_train_eval_finalize_seconds = None
    if train_runtime is not None and train_elapsed is not None:
        post_train_eval_finalize_seconds = round(
            max(
                0.0,
                train_runtime - train_elapsed - (eval_runtime or 0.0),
            ),
            6,
        )

    return {
        "schema_version": "ifv-training-profile-v1",
        "experiment_id": experiment_id,
        "profile_id": profile_id,
        "train_log": str(train_log),
        "line_count": line_count,
        "metric_rows": len(metrics),
        "train_step_metric_rows": len(train_step_metrics),
        "eval_metric_rows": len(eval_metrics),
        "summary_metric_rows": len(summary_metrics),
        "detected_errors": detected_errors,
        "passed_basic_log_gate": not detected_errors,
        "passed_production_gate": passed_production_gate,
        "smoke_only": smoke_only,
        "run_mode": "smoke_only" if smoke_only else "production_candidate",
        "launch": {
            "command": launch_command,
            "max_length": _int_value(_command_value(launch_command, "max_length")),
            "attention_implementation": _command_value(
                launch_command, "attn_impl"
            ),
            "padding_free": (
                (_command_value(launch_command, "padding_free") or "false").lower()
                == "true"
            ),
            "tuner_type": _command_value(launch_command, "tuner_type") or "",
            "fsdp": _command_value(launch_command, "fsdp") or "",
            "deepspeed": _command_value(launch_command, "deepspeed") or "",
        },
        "steps": {
            "observed": steps,
            "last": last_step,
            "configured_max_steps": max_steps,
            "configured_num_train_epochs": num_train_epochs,
            "last_observed_epoch": last_epoch,
            "complete": training_steps_complete,
        },
        "speed_seconds_per_step": {
            "count": len(speed_values),
            "first": speed_values[0] if speed_values else None,
            "last": speed_values[-1] if speed_values else None,
            "mean": mean(speed_values) if speed_values else None,
            "steady_window": steady_window,
            "steady_mean": mean(steady_values) if steady_values else None,
        },
        "step_wall_seconds": {
            "count": len(step_wall_values),
            "values": step_wall_values,
            "mean": step_wall_mean,
            "startup_count": startup_step_count,
            "steady_window": steady_window,
            "steady_count": len(steady_step_wall),
            "steady_mean": steady_step_wall_mean,
            "steady_median": steady_step_wall_median,
            "steady_p90": steady_step_wall_p90,
            "steady_max": steady_step_wall_max,
            "steady_coefficient_of_variation": steady_step_wall_cv,
            "steady_stall_threshold": stall_threshold,
            "steady_stall_count": steady_stall_count,
        },
        "parallelism": {
            "world_size": world_size,
            "sequence_parallel_size": sequence_parallel_size,
            "data_parallel_size": data_parallel_size,
            "per_device_train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation,
            "unique_samples_per_optimizer_step": unique_samples_per_step,
        },
        "throughput": {
            "train_dataset_size": train_dataset_size,
            "observed_train_steps": observed_train_steps,
            "observed_unique_samples": observed_unique_samples,
            "train_elapsed_seconds": train_elapsed,
            "unique_samples_per_second": unique_samples_per_second,
            "steady_unique_samples_per_second": (
                steady_unique_samples_per_second
            ),
            "reported_by_trainer": trainer_throughput,
        },
        "memory_gib": {
            "count": len(memory_values),
            "max": max(memory_values) if memory_values else None,
            "last": memory_values[-1] if memory_values else None,
        },
        "loss": {
            "count": len(loss_values),
            "last": loss_values[-1] if loss_values else None,
        },
        "eval_loss": {
            "count": len(eval_loss_values),
            "last": eval_loss_values[-1] if eval_loss_values else None,
        },
        "runtime_seconds": {
            "train_runtime": train_runtime,
            "eval_runtime_last": eval_runtime,
            "post_train_eval_finalize": post_train_eval_finalize_seconds,
        },
        "validation": {
            "required": validation_required,
            "observed": validation_passed,
            "metric_rows": len(eval_metrics),
            "loss": eval_loss_values[-1] if eval_loss_values else None,
        },
        "checkpoint_save": {
            "required": checkpoint_required,
            "save_only_model": save_only_model,
            "training_state_required": checkpoint_training_state_required,
            "passed": checkpoint_passed,
            "observed": bool(unique_checkpoint_paths),
            "paths": unique_checkpoint_paths,
            "states": saved_checkpoint_states,
            "last_model_checkpoint": last_model_checkpoint,
            "best_model_checkpoint": best_model_checkpoint,
        },
        "resume": {
            "requested": resume_requested,
            "source": resume_state,
            "advanced": resume_advanced,
            "first_observed_step": steps[0] if steps else None,
            "last_observed_step": last_step,
        },
        "process": {
            "train_exit_code": train_exit_code,
            "clean_exit": clean_exit,
            "end_time_observed": end_time_observed,
            "resource_exit_matches": resource_exit_matches,
        },
        "resources": {
            "required": resource_required,
            "passed": resource_passed,
            "acceptance_passed": resource_acceptance_passed,
            "summary_path": str(resource_summary) if resource_summary else "",
            "summary": resource_payload,
        },
        "cached_dataset_gate": {
            "required": cache_required,
            "verification_path": (
                str(cache_verification) if cache_verification else ""
            ),
            "passed": cache_passed,
            "verification": cache_payload,
        },
        "raw_dataset_gate": {
            "required": raw_dataset_required,
            "verification_path": (
                str(dataset_verification) if dataset_verification else ""
            ),
            "passed": raw_dataset_passed,
            "verification": dataset_verification_payload,
        },
        "environment_preflight": {
            "required": environment_preflight_required,
            "report_path": (
                str(environment_preflight) if environment_preflight else ""
            ),
            "passed": environment_preflight_passed,
            "report": environment_preflight_payload,
        },
        "encoded_processor_cache": {
            "required": encode_cache_required,
            "report_path": (
                str(encode_cache_report) if encode_cache_report else ""
            ),
            "passed": encode_cache_passed,
            "report": encode_cache_payload,
        },
        "scheduler_order": {
            "warning_lines": scheduler_order_warning_lines,
            "warning_count": len(scheduler_order_warning_lines),
            "audit_required": scheduler_audit_required,
            "audit_path": str(scheduler_audit) if scheduler_audit else "",
            "passed": scheduler_audit_passed,
            "audit": scheduler_audit_payload,
        },
        "checkpoint_storage_preflight": {
            "required": checkpoint_preflight_required,
            "report_path": (
                str(checkpoint_preflight) if checkpoint_preflight else ""
            ),
            "passed": checkpoint_preflight_passed,
            "report": checkpoint_preflight_payload,
        },
        "checkpoint_io": {
            "required": checkpoint_io_required,
            "report_path": (
                str(checkpoint_io_profile) if checkpoint_io_profile else ""
            ),
            "passed": checkpoint_io_passed,
            "report": checkpoint_io_payload,
        },
    }


def write_training_profile(
    *,
    train_log: Path,
    output: Path,
    steady_window: int = 5,
    experiment_id: str = "",
    profile_id: str = "",
    resource_summary: Path | None = None,
    cache_verification: Path | None = None,
    dataset_verification: Path | None = None,
    environment_preflight: Path | None = None,
    encode_cache_report: Path | None = None,
    scheduler_audit: Path | None = None,
    checkpoint_preflight: Path | None = None,
    checkpoint_io_profile: Path | None = None,
    train_exit_code: int | None = None,
) -> dict[str, Any]:
    result = summarize_training_log(
        train_log,
        steady_window=steady_window,
        experiment_id=experiment_id,
        profile_id=profile_id,
        resource_summary=resource_summary,
        cache_verification=cache_verification,
        dataset_verification=dataset_verification,
        environment_preflight=environment_preflight,
        encode_cache_report=encode_cache_report,
        scheduler_audit=scheduler_audit,
        checkpoint_preflight=checkpoint_preflight,
        checkpoint_io_profile=checkpoint_io_profile,
        train_exit_code=train_exit_code,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
