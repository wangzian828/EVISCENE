from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import deque
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from .profile import read_training_metric_rows, summarize_training_log


SCHEMA_VERSION = "ifv-sft-watchdog-v1"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError:
            return None
        return result if math.isfinite(result) else None
    return None


def _step(row: Mapping[str, Any]) -> tuple[int | None, int | None]:
    direct = row.get("global_step")
    if isinstance(direct, int):
        maximum = row.get("max_steps")
        return direct, maximum if isinstance(maximum, int) else None
    combined = row.get("global_step/max_steps")
    if not isinstance(combined, str) or "/" not in combined:
        return None, None
    left, right = combined.split("/", 1)
    try:
        return int(left.strip()), int(right.strip())
    except ValueError:
        return None, None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _tail_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines: deque[str] = deque(maxlen=max(1, limit))
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                lines.append(line)
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _discover_logging_jsonl(checkpoint_root: Path | None) -> Path | None:
    if checkpoint_root is None or not checkpoint_root.is_dir():
        return None
    candidates = [
        path for path in checkpoint_root.rglob("logging.jsonl") if path.is_file()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _metric_rows(train_log: Path, logging_jsonl: Path | None) -> list[dict[str, Any]]:
    if logging_jsonl is not None:
        rows = _read_jsonl(logging_jsonl)
        normalized: list[dict[str, Any]] = []
        for row in rows:
            if not {
                "loss",
                "eval_loss",
                "global_step/max_steps",
                "train_runtime",
            }.intersection(row):
                continue
            current, maximum = _step(row)
            normalized_row = dict(row)
            if current is not None:
                normalized_row["global_step"] = current
            if maximum is not None:
                normalized_row["max_steps"] = maximum
            normalized.append(normalized_row)
        if normalized:
            return normalized
    return read_training_metric_rows(train_log)


def _checkpoint_descriptors(checkpoint_root: Path | None) -> list[dict[str, Any]]:
    if checkpoint_root is None or not checkpoint_root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in checkpoint_root.rglob("checkpoint-*"):
        if not path.is_dir():
            continue
        state_path = path / "trainer_state.json"
        state: dict[str, Any] = {}
        if state_path.is_file():
            try:
                loaded = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = {}
            if isinstance(loaded, dict):
                state = loaded
        step = state.get("global_step")
        if not isinstance(step, int):
            try:
                step = int(path.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
        files = [item for item in path.rglob("*") if item.is_file()]
        names = [item.name.lower() for item in files]
        rows.append(
            {
                "path": str(path.resolve()),
                "global_step": step,
                "trainer_state": state_path.is_file(),
                "optimizer_state": any("optim" in name for name in names),
                "scheduler_state": "scheduler.pt" in names,
                "rng_state": any(name.startswith("rng_state") for name in names),
                "file_count": len(files),
                "bytes": sum(item.stat().st_size for item in files),
            }
        )
    rows.sort(key=lambda row: int(row["global_step"]))
    return rows


def _load_behavior_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    if path.suffix.lower() == ".jsonl":
        payloads: Any = _read_jsonl(path)
    else:
        payloads = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payloads, Mapping):
            payloads = payloads.get("records", [])
    if not isinstance(payloads, list):
        raise ValueError("behavior metrics must be a JSON array or JSONL")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(payloads):
        if not isinstance(raw, Mapping):
            raise ValueError(f"behavior metrics row {index} must be an object")
        raw_step = raw.get("global_step", raw.get("step", raw.get("checkpoint_step")))
        if not isinstance(raw_step, int):
            checkpoint = str(raw.get("checkpoint") or "")
            try:
                raw_step = int(Path(checkpoint).name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                raise ValueError(
                    f"behavior metrics row {index} requires a checkpoint step"
                ) from None
        metrics = raw.get("metrics")
        if not isinstance(metrics, Mapping):
            metrics = raw
        rows.append(
            {
                "global_step": raw_step,
                "metrics": {
                    str(key): value
                    for key, value in metrics.items()
                    if key
                    not in {
                        "global_step",
                        "step",
                        "checkpoint_step",
                        "checkpoint",
                    }
                },
            }
        )
    rows.sort(key=lambda row: int(row["global_step"]))
    return rows


def _metric(metrics: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = _number(metrics.get(name))
        if value is not None:
            return value
    return None


def _behavior_summary(
    rows: list[dict[str, Any]],
    *,
    accuracy_drop: float,
    evidence_drop: float,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if not rows:
        return {"available": False, "records": 0}, []
    latest = rows[-1]
    metrics = latest["metrics"]
    previous = rows[:-1]
    alerts: list[dict[str, str]] = []

    aliases = {
        "behavior_score": ("behavior_score", "selection_score"),
        "verdict_accuracy": (
            "verdict_accuracy",
            "binary_accuracy",
            "accuracy",
        ),
        "strict_evidence_sufficiency_rate": (
            "strict_evidence_sufficiency_rate",
            "evidence_sufficiency_rate",
            "strong_rate",
        ),
        "trace_audit_pass_rate": (
            "trace_audit_pass_rate",
            "trace_pass_rate",
        ),
        "format_valid_rate": ("format_valid_rate",),
        "think_present_rate": ("think_present_rate",),
        "engineering_failure_rate": ("engineering_failure_rate",),
        "majority_verdict_rate": ("majority_verdict_rate",),
    }
    normalized = {
        name: _metric(metrics, *names) for name, names in aliases.items()
    }
    prior_normalized = [
        {
            name: _metric(row["metrics"], *names)
            for name, names in aliases.items()
        }
        for row in previous
    ]

    def best_prior(name: str, *, lower_is_better: bool = False) -> float | None:
        values = [
            row[name]
            for row in prior_normalized
            if row.get(name) is not None
        ]
        if not values:
            return None
        return min(values) if lower_is_better else max(values)

    accuracy = normalized["verdict_accuracy"]
    prior_accuracy = best_prior("verdict_accuracy")
    if (
        accuracy is not None
        and prior_accuracy is not None
        and prior_accuracy - accuracy >= accuracy_drop
    ):
        alerts.append(
            {
                "severity": "warning",
                "code": "semantic_accuracy_regression",
                "message": (
                    f"verdict accuracy fell from prior best {prior_accuracy:.4f} "
                    f"to {accuracy:.4f}"
                ),
            }
        )

    evidence = normalized["strict_evidence_sufficiency_rate"]
    prior_evidence = best_prior("strict_evidence_sufficiency_rate")
    if (
        evidence is not None
        and prior_evidence is not None
        and prior_evidence - evidence >= evidence_drop
    ):
        alerts.append(
            {
                "severity": "warning",
                "code": "semantic_evidence_regression",
                "message": (
                    f"evidence sufficiency fell from prior best {prior_evidence:.4f} "
                    f"to {evidence:.4f}"
                ),
            }
        )

    for name in ("format_valid_rate", "think_present_rate"):
        value = normalized[name]
        if value is not None and value < 0.95:
            alerts.append(
                {
                    "severity": "critical",
                    "code": f"{name}_collapse",
                    "message": f"{name} is {value:.4f}, below 0.95",
                }
            )
    failure_rate = normalized["engineering_failure_rate"]
    if failure_rate is not None and failure_rate > 0.05:
        alerts.append(
            {
                "severity": "critical",
                "code": "engineering_failure_regression",
                "message": (
                    f"engineering failure rate is {failure_rate:.4f}, above 0.05"
                ),
            }
        )
    majority_rate = normalized["majority_verdict_rate"]
    if majority_rate is not None and majority_rate > 0.90:
        alerts.append(
            {
                "severity": "critical",
                "code": "verdict_mode_collapse",
                "message": (
                    f"one verdict occupies {majority_rate:.4f} of probe outputs"
                ),
            }
        )

    return (
        {
            "available": True,
            "records": len(rows),
            "latest_step": latest["global_step"],
            "latest": normalized,
            "raw_latest": metrics,
        },
        alerts,
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_sft_watchdog_snapshot(
    *,
    train_log: Path,
    checkpoint_root: Path | None = None,
    resource_samples: Path | None = None,
    behavior_metrics: Path | None = None,
    output: Path | None = None,
    stale_seconds: float = 900.0,
    trend_window: int = 5,
    loss_spike_factor: float = 2.0,
    grad_norm_spike_factor: float = 5.0,
    eval_loss_regression_fraction: float = 0.10,
    semantic_accuracy_drop: float = 0.10,
    semantic_evidence_drop: float = 0.15,
    gpu_memory_target_min_mib: int | None = None,
    gpu_memory_target_max_mib: int | None = None,
    gpu_memory_max_imbalance_mib: int | None = None,
    gpu_utilization_target_min_percent: float | None = None,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    train_log = train_log.expanduser().resolve()
    if not train_log.is_file():
        raise FileNotFoundError(f"training log does not exist: {train_log}")
    checkpoint_root = (
        checkpoint_root.expanduser().resolve() if checkpoint_root else None
    )
    resource_samples = (
        resource_samples.expanduser().resolve() if resource_samples else None
    )
    behavior_metrics = (
        behavior_metrics.expanduser().resolve() if behavior_metrics else None
    )
    logging_jsonl = _discover_logging_jsonl(checkpoint_root)
    rows = _metric_rows(train_log, logging_jsonl)
    profile = summarize_training_log(train_log, steady_window=trend_window)
    checkpoints = _checkpoint_descriptors(checkpoint_root)
    behavior_rows = _load_behavior_rows(behavior_metrics)
    semantic, semantic_alerts = _behavior_summary(
        behavior_rows,
        accuracy_drop=semantic_accuracy_drop,
        evidence_drop=semantic_evidence_drop,
    )
    resource_rows = (
        _tail_jsonl(resource_samples, max(5, trend_window))
        if resource_samples
        else []
    )
    last_resource = resource_rows[-1] if resource_rows else {}

    train_rows = [
        row for row in rows if _number(row.get("loss")) is not None and "eval_loss" not in row
    ]
    eval_rows = [row for row in rows if _number(row.get("eval_loss")) is not None]
    latest_train = train_rows[-1] if train_rows else {}
    latest_eval = eval_rows[-1] if eval_rows else {}
    current_step, maximum_step = _step(latest_train)
    if current_step is None:
        current_step = profile.get("steps", {}).get("last")
    if maximum_step is None:
        maximum_step = profile.get("steps", {}).get("configured_max_steps")
    progress_fraction = (
        current_step / maximum_step
        if isinstance(current_step, int)
        and isinstance(maximum_step, int)
        and maximum_step > 0
        else None
    )

    now_epoch = time.time() if now_epoch is None else now_epoch
    update_paths = [train_log]
    if logging_jsonl is not None:
        update_paths.append(logging_jsonl)
    last_update_epoch = max(path.stat().st_mtime for path in update_paths)
    stale_age_seconds = max(0.0, now_epoch - last_update_epoch)
    process_alive = last_resource.get("root_alive")
    if not isinstance(process_alive, bool):
        process_alive = None

    alerts: list[dict[str, str]] = list(semantic_alerts)
    for name, lines in profile.get("detected_errors", {}).items():
        alerts.append(
            {
                "severity": "critical",
                "code": name,
                "message": f"{name} detected at log lines {lines}",
            }
        )

    latest_gpu = last_resource.get("gpu")
    latest_gpu = latest_gpu if isinstance(latest_gpu, Mapping) else {}
    raw_memory = latest_gpu.get("whole_gpu_memory_mib_by_physical_gpu")
    raw_memory = raw_memory if isinstance(raw_memory, Mapping) else {}
    gpu_memory = {
        str(key): value
        for key, raw in raw_memory.items()
        if (value := _number(raw)) is not None
    }
    gpu_memory_values = list(gpu_memory.values())
    gpu_memory_min = min(gpu_memory_values) if gpu_memory_values else None
    gpu_memory_max = max(gpu_memory_values) if gpu_memory_values else None
    gpu_memory_imbalance = (
        gpu_memory_max - gpu_memory_min
        if gpu_memory_min is not None and gpu_memory_max is not None
        else None
    )
    utilization_by_gpu: dict[str, list[float]] = {}
    for resource_row in resource_rows:
        gpu = resource_row.get("gpu")
        gpu = gpu if isinstance(gpu, Mapping) else {}
        raw_utilization = gpu.get("utilization_percent_by_physical_gpu")
        raw_utilization = (
            raw_utilization if isinstance(raw_utilization, Mapping) else {}
        )
        for key, raw in raw_utilization.items():
            value = _number(raw)
            if value is not None:
                utilization_by_gpu.setdefault(str(key), []).append(value)
    gpu_utilization_median = {
        key: median(values) for key, values in utilization_by_gpu.items() if values
    }
    resource_targets_active = current_step is not None and current_step >= 1
    if (
        resource_targets_active
        and gpu_memory_target_min_mib is not None
        and gpu_memory_min is not None
        and gpu_memory_min < gpu_memory_target_min_mib
    ):
        alerts.append(
            {
                "severity": "warning",
                "code": "gpu_memory_below_target",
                "message": (
                    f"lowest GPU memory is {gpu_memory_min:.0f} MiB, below "
                    f"the tuning target {gpu_memory_target_min_mib} MiB"
                ),
            }
        )
    if (
        resource_targets_active
        and gpu_memory_target_max_mib is not None
        and gpu_memory_max is not None
        and gpu_memory_max > gpu_memory_target_max_mib
    ):
        alerts.append(
            {
                "severity": "critical",
                "code": "gpu_memory_headroom_low",
                "message": (
                    f"highest GPU memory is {gpu_memory_max:.0f} MiB, above "
                    f"the safety ceiling {gpu_memory_target_max_mib} MiB"
                ),
            }
        )
    if (
        resource_targets_active
        and gpu_memory_max_imbalance_mib is not None
        and gpu_memory_imbalance is not None
        and gpu_memory_imbalance > gpu_memory_max_imbalance_mib
    ):
        alerts.append(
            {
                "severity": "warning",
                "code": "gpu_memory_rank_imbalance",
                "message": (
                    f"GPU memory spread is {gpu_memory_imbalance:.0f} MiB, above "
                    f"the rank-balance limit {gpu_memory_max_imbalance_mib} MiB"
                ),
            }
        )
    if (
        resource_targets_active
        and gpu_utilization_target_min_percent is not None
        and gpu_utilization_median
        and min(gpu_utilization_median.values())
        < gpu_utilization_target_min_percent
    ):
        alerts.append(
            {
                "severity": "warning",
                "code": "gpu_utilization_below_target",
                "message": (
                    "lowest rolling median GPU utilization is "
                    f"{min(gpu_utilization_median.values()):.1f}%, below "
                    f"{gpu_utilization_target_min_percent:.1f}%"
                ),
            }
        )

    losses = [
        value
        for row in train_rows
        if (value := _number(row.get("loss"))) is not None
    ]
    if len(losses) >= 4:
        prior = losses[-(trend_window + 1) : -1]
        baseline = median(prior)
        if (
            baseline > 0
            and losses[-1] >= baseline * loss_spike_factor
            and losses[-1] - baseline >= 0.10
        ):
            alerts.append(
                {
                    "severity": "warning",
                    "code": "training_loss_spike",
                    "message": (
                        f"latest loss {losses[-1]:.6g} exceeds rolling median "
                        f"{baseline:.6g}"
                    ),
                }
            )

    grad_norms = [
        value
        for row in train_rows
        if (value := _number(row.get("grad_norm"))) is not None
    ]
    if len(grad_norms) >= 4:
        prior = grad_norms[-(trend_window + 1) : -1]
        baseline = median(prior)
        if baseline > 0 and grad_norms[-1] >= baseline * grad_norm_spike_factor:
            alerts.append(
                {
                    "severity": "warning",
                    "code": "gradient_norm_spike",
                    "message": (
                        f"latest grad norm {grad_norms[-1]:.6g} exceeds rolling "
                        f"median {baseline:.6g}"
                    ),
                }
            )

    eval_losses = [
        value
        for row in eval_rows
        if (value := _number(row.get("eval_loss"))) is not None
    ]
    if len(eval_losses) >= 2:
        prior_best = min(eval_losses[:-1])
        latest_eval_loss = eval_losses[-1]
        if (
            prior_best > 0
            and latest_eval_loss >= prior_best * (1.0 + eval_loss_regression_fraction)
            and latest_eval_loss - prior_best >= 0.02
        ):
            alerts.append(
                {
                    "severity": "warning",
                    "code": "eval_loss_regression",
                    "message": (
                        f"latest eval loss {latest_eval_loss:.6g} regressed from "
                        f"prior best {prior_best:.6g}"
                    ),
                }
            )

    latest_loss = _number(latest_train.get("loss"))
    latest_token_accuracy = _number(latest_train.get("token_acc"))
    if (
        progress_fraction is not None
        and progress_fraction <= 0.25
        and latest_loss is not None
        and latest_loss <= 0.02
        and latest_token_accuracy is not None
        and latest_token_accuracy >= 0.995
    ):
        alerts.append(
            {
                "severity": "warning",
                "code": "early_training_saturation",
                "message": (
                    "loss and token accuracy saturated in the first quarter; "
                    "inspect memorization, masking, and data duplication"
                ),
            }
        )

    complete = bool(profile.get("steps", {}).get("complete")) and bool(
        profile.get("process", {}).get("end_time_observed")
    )
    if process_alive is True and stale_age_seconds > stale_seconds:
        alerts.append(
            {
                "severity": "critical",
                "code": "stalled_progress",
                "message": (
                    f"training process is alive but metrics are stale for "
                    f"{stale_age_seconds:.0f} seconds"
                ),
            }
        )
    if process_alive is False and train_rows and not complete:
        alerts.append(
            {
                "severity": "critical",
                "code": "process_exited_before_completion",
                "message": "training process exited before the configured schedule completed",
            }
        )
    if complete and profile.get("validation", {}).get("required") and not eval_rows:
        alerts.append(
            {
                "severity": "critical",
                "code": "required_validation_missing",
                "message": "training completed without the required evaluation metrics",
            }
        )

    severities = {alert["severity"] for alert in alerts}
    health = (
        "critical"
        if "critical" in severities
        else "warning"
        if "warning" in severities
        else "healthy"
    )
    if complete:
        lifecycle = "completed"
    elif process_alive is True:
        lifecycle = "running"
    elif process_alive is False and train_rows:
        lifecycle = "failed"
    elif train_rows:
        lifecycle = "inactive"
    else:
        lifecycle = "waiting"
    if lifecycle == "completed" and health == "healthy":
        status = "completed"
    elif lifecycle == "failed":
        status = "failed"
    elif health == "critical":
        status = "critical"
    elif health == "warning":
        status = "warning"
    else:
        status = lifecycle

    latest_checkpoint = checkpoints[-1] if checkpoints else None
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_epoch": now_epoch,
        "status": status,
        "lifecycle": lifecycle,
        "health": health,
        "paths": {
            "train_log": str(train_log),
            "logging_jsonl": str(logging_jsonl) if logging_jsonl else "",
            "resource_samples": str(resource_samples) if resource_samples else "",
            "checkpoint_root": str(checkpoint_root) if checkpoint_root else "",
            "behavior_metrics": str(behavior_metrics) if behavior_metrics else "",
        },
        "progress": {
            "current_step": current_step,
            "maximum_step": maximum_step,
            "fraction": progress_fraction,
            "percent": (
                round(progress_fraction * 100.0, 3)
                if progress_fraction is not None
                else None
            ),
            "epoch": _number(latest_train.get("epoch")),
            "elapsed_time": latest_train.get("elapsed_time"),
            "remaining_time": latest_train.get("remaining_time"),
            "last_update_age_seconds": round(stale_age_seconds, 3),
            "process_alive": process_alive,
        },
        "optimization": {
            "train_metric_rows": len(train_rows),
            "eval_metric_rows": len(eval_rows),
            "latest_loss": latest_loss,
            "latest_eval_loss": _number(latest_eval.get("eval_loss")),
            "best_eval_loss": min(eval_losses) if eval_losses else None,
            "latest_grad_norm": _number(latest_train.get("grad_norm")),
            "latest_learning_rate": _number(latest_train.get("learning_rate")),
            "latest_token_accuracy": latest_token_accuracy,
            "latest_memory_gib": _number(latest_train.get("memory(GiB)")),
            "latest_seconds_per_step": _number(
                latest_train.get("train_speed(s/it)")
            ),
        },
        "resources": {
            "available": bool(resource_rows),
            "latest": last_resource,
            "gpu_memory_mib_by_physical_gpu": gpu_memory,
            "gpu_memory_min_mib": gpu_memory_min,
            "gpu_memory_max_mib": gpu_memory_max,
            "gpu_memory_imbalance_mib": gpu_memory_imbalance,
            "gpu_utilization_rolling_median_percent_by_physical_gpu": (
                gpu_utilization_median
            ),
            "targets": {
                "memory_min_mib": gpu_memory_target_min_mib,
                "memory_max_mib": gpu_memory_target_max_mib,
                "memory_max_imbalance_mib": gpu_memory_max_imbalance_mib,
                "utilization_min_percent": gpu_utilization_target_min_percent,
            },
        },
        "checkpoints": {
            "count": len(checkpoints),
            "latest": latest_checkpoint,
        },
        "semantic": semantic,
        "alerts": alerts,
        "recommendation": (
            "stop_and_inspect"
            if health == "critical"
            else "inspect_before_selecting_checkpoint"
            if health == "warning"
            else "continue"
            if lifecycle == "running"
            else "select_checkpoint"
            if lifecycle == "completed"
            else "wait"
        ),
    }
    if output is not None:
        _atomic_write_json(output, result)
    return result


def concise_watchdog_status(snapshot: Mapping[str, Any]) -> str:
    progress = snapshot.get("progress")
    progress = progress if isinstance(progress, Mapping) else {}
    optimization = snapshot.get("optimization")
    optimization = optimization if isinstance(optimization, Mapping) else {}
    checkpoints = snapshot.get("checkpoints")
    checkpoints = checkpoints if isinstance(checkpoints, Mapping) else {}
    latest_checkpoint = checkpoints.get("latest")
    latest_checkpoint = (
        latest_checkpoint if isinstance(latest_checkpoint, Mapping) else {}
    )
    resources = snapshot.get("resources")
    resources = resources if isinstance(resources, Mapping) else {}
    memory_min = resources.get("gpu_memory_min_mib")
    memory_max = resources.get("gpu_memory_max_mib")
    memory_text = (
        f"{float(memory_min) / 1024:.1f}-{float(memory_max) / 1024:.1f}GiB"
        if isinstance(memory_min, (int, float))
        and isinstance(memory_max, (int, float))
        else "?"
    )
    step = progress.get("current_step")
    maximum = progress.get("maximum_step")
    step_text = f"{step}/{maximum}" if step is not None and maximum else str(step or "?")
    return (
        f"{str(snapshot.get('status', 'unknown')).upper()} "
        f"step={step_text} "
        f"loss={optimization.get('latest_loss')} "
        f"eval_loss={optimization.get('latest_eval_loss')} "
        f"grad_norm={optimization.get('latest_grad_norm')} "
        f"gpu_memory={memory_text} "
        f"eta={progress.get('remaining_time')} "
        f"checkpoint={latest_checkpoint.get('global_step')} "
        f"alerts={len(snapshot.get('alerts') or [])}"
    )
