from __future__ import annotations

import json
import re
from math import ceil
from pathlib import Path
from typing import Any, Mapping

from .profile import read_training_metric_rows, summarize_training_log


SCHEMA_VERSION = "ifv-sft-checkpoint-validation-v1"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _step_from_path(path: Path) -> int | None:
    match = re.search(r"(?:^|[/\\])checkpoint-(\d+)(?:[/\\]?$)", str(path))
    return int(match.group(1)) if match else None


def _trainer_state(path: Path) -> dict[str, Any]:
    state_path = path / "trainer_state.json"
    if not state_path.is_file():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _checkpoint_descriptor(path: Path) -> dict[str, Any]:
    state = _trainer_state(path)
    state_step = state.get("global_step")
    step = (
        int(state_step)
        if isinstance(state_step, int)
        else _step_from_path(path)
    )
    files = [item for item in path.rglob("*") if item.is_file()]
    names = [item.name.lower() for item in files]
    return {
        "path": str(path.resolve()),
        "exists": path.is_dir(),
        "global_step": step,
        "trainer_state": (path / "trainer_state.json").is_file(),
        "optimizer_state": any("optim" in name for name in names),
        "scheduler_state": "scheduler.pt" in names,
        "rng_state": any(name.startswith("rng_state") for name in names),
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def _discover_checkpoints(root: Path) -> dict[int, dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint root does not exist: {root}")
    discovered: dict[int, dict[str, Any]] = {}
    for path in sorted(root.rglob("checkpoint-*")):
        if not path.is_dir():
            continue
        descriptor = _checkpoint_descriptor(path)
        step = descriptor["global_step"]
        if not isinstance(step, int):
            continue
        prior = discovered.get(step)
        if prior is None or (
            descriptor["file_count"] > int(prior.get("file_count", 0))
        ):
            discovered[step] = descriptor
    return discovered


def _load_behavior_metrics(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"behavior metrics file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        payloads = [
            json.loads(line)
            for line in text.split("\n")
            if line.strip()
        ]
    else:
        payload = json.loads(text)
        if isinstance(payload, list):
            payloads = payload
        elif isinstance(payload, Mapping):
            payloads = payload.get("records", [])
        else:
            payloads = []
    result: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(payloads):
        if not isinstance(raw, Mapping):
            raise ValueError(f"behavior metrics row {index} must be an object")
        raw_step = raw.get("global_step", raw.get("step", raw.get("checkpoint_step")))
        step = _number(raw_step)
        if step is None and raw.get("checkpoint"):
            step = _step_from_path(Path(str(raw["checkpoint"])))
        if step is None or int(step) < 0:
            raise ValueError(
                f"behavior metrics row {index} requires global_step or checkpoint"
            )
        metrics = raw.get("metrics")
        if not isinstance(metrics, Mapping):
            metrics = raw
        result[int(step)] = {
            str(key): value
            for key, value in metrics.items()
            if key not in {"global_step", "step", "checkpoint_step", "checkpoint"}
        }
    return result


def _behavior_score(metrics: Mapping[str, Any]) -> tuple[float | None, dict[str, Any]]:
    explicit = _number(
        metrics.get("behavior_score", metrics.get("selection_score"))
    )
    if explicit is not None:
        return explicit, {
            "source": "explicit_external_score",
            "weights": {},
            "available_metrics": [
                "behavior_score"
                if "behavior_score" in metrics
                else "selection_score"
            ],
            "missing_metrics": [],
        }

    return None, {
        "source": "metrics_only",
        "weights": {},
        "available_metrics": sorted(str(key) for key in metrics),
        "missing_metrics": ["behavior_score"],
    }


def _checkpoint_rows(
    *,
    train_log: Path,
    checkpoint_root: Path,
    behavior_metrics: Path | None,
    train_rows: int | None,
    global_batch: int | None,
) -> dict[str, Any]:
    profile = summarize_training_log(train_log)
    metric_rows = read_training_metric_rows(train_log)
    eval_by_step: dict[int, dict[str, Any]] = {}
    for row in metric_rows:
        step = row.get("global_step")
        loss = _number(row.get("eval_loss"))
        if isinstance(step, int) and loss is not None:
            eval_by_step[step] = {
                "global_step": step,
                "eval_loss": loss,
                "epoch_from_log": _number(row.get("epoch")),
                "eval_runtime": _number(row.get("eval_runtime")),
            }

    checkpoints = _discover_checkpoints(checkpoint_root)
    behavior_by_step = _load_behavior_metrics(behavior_metrics)
    effective_train_rows = train_rows
    if effective_train_rows is None:
        raw_rows = profile.get("throughput", {}).get("train_dataset_size")
        if isinstance(raw_rows, int):
            effective_train_rows = raw_rows
    effective_global_batch = global_batch
    if effective_global_batch is None:
        inferred = profile.get("parallelism", {}).get(
            "unique_samples_per_optimizer_step"
        )
        if isinstance(inferred, int) and inferred > 0:
            effective_global_batch = inferred
    steps_per_epoch = None
    if (
        isinstance(effective_train_rows, int)
        and effective_train_rows > 0
        and isinstance(effective_global_batch, int)
        and effective_global_batch > 0
    ):
        steps_per_epoch = ceil(effective_train_rows / effective_global_batch)

    rows: list[dict[str, Any]] = []
    missing_checkpoints: list[int] = []
    for step, evaluation in sorted(eval_by_step.items()):
        checkpoint = checkpoints.get(step)
        if checkpoint is None:
            missing_checkpoints.append(step)
            continue
        behavior = behavior_by_step.get(step, {})
        behavior_score, behavior_detail = _behavior_score(behavior)
        epoch = evaluation["epoch_from_log"]
        if epoch is None and steps_per_epoch:
            epoch = step / steps_per_epoch
        rows.append(
            {
                **checkpoint,
                "global_step": step,
                "epoch": epoch,
                "eval_loss": evaluation["eval_loss"],
                "eval_runtime": evaluation["eval_runtime"],
                "behavior_metrics": behavior,
                "behavior_score": behavior_score,
                "behavior_score_detail": behavior_detail,
            }
        )

    return {
        "profile": profile,
        "rows": rows,
        "missing_checkpoints_for_eval_steps": missing_checkpoints,
        "behavior_metric_steps": sorted(behavior_by_step),
        "train_rows": effective_train_rows,
        "global_batch": effective_global_batch,
        "optimizer_steps_per_epoch": steps_per_epoch,
    }


def validate_checkpoints(
    *,
    train_log: Path,
    checkpoint_root: Path,
    output: Path,
    behavior_metrics: Path | None = None,
    train_rows: int | None = None,
    global_batch: int | None = None,
    selection: str = "auto",
) -> dict[str, Any]:
    if selection not in {"auto", "eval_loss", "behavior"}:
        raise ValueError("selection must be auto, eval_loss, or behavior")
    assembled = _checkpoint_rows(
        train_log=train_log,
        checkpoint_root=checkpoint_root,
        behavior_metrics=behavior_metrics,
        train_rows=train_rows,
        global_batch=global_batch,
    )
    rows = assembled["rows"]
    behavior_available = any(
        row.get("behavior_score") is not None for row in rows
    )
    mode = (
        "behavior"
        if selection == "auto" and behavior_available
        else "eval_loss"
        if selection == "auto"
        else selection
    )
    if mode == "behavior":
        candidates = [
            row for row in rows if row.get("behavior_score") is not None
        ]
        candidates.sort(
            key=lambda row: (
                -float(row["behavior_score"]),
                float(row["eval_loss"]),
                int(row["global_step"]),
            )
        )
    else:
        candidates = list(rows)
        candidates.sort(
            key=lambda row: (
                float(row["eval_loss"]),
                int(row["global_step"]),
            )
        )

    selected = candidates[0] if candidates else None
    warnings: list[str] = []
    if not rows:
        warnings.append("no checkpoint has a matching eval_loss record")
    if assembled["missing_checkpoints_for_eval_steps"]:
        warnings.append(
            "some eval steps have no saved checkpoint and were excluded"
        )
    if mode == "behavior" and not behavior_available:
        warnings.append(
            "behavior selection requested but no matching behavior metrics "
            "were found; no checkpoint was selected by behavior"
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "passed": bool(selected),
        "selection_mode": mode,
        "train_log": str(train_log.expanduser().resolve()),
        "checkpoint_root": str(checkpoint_root.expanduser().resolve()),
        "behavior_metrics": (
            str(behavior_metrics.expanduser().resolve())
            if behavior_metrics
            else ""
        ),
        "train_rows": assembled["train_rows"],
        "global_batch": assembled["global_batch"],
        "optimizer_steps_per_epoch": assembled["optimizer_steps_per_epoch"],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": selected,
        "missing_checkpoints_for_eval_steps": assembled[
            "missing_checkpoints_for_eval_steps"
        ],
        "warnings": warnings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
