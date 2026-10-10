from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import Any, Iterable


PREFLIGHT_SCHEMA = "ifv-checkpoint-storage-preflight-v1"
PROFILE_SCHEMA = "ifv-checkpoint-io-profile-v1"


def _mount_for(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    best: dict[str, str] = {
        "mount_point": "/",
        "filesystem_type": "",
        "source": "",
    }
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        return best
    for line in mountinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        trailing = right.split()
        if len(fields) < 5 or len(trailing) < 2:
            continue
        mount_point = Path(fields[4].replace("\\040", " "))
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        if len(str(mount_point)) >= len(best["mount_point"]):
            best = {
                "mount_point": str(mount_point),
                "filesystem_type": trailing[0],
                "source": trailing[1],
            }
    return best


def checkpoint_storage_preflight(
    output_dir: Path,
    *,
    estimated_checkpoint_bytes: int,
    reserve_multiplier: float = 1.25,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_dir)
    free_bytes = int(usage.free)
    total_bytes = int(usage.total)
    required_bytes = int(estimated_checkpoint_bytes * reserve_multiplier)
    mount = _mount_for(output_dir)
    checks = {
        "positive_estimate": estimated_checkpoint_bytes > 0,
        "reserve_multiplier_at_least_one": reserve_multiplier >= 1.0,
        "sufficient_free_space": free_bytes >= required_bytes,
    }
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "output_dir": str(output_dir),
        "estimated_checkpoint_bytes": estimated_checkpoint_bytes,
        "reserve_multiplier": reserve_multiplier,
        "required_free_bytes": required_bytes,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "mount": mount,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _all_files(checkpoint: Path) -> list[Path]:
    return sorted(
        path for path in checkpoint.rglob("*") if path.is_file()
    )


def _category(path: Path) -> str:
    name = path.name
    lowered = name.casefold()
    lowered_parts = [part.casefold() for part in path.parts]
    if any(part.startswith("optimizer_") for part in lowered_parts) or lowered in {
        "optimizer.pt",
        "optimizer.bin",
    }:
        return "optimizer_state"
    if any(part.startswith("pytorch_model_fsdp_") for part in lowered_parts):
        return "model_export"
    if (
        lowered.startswith("model-")
        and lowered.endswith(".safetensors")
    ) or lowered in {
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }:
        return "model_export"
    if lowered.endswith("optim_states.pt"):
        return "optimizer_state"
    if lowered.endswith("model_states.pt"):
        return "deepspeed_model_state"
    if lowered.startswith("rng_state") or lowered in {
        "scheduler.pt",
        "trainer_state.json",
        "latest",
        "zero_to_fp32.py",
    }:
        return "recovery_metadata"
    return "model_assets"


def _phase(
    rows: Iterable[dict[str, Any]],
    *,
    start_override_ns: int | None = None,
) -> dict[str, Any]:
    selected = list(rows)
    if not selected:
        return {
            "file_count": 0,
            "bytes": 0,
            "start_mtime_ns": None,
            "end_mtime_ns": None,
            "span_seconds": 0.0,
            "throughput_bytes_per_second": None,
        }
    observed_start = min(int(row["mtime_ns"]) for row in selected)
    start = (
        int(start_override_ns)
        if start_override_ns is not None
        else observed_start
    )
    end = max(int(row["mtime_ns"]) for row in selected)
    span = max(0.0, (end - start) / 1e9)
    total = sum(int(row["bytes"]) for row in selected)
    return {
        "file_count": len(selected),
        "bytes": total,
        "start_mtime_ns": start,
        "first_completion_mtime_ns": observed_start,
        "end_mtime_ns": end,
        "span_seconds": span,
        "throughput_bytes_per_second": (
            total / span if span > 0 else None
        ),
    }


def checkpoint_io_profile(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    rows: list[dict[str, Any]] = []
    for path in _all_files(checkpoint):
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(checkpoint)),
                "category": _category(path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    categories = {
        category: _phase(grouped.get(category, []))
        for category in (
            "model_export",
            "model_assets",
            "deepspeed_model_state",
            "optimizer_state",
            "recovery_metadata",
        )
    }
    first_mtime = min((int(row["mtime_ns"]) for row in rows), default=0)
    last_mtime = max((int(row["mtime_ns"]) for row in rows), default=0)
    model_rows = [
        *grouped.get("model_export", []),
        *grouped.get("model_assets", []),
    ]
    model_phase = _phase(model_rows)
    model_end = int(model_phase["end_mtime_ns"] or first_mtime)
    deepspeed_phase = _phase(
        grouped.get("deepspeed_model_state", []),
        start_override_ns=model_end,
    )
    deepspeed_end = int(deepspeed_phase["end_mtime_ns"] or model_end)
    optimizer_phase = _phase(
        grouped.get("optimizer_state", []),
        start_override_ns=deepspeed_end,
    )
    optimizer_end = int(optimizer_phase["end_mtime_ns"] or deepspeed_end)
    recovery_phase = _phase(
        grouped.get("recovery_metadata", []),
        start_override_ns=optimizer_end,
    )
    total_bytes = sum(int(row["bytes"]) for row in rows)
    optimizer_files = grouped.get("optimizer_state", [])
    model_files = grouped.get("model_export", [])
    checks = {
        "files_present": bool(rows),
        "model_export_present": bool(model_files),
        "optimizer_state_present": bool(optimizer_files),
        "scheduler_state_present": any(
            Path(str(row["path"])).name == "scheduler.pt" for row in rows
        ),
        "rng_state_present": any(
            Path(str(row["path"])).name.startswith("rng_state")
            for row in rows
        ),
        "trainer_state_present": any(
            Path(str(row["path"])).name == "trainer_state.json"
            for row in rows
        ),
    }
    return {
        "schema_version": PROFILE_SCHEMA,
        "checkpoint": str(checkpoint),
        "mount": _mount_for(checkpoint),
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "write_window_seconds": (
            max(0.0, (last_mtime - first_mtime) / 1e9) if rows else 0.0
        ),
        "optimizer_rank_file_count": len(optimizer_files),
        "optimizer_rank_file_bytes": sorted(
            int(row["bytes"]) for row in optimizer_files
        ),
        "categories": categories,
        "pipeline_phases": {
            "model_and_assets": model_phase,
            "deepspeed_model_state": deepspeed_phase,
            "optimizer_state": optimizer_phase,
            "recovery_finalize": recovery_phase,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "files": rows,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
