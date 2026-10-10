from __future__ import annotations

from eviscene.environment import get_env

import csv
import importlib.metadata
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .io import load_json, sha256_file, write_json


TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "ms-swift",
    "deepspeed",
    "vllm",
    "lmdeploy",
    "xformers",
    "flash-attn",
    "flash-linear-attention",
    "causal-conv1d",
    "flashinfer-python",
    "xgrammar",
    "llguidance",
    "compressed-tensors",
    "ray",
    "peft",
    "trl",
    "accelerate",
    "datasets",
    "qwen-vl-utils",
    "rllm",
    "verl",
    "sglang",
)

MS_SWIFT_RELEASE = {
    "name": "ms-swift",
    "version": "4.4.2",
    "git_tag": "v4.4.2",
    "git_commit": "98a09c18cdf95ff07051324b9b8cc90f5184b24b",
}

TRAINING_RUNTIME_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "CUDA_HOME",
    "NPROC_PER_NODE",
    "EVISCENE_DATA_PARALLEL_SIZE",
    "ACCELERATE_USE_FSDP",
    "FSDP_VERSION",
    "CELOSS_PARALLEL_SIZE",
    "NCCL_CUMEM_HOST_ENABLE",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "IMAGE_MAX_TOKEN_NUM",
    "FLASH_ATTENTION_DETERMINISTIC",
)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(path: Path) -> str:
    # Deployed immutable code snapshots have a separate hash binding, no Git.
    if not (path / ".git").exists():
        return ""
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def environment_manifest(repo_root: Path) -> dict[str, Any]:
    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        torch_info = {"error": str(exc)}
    installed = sorted(
        {
            distribution.metadata["Name"].casefold(): distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }.items()
    )
    lock_payload = json.dumps(
        installed,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": "ifv-training-environment-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "conda_environment": get_env("CONDA_DEFAULT_ENV", ""),
        "cuda_visible_devices": get_env("CUDA_VISIBLE_DEVICES", ""),
        "omp_num_threads": get_env("OMP_NUM_THREADS", ""),
        "training_runtime_environment": {
            name: get_env(name, "") for name in TRAINING_RUNTIME_ENVIRONMENT
        },
        "packages": {name: _version(name) for name in TRACKED_PACKAGES},
        "package_lock": {
            "algorithm": "sha256",
            "normalized_distribution_count": len(installed),
            "sha256": hashlib.sha256(lock_payload).hexdigest(),
        },
        "framework_release": MS_SWIFT_RELEASE,
        "torch": torch_info,
        "repository": {
            "path": str(repo_root.resolve()),
            "commit": _git_commit(repo_root),
        },
    }


def artifact_manifest(
    *,
    kind: str,
    root: Path,
    files: Iterable[Path],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    artifacts = []
    for raw_path in files:
        path = raw_path.resolve()
        artifacts.append(
            {
                "path": str(path.relative_to(root.resolve())),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "ifv-training-artifact-manifest-v1",
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "artifacts": artifacts,
    }


def write_environment_manifest(output: Path, repo_root: Path) -> dict[str, Any]:
    value = environment_manifest(repo_root)
    write_json(output, value)
    return value


def _source_dataset_fingerprints(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"source dataset fingerprints are missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"channel", "split", "path", "bytes", "sha256"}
    if not rows:
        raise ValueError(f"source dataset fingerprints are empty: {path}")
    if not required.issubset(rows[0]):
        missing = sorted(required.difference(rows[0]))
        raise ValueError(
            f"source dataset fingerprints lack required columns {missing}: {path}"
        )
    result: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for line_number, row in enumerate(rows, start=2):
        identity = (str(row["channel"]), str(row["split"]))
        if identity in identities:
            raise ValueError(
                f"duplicate source dataset fingerprint {identity}: {path}:{line_number}"
            )
        identities.add(identity)
        item = {
            "channel": identity[0],
            "split": identity[1],
            "path": str(row["path"]),
            "bytes": int(row["bytes"]),
            "sha256": str(row["sha256"]),
        }
        raw_mtime = str(row.get("mtime_ns") or "").strip()
        item["mtime_ns"] = int(raw_mtime) if raw_mtime else None
        result.append(item)
    return result


def _dataset_shape(path: Path) -> dict[str, Any]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required to inspect cached datasets") from exc
    dataset = load_from_disk(str(path))
    return {
        "rows": len(dataset),
        "columns": list(dataset.column_names),
    }


def cached_dataset_manifest(cache_dir: Path) -> dict[str, Any]:
    cache_dir = cache_dir.expanduser().resolve()
    train_dir = cache_dir / "train"
    val_dir = cache_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"cached dataset must contain train/ and val/: {cache_dir}"
        )

    source_fingerprints = _source_dataset_fingerprints(
        cache_dir / "source-dataset-fingerprints.tsv"
    )
    cache_profile_path = cache_dir / "cache-profile.json"
    if not cache_profile_path.is_file():
        raise FileNotFoundError(f"cache profile is missing: {cache_profile_path}")
    cache_profile = load_json(cache_profile_path)
    train_shape = _dataset_shape(train_dir)
    validation_shape = _dataset_shape(val_dir)
    artifact_paths = sorted(
        path
        for path in cache_dir.rglob("*")
        if path.is_file() and path.name != "dataset-manifest.json"
    )
    manifest = artifact_manifest(
        kind="ms-swift-cached-dataset",
        root=cache_dir,
        files=artifact_paths,
        metadata={
            "dataset_version": cache_dir.name,
            "cache_profile": cache_profile,
            "cache_layout": {
                "train": "train",
                "validation": "val",
            },
            "train_rows": train_shape["rows"],
            "validation_rows": validation_shape["rows"],
            "train_columns": train_shape["columns"],
            "validation_columns": validation_shape["columns"],
            "source_dataset_fingerprints": source_fingerprints,
        },
    )
    manifest["schema_version"] = "ifv-cached-dataset-manifest-v2"
    manifest["dataset_version"] = cache_dir.name
    return manifest


def verify_cached_dataset(
    *,
    cache_dir: Path,
    train_dir: Path,
    validation_dir: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    cache_dir = cache_dir.expanduser().resolve()
    train_dir = train_dir.expanduser().resolve()
    validation_dir = validation_dir.expanduser().resolve()
    manifest_path = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else cache_dir / "dataset-manifest.json"
    )
    errors: list[str] = []
    if not manifest_path.is_file():
        errors.append(f"manifest_missing:{manifest_path}")
        return {
            "schema_version": "ifv-cached-dataset-verification-v1",
            "passed": False,
            "cache_dir": str(cache_dir),
            "manifest_path": str(manifest_path),
            "errors": errors,
        }

    manifest = load_json(manifest_path)
    metadata = manifest.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    layout = metadata.get("cache_layout")
    layout = layout if isinstance(layout, dict) else {}
    expected_train = (cache_dir / str(layout.get("train") or "train")).resolve()
    expected_validation = (
        cache_dir / str(layout.get("validation") or "val")
    ).resolve()
    if train_dir != expected_train:
        errors.append(
            f"train_path_mismatch:expected={expected_train}:actual={train_dir}"
        )
    if validation_dir != expected_validation:
        errors.append(
            "validation_path_mismatch:"
            f"expected={expected_validation}:actual={validation_dir}"
        )
    if manifest.get("dataset_version") != cache_dir.name:
        errors.append(
            "dataset_version_mismatch:"
            f"expected={cache_dir.name}:actual={manifest.get('dataset_version')}"
        )

    artifact_results: list[dict[str, Any]] = []
    expected_artifacts: set[str] = set()
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        errors.append("manifest_artifacts_missing")
        raw_artifacts = []
    for raw_artifact in raw_artifacts:
        if not isinstance(raw_artifact, dict):
            errors.append("manifest_artifact_not_object")
            continue
        relative = str(raw_artifact.get("path") or "")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            errors.append(f"manifest_artifact_invalid_path:{relative}")
            continue
        expected_artifacts.add(relative)
        path = cache_dir / relative
        result: dict[str, Any] = {
            "path": relative,
            "exists": path.is_file(),
            "bytes_match": False,
            "sha256_match": False,
        }
        if path.is_file():
            actual_bytes = path.stat().st_size
            actual_sha256 = sha256_file(path)
            result.update(
                {
                    "actual_bytes": actual_bytes,
                    "actual_sha256": actual_sha256,
                    "bytes_match": actual_bytes == raw_artifact.get("bytes"),
                    "sha256_match": actual_sha256 == raw_artifact.get("sha256"),
                }
            )
        if not result["exists"]:
            errors.append(f"cache_artifact_missing:{relative}")
        elif not result["bytes_match"]:
            errors.append(f"cache_artifact_size_mismatch:{relative}")
        elif not result["sha256_match"]:
            errors.append(f"cache_artifact_sha256_mismatch:{relative}")
        artifact_results.append(result)

    actual_artifacts = {
        str(path.relative_to(cache_dir))
        for path in cache_dir.rglob("*")
        if path.is_file() and path.resolve() != manifest_path
    }
    for relative in sorted(actual_artifacts.difference(expected_artifacts)):
        errors.append(f"cache_artifact_unregistered:{relative}")

    source_results: list[dict[str, Any]] = []
    raw_sources = metadata.get("source_dataset_fingerprints")
    if not isinstance(raw_sources, list) or not raw_sources:
        errors.append("source_dataset_fingerprints_missing")
        raw_sources = []
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict):
            errors.append("source_dataset_fingerprint_not_object")
            continue
        source_path = Path(str(raw_source.get("path") or "")).expanduser()
        result = {
            "channel": raw_source.get("channel"),
            "split": raw_source.get("split"),
            "path": str(source_path),
            "exists": source_path.is_file(),
            "bytes_match": False,
            "mtime_match": None,
            "sha256_match": False,
        }
        if source_path.is_file():
            stat = source_path.stat()
            actual_sha256 = sha256_file(source_path)
            expected_mtime = raw_source.get("mtime_ns")
            result.update(
                {
                    "actual_bytes": stat.st_size,
                    "actual_mtime_ns": stat.st_mtime_ns,
                    "actual_sha256": actual_sha256,
                    "bytes_match": stat.st_size == raw_source.get("bytes"),
                    "mtime_match": (
                        stat.st_mtime_ns == expected_mtime
                        if isinstance(expected_mtime, int)
                        else None
                    ),
                    "sha256_match": actual_sha256 == raw_source.get("sha256"),
                }
            )
        identity = f"{raw_source.get('channel')}:{raw_source.get('split')}"
        if not result["exists"]:
            errors.append(f"source_dataset_missing:{identity}")
        elif not result["bytes_match"]:
            errors.append(f"source_dataset_size_mismatch:{identity}")
        elif result["mtime_match"] is False:
            errors.append(f"source_dataset_mtime_mismatch:{identity}")
        elif not result["sha256_match"]:
            errors.append(f"source_dataset_sha256_mismatch:{identity}")
        source_results.append(result)

    provenance_results: list[dict[str, Any]] = []
    cache_profile = metadata.get("cache_profile")
    cache_profile = cache_profile if isinstance(cache_profile, dict) else {}
    historical_manifest = cache_profile.get("historical_manifest")
    if isinstance(historical_manifest, dict):
        historical_path = Path(
            str(historical_manifest.get("path") or "")
        ).expanduser()
        provenance_result = {
            "kind": "historical_manifest",
            "path": str(historical_path),
            "exists": historical_path.is_file(),
            "sha256_match": False,
            "artifact_results": [],
        }
        if historical_path.is_file():
            actual_sha256 = sha256_file(historical_path)
            provenance_result.update(
                {
                    "actual_sha256": actual_sha256,
                    "sha256_match": (
                        actual_sha256 == historical_manifest.get("sha256")
                    ),
                }
            )
            historical_payload = load_json(historical_path)
            historical_artifacts = historical_payload.get("artifacts")
            if not isinstance(historical_artifacts, list):
                errors.append("historical_manifest_artifacts_missing")
            else:
                for raw_artifact in historical_artifacts:
                    if not isinstance(raw_artifact, dict):
                        errors.append("historical_manifest_artifact_not_object")
                        continue
                    relative = str(raw_artifact.get("path") or "")
                    if (
                        not relative
                        or Path(relative).is_absolute()
                        or ".." in Path(relative).parts
                    ):
                        errors.append(
                            f"historical_manifest_artifact_invalid_path:{relative}"
                        )
                        continue
                    artifact_path = cache_dir / relative
                    artifact_result = {
                        "path": relative,
                        "exists": artifact_path.is_file(),
                        "bytes_match": False,
                        "sha256_match": False,
                    }
                    if artifact_path.is_file():
                        actual_bytes = artifact_path.stat().st_size
                        actual_artifact_sha256 = sha256_file(artifact_path)
                        artifact_result.update(
                            {
                                "actual_bytes": actual_bytes,
                                "actual_sha256": actual_artifact_sha256,
                                "bytes_match": (
                                    actual_bytes == raw_artifact.get("bytes")
                                ),
                                "sha256_match": (
                                    actual_artifact_sha256
                                    == raw_artifact.get("sha256")
                                ),
                            }
                        )
                    if not artifact_result["exists"]:
                        errors.append(
                            f"historical_artifact_missing:{relative}"
                        )
                    elif not artifact_result["bytes_match"]:
                        errors.append(
                            f"historical_artifact_size_mismatch:{relative}"
                        )
                    elif not artifact_result["sha256_match"]:
                        errors.append(
                            f"historical_artifact_sha256_mismatch:{relative}"
                        )
                    provenance_result["artifact_results"].append(
                        artifact_result
                    )
        if not provenance_result["exists"]:
            errors.append("historical_manifest_missing")
        elif not provenance_result["sha256_match"]:
            errors.append("historical_manifest_sha256_mismatch")
        provenance_results.append(provenance_result)

    shape_results: dict[str, Any] = {}
    for split, path, row_key, column_key in (
        ("train", train_dir, "train_rows", "train_columns"),
        (
            "validation",
            validation_dir,
            "validation_rows",
            "validation_columns",
        ),
    ):
        try:
            shape = _dataset_shape(path)
        except Exception as exc:
            errors.append(f"{split}_dataset_load_failed:{exc}")
            shape_results[split] = {"error": str(exc)}
            continue
        rows_match = shape["rows"] == metadata.get(row_key)
        columns_match = shape["columns"] == metadata.get(column_key)
        shape_results[split] = {
            **shape,
            "rows_match": rows_match,
            "columns_match": columns_match,
        }
        if not rows_match:
            errors.append(f"{split}_row_count_mismatch")
        if not columns_match:
            errors.append(f"{split}_columns_mismatch")

    return {
        "schema_version": "ifv-cached-dataset-verification-v1",
        "passed": not errors,
        "cache_dir": str(cache_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_version": manifest.get("dataset_version"),
        "cache_profile": metadata.get("cache_profile"),
        "artifact_results": artifact_results,
        "source_results": source_results,
        "provenance_results": provenance_results,
        "shape_results": shape_results,
        "errors": errors,
    }


def write_cached_dataset_verification(
    *,
    train_dirs: list[Path],
    validation_dirs: list[Path],
    output: Path,
    manifest_paths: list[Path] | None = None,
) -> dict[str, Any]:
    if len(train_dirs) != len(validation_dirs):
        raise ValueError(
            "cached train/validation dataset counts differ: "
            f"{len(train_dirs)} != {len(validation_dirs)}"
        )
    if not train_dirs:
        raise ValueError("at least one cached train/validation pair is required")
    if manifest_paths is not None and len(manifest_paths) != len(train_dirs):
        raise ValueError(
            "cached manifest count must match dataset pairs: "
            f"{len(manifest_paths)} != {len(train_dirs)}"
        )
    pair_results: list[dict[str, Any]] = []
    for index, (train_dir, validation_dir) in enumerate(
        zip(train_dirs, validation_dirs, strict=True)
    ):
        train_dir = train_dir.expanduser().resolve()
        validation_dir = validation_dir.expanduser().resolve()
        if train_dir.parent != validation_dir.parent:
            pair_results.append(
                {
                    "schema_version": "ifv-cached-dataset-verification-v1",
                    "passed": False,
                    "train_dir": str(train_dir),
                    "validation_dir": str(validation_dir),
                    "errors": ["cached_pair_parent_mismatch"],
                }
            )
            continue
        manifest_path = (
            manifest_paths[index] if manifest_paths is not None else None
        )
        pair_results.append(
            verify_cached_dataset(
                cache_dir=train_dir.parent,
                train_dir=train_dir,
                validation_dir=validation_dir,
                manifest_path=manifest_path,
            )
        )
    result = {
        "schema_version": "ifv-cached-dataset-gate-v1",
        "passed": all(item.get("passed") is True for item in pair_results),
        "pair_count": len(pair_results),
        "pairs": pair_results,
    }
    write_json(output, result)
    return result
