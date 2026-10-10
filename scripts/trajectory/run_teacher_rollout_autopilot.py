#!/usr/bin/env python3
"""Run the recoverable frozen-teacher trajectory pipeline for one train manifest.

The input manifest may contain evaluator-private facts, labels, and construction
metadata.  This program deliberately creates two separate projections:

* ``runtime-release/`` contains only the exact three-field image runtime contract.
* ``private-gold/`` is used only after a terminal rollout by the frozen SFT judge.

The default mode samples each case once, audits it with the frozen SFT judge, and
then rolls only the cases rejected by that audit for up to three additional
quality rounds.  The first eligible candidate wins; cases rejected in every
quality round are recorded as hard cases.  Engineering failures are refilled with
fresh seeds, while every produced trace is retained for provenance and diagnosis.

``--reroll-from`` can reuse an already completed initial pipeline without rerunning
its successful initial traces. ``--bootstrap-initial-run`` and
``--bootstrap-initial-eligibility`` can instead register one externally completed,
already-audited initial candidate per case, then apply the same quality-reroll
policy without mixing it with another historical rollout group.

Set ``EVISCENE_SERVER_RUNNER`` to a site-specific launcher when needed; otherwise
child commands use the current Python interpreter. It is intentionally
resumable: the output directory is also the durable state and provenance record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.trajectory.stage_accepted_teacher_release import (
    sft_candidate_rank,
    stage_release,
)
from scripts.trajectory.build_sft_case_split import build_split
from eviscene.environment import get_env
from eviscene.eval.release_adapter import (
    DATA_PIPELINE_DECISION_POLICY_VERSION,
    INPUT_MODE,
    RELEASE_SCHEMA_VERSION,
    RUNTIME_CASE_KEYS,
    RUNTIME_CONTRACT_VERSION,
)
from eviscene.eval.evaluator_private_gold import private_gold_index
from eviscene.provider_profiles import resolve_provider_settings


SCHEMA_VERSION = "ifv-teacher-rollout-autopilot-v1"


def _runtime_command(*arguments: str) -> list[str]:
    """Run through an optional site wrapper or the current Python interpreter."""

    configured = get_env("EVISCENE_SERVER_RUNNER", "").strip()
    if configured:
        runner = Path(configured).expanduser().resolve()
        return [str(runner), sys.executable, *arguments]
    return [sys.executable, *arguments]
TERMINAL_RUN_STATUSES = frozenset({"completed", "completed_with_errors"})
VALID_VERDICTS = frozenset({"real", "fake"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sft_judge_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "provider": str(getattr(args, "sft_judge_provider", "gemini")),
        "model": str(getattr(args, "sft_model", "gemini-3.7-flash")),
        "base_url": getattr(args, "sft_judge_base_url", None),
        "api_key_env": getattr(args, "sft_judge_api_key_env", None),
        "wire_api": getattr(args, "sft_judge_wire_api", None),
        "enable_thinking": getattr(args, "sft_judge_enable_thinking", None),
    }


def _rollout_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "profile": str(getattr(args, "rollout_profile", "teacher-gemini")),
        "model": str(getattr(args, "rollout_model", "gemini-3.7-flash")),
    }


def _validate_rollout_profile_model(*, profile: str, recorded_model: str) -> None:
    resolved = resolve_provider_settings(profile_id=profile)
    if resolved.model_name != recorded_model:
        raise ValueError(
            "rollout model does not match the active provider profile: "
            f"profile={profile!r}, recorded={recorded_model!r}, "
            f"resolved={resolved.model_name!r}"
        )
    if profile != "teacher-qwen-server":
        return
    if resolved.vlm_model != recorded_model:
        raise ValueError(
            "teacher-qwen-server must use the same model for the main Agent "
            f"and visual tools: main={recorded_model!r}, "
            f"vision={resolved.vlm_model!r}"
        )
    if not resolved.base_url:
        raise ValueError(
            "teacher-qwen-server requires QWEN_TEACHER_BASE_URL; refusing "
            "an implicit local Qwen endpoint"
        )
    teacher_key = get_env("QWEN_TEACHER_API_KEY", "none").strip() or "none"
    local_key = get_env("QWEN_LOCAL_API_KEY", "none").strip() or "none"
    if local_key != teacher_key:
        raise ValueError(
            "teacher-qwen-server requires QWEN_LOCAL_API_KEY to be mapped "
            "from QWEN_TEACHER_API_KEY by the portable launcher"
        )


def _state_model_config(
    state: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    configured = state.get("model_config")
    if not isinstance(configured, Mapping):
        return _rollout_config(args), _sft_judge_config(args)
    rollout = configured.get("rollout")
    judge = configured.get("sft_judge")
    if not isinstance(rollout, Mapping) or not isinstance(judge, Mapping):
        return _rollout_config(args), _sft_judge_config(args)
    rollout_config = {
        "profile": str(rollout.get("profile") or args.rollout_profile),
        "model": str(rollout.get("model") or args.rollout_model),
    }
    judge_config = {
        "provider": str(
            judge.get("provider") or getattr(args, "sft_judge_provider", "gemini")
        ),
        "model": str(judge.get("model") or args.sft_model),
        "base_url": judge.get("base_url"),
        "api_key_env": (
            judge.get("api_key_env")
            or getattr(args, "sft_judge_api_key_env", None)
        ),
        "wire_api": judge.get("wire_api"),
        "enable_thinking": judge.get("enable_thinking"),
    }
    return rollout_config, judge_config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    pending.replace(path)


def _write_case_list(path: Path, case_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{case_id}\n" for case_id in case_ids),
        encoding="utf-8",
    )


def _case_ids_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    return sorted(
        {
            str(row.get("case_id") or row.get("unified_case_id") or "").strip()
            for row in rows
            if str(row.get("case_id") or row.get("unified_case_id") or "").strip()
        }
    )


def _relative_file(root: Path, rendered: str, *, field: str) -> Path:
    value = Path(rendered)
    if value.is_absolute():
        raise ValueError(f"{field} must be relative: {rendered}")
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes dataset root: {rendered}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} does not exist: {resolved}")
    return resolved


def _case_id(row: Mapping[str, Any]) -> str:
    value = str(
        row.get("unified_case_id")
        or row.get("archive_source_version_id")
        or row.get("case_id")
        or ""
    ).strip()
    if not value:
        raise ValueError("training manifest row lacks unified_case_id")
    if len(value) > 200:
        raise ValueError(f"runtime case_id exceeds 200 characters: {value!r}")
    return value


def _expected_verdict(row: Mapping[str, Any]) -> str:
    supplied = str(row.get("expected_verdict") or row.get("gold_verdict") or "").lower()
    if supplied in VALID_VERDICTS:
        return supplied
    factual_status = str(row.get("factual_status") or "").strip().lower()
    mapping = {"supported": "real", "refuted": "fake"}
    if factual_status not in mapping:
        raise ValueError(
            "private target requires factual_status=supported|refuted or "
            "expected_verdict=real|fake for case "
            f"{_case_id(row)}"
        )
    return mapping[factual_status]


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(source) != _sha256_file(destination):
            raise ValueError(f"existing runtime asset differs from source: {destination}")
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def prepare_runtime_release(
    *,
    dataset_root: Path,
    train_manifest: Path,
    output_dir: Path,
    limit: int | None = None,
    source_access_policy: Path | None = None,
    validation_count: int = 8,
    validation_supported: int | None = None,
    case_split_seed: str = "ifv-qwen35-sft-v1",
    prohibited_runtime: Path | None = None,
) -> dict[str, Any]:
    """Project a train manifest into isolated runtime and private-gold artifacts."""

    dataset_root = dataset_root.expanduser().resolve()
    train_manifest = train_manifest.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    if not train_manifest.is_file():
        raise FileNotFoundError(f"train manifest does not exist: {train_manifest}")
    try:
        train_manifest.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("train manifest must be under dataset root") from exc

    release_root = output_dir / "runtime-release"
    benchmark_path = release_root / "runtime_input" / "cases.jsonl"
    gold_path = output_dir / "private-gold" / "private_gold.jsonl"
    case_split_path = output_dir / "case-split" / "case_split.jsonl"
    preparation_path = output_dir / "preparation.json"
    source_sha256 = _sha256_file(train_manifest)
    policy_path = (
        source_access_policy.expanduser().resolve()
        if source_access_policy is not None
        else None
    )
    if policy_path is not None and not policy_path.is_file():
        raise FileNotFoundError(f"source access policy does not exist: {policy_path}")
    policy_sha256 = _sha256_file(policy_path) if policy_path is not None else ""
    prohibited_path = (
        prohibited_runtime.expanduser().resolve()
        if prohibited_runtime is not None
        else None
    )
    if prohibited_path is not None and not prohibited_path.is_file():
        raise FileNotFoundError(
            f"prohibited runtime manifest does not exist: {prohibited_path}"
        )
    prohibited_sha256 = (
        _sha256_file(prohibited_path) if prohibited_path is not None else ""
    )

    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when supplied")
    if validation_count < 1:
        raise ValueError("validation_count must be at least 1")
    if preparation_path.is_file():
        prior = _read_json(preparation_path)
        if (
            prior.get("train_manifest") != str(train_manifest)
            or prior.get("train_manifest_sha256") != source_sha256
            or prior.get("limit") != limit
            or prior.get("source_access_policy") != str(policy_path or "")
            or prior.get("source_access_policy_sha256") != policy_sha256
            or prior.get("requested_validation_count") != validation_count
            or prior.get("requested_validation_supported") != validation_supported
            or prior.get("case_split_seed") != case_split_seed
            or prior.get("prohibited_runtime") != str(prohibited_path or "")
            or prior.get("prohibited_runtime_sha256") != prohibited_sha256
        ):
            raise ValueError(
                "existing pipeline output was prepared from a different train manifest"
            )
        if (
            not benchmark_path.is_file()
            or not gold_path.is_file()
            or not case_split_path.is_file()
            or not (case_split_path.parent / "manifest.json").is_file()
        ):
            raise FileNotFoundError("existing preparation is missing its projections")
        artifact_hashes = (
            (benchmark_path, "runtime_cases_sha256", "runtime cases"),
            (gold_path, "private_gold_sha256", "private gold"),
            (case_split_path, "case_split_sha256", "case split"),
        )
        for artifact_path, hash_key, label in artifact_hashes:
            expected_sha256 = str(prior.get(hash_key) or "").strip()
            if not expected_sha256:
                raise ValueError(
                    f"existing preparation does not record {label} SHA-256"
                )
            if _sha256_file(artifact_path) != expected_sha256:
                raise ValueError(
                    f"existing prepared {label} no longer matches preparation.json"
                )
        split_manifest = _read_json(case_split_path.parent / "manifest.json")
        split_inputs = split_manifest.get("inputs") or {}
        split_artifacts = split_manifest.get("artifacts") or {}
        if (
            str((split_inputs.get("runtime") or {}).get("sha256") or "")
            != str(prior["runtime_cases_sha256"])
            or str((split_inputs.get("gold") or {}).get("sha256") or "")
            != str(prior["private_gold_sha256"])
            or str(split_artifacts.get("case_split_sha256") or "")
            != str(prior["case_split_sha256"])
        ):
            raise ValueError(
                "existing case-split manifest is not bound to the prepared "
                "runtime, private gold, and split bytes"
            )
        return prior

    if output_dir.exists() and any(output_dir.iterdir()):
        # Operational launchers commonly create ``logs/`` before handing off
        # the foreground process.  It contains no data artifact and must not
        # make a brand-new pipeline output unusable.  Any other pre-existing
        # entry remains a fail-closed collision.
        unexpected_entries = [
            path.name for path in output_dir.iterdir() if path.name != "logs"
        ]
        if unexpected_entries:
            raise FileExistsError(
                "pipeline output must be new/empty (apart from logs/) or contain "
                f"preparation.json: {output_dir}; unexpected={unexpected_entries}"
            )

    rows = _read_jsonl(train_manifest)
    if not rows:
        raise ValueError(f"train manifest is empty: {train_manifest}")
    seen: set[str] = set()
    prepared: list[tuple[str, dict[str, Any], Path]] = []
    for row in rows:
        case_id = _case_id(row)
        if case_id in seen:
            raise ValueError(f"duplicate case_id in train manifest: {case_id}")
        seen.add(case_id)
        if str(row.get("split") or "train").strip().lower() != "train":
            raise ValueError(f"non-train row found in train manifest: {case_id}")
        image_value = str(
            row.get("unified_image_path") or row.get("local_image_path") or ""
        ).strip()
        if not image_value:
            raise ValueError(f"training manifest row lacks unified_image_path: {case_id}")
        prepared.append(
            (
                case_id,
                dict(row),
                _relative_file(dataset_root, image_value, field="unified_image_path"),
            )
        )
    prepared.sort(key=lambda item: item[0])
    if limit is not None:
        prepared = prepared[:limit]

    sidecar_path = (
        dataset_root
        / "evaluator_private"
        / "private-gold-v1"
        / "train-private-gold.jsonl"
    )
    sidecar_index: dict[str, dict[str, Any]] = {}
    if sidecar_path.is_file():
        sidecar_index = private_gold_index(_read_jsonl(sidecar_path))

    runtime_rows: list[dict[str, str]] = []
    gold_rows: list[dict[str, Any]] = []
    materialization: dict[str, int] = {"hardlink": 0, "copy": 0, "existing": 0}
    for index, (case_id, private_row, source_image) in enumerate(prepared, start=1):
        if sidecar_index:
            sidecar_row = sidecar_index.get(case_id)
            if sidecar_row is None:
                raise ValueError(
                    "training private-gold sidecar lacks runtime case_id: "
                    f"{case_id}"
                )
            private_row = dict(sidecar_row)
        suffix = source_image.suffix.lower() or ".jpg"
        asset_relative = Path("assets") / f"{index:05d}{suffix}"
        destination = release_root / "runtime_input" / asset_relative
        method = _link_or_copy(source_image, destination)
        materialization[method] = materialization.get(method, 0) + 1
        runtime_rows.append(
            {
                "case_id": case_id,
                "image_path": asset_relative.as_posix(),
                "image_sha256": _sha256_file(destination),
            }
        )
        private_projection = dict(private_row)
        private_projection["case_id"] = case_id
        private_projection["expected_verdict"] = _expected_verdict(private_projection)
        gold_rows.append(private_projection)

    _write_jsonl(benchmark_path, runtime_rows)
    _write_jsonl(gold_path, gold_rows)
    effective_validation_count = min(validation_count, max(0, len(runtime_rows) - 1))
    supported_count = sum(
        _expected_verdict(row) == "real" for row in gold_rows
    )
    refuted_count = len(gold_rows) - supported_count
    requested_supported = (
        (
            supported_count * effective_validation_count
            + len(gold_rows) // 2
        )
        // len(gold_rows)
        if validation_supported is None
        else validation_supported
    )
    if requested_supported < 0:
        raise ValueError("validation_supported must be non-negative")
    feasible_supported = [
        count
        for count in range(effective_validation_count + 1)
        if count <= supported_count
        and effective_validation_count - count <= refuted_count
    ]
    if not feasible_supported:
        raise ValueError("training labels cannot satisfy a validation split")
    if validation_supported is not None:
        if validation_supported > effective_validation_count:
            raise ValueError(
                "validation_supported cannot exceed the effective validation count"
            )
        if validation_supported not in feasible_supported:
            raise ValueError(
                "validation_supported cannot be satisfied by the available labels"
            )
        supported_targets = [validation_supported]
    else:
        supported_targets = sorted(
            feasible_supported,
            key=lambda count: (abs(count - requested_supported), count),
        )
    prohibited_rows = (
        _read_jsonl(prohibited_path) if prohibited_path is not None else []
    )
    if effective_validation_count:
        split_error: ValueError | None = None
        for effective_validation_supported in supported_targets:
            try:
                split_rows, split_summary = build_split(
                    runtime_rows,
                    gold_rows,
                    validation_count=effective_validation_count,
                    validation_supported=effective_validation_supported,
                    seed=case_split_seed,
                    prohibited_case_ids={
                        str(row.get("case_id") or "").strip()
                        for row in prohibited_rows
                        if str(row.get("case_id") or "").strip()
                    },
                    prohibited_image_hashes={
                        str(row.get("image_sha256") or "").strip()
                        for row in prohibited_rows
                        if str(row.get("image_sha256") or "").strip()
                    },
                )
                break
            except ValueError as exc:
                if "group constraints cannot satisfy" not in str(exc):
                    raise
                split_error = exc
        else:
            raise ValueError(
                "group constraints cannot produce the requested validation size"
            ) from split_error
    else:
        effective_validation_supported = 0
        split_rows = [
            {
                "schema_version": "ifv-sft-case-split-v1",
                "case_id": str(row["case_id"]),
                "image_sha256": str(row["image_sha256"]),
                "split_group_id": hashlib.sha256(
                    f"ifv-single-case:{row['case_id']}".encode("utf-8")
                ).hexdigest()[:20],
                "split": "train",
            }
            for row in runtime_rows
        ]
        split_summary = {
            "schema_version": "ifv-sft-case-split-v1",
            "seed": case_split_seed,
            "case_count": len(split_rows),
            "group_count": len(split_rows),
            "split_counts": {"train": len(split_rows)},
            "split_label_counts": {},
            "validation_group_ids": [],
        }
    _write_jsonl(case_split_path, split_rows)
    _write_json(
        case_split_path.parent / "manifest.json",
        {
            **split_summary,
            "requested_validation_count": validation_count,
            "effective_validation_count": effective_validation_count,
            "requested_validation_supported": validation_supported,
            "effective_validation_supported": effective_validation_supported,
            "inputs": {
                "runtime": {
                    "path": str(benchmark_path),
                    "sha256": _sha256_file(benchmark_path),
                },
                "gold": {
                    "path": str(gold_path),
                    "sha256": _sha256_file(gold_path),
                },
                "prohibited_runtime": (
                    {
                        "path": str(prohibited_path),
                        "sha256": prohibited_sha256,
                    }
                    if prohibited_path is not None
                    else None
                ),
            },
            "artifacts": {
                "case_split": "case_split.jsonl",
                "case_split_sha256": _sha256_file(case_split_path),
            },
        },
    )
    policy_release_path = None
    if policy_path is not None:
        policy_release_path = (
            release_root / "evaluator_private" / "source_access_policy.json"
        )
        policy_release_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(policy_path, policy_release_path)
    _write_json(
        release_root / "manifest.json",
        {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": f"{output_dir.name}-runtime-projection",
            "release_stage": "teacher_rollout_runtime_projection",
            "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
            "input_mode": INPUT_MODE,
            "decision_policy_version": DATA_PIPELINE_DECISION_POLICY_VERSION,
            "runtime_contract": {
                "allowed_keys": sorted(RUNTIME_CASE_KEYS),
                "private_keys_absent": True,
            },
            "artifacts": {"agent_input": "runtime_input/cases.jsonl"},
            "source_access_policy": (
                {
                    "active": True,
                    "path": "evaluator_private/source_access_policy.json",
                    "sha256": policy_sha256,
                }
                if policy_release_path is not None
                else {"active": False}
            ),
        },
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "prepared_at": _now(),
        "dataset_root": str(dataset_root),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": source_sha256,
        "case_count": len(runtime_rows),
        "limit": limit,
        "runtime_release": str(release_root),
        "benchmark": str(benchmark_path),
        "private_gold": str(gold_path),
        "private_gold_source": (
            str(sidecar_path) if sidecar_index else str(train_manifest)
        ),
        "private_gold_source_sha256": (
            _sha256_file(sidecar_path) if sidecar_index else source_sha256
        ),
        "runtime_cases_sha256": _sha256_file(benchmark_path),
        "private_gold_sha256": _sha256_file(gold_path),
        "source_access_policy": str(policy_path or ""),
        "source_access_policy_sha256": policy_sha256,
        "case_split": str(case_split_path),
        "case_split_sha256": _sha256_file(case_split_path),
        "case_split_seed": case_split_seed,
        "requested_validation_count": validation_count,
        "effective_validation_count": effective_validation_count,
        "requested_validation_supported": validation_supported,
        "effective_validation_supported": effective_validation_supported,
        "prohibited_runtime": str(prohibited_path or ""),
        "prohibited_runtime_sha256": prohibited_sha256,
        "runtime_asset_materialization": materialization,
    }
    _write_json(preparation_path, payload)
    return payload


def _trace_case_id(trace: Mapping[str, Any]) -> str:
    state = trace.get("state")
    runtime_case = state.get("runtime_case") if isinstance(state, Mapping) else None
    case_id = runtime_case.get("case_id") if isinstance(runtime_case, Mapping) else ""
    case_id = str(case_id or trace.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("trace lacks state.runtime_case.case_id")
    return case_id


def _trace_episode_id(trace: Mapping[str, Any]) -> str:
    state = trace.get("state")
    episode_id = str(
        trace.get("image_id")
        or (state.get("image_id") if isinstance(state, Mapping) else "")
        or ""
    ).strip()
    if not episode_id:
        raise ValueError("trace lacks image_id")
    return episode_id


def _terminal_success(trace: Mapping[str, Any]) -> bool:
    return (
        str(trace.get("termination") or "") == "success"
        and str(trace.get("verdict") or "").lower() in VALID_VERDICTS
    )


def _trace_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "traces").glob("*.json"))


def _trace_summary(trace: Mapping[str, Any]) -> dict[str, str]:
    """Keep only metadata needed while scanning historical attempts."""

    return {
        "case_id": _trace_case_id(trace),
        "episode_id": _trace_episode_id(trace),
        "termination": str(trace.get("termination") or ""),
        "verdict": str(trace.get("verdict") or "").lower(),
    }


def _attempt_dirs(group_dir: Path) -> list[Path]:
    candidates: list[tuple[int, Path]] = []
    for path in group_dir.glob("attempt-*"):
        if not path.is_dir():
            continue
        try:
            number = int(path.name.removeprefix("attempt-"))
        except ValueError:
            continue
        candidates.append((number, path))
    return [path for _, path in sorted(candidates)]


def _successful_trace_sources(
    attempt_dirs: Sequence[Path],
) -> dict[str, tuple[Path, Path, dict[str, Any]]]:
    """Find successes without retaining every historical trace in memory."""

    selected: dict[str, tuple[Path, Path, dict[str, Any]]] = {}
    for attempt_dir in attempt_dirs:
        seen_in_attempt: set[str] = set()
        for trace_path in _trace_paths(attempt_dir):
            trace = _read_json(trace_path)
            # Early transport failures may lack case identity; leave them to the retry scanner.
            # Only terminal-success traces require merge/audit identity fields.
            if not _terminal_success(trace):
                continue
            summary = _trace_summary(trace)
            case_id = summary["case_id"]
            if case_id in seen_in_attempt:
                raise ValueError(f"duplicate case trace in {attempt_dir}: {case_id}")
            seen_in_attempt.add(case_id)
            if (
                case_id not in selected
            ):
                selected[case_id] = (attempt_dir, trace_path, summary)
    return selected


def _candidate_trace_sources(
    attempt_dirs: Sequence[Path],
) -> dict[str, list[tuple[Path, Path, dict[str, Any]]]]:
    """Collect terminal-success candidates grouped by case without loading bodies."""

    grouped: dict[str, list[tuple[Path, Path, dict[str, Any]]]] = {}
    seen_episode_ids: set[str] = set()
    for attempt_dir in attempt_dirs:
        seen_in_attempt: set[str] = set()
        for trace_path in _trace_paths(attempt_dir):
            trace = _read_json(trace_path)
            # See _successful_trace_sources: provider/transport failures can
            # be valid JSON without a runtime case identity.  They represent a
            # missing candidate, not a malformed successful candidate.
            if not _terminal_success(trace):
                continue
            summary = _trace_summary(trace)
            case_id = summary["case_id"]
            episode_id = summary["episode_id"]
            if episode_id in seen_in_attempt:
                raise ValueError(
                    f"duplicate episode trace in {attempt_dir}: {episode_id}"
                )
            seen_in_attempt.add(episode_id)
            if episode_id in seen_episode_ids:
                raise ValueError(f"duplicate successful episode_id: {episode_id}")
            seen_episode_ids.add(episode_id)
            grouped.setdefault(case_id, []).append(
                (attempt_dir, trace_path, summary)
            )
    for candidates in grouped.values():
        candidates.sort(key=lambda item: (item[0].name, item[1].name))
    return grouped


def _target_case_ids_for_run(
    run_dir: Path,
    gold_case_ids: Iterable[str],
) -> list[str]:
    """Resolve the case scope owned by one rollout group.

    A quality-reroll run contains only a subset of the private-gold cases.
    Its parent group writes ``target-case-list.txt`` before launching the
    attempt. Classification must use that scope; treating every gold case
    absent from a partial reroll as incomplete would enqueue already-settled
    cases again.
    """

    allowed = {
        str(case_id).strip()
        for case_id in gold_case_ids
        if str(case_id).strip()
    }
    candidates = [
        run_dir / "target-case-list.txt",
        run_dir.parent / "target-case-list.txt",
        run_dir.parent.parent / "target-case-list.txt",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        values = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(
                "target case list contains IDs absent from private gold: "
                f"{unknown[:3]}"
            )
        return sorted(set(values))
    return sorted(allowed)


def _copy_or_link(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _inherited_run_manifest(attempts: Sequence[Path]) -> dict[str, Any]:
    """Read non-secret run metadata from the first completed attempt."""

    for attempt_dir in attempts:
        manifest_path = attempt_dir / "run_manifest.json"
        if manifest_path.is_file():
            return _read_json(manifest_path)
    return {}


def _merge_successful_attempts(
    *,
    group_dir: Path,
    group_name: str,
    target_ids: Sequence[str],
) -> tuple[Path, dict[str, Any]]:
    """Materialize one terminal-success trace per case without deleting attempts."""

    merged_dir = group_dir / "merged"
    manifest_path = merged_dir / "run_manifest.json"
    if manifest_path.is_file():
        return merged_dir, _read_json(manifest_path)
    if merged_dir.exists() and any(merged_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite partial merged run: {merged_dir}")

    attempts = _attempt_dirs(group_dir)
    selected = _successful_trace_sources(attempts)
    target_set = set(target_ids)
    unexpected = sorted(set(selected) - target_set)
    if unexpected:
        raise ValueError(f"{group_name} traces include unexpected case IDs: {unexpected[:3]}")
    unresolved = sorted(target_set - set(selected))
    trace_root = merged_dir / "traces"
    trace_root.mkdir(parents=True, exist_ok=True)
    materialization: dict[str, int] = {"hardlink": 0, "copy": 0}
    provenance: list[dict[str, Any]] = []
    for case_id in sorted(selected):
        attempt_dir, source_path, summary = selected[case_id]
        destination = trace_root / source_path.name
        if destination.exists():
            raise FileExistsError(f"duplicate merged trace path: {destination}")
        method = _copy_or_link(source_path, destination)
        materialization[method] += 1
        provenance.append(
            {
                "case_id": case_id,
                "episode_id": summary["episode_id"],
                "source_attempt": attempt_dir.name,
                "source_run": str(attempt_dir),
                "source_trace": str(source_path),
                "verdict": summary["verdict"],
            }
        )
    inherited_manifest = _inherited_run_manifest(attempts)
    manifest = {
        "schema_version": "ifv-merged-teacher-rollout-v1",
        "run_id": merged_dir.name,
        "status": "completed" if not unresolved else "completed_with_errors",
        "started_at": _now(),
        "completed_at": _now(),
        "git_commit": inherited_manifest.get("git_commit"),
        "benchmark": inherited_manifest.get("benchmark"),
        "agent": inherited_manifest.get("agent"),
        "source_access_policy": inherited_manifest.get(
            "source_access_policy",
            {"active": False},
        ),
        "metadata": {
            "kind": "terminal-success-merge",
            "group": group_name,
            "target_case_count": len(target_ids),
            "merged_success_count": len(selected),
            "attempt_dirs": [str(path) for path in attempts],
            "trace_materialization": materialization,
        },
        "result": {
            "num_cases": len(target_ids),
            "num_episodes": len(selected),
            "num_errors": len(unresolved),
            "status_distribution": {
                "success": len(selected),
                "error": len(unresolved),
            },
        },
        "artifacts": {
            "traces": "traces/",
            "trace_provenance": "trace-provenance.jsonl",
            "unresolved_engineering": "unresolved-engineering-case-list.txt",
        },
    }
    _write_jsonl(merged_dir / "trace-provenance.jsonl", provenance)
    _write_case_list(merged_dir / "unresolved-engineering-case-list.txt", unresolved)
    _write_json(manifest_path, manifest)
    return merged_dir, manifest


def _attempt_command_log_path(group_dir: Path, attempt_number: int) -> Path:
    """Keep launcher logs outside the immutable ``run_cases`` output directory."""

    return group_dir / "logs" / f"attempt-{attempt_number:02d}.log"


def _run_command(command: Sequence[str], *, cwd: Path, log_path: Path, env: Mapping[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n[{_now()}] command={json.dumps(list(command))}\n")
        log.flush()
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=dict(env),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"[{_now()}] returncode={completed.returncode}\n")
    return int(completed.returncode)


def _run_engineering_retries(
    *,
    group_name: str,
    benchmark: Path,
    pipeline_dir: Path,
    target_ids: Sequence[str],
    profile: str,
    rollout_concurrency: int,
    base_seed: int,
    timeout: float,
    maximum_attempts: int,
    candidates_per_case: int = 1,
) -> tuple[Path, dict[str, Any]]:
    """Collect terminal candidates, refilling only missing case slots."""

    if maximum_attempts < 1:
        raise ValueError("maximum engineering attempts must be at least 1")
    if candidates_per_case < 1:
        raise ValueError("candidates_per_case must be at least 1")
    group_dir = pipeline_dir / "rollouts" / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    target_ids = list(target_ids)
    if len(target_ids) != len(set(target_ids)):
        raise ValueError(f"{group_name} target case IDs must be unique")
    _write_case_list(group_dir / "target-case-list.txt", target_ids)
    state_path = group_dir / "engineering-retry-state.json"
    attempt_dirs = _attempt_dirs(group_dir)
    next_attempt = 1
    if attempt_dirs:
        next_attempt = max(
            int(path.name.removeprefix("attempt-")) for path in attempt_dirs
        ) + 1

    for _ in range(maximum_attempts):
        candidate_groups = _candidate_trace_sources(_attempt_dirs(group_dir))
        pending_by_slots: dict[int, list[str]] = {}
        for case_id in target_ids:
            missing = candidates_per_case - len(candidate_groups.get(case_id, []))
            if missing > 0:
                pending_by_slots.setdefault(missing, []).append(case_id)
        pending = [
            case_id
            for missing_count in sorted(pending_by_slots)
            for case_id in pending_by_slots[missing_count]
        ]
        _write_case_list(group_dir / "pending-case-list.txt", pending)
        if not pending:
            break

        missing_count, batch_case_ids = min(
            pending_by_slots.items(),
            key=lambda item: (item[0], item[1][0]),
        )
        attempt_number = next_attempt
        next_attempt += 1
        attempt_dir = group_dir / f"attempt-{attempt_number:02d}"
        case_list_path = group_dir / f"attempt-{attempt_number:02d}-case-list.txt"
        _write_case_list(case_list_path, batch_case_ids)
        command = _runtime_command(
            "-m",
            "eviscene.eval.run_cases",
            "--benchmark",
            str(benchmark),
            "--output-dir",
            str(attempt_dir),
            "--profile",
            profile,
            "--concurrency",
            str(rollout_concurrency),
            "--rollouts-per-case",
            str(missing_count),
            "--episode-namespace",
            f"{group_name}-a{attempt_number:02d}-n{missing_count}",
            "--base-sampling-seed",
            str(base_seed + attempt_number - 1),
            "--timeout",
            str(timeout),
            "--skip-preflight-image-hash-verification",
            "--case-list",
            str(case_list_path),
        )
        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "GEMINI_EVAL_MAX_CONCURRENCY": str(rollout_concurrency),
                "GEMINI_MAX_INFLIGHT_REQUESTS": str(rollout_concurrency),
                "PYTHONUNBUFFERED": "1",
            }
        )
        returncode = _run_command(
            command,
            cwd=REPO_ROOT,
            log_path=_attempt_command_log_path(group_dir, attempt_number),
            env=env,
        )
        attempt_payload = {
            "schema_version": SCHEMA_VERSION,
            "group": group_name,
            "attempt": attempt_number,
            "started_case_count": len(batch_case_ids),
            "requested_rollouts_per_case": missing_count,
            "base_sampling_seed": base_seed + attempt_number - 1,
            "returncode": returncode,
            "completed_at": _now(),
            "summary_exists": (attempt_dir / "summary.json").is_file(),
        }
        _write_json(attempt_dir / "autopilot-attempt.json", attempt_payload)
        attempt_dirs = _attempt_dirs(group_dir)
    candidate_groups = _candidate_trace_sources(_attempt_dirs(group_dir))
    unresolved = [
        case_id
        for case_id in target_ids
        if len(candidate_groups.get(case_id, [])) < candidates_per_case
    ]
    complete_case_count = len(target_ids) - len(unresolved)
    trace_count = sum(len(candidate_groups.get(case_id, [])) for case_id in target_ids)
    retry_state = {
        "schema_version": SCHEMA_VERSION,
        "group": group_name,
        "target_case_count": len(target_ids),
        "candidates_per_case": candidates_per_case,
        "successful_case_count": complete_case_count,
        "terminal_trace_count": trace_count,
        "unresolved_engineering_case_count": len(unresolved),
        "maximum_attempts": maximum_attempts,
        "attempt_dirs": [str(path) for path in _attempt_dirs(group_dir)],
        "completed_at": _now(),
    }
    _write_json(state_path, retry_state)
    _write_case_list(group_dir / "unresolved-engineering-case-list.txt", unresolved)
    if candidates_per_case > 1:
        return _merge_candidate_attempts(
            group_dir=group_dir,
            group_name=group_name,
            target_ids=target_ids,
            candidates_per_case=candidates_per_case,
        )
    return _merge_successful_attempts(
        group_dir=group_dir,
        group_name=group_name,
        target_ids=target_ids,
    )


def _merge_candidate_attempts(
    *,
    group_dir: Path,
    group_name: str,
    target_ids: Sequence[str],
    candidates_per_case: int,
) -> tuple[Path, dict[str, Any]]:
    """Materialize up to N terminal candidates per case without deleting attempts."""

    merged_dir = group_dir / "merged"
    manifest_path = merged_dir / "run_manifest.json"
    if manifest_path.is_file():
        return merged_dir, _read_json(manifest_path)
    if candidates_per_case < 1:
        raise ValueError("candidates_per_case must be at least 1")
    if merged_dir.exists() and any(merged_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite partial merged run: {merged_dir}")

    target_set = set(target_ids)
    attempts = _attempt_dirs(group_dir)
    grouped = _candidate_trace_sources(attempts)
    unexpected = sorted(set(grouped) - target_set)
    if unexpected:
        raise ValueError(f"{group_name} traces include unexpected case IDs: {unexpected[:3]}")

    selected: list[tuple[str, int, Path, Path, dict[str, Any]]] = []
    incomplete: list[str] = []
    overflow: list[dict[str, Any]] = []
    for case_id in sorted(target_set):
        candidates = grouped.get(case_id, [])
        if len(candidates) < candidates_per_case:
            incomplete.append(case_id)
        for candidate_index, (attempt_dir, source_path, summary) in enumerate(
            candidates[:candidates_per_case],
            start=1,
        ):
            selected.append(
                (case_id, candidate_index, attempt_dir, source_path, summary)
            )
        for candidate_index, (attempt_dir, source_path, summary) in enumerate(
            candidates[candidates_per_case:],
            start=candidates_per_case + 1,
        ):
            overflow.append(
                {
                    "case_id": case_id,
                    "candidate_index": candidate_index,
                    "source_attempt": attempt_dir.name,
                    "source_trace": str(source_path),
                    "episode_id": summary["episode_id"],
                }
            )

    trace_root = merged_dir / "traces"
    trace_root.mkdir(parents=True, exist_ok=True)
    materialization: dict[str, int] = {"hardlink": 0, "copy": 0}
    provenance: list[dict[str, Any]] = []
    for case_id, candidate_index, attempt_dir, source_path, summary in selected:
        destination = trace_root / source_path.name
        if destination.exists():
            raise FileExistsError(f"duplicate merged trace path: {destination}")
        method = _copy_or_link(source_path, destination)
        materialization[method] += 1
        provenance.append(
            {
                "case_id": case_id,
                "candidate_index": candidate_index,
                "episode_id": summary["episode_id"],
                "source_attempt": attempt_dir.name,
                "source_run": str(attempt_dir),
                "source_trace": str(source_path),
                "verdict": summary["verdict"],
            }
        )

    inherited_manifest = _inherited_run_manifest(attempts)
    manifest = {
        "schema_version": "ifv-merged-teacher-candidates-v1",
        "run_id": merged_dir.name,
        "status": "completed" if not incomplete else "completed_with_errors",
        "started_at": _now(),
        "completed_at": _now(),
        "git_commit": inherited_manifest.get("git_commit"),
        "benchmark": inherited_manifest.get("benchmark"),
        "agent": inherited_manifest.get("agent"),
        "source_access_policy": inherited_manifest.get(
            "source_access_policy",
            {"active": False},
        ),
        "metadata": {
            "kind": "terminal-success-candidate-merge",
            "group": group_name,
            "target_case_count": len(target_ids),
            "candidates_per_case": candidates_per_case,
            "merged_trace_count": len(selected),
            "complete_case_count": len(target_set) - len(incomplete),
            "incomplete_case_count": len(incomplete),
            "overflow_success_count": len(overflow),
            "attempt_dirs": [str(path) for path in attempts],
            "trace_materialization": materialization,
        },
        "result": {
            "num_cases": len(target_ids),
            "num_episodes": len(selected),
            "num_errors": len(incomplete),
            "status_distribution": {
                "success": len(selected),
                "incomplete_case": len(incomplete),
            },
        },
        "artifacts": {
            "traces": "traces/",
            "trace_provenance": "trace-provenance.jsonl",
            "incomplete_cases": "incomplete-case-list.txt",
            "overflow_successes": "overflow-successes.jsonl",
        },
    }
    _write_jsonl(merged_dir / "trace-provenance.jsonl", provenance)
    _write_case_list(merged_dir / "incomplete-case-list.txt", incomplete)
    _write_jsonl(merged_dir / "overflow-successes.jsonl", overflow)
    _write_json(manifest_path, manifest)
    return merged_dir, manifest


def _sft_artifact_index(eligibility_dir: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for path in sorted(eligibility_dir.glob("*.sft_eligibility.json")):
        payload = _read_json(path)
        episode_id = str(
            payload.get("episode_id")
            or (payload.get("rollout") or {}).get("episode_id")
            or ""
        ).strip()
        if not episode_id:
            raise ValueError(f"SFT artifact lacks episode_id: {path}")
        if episode_id in indexed:
            raise ValueError(f"duplicate SFT artifact episode_id: {episode_id}")
        indexed[episode_id] = payload
    return indexed


def _sft_audit_complete(
    eligibility_dir: Path,
    expected_count: int,
    *,
    run_dir: Path | None = None,
) -> bool:
    summary_path = eligibility_dir / "sft_eligibility_summary.json"
    if not summary_path.is_file():
        return False
    summary = _read_json(summary_path)
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_count:
        return False
    artifacts = _sft_artifact_index(eligibility_dir)
    if len(artifacts) != expected_count:
        return False
    if run_dir is None:
        return True
    trace_index: dict[str, Path] = {}
    for trace_path in _trace_paths(run_dir):
        trace = _read_json(trace_path)
        episode_id = _trace_episode_id(trace)
        if episode_id in trace_index:
            return False
        trace_index[episode_id] = trace_path
    row_episode_ids = {
        str(row.get("episode_id") or "").strip()
        for row in rows
        if isinstance(row, Mapping)
    }
    if (
        len(trace_index) != expected_count
        or set(artifacts) != set(trace_index)
        or row_episode_ids != set(trace_index)
    ):
        return False
    summary_run = str(summary.get("run_dir") or "").strip()
    if summary_run and Path(summary_run).expanduser().resolve() != run_dir.resolve():
        return False
    for episode_id, trace_path in trace_index.items():
        source_trace = artifacts[episode_id].get("source_trace") or {}
        if str(source_trace.get("sha256") or "") != _sha256_file(trace_path):
            return False
    return True


def _materialize_bootstrap_file(source: Path, destination: Path) -> str:
    """Preserve an immutable bootstrap artifact, allowing an interrupted resume."""

    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"bootstrap source file does not exist: {source}")
    if destination.exists():
        if not destination.is_file():
            raise ValueError(
                f"bootstrap destination is not a file: {destination}"
            )
        if _sha256_file(destination) != _sha256_file(source):
            raise ValueError(
                "bootstrap destination conflicts with immutable source: "
                f"{destination}"
            )
        return "existing"
    return _copy_or_link(source, destination)


def _bootstrap_completed_initial_pipeline(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    """Attach one completed, audited candidate per case as quality-reroll round 0.

    This is deliberately not a shortcut around SFT eligibility.  The imported
    artifacts must match the exact imported trace bytes, and the source cases must
    equal the freshly prepared evaluator-private scope.  Later quality rounds are
    produced by the ordinary rollout and frozen-judge pipeline.
    """

    dataset_root = args.dataset_root.expanduser().resolve()
    train_manifest = (
        args.train_manifest.expanduser().resolve()
        if args.train_manifest is not None
        else dataset_root / "train-manifest.jsonl"
    )
    pipeline_dir = args.output_dir.expanduser().resolve()
    source_run = args.bootstrap_initial_run.expanduser().resolve()
    source_eligibility = (
        args.bootstrap_initial_eligibility.expanduser().resolve()
    )
    source_manifest_path = source_run / "run_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"bootstrap initial run lacks run manifest: {source_manifest_path}"
        )
    source_manifest = _read_json(source_manifest_path)
    if source_manifest.get("status") not in TERMINAL_RUN_STATUSES:
        raise ValueError(
            "bootstrap initial run must have a terminal status: "
            f"{source_manifest.get('status')!r}"
        )

    preparation = prepare_runtime_release(
        dataset_root=dataset_root,
        train_manifest=train_manifest,
        output_dir=pipeline_dir,
        limit=args.limit,
        source_access_policy=args.source_access_policy,
        validation_count=int(getattr(args, "validation_count", 8)),
        validation_supported=getattr(args, "validation_supported", None),
        case_split_seed=str(
            getattr(args, "case_split_seed", "ifv-qwen35-sft-v1")
        ),
        prohibited_runtime=getattr(args, "prohibited_runtime", None),
    )
    private_gold = Path(str(preparation["private_gold"])).expanduser().resolve()
    private_gold_rows = _read_jsonl(private_gold)
    gold_by_case = {
        _case_id(row): _expected_verdict(row) for row in private_gold_rows
    }
    target_ids = sorted(gold_by_case)
    if not target_ids:
        raise ValueError("bootstrap target scope is empty")

    source_traces: dict[str, tuple[Path, dict[str, Any]]] = {}
    for trace_path in _trace_paths(source_run):
        trace = _read_json(trace_path)
        if not _terminal_success(trace):
            continue
        case_id = _trace_case_id(trace)
        if case_id in source_traces:
            raise ValueError(
                "bootstrap initial run has multiple terminal candidates for "
                f"{case_id}; provide a run with exactly one selected candidate"
            )
        source_traces[case_id] = (trace_path, trace)
    source_case_ids = sorted(source_traces)
    if source_case_ids != target_ids:
        raise ValueError(
            "bootstrap source cases must exactly match the prepared train scope; "
            f"source_only={sorted(set(source_case_ids) - set(target_ids))[:3]}, "
            f"target_only={sorted(set(target_ids) - set(source_case_ids))[:3]}"
        )

    if not _sft_audit_complete(
        source_eligibility,
        len(target_ids),
        run_dir=source_run,
    ):
        raise ValueError(
            "bootstrap SFT eligibility directory is incomplete for its source "
            f"run: {source_eligibility}"
        )
    source_artifacts = _sft_artifact_index(source_eligibility)

    initial_run = pipeline_dir / "rollouts" / "initial" / "merged"
    initial_eligibility = pipeline_dir / "sft-eligibility" / "initial"
    initial_classification = pipeline_dir / "classification"
    state_path = pipeline_dir / "pipeline-state.json"
    bootstrap_descriptor = {
        "source_run": str(source_run),
        "source_run_manifest_sha256": _sha256_file(source_manifest_path),
        "source_sft_eligibility": str(source_eligibility),
        "source_sft_summary_sha256": _sha256_file(
            source_eligibility / "sft_eligibility_summary.json"
        ),
    }

    if state_path.is_file():
        state = _read_json(state_path)
        if state.get("bootstrap_initial") != bootstrap_descriptor:
            raise ValueError(
                "existing pipeline state belongs to a different bootstrap source"
            )
        initial = state.get("initial")
        if not isinstance(initial, Mapping):
            raise ValueError("existing bootstrap pipeline lacks initial state")
        expected_initial_run = Path(str(initial.get("merged_run") or "")).resolve()
        expected_initial_audit = Path(
            str(initial.get("sft_eligibility") or "")
        ).resolve()
        if expected_initial_run != initial_run or expected_initial_audit != initial_eligibility:
            raise ValueError("existing bootstrap pipeline has inconsistent initial paths")
        if not (initial_run / "run_manifest.json").is_file():
            raise FileNotFoundError("existing bootstrap initial run is missing")
        if not _sft_audit_complete(
            initial_eligibility,
            len(target_ids),
            run_dir=initial_run,
        ):
            raise FileNotFoundError("existing bootstrap initial SFT audit is incomplete")
        return state, preparation, initial_run, initial_eligibility

    trace_root = initial_run / "traces"
    trace_root.mkdir(parents=True, exist_ok=True)
    trace_materialization: dict[str, int] = {
        "hardlink": 0,
        "copy": 0,
        "existing": 0,
    }
    artifact_materialization: dict[str, int] = {
        "hardlink": 0,
        "copy": 0,
        "existing": 0,
    }
    provenance: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for case_id in target_ids:
        source_trace_path, trace = source_traces[case_id]
        episode_id = _trace_episode_id(trace)
        artifact = source_artifacts.get(episode_id)
        if artifact is None:
            raise ValueError(
                "bootstrap SFT eligibility lacks trace episode: "
                f"{case_id}/{episode_id}"
            )
        if str(artifact.get("case_id") or "") != case_id:
            raise ValueError(
                "bootstrap eligibility case_id mismatches trace: "
                f"{case_id}/{episode_id}"
            )
        trace_sha256 = _sha256_file(source_trace_path)
        artifact_sha256 = str(
            (artifact.get("source_trace") or {}).get("sha256") or ""
        )
        if artifact_sha256 != trace_sha256:
            raise ValueError(
                "bootstrap eligibility does not match source trace bytes: "
                f"{case_id}/{episode_id}"
            )
        metrics = artifact.get("metrics") or {}
        if str(metrics.get("expected_verdict") or "") != gold_by_case[case_id]:
            raise ValueError(
                "bootstrap eligibility expected verdict mismatches private gold: "
                f"{case_id}"
            )

        trace_destination = trace_root / source_trace_path.name
        method = _materialize_bootstrap_file(
            source_trace_path,
            trace_destination,
        )
        trace_materialization[method] += 1
        source_artifact_path = (
            source_eligibility / f"{episode_id}.sft_eligibility.json"
        )
        artifact_destination = (
            initial_eligibility / f"{episode_id}.sft_eligibility.json"
        )
        method = _materialize_bootstrap_file(
            source_artifact_path,
            artifact_destination,
        )
        artifact_materialization[method] += 1
        provenance.append(
            {
                "case_id": case_id,
                "episode_id": episode_id,
                "source_run": str(source_run),
                "source_trace": str(source_trace_path),
                "source_trace_sha256": trace_sha256,
                "verdict": str(trace.get("verdict") or "").lower(),
            }
        )
        gates = artifact.get("gates") or {}
        summary_rows.append(
            {
                "status": "success",
                "case_id": case_id,
                "episode_id": episode_id,
                "artifact": str(artifact_destination),
                "artifact_id": artifact.get("artifact_id"),
                "sft_eligibility_pass": gates.get("sft_eligibility_pass"),
                "target_scope": metrics.get("target_scope"),
                "decision_support": metrics.get("decision_support"),
                "retrieval_quality": metrics.get("retrieval_quality"),
                "decisive_observation_ids": metrics.get("decisive_observation_ids") or [],
                "fatal_errors": metrics.get("fatal_errors") or [],
                "warnings": metrics.get("warnings") or [],
                "bootstrapped": True,
            }
        )

    source_policy = source_manifest.get("source_access_policy")
    run_manifest = {
        "schema_version": "ifv-bootstrap-merged-teacher-rollout-v1",
        "run_id": initial_run.name,
        "status": "completed",
        "started_at": _now(),
        "completed_at": _now(),
        "git_commit": source_manifest.get("git_commit"),
        "benchmark": source_manifest.get("benchmark"),
        "agent": source_manifest.get("agent"),
        "source_access_policy": (
            dict(source_policy)
            if isinstance(source_policy, Mapping)
            else {"active": False}
        ),
        "metadata": {
            "kind": "bootstrapped-completed-initial-candidates",
            "bootstrap": bootstrap_descriptor,
            "target_case_count": len(target_ids),
            "trace_materialization": trace_materialization,
            "eligibility_materialization": artifact_materialization,
        },
        "result": {
            "num_cases": len(target_ids),
            "num_episodes": len(target_ids),
            "num_errors": 0,
            "status_distribution": {"success": len(target_ids)},
        },
        "artifacts": {
            "traces": "traces/",
            "trace_provenance": "trace-provenance.jsonl",
        },
    }
    _write_jsonl(initial_run / "trace-provenance.jsonl", provenance)
    _write_json(initial_run / "run_manifest.json", run_manifest)
    _write_case_list(initial_run.parent / "target-case-list.txt", target_ids)
    _write_jsonl(
        initial_eligibility / "accepted_episodes.jsonl",
        [
            row
            for row in summary_rows
            if row["sft_eligibility_pass"] is True
        ],
    )
    _write_json(
        initial_eligibility / "sft_eligibility_summary.json",
        {
            "schema_version": "ifv-sft-eligibility-summary-v2",
            "run_dir": str(initial_run),
            "private_gold": {
                "path": str(private_gold),
                "sha256": _sha256_file(private_gold),
            },
            "episode_count": len(summary_rows),
            "passed_count": sum(
                row["sft_eligibility_pass"] is True for row in summary_rows
            ),
            "bootstrap": bootstrap_descriptor,
            "rows": summary_rows,
        },
    )
    classification = _classify_initial_outcomes(
        run_dir=initial_run,
        eligibility_dir=initial_eligibility,
        private_gold=private_gold,
        output_dir=initial_classification,
        candidates_per_case=1,
        target_case_ids=target_ids,
    )
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_dir": str(pipeline_dir),
        "prepared": preparation,
        "bootstrap_initial": bootstrap_descriptor,
        "models": {
            "rollout": args.rollout_model,
            "sft_judge": args.sft_model,
        },
        "model_config": {
            "rollout": _rollout_config(args),
            "sft_judge": _sft_judge_config(args),
        },
        "concurrency": {
            "rollout": args.rollout_concurrency,
            "sft_judge": args.sft_concurrency,
        },
        "initial": {
            "merged_run": str(initial_run),
            "unresolved_engineering_case_count": 0,
            "sft_eligibility": str(initial_eligibility),
            "classification": str(initial_classification),
        },
        "classification": classification,
        "status": "bootstrapped",
        "updated_at": _now(),
    }
    _write_json(state_path, state)
    return state, preparation, initial_run, initial_eligibility


def _run_bootstrapped_initial_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Complete ordinary quality rerolls after registering a preserved round 0."""

    _validate_rollout_profile_model(
        profile=str(getattr(args, "rollout_profile", "teacher-gemini")),
        recorded_model=str(getattr(args, "rollout_model", "gemini-3.7-flash")),
    )
    state, preparation, initial_run, initial_audit = (
        _bootstrap_completed_initial_pipeline(args)
    )
    if state.get("status") == "completed":
        return state
    pipeline_dir = args.output_dir.expanduser().resolve()
    state_path = pipeline_dir / "pipeline-state.json"
    private_gold = Path(str(preparation["private_gold"])).expanduser().resolve()
    initial_classification_dir = pipeline_dir / "classification"
    rollout_config = _rollout_config(args)
    judge_config = _sft_judge_config(args)
    if not (initial_classification_dir / "classification.json").is_file():
        raise FileNotFoundError(
            "bootstrap initial classification is missing: "
            f"{initial_classification_dir}"
        )

    state["status"] = "running"
    state["quality_reroll_config"] = {
        "maximum_rounds": args.quality_reroll_rounds,
        "rollout_concurrency": args.rollout_concurrency,
        "sft_concurrency": args.sft_concurrency,
        "sft_judge": judge_config,
    }
    state["updated_at"] = _now()
    _write_json(state_path, state)
    sources, final_classification = _run_quality_rerolls(
        pipeline_dir=pipeline_dir,
        preparation=preparation,
        private_gold=private_gold,
        initial_run=initial_run,
        initial_audit=initial_audit,
        initial_classification_dir=initial_classification_dir,
        profile=rollout_config["profile"],
        rollout_concurrency=args.rollout_concurrency,
        sft_model=judge_config["model"],
        sft_judge_provider=judge_config["provider"],
        sft_judge_base_url=judge_config["base_url"],
        sft_judge_api_key_env=judge_config["api_key_env"],
        sft_judge_wire_api=judge_config["wire_api"],
        sft_judge_enable_thinking=judge_config["enable_thinking"],
        sft_concurrency=args.sft_concurrency,
        rollout_timeout=args.rollout_timeout,
        sft_timeout=args.sft_timeout,
        maximum_engineering_attempts=args.maximum_engineering_attempts,
        maximum_sft_audit_attempts=args.maximum_sft_audit_attempts,
        base_seed=args.base_seed,
        maximum_rounds=args.quality_reroll_rounds,
        state=state,
        state_path=state_path,
    )
    selected_episode_ids = {
        str(row["episode_id"])
        for row in _selected_classification_rows(
            pipeline_dir / "classification" / "final"
        )
        if str(row.get("episode_id", "")).strip()
    }
    accepted_release = _final_release(
        pipeline_dir=pipeline_dir,
        sources=sources,
        selected_episode_ids=selected_episode_ids,
        output_name="quality-reroll-release",
    )
    package_dir = _build_package(
        pipeline_dir=pipeline_dir,
        accepted_release=accepted_release,
        case_split=Path(str(preparation["case_split"])),
        output_name="quality-reroll-training-package",
    )
    state["classification"] = final_classification
    state["quality_reroll_release"] = str(accepted_release)
    state["quality_reroll_training_package"] = str(package_dir)
    state["status"] = "completed"
    state["updated_at"] = _now()
    _write_json(state_path, state)
    return state


def _run_sft_audit(
    *,
    run_dir: Path,
    gold: Path,
    pipeline_dir: Path,
    label: str,
    model: str,
    provider: str,
    base_url: str | None,
    api_key_env: str | None,
    wire_api: str | None,
    enable_thinking: bool | None,
    concurrency: int,
    timeout: float,
    maximum_attempts: int,
) -> Path:
    expected_count = len(_trace_paths(run_dir))
    if expected_count < 1:
        raise ValueError(f"cannot audit empty merged run: {run_dir}")
    eligibility_dir = pipeline_dir / "sft-eligibility" / label
    if _sft_audit_complete(
        eligibility_dir,
        expected_count,
        run_dir=run_dir,
    ):
        return eligibility_dir
    for attempt in range(1, maximum_attempts + 1):
        command = _runtime_command(
            "-m",
            "eviscene.eval.score_sft_eligibility",
            "--run-dir",
            str(run_dir),
            "--gold",
            str(gold),
            "--output-dir",
            str(eligibility_dir),
            "--cache-dir",
            str(eligibility_dir / "cache"),
            "--storage-dir",
            str(eligibility_dir / "automatic-storage"),
            "--provider",
            provider,
            "--model",
            model,
            "--max-tokens",
            "4096",
            "--timeout",
            str(timeout),
            "--concurrency",
            str(concurrency),
        )
        if base_url:
            command.extend(["--base-url", base_url])
        if api_key_env:
            command.extend(["--api-key-env", api_key_env])
        if wire_api:
            command.extend(["--wire-api", wire_api])
        if enable_thinking is not None:
            command.append(
                "--enable-thinking" if enable_thinking else "--no-enable-thinking"
            )
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
        _run_command(
            command,
            cwd=REPO_ROOT,
            log_path=eligibility_dir / "command.log",
            env=env,
        )
        if _sft_audit_complete(
            eligibility_dir,
            expected_count,
            run_dir=run_dir,
        ):
            return eligibility_dir
    raise RuntimeError(
        f"{label} SFT audit incomplete after {maximum_attempts} attempts: "
        f"{len(_sft_artifact_index(eligibility_dir))}/{expected_count} artifacts"
    )


def _has_early_correct_judgment(trace: Mapping[str, Any], expected: str) -> bool:
    """Require the strict earlier image-only *judgment*, not a proposal."""

    state = trace.get("state")
    steps = state.get("all_steps") if isinstance(state, Mapping) else []
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        if step.get("stage") != "unified_judgment":
            continue
        output = step.get("output")
        verdict = str(output.get("verdict") if isinstance(output, Mapping) else "").lower()
        if verdict == expected:
            return True
    return False


def _classify_initial_outcomes(
    *,
    run_dir: Path,
    eligibility_dir: Path,
    private_gold: Path,
    output_dir: Path,
    candidates_per_case: int = 1,
    target_case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Classify every candidate and select one winner per case.

    With four candidates, quality rerolling is finished inside the case group:
    all candidates are judged once, the best eligible candidate is selected, and
    cases without an eligible candidate become hard cases.  The legacy one-
    candidate mode retains its old reroll list for compatibility with older
    smoke callers.
    """

    result_path = output_dir / "classification.json"
    if result_path.is_file():
        return _read_json(result_path)
    gold_by_case = {
        _case_id(row): _expected_verdict(row) for row in _read_jsonl(private_gold)
    }
    scoped_case_ids = (
        sorted(
            {
                str(case_id).strip()
                for case_id in target_case_ids
                if str(case_id).strip()
            }
        )
        if target_case_ids is not None
        else _target_case_ids_for_run(run_dir, gold_by_case)
    )
    unknown_scoped_ids = sorted(set(scoped_case_ids) - set(gold_by_case))
    if unknown_scoped_ids:
        raise ValueError(
            "classification target contains IDs absent from private gold: "
            f"{unknown_scoped_ids[:3]}"
        )
    artifacts = _sft_artifact_index(eligibility_dir)
    early: list[dict[str, Any]] = []
    final_only: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    candidates_by_case: dict[str, list[dict[str, Any]]] = {}
    reroll_ids: list[str] = []
    seen_episode_ids: set[str] = set()
    for trace_path in _trace_paths(run_dir):
        trace = _read_json(trace_path)
        case_id = _trace_case_id(trace)
        expected = gold_by_case.get(case_id)
        if expected is None:
            raise ValueError(f"private gold lacks successful case: {case_id}")
        episode_id = _trace_episode_id(trace)
        if episode_id in seen_episode_ids:
            raise ValueError(f"duplicate candidate trace in {run_dir}: {episode_id}")
        seen_episode_ids.add(episode_id)
        artifact = artifacts.get(episode_id)
        if artifact is None:
            raise ValueError(f"SFT audit artifact missing for episode: {episode_id}")
        passed = (artifact.get("gates") or {}).get("sft_eligibility_pass") is True
        metrics = artifact.get("metrics") or {}
        row = {
            "case_id": case_id,
            "episode_id": episode_id,
            "trace_path": str(trace_path),
            "expected_verdict": expected,
            "final_verdict": str(trace.get("verdict") or "").lower(),
            "sft_eligibility_pass": passed,
            "target_scope": metrics.get("target_scope"),
            "decision_support": metrics.get("decision_support"),
            "retrieval_quality": metrics.get("retrieval_quality"),
            "trajectory_conduct": metrics.get("trajectory_conduct"),
            "overclaiming": metrics.get("overclaiming"),
            "boundary_assessment": metrics.get("boundary_assessment"),
            "fatal_errors": metrics.get("fatal_errors") or [],
            "warnings": metrics.get("warnings") or [],
            "candidate_rank": list(
                sft_candidate_rank(
                    artifact,
                    episode_id=episode_id,
                )
            ),
        }
        if not passed:
            row["bucket"] = "sft_rejected"
            row["rejection_reasons"] = (artifact.get("metrics") or {}).get(
                "fatal_errors"
            ) or []
            rejected.append(row)
        elif _has_early_correct_judgment(trace, expected):
            row["bucket"] = "early_correct_judgment"
            early.append(row)
        else:
            row["bucket"] = "final_only_judgment"
            final_only.append(row)
        candidates_by_case.setdefault(case_id, []).append(row)
        if (
            candidates_per_case == 1
            and row["bucket"] != "early_correct_judgment"
            and case_id not in reroll_ids
        ):
            reroll_ids.append(case_id)

    selected: list[dict[str, Any]] = []
    hard_cases: list[dict[str, Any]] = []
    incomplete_cases: list[dict[str, Any]] = []
    for case_id in scoped_case_ids:
        candidates = candidates_by_case.get(case_id, [])
        candidates.sort(key=lambda item: tuple(item["candidate_rank"]), reverse=True)
        eligible = [
            item for item in candidates if item["sft_eligibility_pass"] is True
        ]
        complete_candidate_set = len(candidates) >= candidates_per_case
        if not complete_candidate_set:
            incomplete_cases.append(
                {
                    "case_id": case_id,
                    "candidate_count": len(candidates),
                    "expected_candidate_count": candidates_per_case,
                    "status": "incomplete_candidate_set",
                }
            )
        if complete_candidate_set and eligible:
            winner = dict(eligible[0])
            winner["candidate_count"] = len(candidates)
            winner["eligible_candidate_count"] = len(eligible)
            winner["selected_rank"] = 1
            selected.append(winner)
        elif complete_candidate_set:
            hard_cases.append(
                {
                    "case_id": case_id,
                    "expected_verdict": gold_by_case[case_id],
                    "candidate_count": len(candidates),
                    "candidate_episode_ids": [
                        item["episode_id"] for item in candidates
                    ],
                    "rejection_reasons": sorted(
                        {
                            reason
                            for item in candidates
                            for reason in item.get("fatal_errors", [])
                        }
                    ),
                    "status": "hard_case",
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "early-correct-judgment.jsonl", early)
    _write_jsonl(output_dir / "final-only-judgment.jsonl", final_only)
    _write_jsonl(output_dir / "sft-rejected.jsonl", rejected)
    _write_jsonl(output_dir / "candidate-outcomes.jsonl", [
        item
        for case_candidates in candidates_by_case.values()
        for item in case_candidates
    ])
    _write_jsonl(output_dir / "selected-candidates.jsonl", selected)
    _write_jsonl(output_dir / "hard-cases.jsonl", hard_cases)
    _write_jsonl(output_dir / "incomplete-cases.jsonl", incomplete_cases)
    _write_case_list(output_dir / "quality-reroll-case-list.txt", reroll_ids)
    result = {
        "schema_version": SCHEMA_VERSION,
        "source_run": str(run_dir),
        "source_eligibility": str(eligibility_dir),
        "candidates_per_case": candidates_per_case,
        "case_count": len(scoped_case_ids),
        "target_case_ids": scoped_case_ids,
        "successful_case_count": len(candidates_by_case),
        "candidate_count": sum(len(items) for items in candidates_by_case.values()),
        "early_correct_judgment_count": len(early),
        "final_only_judgment_count": len(final_only),
        "sft_rejected_count": len(rejected),
        "selected_case_count": len(selected),
        "hard_case_count": len(hard_cases),
        "incomplete_case_count": len(incomplete_cases),
        "quality_reroll_count": len(reroll_ids),
        "created_at": _now(),
    }
    _write_json(result_path, result)
    return result


def _quality_reroll_case_ids(classification_dir: Path) -> list[str]:
    """Return cases without any selected SFT candidate."""

    classification_path = classification_dir / "classification.json"
    if not classification_path.is_file():
        # Keep the small legacy helper contract used by older offline callers.
        # Production classifications always have classification.json and are
        # scoped below, so this fallback cannot widen a real reroll queue.
        return sorted(
            set(
                _case_ids_from_rows(
                    _read_jsonl(classification_dir / "sft-rejected.jsonl")
                )
                + (
                    _case_ids_from_rows(
                        _read_jsonl(classification_dir / "incomplete-cases.jsonl")
                    )
                    if (classification_dir / "incomplete-cases.jsonl").is_file()
                    else []
                )
            )
        )
    classification = _read_json(classification_path)
    target_case_ids = classification.get("target_case_ids")
    if not isinstance(target_case_ids, list):
        source_run = str(classification.get("source_run") or "").strip()
        if source_run:
            gold_candidates = [
                classification_dir.parent
                / "private-gold"
                / "private_gold.jsonl",
                classification_dir.parent.parent
                / "private-gold"
                / "private_gold.jsonl",
            ]
            gold_path = next(
                (path for path in gold_candidates if path.is_file()),
                None,
            )
            if gold_path is None:
                raise FileNotFoundError(
                    "private gold is missing beside classification directory: "
                    f"{classification_dir}"
                )
            gold_case_ids = {
                str(row.get("case_id") or "").strip()
                for row in _read_jsonl(gold_path)
                if str(row.get("case_id") or "").strip()
            }
            target_case_ids = _target_case_ids_for_run(
                Path(source_run),
                gold_case_ids,
            )
    scoped = {
        str(case_id).strip()
        for case_id in (target_case_ids or [])
        if str(case_id).strip()
    }
    hard_path = classification_dir / "hard-cases.jsonl"
    rejected = (
        _case_ids_from_rows(_read_jsonl(hard_path))
        if hard_path.is_file()
        else []
    )
    incomplete_path = classification_dir / "incomplete-cases.jsonl"
    if incomplete_path.is_file():
        rejected.extend(_case_ids_from_rows(_read_jsonl(incomplete_path)))
    return sorted(set(rejected).intersection(scoped))


def _selected_classification_rows(classification_dir: Path) -> list[dict[str, Any]]:
    path = classification_dir / "selected-candidates.jsonl"
    return _read_jsonl(path) if path.is_file() else []


def _build_quality_reroll_summary(
    *,
    pipeline_dir: Path,
    private_gold: Path,
    classification_dirs: Sequence[Path],
    quality_rounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine one-candidate rounds into the final per-case quality result."""

    gold_case_ids = _case_ids_from_rows(_read_jsonl(private_gold))
    selected: list[dict[str, Any]] = []
    selected_case_ids: set[str] = set()
    for classification_dir in classification_dirs:
        for row in _selected_classification_rows(classification_dir):
            case_id = str(row.get("case_id") or "").strip()
            if not case_id:
                continue
            if case_id in selected_case_ids:
                raise ValueError(
                    f"quality reroll selected the same case twice: {case_id}"
                )
            selected_case_ids.add(case_id)
            selected.append(row)

    final_rejected: list[dict[str, Any]] = []
    final_incomplete: list[dict[str, Any]] = []
    if classification_dirs:
        last_dir = classification_dirs[-1]
        rejected_path = last_dir / "sft-rejected.jsonl"
        incomplete_path = last_dir / "incomplete-cases.jsonl"
        if rejected_path.is_file():
            final_rejected = _read_jsonl(rejected_path)
        if incomplete_path.is_file():
            final_incomplete = _read_jsonl(incomplete_path)

    hard_by_case: dict[str, dict[str, Any]] = {}
    for row in final_rejected:
        case_id = str(row.get("case_id") or "").strip()
        if case_id:
            hard_by_case[case_id] = {
                "case_id": case_id,
                "expected_verdict": row.get("expected_verdict"),
                "candidate_episode_ids": [row.get("episode_id")],
                "rejection_reasons": list(row.get("rejection_reasons") or []),
                "status": "hard_case",
            }
    for row in final_incomplete:
        case_id = str(row.get("case_id") or "").strip()
        if case_id:
            hard_by_case.setdefault(
                case_id,
                {
                    "case_id": case_id,
                    "expected_verdict": None,
                    "candidate_episode_ids": [],
                    "rejection_reasons": ["unresolved_engineering"],
                    "status": "hard_case",
                },
            )
    for case_id in sorted(set(gold_case_ids) - selected_case_ids):
        hard_by_case.setdefault(
            case_id,
            {
                "case_id": case_id,
                "expected_verdict": None,
                "candidate_episode_ids": [],
                "rejection_reasons": ["not_selected_after_quality_rerolls"],
                "status": "hard_case",
            },
        )

    early = [
        row
        for row in selected
        if row.get("bucket") == "early_correct_judgment"
    ]
    final_only = [
        row
        for row in selected
        if row.get("bucket") == "final_only_judgment"
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "case_count": len(gold_case_ids),
        "selected_case_count": len(selected),
        "early_correct_judgment_count": len(early),
        "final_only_judgment_count": len(final_only),
        "hard_case_count": len(hard_by_case),
        "quality_round_count": len(quality_rounds),
        "quality_rounds": [dict(row) for row in quality_rounds],
        "created_at": _now(),
    }
    output_dir = pipeline_dir / "classification" / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "selected-candidates.jsonl", selected)
    _write_jsonl(output_dir / "early-correct-judgment.jsonl", early)
    _write_jsonl(output_dir / "final-only-judgment.jsonl", final_only)
    _write_jsonl(output_dir / "hard-cases.jsonl", hard_by_case.values())
    _write_json(output_dir / "classification.json", result)
    return result


def _run_quality_rerolls(
    *,
    pipeline_dir: Path,
    preparation: Mapping[str, Any],
    private_gold: Path,
    initial_run: Path,
    initial_audit: Path,
    initial_classification_dir: Path,
    profile: str,
    rollout_concurrency: int,
    sft_model: str,
    sft_judge_provider: str,
    sft_judge_base_url: str | None,
    sft_judge_api_key_env: str | None,
    sft_judge_wire_api: str | None,
    sft_judge_enable_thinking: bool | None,
    sft_concurrency: int,
    rollout_timeout: float,
    sft_timeout: float,
    maximum_engineering_attempts: int,
    maximum_sft_audit_attempts: int,
    base_seed: int,
    maximum_rounds: int,
    state: dict[str, Any],
    state_path: Path,
) -> tuple[list[tuple[Path, Path]], dict[str, Any]]:
    """Reroll only SFT-rejected cases for up to ``maximum_rounds`` rounds."""

    if maximum_rounds < 0:
        raise ValueError("quality reroll rounds must be non-negative")
    sources: list[tuple[Path, Path]] = [(initial_run, initial_audit)]
    classification_dirs = [initial_classification_dir]
    quality_rounds: list[dict[str, Any]] = []
    pending = _quality_reroll_case_ids(initial_classification_dir)

    prior_rounds = state.get("quality_rerolls")
    if isinstance(prior_rounds, list):
        for item in prior_rounds:
            if not isinstance(item, Mapping):
                continue
            round_number = int(item.get("round", 0) or 0)
            classification_path = item.get("classification")
            audit_path = item.get("sft_eligibility")
            merged_path = item.get("merged_run")
            if (
                round_number < 1
                or not classification_path
                or not audit_path
                or not merged_path
            ):
                continue
            classification_dir = Path(str(classification_path))
            audit_dir = Path(str(audit_path))
            merged_dir = Path(str(merged_path))
            if (
                (classification_dir / "classification.json").is_file()
                and (audit_dir / "sft_eligibility_summary.json").is_file()
                and (merged_dir / "run_manifest.json").is_file()
            ):
                if merged_dir not in [run for run, _ in sources]:
                    sources.append((merged_dir, audit_dir))
                if classification_dir not in classification_dirs:
                    classification_dirs.append(classification_dir)
                quality_rounds.append(dict(item))
                pending = _quality_reroll_case_ids(classification_dir)

    completed_round_numbers = {
        int(item.get("round", 0) or 0)
        for item in quality_rounds
        if isinstance(item, Mapping)
    }
    next_round = max(completed_round_numbers, default=0) + 1
    while pending and next_round <= maximum_rounds:
        group_name = f"quality-reroll-{next_round:02d}"
        merged_run, merged_manifest = _run_engineering_retries(
            group_name=group_name,
            benchmark=Path(str(preparation["benchmark"])),
            pipeline_dir=pipeline_dir,
            target_ids=pending,
            profile=profile,
            rollout_concurrency=rollout_concurrency,
            base_seed=base_seed + next_round,
            timeout=rollout_timeout,
            maximum_attempts=maximum_engineering_attempts,
            candidates_per_case=1,
        )
        successful_count = int(merged_manifest["result"]["num_episodes"])
        round_info: dict[str, Any] = {
            "round": next_round,
            "target_case_count": len(pending),
            "merged_run": str(merged_run),
            "unresolved_engineering_case_count": int(
                merged_manifest["result"]["num_errors"]
            ),
            "status": "completed",
        }
        if successful_count:
            audit_dir = _run_sft_audit(
                run_dir=merged_run,
                gold=private_gold,
                pipeline_dir=pipeline_dir,
                label=group_name,
                model=sft_model,
                provider=sft_judge_provider,
                base_url=sft_judge_base_url,
                api_key_env=sft_judge_api_key_env,
                wire_api=sft_judge_wire_api,
                enable_thinking=sft_judge_enable_thinking,
                concurrency=sft_concurrency,
                timeout=sft_timeout,
                maximum_attempts=maximum_sft_audit_attempts,
            )
            classification_dir = pipeline_dir / "classification" / group_name
            classification = _classify_initial_outcomes(
                run_dir=merged_run,
                eligibility_dir=audit_dir,
                private_gold=private_gold,
                output_dir=classification_dir,
                candidates_per_case=1,
            )
            sources.append((merged_run, audit_dir))
            classification_dirs.append(classification_dir)
            round_info.update(
                {
                    "sft_eligibility": str(audit_dir),
                    "classification": str(classification_dir),
                    "sft_rejected_count": int(
                        classification["sft_rejected_count"]
                    ),
                    "selected_case_count": int(
                        classification["selected_case_count"]
                    ),
                    "hard_case_count": int(classification["hard_case_count"]),
                }
            )
            pending = _quality_reroll_case_ids(classification_dir)
        else:
            round_info.update(
                {
                    "sft_eligibility": None,
                    "classification": None,
                    "sft_rejected_count": 0,
                    "selected_case_count": 0,
                    "hard_case_count": len(pending),
                }
            )
            # Keep unresolved engineering cases queued for the next quality
            # round, but do not invent an SFT result for them.
            pending = list(pending)

        quality_rounds.append(round_info)
        state["quality_rerolls"] = quality_rounds
        state["updated_at"] = _now()
        _write_json(state_path, state)
        next_round += 1

    final_classification = _build_quality_reroll_summary(
        pipeline_dir=pipeline_dir,
        private_gold=private_gold,
        classification_dirs=classification_dirs,
        quality_rounds=quality_rounds,
    )
    final_classification["remaining_quality_reroll_case_count"] = len(pending)
    _write_json(
        pipeline_dir / "classification" / "final" / "classification.json",
        final_classification,
    )
    return sources, final_classification


def _final_release(
    *,
    pipeline_dir: Path,
    sources: Sequence[tuple[Path, Path]],
    selected_episode_ids: set[str] | None = None,
    output_name: str = "accepted-release",
) -> Path:
    output_dir = pipeline_dir / output_name
    manifest_path = output_dir / "accepted_release_manifest.json"
    if manifest_path.is_file():
        return output_dir
    staged_sources = [(run, audit, None) for run, audit in sources]
    stage_release(
        staged_sources,
        output_dir,
        minimum_accepted_cases=1,
        selected_episode_ids=selected_episode_ids,
    )
    return output_dir


def _build_package(
    *,
    pipeline_dir: Path,
    accepted_release: Path,
    case_split: Path,
    output_name: str = "sft-training-package",
) -> Path:
    package_dir = pipeline_dir / output_name
    manifest_path = package_dir / "MANIFEST.json"
    if manifest_path.is_file():
        return package_dir
    command = _runtime_command(
        "scripts/trajectory/build_sft_training_package.py",
        "--accepted-release",
        str(accepted_release),
        "--case-split",
        str(case_split),
        "--output-dir",
        str(package_dir),
        "--minimum-accepted-cases",
        "1",
    )
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    returncode = _run_command(
        command,
        cwd=REPO_ROOT,
        log_path=package_dir / "command.log",
        env=env,
    )
    if returncode != 0 or not manifest_path.is_file():
        raise RuntimeError(f"SFT package construction failed: {package_dir}")
    return package_dir


def _continue_quality_rerolls(args: argparse.Namespace) -> dict[str, Any]:
    """Attach quality rerolls to a completed initial pipeline."""

    pipeline_dir = args.reroll_from.expanduser().resolve()
    state_path = pipeline_dir / "pipeline-state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"completed pipeline state does not exist: {state_path}")
    state = _read_json(state_path)
    rollout_config, judge_config = _state_model_config(state, args)
    _validate_rollout_profile_model(
        profile=rollout_config["profile"],
        recorded_model=rollout_config["model"],
    )
    preparation = state.get("prepared")
    initial = state.get("initial")
    if not isinstance(preparation, Mapping) or not isinstance(initial, Mapping):
        raise ValueError(f"pipeline lacks initial preparation state: {pipeline_dir}")
    private_gold = Path(str(preparation["private_gold"])).expanduser().resolve()
    initial_run = Path(str(initial["merged_run"])).expanduser().resolve()
    initial_audit = Path(str(initial["sft_eligibility"])).expanduser().resolve()
    initial_classification_dir = pipeline_dir / "classification"
    if not (initial_run / "run_manifest.json").is_file():
        raise FileNotFoundError(f"initial merged run is missing: {initial_run}")
    if not (initial_audit / "sft_eligibility_summary.json").is_file():
        raise FileNotFoundError(f"initial SFT audit is missing: {initial_audit}")
    if not (initial_classification_dir / "classification.json").is_file():
        raise FileNotFoundError(
            f"initial classification is missing: {initial_classification_dir}"
        )

    state["status"] = "running"
    state["quality_reroll_config"] = {
        "maximum_rounds": args.quality_reroll_rounds,
        "rollout_concurrency": args.rollout_concurrency,
        "sft_concurrency": args.sft_concurrency,
        "sft_judge": judge_config,
    }
    state["updated_at"] = _now()
    _write_json(state_path, state)
    sources, final_classification = _run_quality_rerolls(
        pipeline_dir=pipeline_dir,
        preparation=preparation,
        private_gold=private_gold,
        initial_run=initial_run,
        initial_audit=initial_audit,
        initial_classification_dir=initial_classification_dir,
        profile=rollout_config["profile"],
        rollout_concurrency=args.rollout_concurrency,
        sft_model=judge_config["model"],
        sft_judge_provider=judge_config["provider"],
        sft_judge_base_url=judge_config["base_url"],
        sft_judge_api_key_env=judge_config["api_key_env"],
        sft_judge_wire_api=judge_config["wire_api"],
        sft_judge_enable_thinking=judge_config["enable_thinking"],
        sft_concurrency=args.sft_concurrency,
        rollout_timeout=args.rollout_timeout,
        sft_timeout=args.sft_timeout,
        maximum_engineering_attempts=args.maximum_engineering_attempts,
        maximum_sft_audit_attempts=args.maximum_sft_audit_attempts,
        base_seed=args.base_seed,
        maximum_rounds=args.quality_reroll_rounds,
        state=state,
        state_path=state_path,
    )
    selected_episode_ids = {
        str(row["episode_id"])
        for row in _selected_classification_rows(
            pipeline_dir / "classification" / "final"
        )
        if str(row.get("episode_id", "")).strip()
    }
    accepted_release = _final_release(
        pipeline_dir=pipeline_dir,
        sources=sources,
        selected_episode_ids=selected_episode_ids,
        output_name="quality-reroll-release",
    )
    package_dir = _build_package(
        pipeline_dir=pipeline_dir,
        accepted_release=accepted_release,
        case_split=Path(str(preparation["case_split"])),
        output_name="quality-reroll-training-package",
    )
    state["classification"] = final_classification
    state["quality_reroll_release"] = str(accepted_release)
    state["quality_reroll_training_package"] = str(package_dir)
    state["status"] = "completed"
    state["updated_at"] = _now()
    _write_json(state_path, state)
    return state


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.expanduser().resolve()
    train_manifest = (
        args.train_manifest.expanduser().resolve()
        if args.train_manifest is not None
        else dataset_root / "train-manifest.jsonl"
    )
    pipeline_dir = args.output_dir.expanduser().resolve()
    rollout_config = _rollout_config(args)
    _validate_rollout_profile_model(
        profile=rollout_config["profile"],
        recorded_model=rollout_config["model"],
    )
    judge_config = _sft_judge_config(args)
    preparation = prepare_runtime_release(
        dataset_root=dataset_root,
        train_manifest=train_manifest,
        output_dir=pipeline_dir,
        limit=args.limit,
        source_access_policy=args.source_access_policy,
        validation_count=args.validation_count,
        validation_supported=args.validation_supported,
        case_split_seed=args.case_split_seed,
        prohibited_runtime=args.prohibited_runtime,
    )
    state_path = pipeline_dir / "pipeline-state.json"
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_dir": str(pipeline_dir),
        "prepared": preparation,
        "models": {
            "rollout": args.rollout_model,
            "sft_judge": args.sft_model,
        },
        "model_config": {
            "rollout": _rollout_config(args),
            "sft_judge": judge_config,
        },
        "concurrency": {
            "rollout": args.rollout_concurrency,
            "sft_judge": args.sft_concurrency,
        },
        "status": "running",
        "updated_at": _now(),
    }
    _write_json(state_path, state)
    if args.prepare_only:
        state["status"] = "prepared"
        state["updated_at"] = _now()
        _write_json(state_path, state)
        return state

    private_gold_path = Path(str(preparation["private_gold"])).expanduser().resolve()
    all_case_ids = [str(row["case_id"]) for row in _read_jsonl(private_gold_path)]
    initial_run, initial_manifest = _run_engineering_retries(
        group_name="initial",
        benchmark=Path(preparation["benchmark"]),
        pipeline_dir=pipeline_dir,
        target_ids=all_case_ids,
        profile=rollout_config["profile"],
        rollout_concurrency=args.rollout_concurrency,
        base_seed=args.base_seed,
        timeout=args.rollout_timeout,
        maximum_attempts=args.maximum_engineering_attempts,
        candidates_per_case=args.candidates_per_case,
    )
    state["initial"] = {
        "merged_run": str(initial_run),
        "unresolved_engineering_case_count": initial_manifest["result"]["num_errors"],
    }
    state["updated_at"] = _now()
    _write_json(state_path, state)

    initial_audit = _run_sft_audit(
        run_dir=initial_run,
        gold=private_gold_path,
        pipeline_dir=pipeline_dir,
        label="initial",
        model=judge_config["model"],
        provider=judge_config["provider"],
        base_url=judge_config["base_url"],
        api_key_env=judge_config["api_key_env"],
        wire_api=judge_config["wire_api"],
        enable_thinking=judge_config["enable_thinking"],
        concurrency=args.sft_concurrency,
        timeout=args.sft_timeout,
        maximum_attempts=args.maximum_sft_audit_attempts,
    )
    classification = _classify_initial_outcomes(
        run_dir=initial_run,
        eligibility_dir=initial_audit,
        private_gold=private_gold_path,
        output_dir=pipeline_dir / "classification",
        candidates_per_case=args.candidates_per_case,
    )
    state["initial"]["sft_eligibility"] = str(initial_audit)
    state["initial"]["classification"] = str(pipeline_dir / "classification")
    state["classification"] = classification
    state["updated_at"] = _now()
    _write_json(state_path, state)

    sources, final_classification = _run_quality_rerolls(
        pipeline_dir=pipeline_dir,
        preparation=preparation,
        private_gold=private_gold_path,
        initial_run=initial_run,
        initial_audit=initial_audit,
        initial_classification_dir=pipeline_dir / "classification",
        profile=rollout_config["profile"],
        rollout_concurrency=args.rollout_concurrency,
        sft_model=judge_config["model"],
        sft_judge_provider=judge_config["provider"],
        sft_judge_base_url=judge_config["base_url"],
        sft_judge_api_key_env=judge_config["api_key_env"],
        sft_judge_wire_api=judge_config["wire_api"],
        sft_judge_enable_thinking=judge_config["enable_thinking"],
        sft_concurrency=args.sft_concurrency,
        rollout_timeout=args.rollout_timeout,
        sft_timeout=args.sft_timeout,
        maximum_engineering_attempts=args.maximum_engineering_attempts,
        maximum_sft_audit_attempts=args.maximum_sft_audit_attempts,
        base_seed=args.base_seed,
        maximum_rounds=args.quality_reroll_rounds,
        state=state,
        state_path=state_path,
    )
    state["classification"] = final_classification
    state["updated_at"] = _now()
    _write_json(state_path, state)

    selected_episode_ids = {
        str(row["episode_id"])
        for row in _read_jsonl(
            pipeline_dir / "classification" / "final" / "selected-candidates.jsonl"
        )
        if str(row.get("episode_id", "")).strip()
    }
    accepted_release = _final_release(
        pipeline_dir=pipeline_dir,
        sources=sources,
        selected_episode_ids=selected_episode_ids,
    )
    package_dir = _build_package(
        pipeline_dir=pipeline_dir,
        accepted_release=accepted_release,
        case_split=Path(str(preparation["case_split"])),
    )
    state["accepted_release"] = str(accepted_release)
    state["sft_training_package"] = str(package_dir)
    state["status"] = "completed"
    state["updated_at"] = _now()
    _write_json(state_path, state)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        help="defaults to <dataset-root>/train-manifest.jsonl",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--source-access-policy",
        type=Path,
        default=(
            REPO_ROOT
            / "configs"
            / "source-access-policy-web-refuted-v3-expanded-20260821.json"
        ),
        help="Evaluator-side deny policy copied into the runtime release.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="deterministic case-count cap for an isolated smoke run",
    )
    parser.add_argument(
        "--validation-count",
        type=int,
        default=8,
        help="pre-freeze this many validation cases before teacher rollout",
    )
    parser.add_argument(
        "--validation-supported",
        type=int,
        default=None,
        help="supported validation cases; defaults to the source label proportion",
    )
    parser.add_argument(
        "--case-split-seed",
        default="ifv-qwen35-sft-v1",
    )
    parser.add_argument(
        "--prohibited-runtime",
        type=Path,
        default=None,
        help="optional evaluator runtime manifest whose IDs/images must not overlap",
    )
    parser.add_argument("--rollout-profile", default="teacher-gemini")
    parser.add_argument(
        "--rollout-model",
        default="gemini-3.7-flash",
        help=(
            "recorded expected profile model; teacher-gemini and "
            "teacher-gemini36 are pinned to their provider-profile models"
        ),
    )
    parser.add_argument(
        "--sft-model",
        default=get_env("EVISCENE_SFT_ELIGIBILITY_MODEL", "gemini-3.7-flash"),
    )
    parser.add_argument(
        "--sft-judge-provider",
        default=get_env("EVISCENE_SFT_ELIGIBILITY_PROVIDER", "gemini"),
        choices=("gemini", "qwen", "qwen_local", "lmdeploy"),
        help="Independent frozen SFT judge provider; does not change teacher rollout.",
    )
    parser.add_argument(
        "--sft-judge-base-url",
        default=get_env("EVISCENE_SFT_ELIGIBILITY_BASE_URL"),
    )
    parser.add_argument(
        "--sft-judge-api-key-env",
        default=get_env("EVISCENE_SFT_ELIGIBILITY_API_KEY_ENV"),
        help="Environment-variable name containing the independent judge key.",
    )
    parser.add_argument(
        "--sft-judge-wire-api",
        default=get_env("EVISCENE_SFT_ELIGIBILITY_WIRE_API"),
        choices=("chat_completions", "responses", "interactions"),
    )
    parser.add_argument(
        "--sft-judge-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=(
            get_env("EVISCENE_SFT_ELIGIBILITY_ENABLE_THINKING", "")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
            if get_env("EVISCENE_SFT_ELIGIBILITY_ENABLE_THINKING") is not None
            else None
        ),
        help="Enable native thinking for a Qwen SFT judge when supported.",
    )
    parser.add_argument("--rollout-concurrency", type=int, default=10)
    parser.add_argument("--sft-concurrency", type=int, default=10)
    parser.add_argument(
        "--candidates-per-case",
        type=int,
        default=1,
        help="Initial candidates per case; quality rerolls always add one candidate.",
    )
    parser.add_argument(
        "--quality-reroll-rounds",
        type=int,
        default=3,
        help="Maximum additional rounds for cases rejected by the previous SFT audit.",
    )
    parser.add_argument(
        "--reroll-from",
        type=Path,
        default=None,
        help="Continue quality rerolls from an existing completed pipeline directory.",
    )
    parser.add_argument(
        "--bootstrap-initial-run",
        type=Path,
        default=None,
        help=(
            "Completed external run with exactly one terminal trace per case. "
            "Registers it as round 0 before ordinary quality rerolls."
        ),
    )
    parser.add_argument(
        "--bootstrap-initial-eligibility",
        type=Path,
        default=None,
        help=(
            "Frozen SFT eligibility directory matching --bootstrap-initial-run "
            "trace bytes exactly."
        ),
    )
    parser.add_argument("--base-seed", type=int, default=2026082401)
    parser.add_argument("--rollout-timeout", type=float, default=1800.0)
    parser.add_argument("--sft-timeout", type=float, default=180.0)
    parser.add_argument("--maximum-engineering-attempts", type=int, default=12)
    parser.add_argument("--maximum-sft-audit-attempts", type=int, default=12)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="only build and validate the isolated runtime/private projections",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bootstrap_requested = (
        args.bootstrap_initial_run is not None
        or args.bootstrap_initial_eligibility is not None
    )
    if (
        args.bootstrap_initial_run is None
    ) != (
        args.bootstrap_initial_eligibility is None
    ):
        raise SystemExit(
            "--bootstrap-initial-run and --bootstrap-initial-eligibility "
            "must be supplied together"
        )
    if args.reroll_from is not None and bootstrap_requested:
        raise SystemExit(
            "--reroll-from cannot be combined with bootstrap initial inputs"
        )
    if args.reroll_from is None:
        missing = [
            option
            for option, value in (
                ("--dataset-root", args.dataset_root),
                ("--output-dir", args.output_dir),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(", ".join(missing) + " is required")
    elif args.output_dir is not None or args.dataset_root is not None:
        raise SystemExit(
            "--dataset-root and --output-dir are only valid for a new pipeline; "
            "use --reroll-from for an existing pipeline"
        )
    if bootstrap_requested and args.prepare_only:
        raise SystemExit(
            "--prepare-only cannot be combined with bootstrap initial inputs"
        )
    if bootstrap_requested and args.candidates_per_case != 1:
        raise SystemExit(
            "bootstrap initial mode requires --candidates-per-case 1; "
            "quality rerolls add the remaining candidates one round at a time"
        )
    numeric_values = {
        "--rollout-concurrency": args.rollout_concurrency,
        "--sft-concurrency": args.sft_concurrency,
        "--maximum-engineering-attempts": args.maximum_engineering_attempts,
        "--maximum-sft-audit-attempts": args.maximum_sft_audit_attempts,
    }
    invalid = [name for name, value in numeric_values.items() if value < 1]
    if invalid:
        raise SystemExit(", ".join(invalid) + " must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1 when supplied")
    if args.quality_reroll_rounds < 0:
        raise SystemExit("--quality-reroll-rounds must be non-negative")
    if args.validation_count < 1:
        raise SystemExit("--validation-count must be at least 1")
    if args.validation_supported is not None and args.validation_supported < 0:
        raise SystemExit("--validation-supported must be non-negative")
    pinned_rollout_models = {
        "teacher-gemini": "gemini-3.7-flash",
        "teacher-gemini36": "gemini-3.6-flash",
    }
    expected_model = pinned_rollout_models.get(args.rollout_profile)
    if expected_model is not None and args.rollout_model != expected_model:
        raise SystemExit(
            f"{args.rollout_profile} is pinned to {expected_model}; do not "
            "supply a different --rollout-model"
        )
    result = (
        _continue_quality_rerolls(args)
        if args.reroll_from is not None
        else (
            _run_bootstrapped_initial_pipeline(args)
            if bootstrap_requested
            else run_pipeline(args)
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
