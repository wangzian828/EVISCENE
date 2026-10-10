#!/usr/bin/env python3
"""Run frozen teacher SFT eligibility and build the complete training package.

This is the single entry point for the frozen-teacher SFT path:

    rollout + frozen judge
      -> accepted release
      -> full-trajectory dataset export
      -> ms-swift policy/perception conversion
      -> strict audits

An existing eligibility directory or accepted release can be supplied to reuse
already completed judge work.  The script never loads private gold into a model
training row; private gold is used only by the eligibility stage.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


# Resolve the checkout before importing the project configuration helper.
_configured_repo_root = os.environ.get(
    "EVISCENE_REPO_ROOT", os.environ.get("IFV_REPO_ROOT", "")
).strip()
REPO_ROOT = (
    Path(_configured_repo_root)
    if _configured_repo_root
    else Path(__file__).resolve().parents[2]
).expanduser().resolve()
TRAINING_ROOT = REPO_ROOT / "training"
for import_root in (REPO_ROOT, TRAINING_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.trajectory.export_dataset import (
    DEFAULT_SHORT_TRAJECTORY_MAX_TOKENS,
    UTF8_BYTES_PER_APPROX_TOKEN,
    export_dataset,
)
from eviscene.environment import get_env
from eviscene.eval.score_sft_eligibility import (
    _run as run_sft_eligibility,
)
from scripts.trajectory.stage_accepted_teacher_release import stage_release
from eviscene_training.audit import audit_derived_dataset
from eviscene_training.io import sha256_file, write_json
from eviscene_training.perception import convert_accepted_perception_dataset
from eviscene_training.policy import convert_policy_dataset


PACKAGE_SCHEMA_VERSION = "ifv-sft-training-package-v2"
LENGTH_BUCKETS = (
    "within_32k_estimate",
    "within_128k_estimate",
    "over_128k_estimate",
    "unknown",
)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _copy_into_package(source: Path, destination: Path) -> Path:
    source = source.expanduser().resolve()
    if source.is_dir():
        if destination.exists():
            raise FileExistsError(f"package destination already exists: {destination}")
        shutil.copytree(
            source,
            destination,
            copy_function=_hardlink_or_copy,
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _hardlink_or_copy(source, destination)
    return destination


def _hardlink_or_copy(source: str | Path, destination: str | Path) -> str:
    source_path = Path(source)
    destination_path = Path(destination)
    try:
        os.link(source_path, destination_path)
    except OSError:
        shutil.copy2(source_path, destination_path)
    return str(destination_path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _release_runtime_archive(release_path: Path) -> dict[str, Any]:
    manifest = _load_json(release_path / "accepted_release_manifest.json")
    schema_version = str(manifest.get("schema_version", "")).strip()
    archive = manifest.get("runtime_store_archive")
    archive = dict(archive) if isinstance(archive, Mapping) else {}
    selected_rows = _load_jsonl(release_path / "selected_episodes.jsonl")
    archived_rows = 0
    missing_rows: list[str] = []
    for row in selected_rows:
        episode_id = str(row.get("episode_id", "")).strip()
        relative = str(row.get("runtime_store_path", "")).strip()
        if not relative:
            missing_rows.append(episode_id)
            continue
        runtime_store = (release_path / relative).resolve()
        release_root = release_path.resolve()
        if runtime_store != release_root and release_root not in runtime_store.parents:
            raise ValueError(
                "accepted release runtime store escapes release root: "
                f"{episode_id}"
            )
        if (
            not runtime_store.is_dir()
            or not any((runtime_store / "context").glob("*.json"))
            or not (runtime_store / "artifacts" / "sha256").is_dir()
        ):
            missing_rows.append(episode_id)
            continue
        archived_rows += 1
    if schema_version == "ifv-accepted-teacher-release-v4" and missing_rows:
        raise ValueError(
            "accepted release v4 is missing portable runtime stores for "
            + ", ".join(missing_rows[:10])
        )
    declared_selected = int(archive.get("selected_count", 0) or 0)
    if (
        schema_version == "ifv-accepted-teacher-release-v4"
        and declared_selected != archived_rows
    ):
        raise ValueError(
            "accepted release runtime-store count mismatch: "
            f"manifest={declared_selected} actual={archived_rows}"
        )
    return {
        "schema_version": schema_version,
        "portable": (
            schema_version == "ifv-accepted-teacher-release-v4"
            and archived_rows == len(selected_rows)
        ),
        "selected_count": archived_rows,
        "selected_release_count": len(selected_rows),
        "index_path": (
            "runtime_store_index.jsonl"
            if (release_path / "runtime_store_index.jsonl").is_file()
            else None
        ),
        "manifest": archive,
    }


def _build_default_case_split(
    release_path: Path,
    destination: Path,
    *,
    all_train: bool = False,
) -> dict[str, Any]:
    """Create a deterministic fallback split from selected accepted cases.

    A production run should pass the split frozen over the complete candidate
    population.  This fallback exists so a small already-judged smoke batch can
    still produce a self-contained package without another manual command.
    """

    selected = _load_jsonl(release_path / "selected_episodes.jsonl")
    if not selected:
        raise ValueError("accepted release has no selected episodes")
    if len(selected) > 100 and not all_train:
        raise ValueError(
            "large accepted releases require --case-split frozen over the "
            "pre-rollout candidate population"
        )
    trace_by_episode = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (release_path / "traces").glob("*.json")
    }
    ordered = sorted(
        selected,
        key=lambda row: hashlib.sha256(
            f"ifv-sft-package-fallback:{row.get('case_id', '')}".encode("utf-8")
        ).hexdigest(),
    )
    policy_selected = [
        row
        for row in ordered
        if int(row.get("trajectory_sft_rows", 0) or 0) > 0
        or "reasoning_sft" in (row.get("training_buckets") or [])
    ]
    validation_count = 0 if all_train else (1 if len(policy_selected) >= 2 else 0)
    validation_ids = {
        str(row.get("case_id", "")) for row in policy_selected[:validation_count]
    }
    rows: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: str(item.get("case_id", ""))):
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("selected episode lacks case_id")
        episode_id = str(row.get("episode_id", "")).strip()
        trace = trace_by_episode.get(episode_id)
        if trace is None:
            raise ValueError(f"selected episode trace is missing: {episode_id}")
        state = trace.get("state") if isinstance(trace.get("state"), dict) else {}
        runtime_case = state.get("runtime_case") if isinstance(state, dict) else {}
        image_sha256 = str(
            runtime_case.get("image_sha256", "")
            if isinstance(runtime_case, dict)
            else ""
        ).strip()
        if not image_sha256:
            raise ValueError(f"selected trace lacks runtime image_sha256: {episode_id}")
        group_id = hashlib.sha256(
            f"ifv-sft-package-group:{case_id}".encode("utf-8")
        ).hexdigest()[:20]
        rows.append(
            {
                "schema_version": "ifv-sft-case-split-v1",
                "case_id": case_id,
                "image_sha256": image_sha256,
                "split_group_id": group_id,
                "split": "validation" if case_id in validation_ids else "train",
            }
        )
    _write_text(
        destination,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )
    return {
        "mode": "accepted_release_fallback",
        "case_count": len(rows),
        "validation_count": validation_count,
        "path": str(destination),
        "sha256": sha256_file(destination),
    }


def _effective_short_max_tokens(release_path: Path, requested: int | None) -> int:
    """Return the formal maximum length for converted training rows.

    The canonical exporter calls rows above this budget ``long_holdout`` and
    excludes them from the converted ms-swift files.  A positive explicit
    value still allows callers to request a narrower short/holdout split.
    """

    if requested is not None and requested > 0:
        return requested
    # Keep the formal admission gate at 128K approximate model tokens.
    # Episodes above it remain preserved in long_holdout.jsonl.
    return DEFAULT_SHORT_TRAJECTORY_MAX_TOKENS


def _assert_new_or_empty(path: Path) -> None:
    if path.exists():
        unexpected = [
            child.name
            for child in path.iterdir()
            if child.name != "command.log"
        ]
        if unexpected:
            raise FileExistsError(
                f"package output must be new or contain only command.log: "
                f"{path}; unexpected={unexpected}"
            )
    path.mkdir(parents=True, exist_ok=True)


def _load_artifact_index(root: Path) -> dict[str, dict[str, Any]]:
    """Index eligibility artifacts without exposing them to model rows."""

    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.sft_eligibility.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        episode_id = str(
            payload.get("episode_id")
            or (payload.get("rollout") or {}).get("episode_id", "")
        ).strip()
        if episode_id:
            result[episode_id] = {"payload": payload, "path": path}
    return result


def _length_bucket(token_estimate: int | None) -> str:
    """Bucket the provisional UTF-8-byte estimate as approximate tokens.

    The exporter currently uses ``Utf8ByteTokenizer``.  Its field is a byte
    count, not a Qwen tokenizer count, so the bucket is intentionally labelled
    provisional and uses a conservative bytes/3 conversion.  The real
    ms-swift processor audit remains the final length gate.
    """

    if token_estimate is None or token_estimate <= 0:
        return "unknown"
    approximate_tokens = (
        token_estimate + UTF8_BYTES_PER_APPROX_TOKEN - 1
    ) // UTF8_BYTES_PER_APPROX_TOKEN
    if approximate_tokens <= 32768:
        return "within_32k_estimate"
    if approximate_tokens <= 131072:
        return "within_128k_estimate"
    return "over_128k_estimate"


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _quality_bucket(
    *,
    trace: Mapping[str, Any],
    eligibility: Mapping[str, Any] | None,
    selected: bool,
    rejection_reasons: list[str],
    red_flags: list[str],
) -> tuple[str, str]:
    """Return a conservative quality bucket and the reason for it.

    This is deliberately orthogonal to length.  A long high-quality trace is
    still high quality; it is merely placed in a different length bucket.
    """

    gates = (eligibility or {}).get("gates") or {}
    engineering_valid = gates.get("engineering_valid") is True
    successful = str(trace.get("termination", "")) == "success"
    if not successful or not engineering_valid or "engineering_invalid" in rejection_reasons:
        return "engineering_error", "trace_or_engineering_gate_failed"
    if not selected or gates.get("sft_eligibility_pass") is not True:
        return "rejected", "sft_eligibility_rejected"
    metrics = (eligibility or {}).get("metrics") or {}
    gates = (eligibility or {}).get("gates") or {}
    judgment = (eligibility or {}).get("structured_judgment") or {}
    warnings = _as_string_list(metrics.get("warnings"))
    audit_warnings = metrics.get("audit_warnings") or gates.get("audit_warnings") or []
    conduct = str(
        metrics.get("trajectory_conduct") or judgment.get("trajectory_conduct", "")
    ).strip().lower()
    confidence = float(metrics.get("confidence", 0.0) or 0.0)
    has_correct_verdict = metrics.get("verdict_correct") is True
    decision_support = str(
        metrics.get("decision_support") or judgment.get("decision_support", "")
    )
    has_expected_support = decision_support == str(
        metrics.get("expected_decision_support", "")
    )
    overclaiming = str(judgment.get("overclaiming", ""))
    boundary_assessment = str(judgment.get("boundary_assessment", ""))
    # ``missing_training_quality_score`` is provenance metadata from the
    # older rollout scorer.  It must remain visible in the catalog, but it is
    # not by itself a quality downgrade when the frozen SFT judge is clean.
    substantive_red_flags = [
        flag for flag in red_flags if flag != "missing_training_quality_score"
    ]
    if (
        substantive_red_flags
        or warnings
        or audit_warnings
        or conduct not in {"", "clean", "good", "pass", "passed"}
        or not has_correct_verdict
        or not has_expected_support
        or overclaiming not in {"", "none"}
        or boundary_assessment not in {"", "respected"}
        or confidence < 0.9
    ):
        return "usable", "accepted_with_warnings_or_recovered_conduct"
    return "high", "accepted_cleanly"


def _build_trajectory_catalog(
    release_path: Path,
    package_dir: Path,
) -> dict[str, Any]:
    """Copy every staged trace once and build quality/length indexes.

    The accepted release is the frozen source of truth.  It contains both the
    selected and rejected traces, so this function never reconstructs a trace
    from a model-visible export and never deletes a source artifact.
    """

    selected_rows = _load_jsonl(release_path / "selected_episodes.jsonl")
    rejected_rows = _load_jsonl(release_path / "rejected_episodes.jsonl")
    eligibility = _load_artifact_index(release_path / "eligibility")
    eligibility.update(_load_artifact_index(release_path / "rejected" / "eligibility"))
    selected_by_episode = {
        str(row.get("episode_id", "")): row for row in selected_rows
    }
    raw_root = package_dir / "all-trajectories"
    bucket_root = package_dir / "trajectory-buckets"
    catalog_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    candidates: list[tuple[dict[str, Any], bool]] = [
        (row, True) for row in selected_rows
    ] + [(row, False) for row in rejected_rows]
    for row, is_selected in candidates:
        episode_id = str(row.get("episode_id", "")).strip()
        if not episode_id:
            continue
        if is_selected:
            trace_path = release_path / "traces" / f"{episode_id}.json"
        else:
            relative = str(row.get("trace_path", "")).strip()
            trace_path = release_path / relative
        if not trace_path.is_file():
            raise FileNotFoundError(f"staged trajectory is missing: {trace_path}")
        trace_sha256 = sha256_file(trace_path)
        trace_key = str(row.get("source_trace_sha256", "")).strip() or trace_sha256
        if trace_key in seen or episode_id in seen:
            continue
        seen.add(trace_key)
        seen.add(episode_id)
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        artifact = eligibility.get(episode_id)
        eligibility_payload = artifact["payload"] if artifact else None
        reasons = _as_string_list(row.get("rejection_reasons"))
        selected_metadata = selected_by_episode.get(episode_id, {})
        red_flags = _as_string_list(
            selected_metadata.get("deterministic_red_flags")
            or row.get("deterministic_red_flags")
        )
        token_estimate_raw = (
            selected_metadata.get("trajectory_token_count_estimate")
            or row.get("trajectory_token_count_estimate")
        )
        try:
            token_estimate = int(token_estimate_raw) if token_estimate_raw else None
        except (TypeError, ValueError):
            token_estimate = None
        if token_estimate is None:
            token_estimate = max(1, len(trace_path.read_bytes()))
        quality, quality_reason = _quality_bucket(
            trace=trace,
            eligibility=eligibility_payload,
            selected=is_selected,
            rejection_reasons=reasons,
            red_flags=red_flags,
        )
        length = _length_bucket(token_estimate)
        source_sha = str(
            selected_metadata.get("source_trace_sha256")
            or row.get("source_trace_sha256")
            or trace_sha256
        )
        stable_name = f"{episode_id}--{source_sha[:12]}.json"
        all_destination = raw_root / stable_name
        all_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trace_path, all_destination)
        # These are explicit bucket copies, not a destructive move.  The
        # catalog is the authoritative one-row-per-trace mapping.
        bucket_destination = bucket_root / f"quality-{quality}" / stable_name
        bucket_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trace_path, bucket_destination)
        metrics = (eligibility_payload or {}).get("metrics") or {}
        gates = (eligibility_payload or {}).get("gates") or {}
        judgment = (eligibility_payload or {}).get("structured_judgment") or {}
        catalog_rows.append(
            {
                "schema_version": "ifv-trajectory-catalog-row-v1",
                "case_id": str(row.get("case_id", "")).strip(),
                "episode_id": episode_id,
                "quality_bucket": quality,
                "quality_reason": quality_reason,
                "length_bucket": length,
                "token_count_estimate": token_estimate,
                "approx_token_count_estimate_utf8_div4": (
                    (token_estimate + 3) // 4
                    if token_estimate > 0
                    else None
                ),
                "token_estimate_kind": "utf8_bytes_provisional",
                "sft_eligibility_pass": gates.get("sft_eligibility_pass"),
                "engineering_valid": gates.get("engineering_valid"),
                "deterministic_gate_pass": selected_metadata.get(
                    "deterministic_hard_gate_pass"
                ),
                "verdict": trace.get("verdict"),
                "verdict_correct": metrics.get("verdict_correct"),
                "trajectory_conduct": metrics.get("trajectory_conduct"),
                "llm_judge": {
                    "target_scope": judgment.get("target_scope"),
                    "decision_support": judgment.get("decision_support"),
                    "retrieval_quality": judgment.get("retrieval_quality"),
                    "overclaiming": judgment.get("overclaiming"),
                    "boundary_assessment": judgment.get("boundary_assessment"),
                    "trajectory_conduct": judgment.get("trajectory_conduct"),
                    "confidence": judgment.get("confidence"),
                },
                "fatal_errors": metrics.get("fatal_errors") or [],
                "warnings": metrics.get("warnings") or [],
                "teacher_score": selected_metadata.get("teacher_score"),
                "rejection_reasons": reasons,
                "accepted_for_policy_sft": bool(is_selected),
                "accepted_for_perception_sft": bool(
                    is_selected and selected_metadata.get("perception_example_count", 0)
                ),
                "training_action": (
                    "train_candidate"
                    if quality in {"high", "usable"} and is_selected
                    else "audit_only"
                ),
                "source_trace_sha256": source_sha,
                "raw_trace_path": all_destination.relative_to(package_dir).as_posix(),
                "quality_bucket_path": bucket_destination.relative_to(package_dir).as_posix(),
                "source_release_trace_path": trace_path.relative_to(release_path).as_posix(),
                "eligibility_path": (
                    artifact["path"].relative_to(release_path).as_posix()
                    if artifact
                    else None
                ),
            }
        )

    catalog_rows.sort(key=lambda item: str(item["episode_id"]))
    catalog_path = package_dir / "trajectory_catalog.jsonl"
    _write_text(
        catalog_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in catalog_rows),
    )
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in catalog_rows:
        by_bucket.setdefault(str(row["length_bucket"]), []).append(row)
    for length, rows in by_bucket.items():
        _write_text(
            bucket_root / "length" / f"{length}.jsonl",
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        )
    quality_counts: dict[str, int] = {}
    length_counts: dict[str, int] = {}
    for row in catalog_rows:
        quality_counts[row["quality_bucket"]] = quality_counts.get(row["quality_bucket"], 0) + 1
        length_counts[row["length_bucket"]] = length_counts.get(row["length_bucket"], 0) + 1
    return {
        "path": "trajectory_catalog.jsonl",
        "count": len(catalog_rows),
        "quality_counts": quality_counts,
        "length_counts": length_counts,
        "all_trajectories_dir": "all-trajectories",
        "quality_buckets_dir": "trajectory-buckets/quality-*",
        "length_indexes_dir": "trajectory-buckets/length",
        "sha256": sha256_file(catalog_path),
    }


def _training_command(args: argparse.Namespace, package_dir: Path) -> list[str] | None:
    if not args.run_training:
        return None
    required = {
        "--training-script": args.training_script,
        "--model-profile": args.model_profile,
        "--sft-profile": args.sft_profile,
        "--experiment-id": args.experiment_id,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("--run-training requires " + ", ".join(missing))
    train_path = package_dir / "ms-swift-policy" / "train.jsonl"
    validation_path = package_dir / "ms-swift-policy" / "validation.jsonl"
    if not train_path.stat().st_size or not validation_path.stat().st_size:
        raise ValueError(
            "--run-training requires non-empty policy train.jsonl and "
            "validation.jsonl; package with a validation split first"
        )
    command = [
        str(args.training_script.expanduser().resolve()),
        str(args.model_profile.expanduser().resolve()),
        str(args.sft_profile.expanduser().resolve()),
        str(train_path),
        str(validation_path),
        str(args.experiment_id),
    ]
    if args.resume_checkpoint is not None:
        command.append(str(args.resume_checkpoint.expanduser().resolve()))
    return command


def _judge_namespace(args: argparse.Namespace, package_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=args.run_dir.expanduser().resolve(),
        gold=args.gold.expanduser().resolve(),
        private_gold_sidecar=None,
        output_dir=(package_dir / "sft-eligibility").resolve(),
        cache_dir=(package_dir / "sft-eligibility" / "cache").resolve(),
        image_root=(args.image_root.expanduser().resolve() if args.image_root else None),
        storage_dir=(package_dir / "accepted-release").resolve(),
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        wire_api=args.wire_api,
        enable_thinking=args.enable_thinking,
        max_tokens=args.judge_max_tokens,
        provider_retries=int(
            get_env("SFT_ELIGIBILITY_PROVIDER_RETRIES", "3")
        ),
        trace_retries=int(
            get_env("SFT_ELIGIBILITY_TRACE_RETRIES", "12")
        ),
        trace_retry_delay=float(
            get_env("SFT_ELIGIBILITY_TRACE_RETRY_DELAY", "30")
        ),
        timeout=args.timeout,
        concurrency=args.concurrency,
        force=args.force_judge,
    )


def _build_release(args: argparse.Namespace, package_dir: Path) -> tuple[Path, dict[str, Any]]:
    release_path = (package_dir / "accepted-release").resolve()
    if args.accepted_release is not None:
        source = args.accepted_release.expanduser().resolve()
        if source != release_path:
            _copy_into_package(source, release_path)
        manifest_path = release_path / "accepted_release_manifest.json"
        return release_path, {
            "mode": "reused_accepted_release",
            "path": str(release_path),
            "manifest_sha256": sha256_file(manifest_path),
        }

    run_dir = args.run_dir.expanduser().resolve()
    if args.eligibility_dir is None:
        if args.gold is None:
            raise ValueError("--gold is required when --eligibility-dir is omitted")
        summary = asyncio.run(run_sft_eligibility(_judge_namespace(args, package_dir)))
        return release_path, {
            "mode": "fresh_frozen_judge",
            "path": str(release_path),
            "manifest_sha256": sha256_file(release_path / "accepted_release_manifest.json"),
            "summary": summary,
        }

    eligibility_dir = args.eligibility_dir.expanduser().resolve()
    result = stage_release(
        [(run_dir, eligibility_dir, None)],
        release_path,
        minimum_accepted_cases=args.minimum_accepted_cases,
    )
    return release_path, {
        "mode": "reused_sft_eligibility",
        "path": str(release_path),
        "manifest_sha256": sha256_file(release_path / "accepted_release_manifest.json"),
        "summary": result,
    }


def _package_readme(manifest: Mapping[str, Any]) -> str:
    counts = manifest["counts"]
    training = manifest["training"]
    return f"""# EviScene frozen-teacher SFT training package

This directory is the complete output of the frozen-teacher SFT pipeline:

`rollout -> SFT eligibility judge -> accepted release -> full-trajectory export -> ms-swift conversion -> strict audit`

## Direct training inputs

- `ms-swift-policy/`: primary Agent SFT input. Use this directory with the existing `training/scripts/train/run_sft.sh` flow.
- `ms-swift-perception/`: optional independent perception SFT input.

The accepted release contains {counts['selected_release_cases']} selected case(s):

- policy reasoning SFT: {counts['policy_rows']} complete episode row(s);
- action-only executable traces: {counts['action_only_rows']} row(s), kept separately and not mixed into reasoning SFT;
- independent perception SFT: {counts['perception_rows']} row(s).

Each policy row is one complete episode; this package does not use the legacy
step-level `policy_trajectories.jsonl` format. Episodes above the 128K
approximate-token admission gate remain in `accepted-dataset/long_holdout.jsonl`
and are not converted into the direct policy training files. Action-only rows
are available at `accepted-dataset/action_only.jsonl` for separate action/RL
workflows.

## Reproducibility and audit artifacts

- `accepted-release/`: frozen selected traces and the eligibility artifacts used for selection.
- `accepted-release/runtime-stores/`: archived request context and media needed to reproduce candidate/crop images without the original rollout directory.
- `all-trajectories/`: every staged teacher trajectory, including rejected and engineering-error traces.
- `trajectory-buckets/`: quality copies and length indexes.  Bucket membership never deletes the source trace.
- `trajectory_catalog.jsonl`: one audit row per complete trajectory with quality and length dimensions.
- `accepted-dataset/`: provider-neutral full-trajectory dataset before ms-swift conversion.
- `case-split/case_split.jsonl`: frozen leakage-safe split consumed by the exporter.
- `audits/`: strict audits for both converted datasets.
- `MANIFEST.json`: hashes, counts, source mode, and audit results.

Training was requested: `{training['requested']}`; started: `{training['started']}`; status: `{training['status']}`.
The package-only default writes the exact command to `training_plan.json` without starting a GPU job.

`accepted-release/` and `sft-eligibility/` are provenance/audit artifacts, not model input. Private gold is not copied into this package and no evaluator-private fields are included in the converted model-visible rows.

Generated with schema `{manifest['schema_version']}`.
"""


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.output_dir.expanduser().resolve()
    _assert_new_or_empty(package_dir)

    release_path, selection = _build_release(args, package_dir)
    runtime_archive = _release_runtime_archive(release_path)
    catalog = _build_trajectory_catalog(release_path, package_dir)
    split_destination = package_dir / "case-split" / "case_split.jsonl"
    if args.case_split is not None:
        _copy_into_package(args.case_split, split_destination)
        split_info = {
            "mode": "supplied_frozen_split",
            "path": str(split_destination),
            "sha256": sha256_file(split_destination),
        }
    else:
        split_info = _build_default_case_split(
            release_path,
            split_destination,
            all_train=args.all_train,
        )

    accepted_dataset = package_dir / "accepted-dataset"
    effective_short_max_tokens = _effective_short_max_tokens(
        release_path,
        args.short_max_tokens,
    )
    dataset_manifest = export_dataset(
        [],
        accepted_dataset,
        accepted_release_path=release_path,
        case_split_path=split_destination,
        minimum_accepted_cases=args.minimum_accepted_cases,
        short_max_tokens=effective_short_max_tokens,
        export_concurrency=args.export_concurrency,
    )

    policy_dir = package_dir / "ms-swift-policy"
    perception_dir = package_dir / "ms-swift-perception"
    policy_manifest = convert_policy_dataset(accepted_dataset, policy_dir)
    perception_manifest = convert_accepted_perception_dataset(
        accepted_dataset,
        perception_dir,
    )

    audits_dir = package_dir / "audits"
    policy_audit = audit_derived_dataset(policy_dir)
    perception_audit = audit_derived_dataset(perception_dir)
    write_json(audits_dir / "policy.json", policy_audit)
    write_json(audits_dir / "perception.json", perception_audit)
    if not policy_audit["passed"] or not perception_audit["passed"]:
        raise RuntimeError(
            "strict converted-dataset audit failed; see package/audits/*.json"
        )

    training_command = _training_command(args, package_dir)
    training: dict[str, Any] = {
        "requested": bool(args.run_training),
        "started": False,
        "status": "not_started",
        "command": training_command,
    }
    write_json(
        package_dir / "training_plan.json",
        {
            "schema_version": "ifv-training-plan-v1",
            "requested": bool(args.run_training),
            "command": training_command,
            "policy_train": "ms-swift-policy/train.jsonl",
            "policy_validation": "ms-swift-policy/validation.jsonl",
            "note": (
                "Training is opt-in. Pass --run-training with profiles and an "
                "experiment id to execute it after package generation."
            ),
        },
    )
    manifest: dict[str, Any] = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "selection": selection,
        "accepted_release": {
            "path": "accepted-release",
            "manifest_sha256": sha256_file(
                release_path / "accepted_release_manifest.json"
            ),
            "runtime_store_archive": runtime_archive,
        },
        "trajectory_catalog": catalog,
        "case_split": split_info,
        "effective_short_max_tokens": effective_short_max_tokens,
        "accepted_dataset": dataset_manifest,
        "policy_dataset": policy_manifest,
        "perception_dataset": perception_manifest,
        "audits": {
            "policy": policy_audit,
            "perception": perception_audit,
        },
        "counts": {
            "accepted_cases": int(dataset_manifest.get("accepted_case_count", 0)),
            "selected_release_cases": int(
                dataset_manifest.get(
                    "selected_release_case_count",
                    dataset_manifest.get("accepted_case_count", 0),
                )
            ),
            "policy_rows": int(policy_manifest.get("example_count", 0)),
            "perception_rows": int(perception_manifest.get("example_count", 0)),
            "action_only_rows": int(
                dataset_manifest.get("action_only_episode_count", 0)
            ),
            "long_holdout_rows": int(
                dataset_manifest.get("long_holdout_episode_count", 0)
            ),
        },
        "training_inputs": {
            "policy": "ms-swift-policy",
            "perception": "ms-swift-perception",
            "action_only": "accepted-dataset/action_only.jsonl",
            "legacy_step_level_policy": False,
        },
        "training": training,
    }
    write_json(package_dir / "MANIFEST.json", manifest)
    _write_text(package_dir / "README.md", _package_readme(manifest))
    if training_command is not None:
        training["started"] = True
        training["status"] = "running"
        write_json(package_dir / "MANIFEST.json", manifest)
        completed = subprocess.run(training_command, cwd=str(REPO_ROOT), check=False)
        training["status"] = "completed" if completed.returncode == 0 else "failed"
        training["returncode"] = completed.returncode
        write_json(package_dir / "MANIFEST.json", manifest)
        _write_text(package_dir / "README.md", _package_readme(manifest))
        if completed.returncode != 0:
            raise RuntimeError(
                f"SFT training command failed with exit code {completed.returncode}"
            )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--accepted-release", type=Path)
    source.add_argument("--run-dir", type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--eligibility-dir", type=Path)
    parser.add_argument(
        "--case-split",
        type=Path,
        help=(
            "pre-frozen full-population split; if omitted, create a deterministic "
            "fallback split from selected accepted cases"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument(
        "--provider",
        choices=("gemini", "qwen", "qwen_local", "lmdeploy"),
        default=get_env("EVISCENE_SFT_ELIGIBILITY_PROVIDER", "gemini"),
    )
    parser.add_argument(
        "--model",
        default=get_env("EVISCENE_SFT_ELIGIBILITY_MODEL", "gemini-3.7-flash"),
    )
    parser.add_argument(
        "--base-url",
        default=get_env("EVISCENE_SFT_ELIGIBILITY_BASE_URL"),
    )
    parser.add_argument(
        "--api-key-env",
        default=get_env("EVISCENE_SFT_ELIGIBILITY_API_KEY_ENV"),
    )
    parser.add_argument(
        "--wire-api",
        choices=("chat_completions", "responses", "interactions"),
        default=get_env("EVISCENE_SFT_ELIGIBILITY_WIRE_API"),
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=(
            get_env("EVISCENE_SFT_ELIGIBILITY_ENABLE_THINKING", "")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
            if get_env("EVISCENE_SFT_ELIGIBILITY_ENABLE_THINKING") is not None
            else None
        ),
    )
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--export-concurrency",
        type=int,
        default=8,
        help=(
            "number of concurrent trace-to-SFT exporter workers; this is "
            "separate from --concurrency, which controls the LLM judge"
        ),
    )
    parser.add_argument("--force-judge", action="store_true")
    parser.add_argument("--minimum-accepted-cases", type=int, default=1)
    parser.add_argument(
        "--short-max-tokens",
        type=int,
        default=None,
        help=(
            "optional short-trajectory budget in approximate model tokens; "
            "default is 131072 and longer episodes remain in long_holdout.jsonl"
        ),
    )
    parser.add_argument(
        "--all-train",
        action="store_true",
        help="put the fallback split entirely in train (no validation split)",
    )
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="after packaging, execute the opt-in policy SFT command",
    )
    parser.add_argument("--training-script", type=Path)
    parser.add_argument("--model-profile", type=Path)
    parser.add_argument("--sft-profile", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.accepted_release is not None and (
        args.gold is not None or args.eligibility_dir is not None
    ):
        raise SystemExit(
            "--accepted-release cannot be combined with --gold or --eligibility-dir"
        )
    if args.run_dir is not None and args.eligibility_dir is None and args.gold is None:
        raise SystemExit("--gold is required for a fresh SFT eligibility run")
    if args.minimum_accepted_cases < 0:
        raise SystemExit("--minimum-accepted-cases must be non-negative")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.export_concurrency < 1:
        raise SystemExit("--export-concurrency must be at least 1")
    result = build_package(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
