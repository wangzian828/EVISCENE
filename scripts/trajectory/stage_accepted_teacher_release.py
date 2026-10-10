#!/usr/bin/env python3
"""Stage one strictly accepted teacher rollout per case for SFT export.

Teacher retries historically used one rollout per invocation, so their public
episode IDs equal case IDs and collide across run directories.  This tool selects
the best strict/structured candidate per case after deterministic hard-fail
screening and stages only that candidate into one canonical run directory.
Semantic reward artifacts may be copied as optional diagnostics, but they are
never an eligibility gate.  It never reads evaluator-private gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eviscene.trajectory.exporter import (
    export_trajectory_action_only_example,
    export_trajectory_sft_example,
    unified_react_training_buckets,
)
from eviscene.trajectory.perception_exporter import export_perception_example


SCHEMA_VERSION = "ifv-accepted-teacher-release-v4"
DETERMINISTIC_FATAL_TEACHER_REASONS = frozenset(
    {
        "incorrect_result",
        "engineering_error",
        "legacy_core_ownership",
    }
)
JSONL_ARTIFACTS = (
    "trajectory_scores.jsonl",
    "run_results.jsonl",
    "rollout_groups.jsonl",
    "process_metrics.jsonl",
    "post_rollout_rewards.jsonl",
)


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_store_root(trace: Mapping[str, Any]) -> Path | None:
    state = trace.get("state")
    if not isinstance(state, Mapping):
        return None
    runtime_store = state.get("runtime_store")
    if not isinstance(runtime_store, Mapping):
        return None
    raw = str(runtime_store.get("runtime_path", "")).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path.resolve() if path.is_dir() else None


def _stage_runtime_store(
    *,
    output_dir: Path,
    trace: Mapping[str, Any],
    episode_id: str,
    trace_sha256: str,
    relative_root: Path,
    required: bool,
) -> Dict[str, Any]:
    source = _runtime_store_root(trace)
    if source is None:
        if required:
            raise ValueError(
                "accepted unified-ReAct trace has no available runtime store: "
                f"{episode_id}"
            )
        return {}
    if not any((source / "context").glob("*.json")):
        if required:
            raise ValueError(
                "accepted unified-ReAct runtime store has no context manifests: "
                f"{episode_id}"
            )
        return {"runtime_store_error": "runtime store has no context manifests"}
    if not (source / "artifacts" / "sha256").is_dir():
        if required:
            raise ValueError(
                "accepted unified-ReAct runtime store has no artifact store: "
                f"{episode_id}"
            )
        return {"runtime_store_error": "runtime store has no artifact store"}

    safe_episode = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:12]
    destination = (
        output_dir
        / relative_root
        / f"{trace_sha256[:12]}--{safe_episode}"
    )
    if destination.exists():
        raise FileExistsError(f"runtime store destination already exists: {destination}")

    tree_digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    hardlink_count = 0
    copy_count = 0
    for source_path in sorted(
        source.rglob("*"),
        key=lambda path: path.relative_to(source).as_posix(),
    ):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_symlink():
            raise ValueError(
                f"runtime store cannot contain symbolic links: {source_path}"
            )
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            continue
        if not source_path.is_file():
            raise ValueError(
                f"runtime store contains unsupported entry: {source_path}"
            )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_path, destination_path)
            hardlink_count += 1
        except OSError:
            shutil.copy2(source_path, destination_path)
            copy_count += 1
        digest = _sha256(destination_path)
        size = destination_path.stat().st_size
        tree_digest.update(relative.as_posix().encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(str(size).encode("ascii"))
        tree_digest.update(b"\n")
        file_count += 1
        byte_count += size

    if required and file_count < 1:
        raise ValueError(f"accepted runtime store is empty: {episode_id}")
    return {
        "runtime_store_path": destination.relative_to(output_dir).as_posix(),
        "runtime_store_tree_sha256": tree_digest.hexdigest(),
        "runtime_store_file_count": file_count,
        "runtime_store_byte_count": byte_count,
        "runtime_store_materialization": {
            "hardlink_files": hardlink_count,
            "copied_files": copy_count,
        },
    }


def _gate_index(root: Path, suffix: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.glob(f"*{suffix}")):
        payload = _load_json(path)
        episode_id = str(
            payload.get("episode_id")
            or (payload.get("rollout") or {}).get("episode_id", "")
        ).strip()
        if not episode_id or episode_id in result:
            raise ValueError(f"invalid or duplicate gate artifact: {path}")
        result[episode_id] = {"payload": payload, "path": path}
    return result


def _score_index(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in _load_jsonl(run_dir / "trajectory_scores.jsonl"):
        episode_id = str(row.get("episode_id") or row.get("case_id", "")).strip()
        if episode_id:
            result[episode_id] = row
    return result


def _deterministic_teacher_quality(score: Mapping[str, Any]) -> Dict[str, Any]:
    if not score:
        return {
            "hard_gate_pass": True,
            "fatal_reasons": [],
            "red_flags": ["missing_training_quality_score"],
        }
    reasons = [
        str(item)
        for item in score.get("training_exclusion_reasons", []) or []
        if str(item).strip()
    ]
    fatal_reasons = [
        reason
        for reason in reasons
        if reason in DETERMINISTIC_FATAL_TEACHER_REASONS
    ]
    if score.get("training_eligible") is False and not reasons:
        fatal_reasons.append("teacher_quality_gate_failed")
    return {
        "hard_gate_pass": not fatal_reasons,
        "fatal_reasons": list(dict.fromkeys(fatal_reasons)),
        "red_flags": [
            reason
            for reason in reasons
            if reason not in DETERMINISTIC_FATAL_TEACHER_REASONS
        ],
    }


def sft_candidate_rank(
    eligibility: Mapping[str, Any],
    *,
    teacher_score: float = 0.0,
    episode_id: str = "",
) -> tuple[object, ...]:
    """Rank candidates after SFT judging, with safety gates before style."""

    metrics = eligibility.get("metrics") or {}
    gates = eligibility.get("gates") or {}
    passed = int(gates.get("sft_eligibility_pass") is True)
    target_scope = {
        "direct_target": 3,
        "decisive_subfact": 2,
        "related_but_incomplete": 1,
        "unrelated_fact": 0,
        "unclear": 0,
    }.get(str(metrics.get("target_scope", "")), 0)
    retrieval = {
        "effective": 2,
        "mixed": 1,
        "poor": 0,
    }.get(str(metrics.get("retrieval_quality", "")), 0)
    conduct = {
        "clean": 2,
        "recovered_minor": 1,
        "degraded_repetition": 0,
        "unresolved": 0,
    }.get(str(metrics.get("trajectory_conduct", "")), 0)
    overclaiming = {
        "none": 2,
        "minor": 1,
        "major": 0,
    }.get(str(metrics.get("overclaiming", "")), 0)
    boundary = {
        "respected": 2,
        "minor_issue": 1,
        "major_issue": 0,
    }.get(str(metrics.get("boundary_assessment", "")), 0)
    warnings = len(metrics.get("warnings") or [])
    fatal_errors = len(metrics.get("fatal_errors") or [])
    confidence = int(round(float(metrics.get("confidence", 0.0) or 0.0) * 1000))
    return (
        passed,
        target_scope,
        retrieval,
        conduct,
        overclaiming,
        boundary,
        -fatal_errors,
        -warnings,
        confidence,
        int(round(float(teacher_score or 0.0) * 1000)),
        str(episode_id),
    )


def _eligible(
    trace: Mapping[str, Any],
    trace_sha256: str,
    eligibility: Mapping[str, Any],
) -> bool:
    eligibility_gates = eligibility.get("gates") or {}
    state = trace.get("state") if isinstance(trace.get("state"), Mapping) else {}
    policy_version = str(
        trace.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    )
    eligibility_policy = str(
        (eligibility.get("source_trace") or {}).get(
            "decision_policy_version",
            "",
        )
    )
    return bool(
        str(trace.get("termination", "")) == "success"
        and str(trace.get("verdict", "")) in {"real", "fake"}
        and eligibility_gates.get("engineering_valid") is True
        and eligibility_gates.get("sft_eligibility_pass") is True
        and str((eligibility.get("source_trace") or {}).get("sha256", ""))
        == trace_sha256
        and (
            policy_version != "unified-react-v1"
            or eligibility_policy == policy_version
        )
    )


def _rejection_reasons(
    *,
    trace: Mapping[str, Any],
    trace_sha256: str,
    eligibility: Mapping[str, Any] | None,
) -> list[str]:
    if eligibility is None:
        return ["missing_sft_eligibility_artifact"]
    reasons: list[str] = []
    gates = eligibility.get("gates") or {}
    if str(trace.get("termination", "")) != "success":
        reasons.append("trace_not_successful")
    if str(trace.get("verdict", "")) not in {"real", "fake"}:
        reasons.append("missing_binary_verdict")
    if gates.get("engineering_valid") is not True:
        reasons.append("engineering_invalid")
    if gates.get("sft_eligibility_pass") is not True:
        reasons.append("sft_judge_rejected")
    if str((eligibility.get("source_trace") or {}).get("sha256", "")) != trace_sha256:
        reasons.append("trace_sha256_mismatch")
    state = trace.get("state") if isinstance(trace.get("state"), Mapping) else {}
    trace_policy = str(
        trace.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    )
    eligibility_policy = str(
        (eligibility.get("source_trace") or {}).get(
            "decision_policy_version",
            "",
        )
    )
    if trace_policy == "unified-react-v1" and eligibility_policy != trace_policy:
        reasons.append("decision_policy_version_mismatch")
    metrics = eligibility.get("metrics") or {}
    if metrics.get("fatal_errors"):
        reasons.append("sft_judge_fatal_errors")
    return list(dict.fromkeys(reasons))


def _stage_rejected_trace(
    *,
    output_dir: Path,
    trace_path: Path,
    eligibility_path: Path | None,
    trace_sha256: str,
) -> Dict[str, Any]:
    trace_destination = (
        output_dir
        / "rejected"
        / "traces"
        / f"{trace_sha256[:12]}--{trace_path.name}"
    )
    trace_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trace_path, trace_destination)
    result = {
        "trace_path": trace_destination.relative_to(output_dir).as_posix(),
    }
    trace = _load_json(trace_path)
    result.update(
        _stage_runtime_store(
            output_dir=output_dir,
            trace=trace,
            episode_id=str(trace.get("image_id") or trace_path.stem),
            trace_sha256=trace_sha256,
            relative_root=Path("rejected") / "runtime-stores",
            required=False,
        )
    )
    if eligibility_path is not None and eligibility_path.is_file():
        eligibility_destination = (
            output_dir / "rejected" / "eligibility" / eligibility_path.name
        )
        eligibility_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(eligibility_path, eligibility_destination)
        result["eligibility_path"] = (
            eligibility_destination.relative_to(output_dir).as_posix()
        )
    return result


def _matching_semantic_row(
    *,
    semantic_index: Mapping[str, Mapping[str, Any]],
    episode_id: str,
    case_id: str,
    trace_sha256: str,
) -> Mapping[str, Any] | None:
    row = semantic_index.get(episode_id)
    if row is None:
        return None
    payload = row["payload"]
    semantic_episode_id = str(
        (payload.get("rollout") or {}).get("episode_id")
        or payload.get("episode_id", "")
    )
    if (
        str(payload.get("case_id", "")) != case_id
        or semantic_episode_id != episode_id
        or str((payload.get("source_trace") or {}).get("sha256", ""))
        != trace_sha256
    ):
        raise ValueError(
            "semantic diagnostic artifact does not match trace: "
            f"{episode_id}"
        )
    return row


def stage_release(
    sources: list[tuple[Path, Path, Path | None]],
    output_dir: Path,
    *,
    minimum_accepted_cases: int,
    selected_episode_ids: set[str] | None = None,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output_dir}")
    if minimum_accepted_cases < 0:
        raise ValueError("minimum_accepted_cases must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected: Dict[str, Dict[str, Any]] = {}
    rejected_rows: list[Dict[str, Any]] = []
    source_manifests: list[Dict[str, Any]] = []

    for run_dir, eligibility_dir, semantic_dir in sources:
        manifest = _load_json(run_dir / "run_manifest.json")
        if manifest.get("status") not in {"completed", "completed_with_errors"}:
            raise ValueError(f"teacher run is not terminal: {run_dir}")
        source_manifests.append(
            {
                "run_id": manifest.get("run_id"),
                "run_dir": str(run_dir),
                "eligibility_dir": str(eligibility_dir),
                "semantic_dir": str(semantic_dir) if semantic_dir else None,
                "git_commit": manifest.get("git_commit"),
            }
        )
        eligibility_index = _gate_index(eligibility_dir, ".sft_eligibility.json")
        semantic_index = (
            _gate_index(semantic_dir, ".semantic_reward.json")
            if semantic_dir is not None
            else {}
        )
        score_index = _score_index(run_dir)
        for trace_path in sorted((run_dir / "traces").glob("*.json")):
            trace = _load_json(trace_path)
            state = trace.get("state") if isinstance(trace.get("state"), Mapping) else {}
            runtime_case = state.get("runtime_case") if isinstance(state, Mapping) else {}
            case_id = str((runtime_case or {}).get("case_id", "")).strip()
            episode_id = str(trace.get("image_id", "")).strip()
            if not case_id or not episode_id:
                raise ValueError(f"trace has no case/episode ID: {trace_path}")
            if selected_episode_ids is not None and episode_id not in selected_episode_ids:
                continue
            eligibility_row = eligibility_index.get(episode_id)
            if not eligibility_row:
                trace_sha256 = _sha256(trace_path)
                stored = _stage_rejected_trace(
                    output_dir=output_dir,
                    trace_path=trace_path,
                    eligibility_path=None,
                    trace_sha256=trace_sha256,
                )
                rejected_rows.append(
                    {
                        "case_id": case_id,
                        "episode_id": episode_id,
                        "source_run_id": str(
                            manifest.get("run_id", run_dir.name)
                        ),
                        "source_trace_sha256": trace_sha256,
                        "rejection_reasons": [
                            "missing_sft_eligibility_artifact"
                        ],
                        **stored,
                    }
                )
                continue
            trace_sha256 = _sha256(trace_path)
            eligibility = eligibility_row["payload"]
            if not _eligible(trace, trace_sha256, eligibility):
                stored = _stage_rejected_trace(
                    output_dir=output_dir,
                    trace_path=trace_path,
                    eligibility_path=eligibility_row["path"],
                    trace_sha256=trace_sha256,
                )
                rejected_rows.append(
                    {
                        "case_id": case_id,
                        "episode_id": episode_id,
                        "source_run_id": str(
                            manifest.get("run_id", run_dir.name)
                        ),
                        "source_trace_sha256": trace_sha256,
                        "rejection_reasons": _rejection_reasons(
                            trace=trace,
                            trace_sha256=trace_sha256,
                            eligibility=eligibility,
                        ),
                        **stored,
                    }
                )
                continue
            score_row = score_index.get(episode_id) or score_index.get(case_id) or {}
            deterministic_quality = _deterministic_teacher_quality(score_row)
            if not deterministic_quality["hard_gate_pass"]:
                stored = _stage_rejected_trace(
                    output_dir=output_dir,
                    trace_path=trace_path,
                    eligibility_path=eligibility_row["path"],
                    trace_sha256=trace_sha256,
                )
                rejected_rows.append(
                    {
                        "case_id": case_id,
                        "episode_id": episode_id,
                        "source_run_id": str(
                            manifest.get("run_id", run_dir.name)
                        ),
                        "source_trace_sha256": trace_sha256,
                        "rejection_reasons": (
                            deterministic_quality["fatal_reasons"]
                        ),
                        **stored,
                    }
                )
                continue
            semantic_row = _matching_semantic_row(
                semantic_index=semantic_index,
                episode_id=episode_id,
                case_id=case_id,
                trace_sha256=trace_sha256,
            )
            score = float(
                score_row.get("total", 0.0)
                or 0.0
            )
            candidate = {
                "case_id": case_id,
                "episode_id": episode_id,
                "score": score,
                "trace_path": trace_path,
                "trace_sha256": trace_sha256,
                "run_dir": run_dir,
                "run_id": str(manifest.get("run_id", run_dir.name)),
                "eligibility_path": eligibility_row["path"],
                "eligibility": eligibility,
                "semantic_path": (
                    semantic_row["path"] if semantic_row is not None else None
                ),
                "semantic": (
                    semantic_row["payload"] if semantic_row is not None else {}
                ),
                "deterministic_fatal_reasons": deterministic_quality[
                    "fatal_reasons"
                ],
                "deterministic_red_flags": deterministic_quality["red_flags"],
                "decision_policy_version": str(
                    trace.get("decision_policy_version")
                    or state.get("decision_policy_version")
                    or ""
                ),
                "source_metadata": {
                    "source_run_id": str(manifest.get("run_id", run_dir.name)),
                    "runtime_commit": str(manifest.get("git_commit", "")),
                    "release_id": str((manifest.get("benchmark") or {}).get("release_id", "")),
                    "runtime_contract_version": str(
                        (manifest.get("benchmark") or {}).get("runtime_contract_version", "")
                    ),
                    "process_reference_protocol_version": str(
                        manifest.get("process_reference_protocol_version", "")
                    ),
                },
            }
            current = selected.get(case_id)
            if current is None or (
                sft_candidate_rank(
                    candidate["eligibility"],
                    teacher_score=score,
                    episode_id=episode_id,
                )
                > sft_candidate_rank(
                    current["eligibility"],
                    teacher_score=current["score"],
                    episode_id=current["episode_id"],
                )
            ):
                selected[case_id] = candidate

    if len(selected) < minimum_accepted_cases:
        raise ValueError(
            f"accepted too few unique cases: {len(selected)} < {minimum_accepted_cases}"
        )
    policy_versions = {
        str(candidate["decision_policy_version"])
        for candidate in selected.values()
    }
    if len(policy_versions) > 1:
        raise ValueError(
            "accepted release cannot mix decision_policy_version values: "
            + ", ".join(sorted(policy_versions))
        )
    decision_policy_version = next(iter(policy_versions), "")

    selected_rows = []
    selected_by_run: Dict[Path, set[str]] = {}
    selected_cases_by_run: Dict[Path, set[str]] = {}
    trajectory_sft_rows: list[Dict[str, Any]] = []
    action_only_rows: list[Dict[str, Any]] = []
    rl_candidate_rows: list[Dict[str, Any]] = []
    perception_rows: list[Dict[str, Any]] = []
    (output_dir / "eligibility").mkdir(parents=True, exist_ok=True)
    has_semantic_diagnostics = any(
        candidate.get("semantic_path") is not None
        for candidate in selected.values()
    )
    if has_semantic_diagnostics:
        (output_dir / "semantic_rewards").mkdir(parents=True, exist_ok=True)
    for case_id, candidate in sorted(selected.items()):
        trace = _load_json(candidate["trace_path"])
        training_buckets = (
            unified_react_training_buckets(trace)
            if candidate["decision_policy_version"] == "unified-react-v1"
            else ["reasoning_sft"]
        )
        try:
            exported_trajectory = (
                export_trajectory_sft_example(
                    trace,
                    source_metadata=candidate["source_metadata"],
                )
                if "reasoning_sft" in training_buckets
                else export_trajectory_action_only_example(
                    trace,
                    source_metadata=candidate["source_metadata"],
                )
            )
        except ValueError as exc:
            stored = _stage_rejected_trace(
                output_dir=output_dir,
                trace_path=candidate["trace_path"],
                eligibility_path=candidate["eligibility_path"],
                trace_sha256=candidate["trace_sha256"],
            )
            rejected_rows.append(
                {
                    "case_id": case_id,
                    "episode_id": candidate["episode_id"],
                    "source_run_id": candidate["run_id"],
                    "source_trace_sha256": candidate["trace_sha256"],
                    "rejection_reasons": [
                        "trajectory_sft_export_failed",
                        f"trajectory_sft_export_error: {exc}",
                    ],
                    **stored,
                }
            )
            continue

        selected_by_run.setdefault(candidate["run_dir"], set()).add(
            candidate["episode_id"]
        )
        selected_cases_by_run.setdefault(candidate["run_dir"], set()).add(case_id)
        destination = output_dir / "traces" / f"{candidate['episode_id']}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate["trace_path"], destination)
        shutil.copy2(
            candidate["eligibility_path"],
            output_dir / "eligibility" / candidate["eligibility_path"].name,
        )
        if candidate["semantic_path"] is not None:
            shutil.copy2(
                candidate["semantic_path"],
                output_dir / "semantic_rewards" / candidate["semantic_path"].name,
            )
        if "reasoning_sft" in training_buckets:
            trajectory_sft_rows.append(exported_trajectory.model_dump(mode="json"))
        else:
            action_only_rows.append(exported_trajectory.model_dump(mode="json"))
        if "rl_candidate" in training_buckets:
            rl_candidate_rows.append(
                {
                    "case_id": case_id,
                    "episode_id": candidate["episode_id"],
                    "source_trace_sha256": candidate["trace_sha256"],
                    "decision_policy_version": candidate[
                        "decision_policy_version"
                    ],
                    "training_buckets": training_buckets,
                }
            )
        perception_export_error = ""
        try:
            exported_perception = export_perception_example(
                trace,
                source_metadata=candidate["source_metadata"],
            )
        except ValueError as exc:
            exported_perception = None
            perception_export_error = str(exc)
        if exported_perception is not None:
            perception_rows.append(
                exported_perception.model_dump(mode="json")
            )
        runtime_store = _stage_runtime_store(
            output_dir=output_dir,
            trace=trace,
            episode_id=candidate["episode_id"],
            trace_sha256=candidate["trace_sha256"],
            relative_root=Path("runtime-stores"),
            required=candidate["decision_policy_version"] == "unified-react-v1",
        )
        selected_rows.append(
            {
                "case_id": case_id,
                "episode_id": candidate["episode_id"],
                "source_run_id": candidate["run_id"],
                "source_run_dir": str(candidate["run_dir"]),
                "source_trace_sha256": candidate["trace_sha256"],
                "teacher_score": candidate["score"],
                "sft_candidate_rank": list(
                    sft_candidate_rank(
                        candidate["eligibility"],
                        teacher_score=candidate["score"],
                        episode_id=candidate["episode_id"],
                    )
                ),
                "deterministic_hard_gate_pass": True,
                "deterministic_fatal_reasons": candidate[
                    "deterministic_fatal_reasons"
                ],
                "deterministic_red_flags": candidate["deterministic_red_flags"],
                "sft_eligibility_artifact_id": candidate["eligibility"].get("artifact_id"),
                "semantic_reward_artifact_id": candidate["semantic"].get("artifact_id"),
                "decision_policy_version": candidate["decision_policy_version"],
                "training_buckets": training_buckets,
                "trajectory_sft_rows": int("reasoning_sft" in training_buckets),
                "trajectory_token_count_estimate": (
                    exported_trajectory.token_count_estimate
                ),
                "tool_call_count": exported_trajectory.tool_call_count,
                "perception_example_count": int(exported_perception is not None),
                "perception_export_error": perception_export_error,
                **runtime_store,
            }
        )

    _write_jsonl(output_dir / "trajectory_sft.jsonl", trajectory_sft_rows)
    _write_jsonl(output_dir / "action_only.jsonl", action_only_rows)
    _write_jsonl(output_dir / "rl_candidates.jsonl", rl_candidate_rows)
    _write_jsonl(output_dir / "perception_trajectories.jsonl", perception_rows)
    for artifact_name in JSONL_ARTIFACTS:
        rows: list[Dict[str, Any]] = []
        for run_dir, episode_ids in selected_by_run.items():
            case_ids = selected_cases_by_run[run_dir]
            for row in _load_jsonl(run_dir / artifact_name):
                episode_id = str(row.get("episode_id") or row.get("image_id") or "")
                case_id = str(row.get("case_id") or "")
                if episode_id in episode_ids or (not episode_id and case_id in case_ids):
                    rows.append(row)
        _write_jsonl(output_dir / artifact_name, rows)

    first_manifest = _load_json(sources[0][0] / "run_manifest.json")
    first_manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": output_dir.name,
            "status": "completed",
            "accepted_case_count": len(selected_rows),
            "rejected_case_count": len(rejected_rows),
            "decision_policy_version": decision_policy_version,
            "accepted_sources": source_manifests,
            "selected_episodes": "selected_episodes.jsonl",
            "rejected_episodes": "rejected_episodes.jsonl",
            "source_run_kind": "strict-structured-selected",
            "canonical_training_source": {
                "trajectory_sft_export": "canonical_trace",
                "action_only_export": "canonical_trace",
                "rl_candidate_manifest": "rl_candidates.jsonl",
                "perception_export": "canonical_trace",
                "legacy_step_policy_export": "disabled",
            },
        }
    )
    _write_json(output_dir / "run_manifest.json", first_manifest)
    _write_jsonl(output_dir / "selected_episodes.jsonl", selected_rows)
    _write_jsonl(output_dir / "rejected_episodes.jsonl", rejected_rows)
    runtime_store_rows = [
        {
            "case_id": row.get("case_id"),
            "episode_id": row.get("episode_id"),
            "selected": selected,
            "runtime_store_path": row.get("runtime_store_path"),
            "runtime_store_tree_sha256": row.get("runtime_store_tree_sha256"),
            "runtime_store_file_count": row.get("runtime_store_file_count"),
            "runtime_store_byte_count": row.get("runtime_store_byte_count"),
            "runtime_store_materialization": row.get(
                "runtime_store_materialization"
            ),
            "runtime_store_error": row.get("runtime_store_error"),
        }
        for selected, rows in ((True, selected_rows), (False, rejected_rows))
        for row in rows
        if row.get("runtime_store_path") or row.get("runtime_store_error")
    ]
    _write_jsonl(output_dir / "runtime_store_index.jsonl", runtime_store_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "accepted_case_count": len(selected_rows),
        "rejected_case_count": len(rejected_rows),
        "decision_policy_version": decision_policy_version,
        "sources": source_manifests,
        "artifacts": {
            "run_dir": str(output_dir),
            "selected_episodes": "selected_episodes.jsonl",
            "rejected_episodes": "rejected_episodes.jsonl",
            "rejected_dir": "rejected",
            "eligibility_dir": "eligibility",
            "semantic_rewards_dir": (
                "semantic_rewards" if has_semantic_diagnostics else None
            ),
            "trajectory_sft": "trajectory_sft.jsonl",
            "action_only": "action_only.jsonl",
            "rl_candidates": "rl_candidates.jsonl",
            "runtime_store_index": "runtime_store_index.jsonl",
            "runtime_stores_dir": "runtime-stores",
            "rejected_runtime_stores_dir": "rejected/runtime-stores",
        },
        "runtime_store_archive": {
            "selected_count": sum(
                1 for row in selected_rows if row.get("runtime_store_path")
            ),
            "rejected_count": sum(
                1 for row in rejected_rows if row.get("runtime_store_path")
            ),
            "missing_or_invalid_rejected_count": sum(
                1 for row in rejected_rows if row.get("runtime_store_error")
            ),
        },
        "canonical_training_source": {
            "trajectory_sft_export": "canonical_trace",
            "action_only_export": "canonical_trace",
            "rl_candidate_manifest": "rl_candidates.jsonl",
            "perception_export": "canonical_trace",
            "legacy_step_policy_export": "disabled",
        },
    }
    _write_json(output_dir / "accepted_release_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        nargs="+",
        metavar="PATH",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-accepted-cases", type=int, default=1)
    args = parser.parse_args()
    sources = []
    for source in args.source:
        if len(source) not in {2, 3}:
            raise SystemExit("--source requires RUN_DIR ELIGIBILITY_DIR [SEMANTIC_DIR]")
        run_dir = Path(source[0]).expanduser().resolve()
        eligibility_dir = Path(source[1]).expanduser().resolve()
        semantic_dir = Path(source[2]).expanduser().resolve() if len(source) == 3 else None
        sources.append((run_dir, eligibility_dir, semantic_dir))
    result = stage_release(
        sources, args.output_dir.expanduser().resolve(), minimum_accepted_cases=args.minimum_accepted_cases
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
