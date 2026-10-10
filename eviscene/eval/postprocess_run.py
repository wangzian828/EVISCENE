from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from scripts.audit_real_trace import audit_trace
from eviscene.eval.run_artifacts import (
    ROLLOUT_MEMBER_SCHEMA_VERSION,
    file_descriptor as _file_descriptor,
    load_benchmark as _load_benchmark,
    load_json_object as _load_json_object,
    load_jsonl_objects as _load_jsonl,
    private_index as _private_index,
    write_json as _write_json,
    write_jsonl as _write_jsonl,
)
from eviscene.eval.release_adapter import load_runtime_release
from eviscene.orchestrator.source_access import SourceAccessPolicy
from eviscene.eval.scoring_release_adapter import (
    SCORING_PACKAGE_SCHEMA_VERSION,
    ScoringRuntimeRelease,
    adapt_scoring_gold_for_process,
    load_scoring_release,
)
from eviscene.trajectory.exporter import (
    export_trajectory_sft_example,
    trajectory_policy_step_ids,
)
from eviscene.trajectory.perception_exporter import export_perception_example
from eviscene.trajectory.reference_chain import score_reference_chain_trace
from eviscene.trajectory.scoring import score_process_trace


POSTPROCESS_SCHEMA_VERSION = "ifv-run-postprocess-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess a lightweight run_cases output directory. This step "
            "loads evaluator-private gold, scores traces, and exports optional "
            "training artifacts; it does not run the Agent."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help=(
            "Optional runtime_input/cases.jsonl override. Defaults to the "
            "benchmark path recorded in run_manifest.json."
        ),
    )
    parser.add_argument(
        "--training-prohibited",
        action="store_true",
        help="Mark derived trajectory members ineligible for training.",
    )
    return parser.parse_args()


def _release_for_benchmark(benchmark_path: Path) -> Any:
    manifest = _load_json_object(
        benchmark_path.expanduser().resolve().parent.parent / "manifest.json",
        name="release manifest",
    )
    if str(manifest.get("schema_version") or "") == SCORING_PACKAGE_SCHEMA_VERSION:
        return load_scoring_release(benchmark_path)
    return load_runtime_release(benchmark_path)


def _trace_path(run_dir: Path, row: Mapping[str, Any]) -> Path:
    recorded = str(row.get("trace_path") or "").strip()
    if recorded:
        path = (run_dir / recorded).resolve()
    else:
        episode_id = str(row.get("episode_id") or row.get("case_id") or "")
        safe_id = "".join(
            character if character.isalnum() or character in "-_."
            else "_"
            for character in episode_id
        )
        candidates = [
            run_dir / "traces" / f"{episode_id}.json",
            run_dir / "traces" / f"{safe_id}.json",
        ]
        path = next(
            (candidate.resolve() for candidate in candidates if candidate.is_file()),
            candidates[-1].resolve(),
        )
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"trace_path escapes run_dir: {path}") from exc
    return path


def _classification_prediction(row: Mapping[str, Any]) -> Dict[str, Any] | None:
    verdict = str(row.get("verdict") or "").strip()
    if verdict not in {"real", "fake"} or str(row.get("status") or "") != "success":
        return None
    return {
        "case_id": row.get("case_id"),
        "verdict": verdict,
        "episode_id": row.get("episode_id"),
        "prompt_group_id": row.get("prompt_group_id"),
        "rollout_index": row.get("rollout_index"),
    }


def _group_sizes(rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    for row in rows:
        prompt_group_id = str(row.get("prompt_group_id") or "")
        sizes[prompt_group_id] = sizes.get(prompt_group_id, 0) + 1
    return sizes


async def _postprocess_run(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = _load_json_object(manifest_path)
    benchmark_path = (
        args.benchmark.expanduser().resolve()
        if args.benchmark
        else Path(str(manifest.get("benchmark", {}).get("path") or "")).expanduser().resolve()
    )
    release = _release_for_benchmark(benchmark_path)
    run_results = _load_jsonl(run_dir / "run_results.jsonl")
    if not run_results:
        raise ValueError(f"run_results.jsonl has no rows: {run_dir}")

    evaluation_gold_index = _private_index(
        _load_benchmark(release.artifacts.evaluation_gold),
        key="case_id",
        name="evaluation-gold",
    )
    missing_gold = [
        str(row.get("case_id") or "")
        for row in run_results
        if str(row.get("case_id") or "") not in evaluation_gold_index
    ]
    if missing_gold:
        raise RuntimeError(
            "evaluation-gold rows missing run case IDs: "
            + ", ".join(missing_gold[:10])
        )

    source_policy = manifest.get("source_access_policy")
    enforce_source_policy = bool(
        isinstance(source_policy, Mapping) and source_policy.get("active")
    )
    source_policy_object = (
        SourceAccessPolicy.load(release.artifacts.source_access_policy)
        if enforce_source_policy and release.artifacts.source_access_policy is not None
        else None
    )
    gold_descriptor = _file_descriptor(release.artifacts.evaluation_gold)
    process_metadata: Dict[str, Any] = {"evaluation_gold": gold_descriptor}
    if release.artifacts.process_reference_protocol is not None:
        process_metadata["process_reference_protocol"] = _file_descriptor(
            release.artifacts.process_reference_protocol
        )

    process_metrics: List[Dict[str, Any]] = []
    reference_chain_metrics: List[Dict[str, Any]] = []
    trajectory_scores: List[Dict[str, Any]] = []
    trajectory_sft: List[Dict[str, Any]] = []
    perception_trajectories: List[Dict[str, Any]] = []
    rollout_members: List[Dict[str, Any]] = []
    post_rollout_rewards: List[Dict[str, Any]] = []
    episode_predictions: List[Dict[str, Any]] = []
    group_sizes = _group_sizes(run_results)

    for row in run_results:
        case_id = str(row.get("case_id") or "")
        episode_id = str(row.get("episode_id") or case_id)
        prompt_group_id = str(row.get("prompt_group_id") or case_id)
        trace_path = _trace_path(run_dir, row)
        member = {
            "schema_version": ROLLOUT_MEMBER_SCHEMA_VERSION,
            "prompt_group_id": prompt_group_id,
            "case_id": case_id,
            "episode_id": episode_id,
            "rollout_index": int(row.get("rollout_index", 0) or 0),
            "group_size": group_sizes.get(prompt_group_id, 1),
            "sampling_seed": row.get("sampling_seed"),
            "policy_revision": manifest.get("git_commit"),
            "trace_path": (
                trace_path.relative_to(run_dir).as_posix()
                if trace_path.exists()
                else None
            ),
            "training_prohibited": (
                bool(getattr(args, "training_prohibited", False))
                or release.release_stage == "development_subset"
            ),
            "training_eligible": False,
            "sft_structured_eligibility_required": isinstance(
                release, ScoringRuntimeRelease
            ),
        }
        rollout_members.append(member)

        prediction = _classification_prediction(row)
        if prediction is not None:
            episode_predictions.append(prediction)

        if not trace_path.exists():
            process_metrics.append(
                {
                    "schema_version": "ifv-process-metrics-v2",
                    "case_id": case_id,
                    "episode_id": episode_id,
                    "engineering_error": True,
                    "runtime_verdict": row.get("verdict"),
                    "result_correct": False,
                    "first_error": {"error": "canonical trace is missing"},
                }
            )
            reference_chain_metrics.append(
                {
                    "schema_version": "ifv-reference-chain-metrics-v2",
                    "case_id": case_id,
                    "episode_id": episode_id,
                    "engineering_error": True,
                    "error": "canonical trace is missing",
                }
            )
            trajectory_scores.append(
                {
                    "schema_version": "ifv-trajectory-score-v2",
                    "case_id": case_id,
                    "episode_id": episode_id,
                    "components": {},
                    "total": 0.0,
                }
            )
            post_rollout_rewards.append(
                {
                    "schema_version": "ifv-post-rollout-deterministic-v1",
                    "prompt_group_id": prompt_group_id,
                    "case_id": case_id,
                    "episode_id": episode_id,
                    "classification_correct": False,
                    "fatal_engineering_error": True,
                    "strict_trace_audit_pass": False,
                    "training_prohibited": member["training_prohibited"],
                    "step_ids": [],
                    "process_components": {},
                }
            )
            continue

        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        trace_state = trace.get("state")
        if isinstance(trace_state, Mapping) and isinstance(
            trace_state.get("perception"), Mapping
        ):
            perception_trajectories.append(
                export_perception_example(
                    trace,
                    source_metadata={
                        "source_run_id": manifest.get("run_id"),
                        "runtime_commit": manifest.get("git_commit"),
                        "release_id": release.release_id,
                        "runtime_contract_version": release.runtime_contract_version,
                    },
                ).model_dump(mode="json")
            )

        private_gold = evaluation_gold_index[case_id]
        process_gold = (
            adapt_scoring_gold_for_process(private_gold)
            if isinstance(release, ScoringRuntimeRelease)
            else private_gold
        )
        metrics, teacher_score = score_process_trace(
            trace,
            process_gold,
            score_metadata=process_metadata,
        )
        metrics["episode_id"] = episode_id
        metrics["prompt_group_id"] = prompt_group_id
        teacher_score["episode_id"] = episode_id
        teacher_score["prompt_group_id"] = prompt_group_id

        audit_report = audit_trace(
            trace_path,
            enforce_source_access_policy=enforce_source_policy,
            source_access_policy=source_policy_object,
        )
        strict_failures = audit_report.failures(strict_scheduler=True)
        hard_failures = audit_report.failures(strict_scheduler=False)
        strict_trace_audit_pass = not strict_failures
        hard_trace_audit_pass = not hard_failures
        metrics["strict_trace_audit_pass"] = strict_trace_audit_pass
        metrics["hard_trace_audit_pass"] = hard_trace_audit_pass
        metrics["strict_trace_audit_failures"] = [
            asdict(item) for item in strict_failures
        ]
        process_metrics.append(metrics)

        if isinstance(release, ScoringRuntimeRelease):
            reference_score = {
                "schema_version": "ifv-scoring-reference-chain-v1",
                "case_id": case_id,
                "applicable": False,
                "reason": (
                    "ifv-scoring-gold-v1 uses a structured relation target gate "
                    "instead of the v0.3 reference protocol"
                ),
            }
        else:
            reference_score = await score_reference_chain_trace(
                trace,
                private_gold,
                score_metadata=process_metadata,
            )
        reference_score["episode_id"] = episode_id
        reference_score["prompt_group_id"] = prompt_group_id
        reference_chain_metrics.append(reference_score)
        trajectory_scores.append(teacher_score)

        try:
            exported_trajectory = export_trajectory_sft_example(
                trace,
                source_metadata={
                    "source_run_id": manifest.get("run_id"),
                    "runtime_commit": manifest.get("git_commit"),
                    "release_id": release.release_id,
                    "runtime_contract_version": release.runtime_contract_version,
                    "process_reference_protocol_version": release.manifest.get(
                        "process_reference_protocol_version",
                        "",
                    ),
                },
            )
        except ValueError as exc:
            metrics["training_eligible"] = False
            reasons = list(metrics.get("training_exclusion_reasons", []) or [])
            reasons.append(f"trajectory_sft_export_rejected: {exc}")
            metrics["training_exclusion_reasons"] = list(dict.fromkeys(reasons))
            teacher_score["training_eligible"] = False
            teacher_score["training_exclusion_reasons"] = list(
                metrics["training_exclusion_reasons"]
            )
            exported_trajectory = None
        if exported_trajectory is not None:
            trajectory_sft.append(
                exported_trajectory.model_dump(mode="json")
            )
        policy_step_ids = trajectory_policy_step_ids(
            trace,
            episode_id=episode_id,
        )
        member["sft_training_eligible"] = bool(
            strict_trace_audit_pass
            and bool(metrics.get("training_eligible", False))
            and exported_trajectory is not None
            and not member["training_prohibited"]
            and not isinstance(release, ScoringRuntimeRelease)
        )
        member["rl_reward_eligible"] = bool(
            not metrics.get("engineering_error", False)
            and bool(policy_step_ids)
            and not member["training_prohibited"]
        )
        member["training_eligible"] = member["sft_training_eligible"]
        post_rollout_rewards.append(
            {
                "schema_version": "ifv-post-rollout-deterministic-v1",
                "prompt_group_id": prompt_group_id,
                "case_id": case_id,
                "episode_id": episode_id,
                "classification_correct": bool(
                    metrics.get("result_correct", False)
                ),
                "fatal_engineering_error": bool(
                    metrics.get("engineering_error", False)
                ),
                "strict_trace_audit_pass": bool(strict_trace_audit_pass),
                "hard_trace_audit_pass": bool(hard_trace_audit_pass),
                "strict_trace_audit_failure_codes": [
                    str(item.get("code", ""))
                    for item in metrics.get("strict_trace_audit_failures", [])
                    if str(item.get("code", ""))
                ],
                "training_prohibited": member["training_prohibited"],
                "rl_reward_eligible": member["rl_reward_eligible"],
                "step_ids": policy_step_ids,
                "process_components": dict(
                    teacher_score.get("components") or {}
                ),
            }
        )

    _write_jsonl(run_dir / "episode_predictions.jsonl", episode_predictions)
    _write_jsonl(run_dir / "process_metrics.jsonl", process_metrics)
    _write_jsonl(run_dir / "reference_chain_metrics.jsonl", reference_chain_metrics)
    _write_jsonl(run_dir / "trajectory_scores.jsonl", trajectory_scores)
    _write_jsonl(run_dir / "trajectory_sft.jsonl", trajectory_sft)
    _write_jsonl(run_dir / "perception_trajectories.jsonl", perception_trajectories)
    _write_jsonl(run_dir / "rollout_groups.jsonl", rollout_members)
    _write_jsonl(run_dir / "post_rollout_rewards.jsonl", post_rollout_rewards)

    summary = {
        "schema_version": POSTPROCESS_SCHEMA_VERSION,
        "run_id": manifest.get("run_id"),
        "episode_count": len(run_results),
        "process_metric_rows": len(process_metrics),
        "reference_chain_metric_rows": len(reference_chain_metrics),
        "trajectory_score_rows": len(trajectory_scores),
        "trajectory_sft_rows": len(trajectory_sft),
        "perception_trajectory_rows": len(perception_trajectories),
        "rollout_member_rows": len(rollout_members),
        "post_rollout_reward_rows": len(post_rollout_rewards),
    }
    _write_json(run_dir / "postprocess_summary.json", summary)

    artifacts = dict(manifest.get("artifacts") or {})
    artifacts.update(
        {
            "episode_predictions": "episode_predictions.jsonl",
            "process_metrics": "process_metrics.jsonl",
            "reference_chain_metrics": "reference_chain_metrics.jsonl",
            "trajectory_scores": "trajectory_scores.jsonl",
            "trajectory_sft": "trajectory_sft.jsonl",
            "perception_trajectories": "perception_trajectories.jsonl",
            "rollout_groups": "rollout_groups.jsonl",
            "post_rollout_rewards": "post_rollout_rewards.jsonl",
            "postprocess_summary": "postprocess_summary.json",
        }
    )
    manifest["artifacts"] = artifacts
    manifest["postprocess"] = {
        "schema_version": POSTPROCESS_SCHEMA_VERSION,
        "status": "completed",
        "evaluation_gold": _file_descriptor(release.artifacts.evaluation_gold),
        "training_prohibited": bool(getattr(args, "training_prohibited", False)),
    }
    _write_json(manifest_path, manifest)
    return summary


def main() -> None:
    summary = asyncio.run(_postprocess_run(_parse_args()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
