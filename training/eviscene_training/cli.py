from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_derived_dataset
from .checkpoints import (
    audit_full_parameter_checkpoint,
    build_checkpoint_manifest,
    build_serving_profile,
)
from .checkpoint_io import (
    checkpoint_io_profile,
    checkpoint_storage_preflight,
    write_json as write_checkpoint_json,
)
from .checkpoint_selection import validate_checkpoints
from .encode_cache import aggregate_encode_cache_metrics
from .io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from .manifests import (
    cached_dataset_manifest,
    write_cached_dataset_verification,
    write_environment_manifest,
)
from .long_context import build_long_context_plan, validate_128k_stage
from .perception import (
    convert_accepted_perception_dataset,
    convert_perception_runs,
)
from .policy import convert_policy_dataset, repair_derived_policy_dataset
from .profile import write_training_profile
from .watchdog import build_sft_watchdog_snapshot
from .psd import (
    audit_psd_hints,
    build_psd_target_package,
    materialize_psd_topk_cache,
)
from .psd_candidates import build_psd_candidate_package
from .psd_datums import build_sparse_topk_package
from .psd_repairs import assemble_psd_repair_package
from .psd_repair_finalize import finalize_psd_repair_run
from .psd_preflight import verify_psd_training_input
from .psd_round import complete_psd_round, verify_psd_round_rollout
from .psd_topk import collect_psd_topk_cache
from .rewards import (
    build_and_write_ledger,
    build_ledgers_from_run_artifacts,
    build_standard_grpo_groups,
    export_framework_reward,
    load_reward_profile,
    validate_reward_ledger,
    validate_semantic_reward_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eviscene-train",
        description="SFT/PSD preparation utilities (GPU training is separate).",
        epilog="Use COMMAND --help for arguments. Additional compatibility commands remain available.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND", title="Common commands")

    policy = subparsers.add_parser("convert-policy", help="Convert reviewed policy trajectories")
    policy.add_argument("--input", type=Path, required=True)
    policy.add_argument("--output", type=Path, required=True)

    policy_repair = subparsers.add_parser("repair-policy-contract")
    policy_repair.add_argument("--input", type=Path, required=True)
    policy_repair.add_argument("--output", type=Path, required=True)

    perception = subparsers.add_parser("convert-perception")
    perception.add_argument("--run-dir", action="append", type=Path, required=True)
    perception.add_argument("--split-map", type=Path, required=True)
    perception.add_argument("--output", type=Path, required=True)

    accepted_perception = subparsers.add_parser("convert-accepted-perception")
    accepted_perception.add_argument("--input", type=Path, required=True)
    accepted_perception.add_argument("--output", type=Path, required=True)

    audit = subparsers.add_parser("audit", help="Audit prepared training examples")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    audit.add_argument("--strict", action="store_true")

    environment = subparsers.add_parser("environment-manifest")
    environment.add_argument("--output", type=Path, required=True)
    environment.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )

    checkpoint = subparsers.add_parser("checkpoint-manifest")
    checkpoint.add_argument("--checkpoint-dir", type=Path, required=True)
    checkpoint.add_argument("--dataset-manifest", type=Path, required=True)
    checkpoint.add_argument("--output", type=Path, required=True)
    checkpoint.add_argument("--base-model-id", required=True)
    checkpoint.add_argument("--model-revision", default="")
    checkpoint.add_argument("--processor-revision", default="")
    checkpoint.add_argument("--state-checkpoint-dir", type=Path)
    checkpoint.add_argument(
        "--method",
        choices=("lora", "qlora", "full", "grpo-lora", "grpo-full"),
        required=True,
    )

    full_audit = subparsers.add_parser("audit-full-checkpoint")
    full_audit.add_argument("--checkpoint-dir", type=Path, required=True)
    full_audit.add_argument("--base-model-dir", type=Path, required=True)
    full_audit.add_argument("--state-checkpoint-dir", type=Path)
    full_audit.add_argument("--output", type=Path, required=True)

    cached_manifest = subparsers.add_parser("cached-dataset-manifest")
    cached_manifest.add_argument("--cache-dir", type=Path, required=True)
    cached_manifest.add_argument("--output", type=Path, required=True)

    cached_verify = subparsers.add_parser("verify-cached-dataset")
    cached_verify.add_argument(
        "--train-dir",
        action="append",
        type=Path,
        required=True,
    )
    cached_verify.add_argument(
        "--validation-dir",
        action="append",
        type=Path,
        required=True,
    )
    cached_verify.add_argument("--manifest", action="append", type=Path)
    cached_verify.add_argument("--output", type=Path, required=True)

    serving = subparsers.add_parser("serving-profile")
    serving.add_argument("--output", type=Path, required=True)
    serving.add_argument("--profile-id", required=True)
    serving.add_argument("--model-path", required=True)
    serving.add_argument("--engine", choices=("vllm", "lmdeploy"), required=True)
    serving.add_argument("--port", type=int, required=True)
    serving.add_argument("--tensor-parallel-size", type=int, required=True)
    serving.add_argument("--dtype", default="bfloat16")
    serving.add_argument("--context-length", type=int, required=True)
    serving.add_argument("--tool-call-parser", required=True)
    serving.add_argument("--reasoning-parser", required=True)
    serving.add_argument(
        "--thinking-enabled",
        choices=("true", "false"),
        required=True,
    )
    serving.add_argument("--checkpoint-manifest", type=Path)
    serving.add_argument("--engine-model-path")
    serving.add_argument("--adapter-path")

    semantic_audit = subparsers.add_parser("audit-semantic-reward")
    semantic_audit.add_argument("--input", type=Path, required=True)
    semantic_audit.add_argument("--output", type=Path)
    semantic_audit.add_argument("--strict", action="store_true")

    reward_ledger = subparsers.add_parser("reward-ledger")
    reward_ledger.add_argument("--deterministic", type=Path, required=True)
    reward_ledger.add_argument("--semantic-artifact", type=Path)
    reward_ledger.add_argument("--profile", type=Path)
    reward_ledger.add_argument("--output", type=Path, required=True)

    ledger_audit = subparsers.add_parser("audit-reward-ledger")
    ledger_audit.add_argument("--input", type=Path, required=True)
    ledger_audit.add_argument("--output", type=Path)
    ledger_audit.add_argument("--strict", action="store_true")

    reward_export = subparsers.add_parser("export-reward")
    reward_export.add_argument("--ledger", type=Path, required=True)
    reward_export.add_argument(
        "--framework",
        choices=("rllm", "verl"),
        required=True,
    )
    reward_export.add_argument("--output", type=Path, required=True)

    grpo_groups = subparsers.add_parser("build-grpo-groups")
    grpo_groups.add_argument("--ledgers", type=Path, required=True)
    grpo_groups.add_argument("--rollout-members", type=Path, required=True)
    grpo_groups.add_argument("--output", type=Path, required=True)
    grpo_groups.add_argument("--minimum-valid-members", type=int, default=2)

    psd_audit = subparsers.add_parser("audit-psd-hints", help="Check privileged hints for answer leakage")
    psd_audit.add_argument("--input", type=Path, required=True)
    psd_audit.add_argument("--output", type=Path, required=True)

    psd_targets = subparsers.add_parser("build-psd-targets")
    psd_targets.add_argument("--repairs", type=Path, required=True)
    psd_targets.add_argument("--preservation", type=Path)
    psd_targets.add_argument("--output-dir", type=Path, required=True)
    psd_targets.add_argument("--topk", type=int, default=20)
    psd_targets.add_argument("--allow-local-only", action="store_true")

    psd_candidates = subparsers.add_parser("build-psd-candidates")
    psd_candidates.add_argument("--run-dir", type=Path, required=True)
    psd_candidates.add_argument("--train-cases", type=Path, required=True)
    psd_candidates.add_argument("--rollout-gate", type=Path, required=True)
    psd_candidates.add_argument("--output-dir", type=Path, required=True)

    psd_rollout_gate = subparsers.add_parser("verify-psd-round-rollout")
    psd_rollout_gate.add_argument("--round-index", type=int, required=True)
    psd_rollout_gate.add_argument("--run-dir", type=Path, required=True)
    psd_rollout_gate.add_argument("--train-cases", type=Path, required=True)
    psd_rollout_gate.add_argument("--serving-profile", type=Path, required=True)
    psd_rollout_gate.add_argument(
        "--round-start-checkpoint-manifest", type=Path, required=True
    )
    psd_rollout_gate.add_argument("--previous-round-completion", type=Path)
    psd_rollout_gate.add_argument("--output", type=Path, required=True)

    psd_round_completion = subparsers.add_parser("complete-psd-round")
    psd_round_completion.add_argument("--rollout-gate", type=Path, required=True)
    psd_round_completion.add_argument(
        "--training-profile", type=Path, required=True
    )
    psd_round_completion.add_argument(
        "--output-checkpoint-manifest", type=Path, required=True
    )
    psd_round_completion.add_argument("--output", type=Path, required=True)
    psd_round_completion.add_argument("--initialization-gate", type=Path, required=True)

    psd_topk = subparsers.add_parser("materialize-psd-topk")
    psd_topk.add_argument("--targets", type=Path, required=True)
    psd_topk.add_argument("--cache", type=Path, required=True)
    psd_topk.add_argument("--output-dir", type=Path, required=True)
    psd_topk.add_argument("--topk", type=int, default=20)

    psd_topk_collect = subparsers.add_parser("collect-psd-topk")
    psd_topk_collect.add_argument("--targets", type=Path, required=True)
    psd_topk_collect.add_argument(
        "--serving-profile", type=Path, required=True
    )
    psd_topk_collect.add_argument(
        "--round-start-checkpoint-manifest", type=Path, required=True
    )
    psd_topk_collect.add_argument("--output-dir", type=Path, required=True)
    psd_topk_collect.add_argument("--topk", type=int, default=20)
    psd_topk_collect.add_argument("--retries", type=int, default=3)
    psd_topk_collect.add_argument("--timeout", type=float, default=1800.0)
    psd_topk_collect.add_argument("--limit", type=int)
    psd_topk_collect.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    psd_topk_collect.add_argument("--device", default="cuda:0")

    psd_repairs = subparsers.add_parser("assemble-psd-repairs")
    psd_repairs.add_argument("--repair-candidates", type=Path, required=True)
    psd_repairs.add_argument("--repair-attempts", type=Path, required=True)
    psd_repairs.add_argument(
        "--preservation-candidates",
        type=Path,
        required=True,
    )
    psd_repairs.add_argument("--output-dir", type=Path, required=True)

    psd_finalize = subparsers.add_parser("finalize-psd-repair-run")
    psd_finalize.add_argument("--run-dir", type=Path, required=True)
    psd_finalize.add_argument("--source-trace", type=Path, required=True)
    psd_finalize.add_argument("--gold", type=Path, required=True)
    psd_finalize.add_argument(
        "--verification-bundle", type=Path, required=True
    )
    psd_finalize.add_argument("--require-all", action="store_true")

    psd_datums = subparsers.add_parser("build-psd-datums", help="Build sparse teacher-distribution targets")
    psd_datums.add_argument("--targets", type=Path, required=True)
    psd_datums.add_argument("--output-dir", type=Path, required=True)
    psd_datums.add_argument("--topk", type=int, default=20)
    psd_datums.add_argument(
        "--max-sequence-length",
        type=int,
        default=131_072,
    )
    psd_datums.add_argument(
        "--allow-single-kind",
        action="store_false",
        dest="require_both_kinds",
        help="Allow a diagnostic/ablation package without both repair and preservation.",
    )
    psd_datums.set_defaults(require_both_kinds=True)

    psd_input = subparsers.add_parser("verify-psd-datums", help="Validate PSD training inputs")
    psd_input.add_argument("--datums", type=Path, required=True)
    psd_input.add_argument("--manifest", type=Path, required=True)
    psd_input.add_argument("--expected-topk", type=int, default=20)
    psd_input.add_argument("--max-context", type=int, required=True)
    psd_input.add_argument("--output", type=Path, required=True)

    psd_locate = subparsers.add_parser("locate-psd-failure")
    psd_locate.add_argument("--trace", type=Path, required=True)
    psd_locate.add_argument("--audit", type=Path)
    psd_locate.add_argument("--semantic-verification", type=Path)
    psd_locate.add_argument("--output", type=Path, required=True)

    psd_verify = subparsers.add_parser("verify-psd-episode")
    psd_verify.add_argument("--source-trace", type=Path, required=True)
    psd_verify.add_argument("--hinted-trace", type=Path, required=True)
    psd_verify.add_argument("--gold", type=Path, required=True)
    psd_verify.add_argument("--local-verification", type=Path, required=True)
    psd_verify.add_argument("--repair-step-id", required=True)
    psd_verify.add_argument("--hint-sha256", required=True)
    psd_verify.add_argument("--teacher-prompt-sha256", required=True)
    psd_verify.add_argument("--teacher-completion-sha256", required=True)
    psd_verify.add_argument("--output", type=Path, required=True)
    psd_verify.add_argument("--downstream-patch-count", type=int, default=0)

    run_rewards = subparsers.add_parser("build-run-rewards")
    run_rewards.add_argument("--deterministic", type=Path, required=True)
    run_rewards.add_argument("--rollout-members", type=Path, required=True)
    run_rewards.add_argument("--semantic-artifacts", type=Path)
    run_rewards.add_argument("--profile", type=Path)
    run_rewards.add_argument("--ledger-output", type=Path, required=True)
    run_rewards.add_argument("--group-output", type=Path, required=True)
    run_rewards.add_argument("--minimum-valid-members", type=int, default=2)

    training_profile = subparsers.add_parser("training-profile")
    training_profile.add_argument("--train-log", type=Path, required=True)
    training_profile.add_argument("--output", type=Path, required=True)
    training_profile.add_argument("--steady-window", type=int, default=5)
    training_profile.add_argument("--experiment-id", default="")
    training_profile.add_argument("--profile-id", default="")
    training_profile.add_argument("--resource-summary", type=Path)
    training_profile.add_argument("--cache-verification", type=Path)
    training_profile.add_argument("--dataset-verification", type=Path)
    training_profile.add_argument("--environment-preflight", type=Path)
    training_profile.add_argument("--encode-cache-report", type=Path)
    training_profile.add_argument("--scheduler-audit", type=Path)
    training_profile.add_argument("--checkpoint-preflight", type=Path)
    training_profile.add_argument("--checkpoint-io-profile", type=Path)
    training_profile.add_argument("--train-exit-code", type=int)

    checkpoint_validation = subparsers.add_parser("validate-checkpoints")
    checkpoint_validation.add_argument("--train-log", type=Path, required=True)
    checkpoint_validation.add_argument(
        "--checkpoint-root", type=Path, required=True
    )
    checkpoint_validation.add_argument("--output", type=Path, required=True)
    checkpoint_validation.add_argument("--behavior-metrics", type=Path)
    checkpoint_validation.add_argument("--train-rows", type=int)
    checkpoint_validation.add_argument("--global-batch", type=int)
    checkpoint_validation.add_argument(
        "--selection",
        choices=("auto", "eval_loss", "behavior"),
        default="auto",
    )

    encode_cache_report = subparsers.add_parser("encode-cache-report")
    encode_cache_report.add_argument("--metrics-dir", type=Path, required=True)
    encode_cache_report.add_argument("--output", type=Path, required=True)

    checkpoint_preflight = subparsers.add_parser("checkpoint-storage-preflight")
    checkpoint_preflight.add_argument("--output-dir", type=Path, required=True)
    checkpoint_preflight.add_argument(
        "--estimated-checkpoint-bytes",
        type=int,
        required=True,
    )
    checkpoint_preflight.add_argument(
        "--reserve-multiplier",
        type=float,
        default=1.25,
    )
    checkpoint_preflight.add_argument("--output", type=Path, required=True)

    checkpoint_profile = subparsers.add_parser("checkpoint-io-profile")
    checkpoint_profile.add_argument("--checkpoint", type=Path, required=True)
    checkpoint_profile.add_argument("--output", type=Path, required=True)

    monitor_sft = subparsers.add_parser("monitor-sft")
    monitor_sft.add_argument("--train-log", type=Path, required=True)
    monitor_sft.add_argument("--checkpoint-root", type=Path)
    monitor_sft.add_argument("--resource-samples", type=Path)
    monitor_sft.add_argument("--behavior-metrics", type=Path)
    monitor_sft.add_argument("--output", type=Path)
    monitor_sft.add_argument("--stale-seconds", type=float, default=900.0)
    monitor_sft.add_argument("--trend-window", type=int, default=5)
    monitor_sft.add_argument("--strict", action="store_true")

    long_plan = subparsers.add_parser("long-context-plan")
    long_plan.add_argument("--processor-report", type=Path, required=True)
    long_plan.add_argument("--train-jsonl", type=Path, required=True)
    long_plan.add_argument("--sft-profile", type=Path, action="append", required=True)
    long_plan.add_argument("--empirical-profile", type=Path, action="append")
    long_plan.add_argument("--epochs", type=float, default=1.0)
    long_plan.add_argument("--parameter-count", type=int, default=9_000_000_000)
    long_plan.add_argument("--output", type=Path, required=True)

    stage_gate = subparsers.add_parser("validate-128k-stage")
    stage_gate.add_argument("--stage", choices=("memory-probe", "canary", "resume"), required=True)
    stage_gate.add_argument("--training-profile", type=Path, required=True)
    stage_gate.add_argument("--sft-profile", type=Path, required=True)
    stage_gate.add_argument("--prior-gate", type=Path)
    stage_gate.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "convert-policy":
        result = convert_policy_dataset(args.input, args.output)
    elif args.command == "repair-policy-contract":
        result = repair_derived_policy_dataset(args.input, args.output)
    elif args.command == "convert-perception":
        result = convert_perception_runs(
            args.run_dir,
            args.output,
            split_map_path=args.split_map,
        )
    elif args.command == "convert-accepted-perception":
        result = convert_accepted_perception_dataset(args.input, args.output)
    elif args.command == "audit":
        result = audit_derived_dataset(args.input)
        if args.output:
            write_json(args.output, result)
        if args.strict and not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "environment-manifest":
        result = write_environment_manifest(args.output, args.repo_root)
    elif args.command == "checkpoint-manifest":
        result = build_checkpoint_manifest(
            checkpoint_dir=args.checkpoint_dir,
            dataset_manifest_path=args.dataset_manifest,
            output_path=args.output,
            base_model_id=args.base_model_id,
            model_revision=args.model_revision,
            processor_revision=args.processor_revision,
            method=args.method,
            state_checkpoint_dir=args.state_checkpoint_dir,
        )
    elif args.command == "audit-full-checkpoint":
        result = audit_full_parameter_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            base_model_dir=args.base_model_dir,
            state_checkpoint_dir=args.state_checkpoint_dir,
            output_path=args.output,
        )
        if not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "cached-dataset-manifest":
        result = cached_dataset_manifest(args.cache_dir)
        write_json(args.output, result)
    elif args.command == "verify-cached-dataset":
        result = write_cached_dataset_verification(
            train_dirs=args.train_dir,
            validation_dirs=args.validation_dir,
            manifest_paths=args.manifest,
            output=args.output,
        )
        if not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "serving-profile":
        result = build_serving_profile(
            output_path=args.output,
            profile_id=args.profile_id,
            model_path=args.model_path,
            engine=args.engine,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            context_length=args.context_length,
            tool_call_parser=args.tool_call_parser,
            reasoning_parser=args.reasoning_parser,
            thinking_enabled=args.thinking_enabled == "true",
            checkpoint_manifest_path=args.checkpoint_manifest,
            engine_model_path=args.engine_model_path,
            adapter_path=args.adapter_path,
        )
    elif args.command == "audit-semantic-reward":
        result = validate_semantic_reward_artifact(load_json(args.input))
        if args.output:
            write_json(args.output, result)
        if args.strict and not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "reward-ledger":
        result = build_and_write_ledger(
            deterministic_path=args.deterministic,
            profile_path=args.profile,
            semantic_artifact_path=args.semantic_artifact,
            output_path=args.output,
        )
    elif args.command == "audit-reward-ledger":
        result = validate_reward_ledger(load_json(args.input))
        if args.output:
            write_json(args.output, result)
        if args.strict and not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "export-reward":
        result = export_framework_reward(
            load_json(args.ledger),
            framework=args.framework,
        )
        write_json(args.output, result)
    elif args.command == "build-grpo-groups":
        result = build_standard_grpo_groups(
            load_jsonl(args.ledgers),
            load_jsonl(args.rollout_members),
            minimum_valid_members=args.minimum_valid_members,
        )
        write_jsonl(args.output, result)
    elif args.command == "audit-psd-hints":
        result = audit_psd_hints(args.input, args.output)
    elif args.command == "build-psd-targets":
        result = build_psd_target_package(
            repairs_path=args.repairs,
            preservation_path=args.preservation,
            output_dir=args.output_dir,
            topk=args.topk,
            allow_local_only=args.allow_local_only,
        )
    elif args.command == "build-psd-candidates":
        result = build_psd_candidate_package(
            run_dir=args.run_dir,
            train_cases_path=args.train_cases,
            rollout_gate_path=args.rollout_gate,
            output_dir=args.output_dir,
        )
    elif args.command == "verify-psd-round-rollout":
        result = verify_psd_round_rollout(
            round_index=args.round_index,
            run_dir=args.run_dir,
            train_cases_path=args.train_cases,
            serving_profile_path=args.serving_profile,
            round_start_checkpoint_manifest_path=(
                args.round_start_checkpoint_manifest
            ),
            previous_round_completion_path=args.previous_round_completion,
            output=args.output,
        )
        if not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "complete-psd-round":
        result = complete_psd_round(
            rollout_gate_path=args.rollout_gate,
            training_profile_path=args.training_profile,
            output_checkpoint_manifest_path=args.output_checkpoint_manifest,
            output=args.output,
            initialization_gate_path=args.initialization_gate,
        )
        if not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "materialize-psd-topk":
        result = materialize_psd_topk_cache(
            targets_path=args.targets,
            cache_path=args.cache,
            output_dir=args.output_dir,
            topk=args.topk,
        )
    elif args.command == "collect-psd-topk":
        result = collect_psd_topk_cache(
            backend=args.backend,
            device=args.device,
            targets_path=args.targets,
            serving_profile_path=args.serving_profile,
            checkpoint_manifest_path=args.round_start_checkpoint_manifest,
            output_dir=args.output_dir,
            topk=args.topk,
            retries=args.retries,
            timeout=args.timeout,
            limit=args.limit,
        )
        if result["status"] == "blocked_collection_errors":
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "assemble-psd-repairs":
        result = assemble_psd_repair_package(
            repair_candidates_path=args.repair_candidates,
            repair_attempts_path=args.repair_attempts,
            preservation_candidates_path=args.preservation_candidates,
            output_dir=args.output_dir,
        )
    elif args.command == "finalize-psd-repair-run":
        result = finalize_psd_repair_run(
            run_dir=args.run_dir,
            source_trace_path=args.source_trace,
            gold_path=args.gold,
            verification_bundle_path=args.verification_bundle,
            require_all=args.require_all,
        )
        if result["status"] == "blocked_unresolved_verifiers":
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "build-psd-datums":
        result = build_sparse_topk_package(
            targets_path=args.targets,
            output_dir=args.output_dir,
            topk=args.topk,
            max_sequence_length=args.max_sequence_length,
            require_both_kinds=args.require_both_kinds,
        )
    elif args.command == "verify-psd-datums":
        result = verify_psd_training_input(
            datums_path=args.datums,
            manifest_path=args.manifest,
            expected_topk=args.expected_topk,
            max_context=args.max_context,
        )
        write_json(args.output, result)
        if not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "locate-psd-failure":
        from .psd_repair import locate_failure_site

        trace = load_json(args.trace)
        audit = load_json(args.audit) if args.audit else {}
        semantic_verification = (
            load_json(args.semantic_verification)
            if args.semantic_verification
            else None
        )
        site = locate_failure_site(trace, audit, semantic_verification)
        result = {
            "found": site is not None,
            "failure_site": site.public_record() if site is not None else None,
        }
        write_json(args.output, result)
    elif args.command == "verify-psd-episode":
        from .psd_repair_verifier import verify_causal_episode

        source_trace = load_json(args.source_trace)
        hinted_trace = load_json(args.hinted_trace)
        gold = load_json(args.gold)
        local_verification = load_json(args.local_verification)
        verification = verify_causal_episode(
            hinted_trace,
            source_trace=source_trace,
            source_trace_sha256=sha256_file(args.source_trace),
            gold=gold,
            local_verification=local_verification,
            repair_step_id=args.repair_step_id,
            hint_sha256=args.hint_sha256,
            teacher_prompt_sha256=args.teacher_prompt_sha256,
            teacher_completion_sha256=args.teacher_completion_sha256,
            downstream_patch_count=args.downstream_patch_count,
        )
        result = {
            "accepted_for_primary_psd": verification.accepted_for_primary_psd,
            "source_rollout_failed": verification.source_rollout_failed,
            "hinted_local_pass": verification.hinted_local_pass,
            "hinted_episode_pass": verification.hinted_episode_pass,
            "hinted_strict_trace_audit_pass": (
                verification.hinted_strict_trace_audit_pass
            ),
            "hinted_recorded_verdict": verification.hinted_recorded_verdict,
            "expected_verdict": verification.expected_verdict,
            "repair_tier": verification.repair_tier,
            "reasons": list(verification.reasons),
        }
        write_json(args.output, result)
        if not verification.accepted_for_primary_psd:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "build-run-rewards":
        semantic_artifacts = []
        if args.semantic_artifacts:
            semantic_paths = sorted(args.semantic_artifacts.glob("*.semantic_reward.json"))
            semantic_artifacts = [load_json(path) for path in semantic_paths]
        profile = load_reward_profile(args.profile)
        ledgers = build_ledgers_from_run_artifacts(
            deterministic_rows=load_jsonl(args.deterministic),
            semantic_artifacts=semantic_artifacts,
            profile=profile,
        )
        groups = build_standard_grpo_groups(
            ledgers,
            load_jsonl(args.rollout_members),
            minimum_valid_members=args.minimum_valid_members,
        )
        write_jsonl(args.ledger_output, ledgers)
        write_jsonl(args.group_output, groups)
        result = {
            "ledger_count": len(ledgers),
            "group_count": len(groups),
            "trainable_group_count": sum(
                1 for group in groups if group.get("trainable") is True
            ),
        }
    elif args.command == "training-profile":
        result = write_training_profile(
            train_log=args.train_log,
            output=args.output,
            steady_window=args.steady_window,
            experiment_id=args.experiment_id,
            profile_id=args.profile_id,
            resource_summary=args.resource_summary,
            cache_verification=args.cache_verification,
            dataset_verification=args.dataset_verification,
            environment_preflight=args.environment_preflight,
            encode_cache_report=args.encode_cache_report,
            scheduler_audit=args.scheduler_audit,
            checkpoint_preflight=args.checkpoint_preflight,
            checkpoint_io_profile=args.checkpoint_io_profile,
            train_exit_code=args.train_exit_code,
        )
    elif args.command == "validate-checkpoints":
        result = validate_checkpoints(
            train_log=args.train_log,
            checkpoint_root=args.checkpoint_root,
            output=args.output,
            behavior_metrics=args.behavior_metrics,
            train_rows=args.train_rows,
            global_batch=args.global_batch,
            selection=args.selection,
        )
        if not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "encode-cache-report":
        result = aggregate_encode_cache_metrics(
            args.metrics_dir,
            output=args.output,
        )
        if not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "checkpoint-storage-preflight":
        result = checkpoint_storage_preflight(
            args.output_dir,
            estimated_checkpoint_bytes=args.estimated_checkpoint_bytes,
            reserve_multiplier=args.reserve_multiplier,
        )
        write_checkpoint_json(args.output, result)
        if not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "checkpoint-io-profile":
        result = checkpoint_io_profile(args.checkpoint)
        write_checkpoint_json(args.output, result)
        if not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "monitor-sft":
        result = build_sft_watchdog_snapshot(
            train_log=args.train_log,
            checkpoint_root=args.checkpoint_root,
            resource_samples=args.resource_samples,
            behavior_metrics=args.behavior_metrics,
            output=args.output,
            stale_seconds=args.stale_seconds,
            trend_window=args.trend_window,
        )
        if args.strict and result["health"] == "critical":
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "long-context-plan":
        result = build_long_context_plan(
            processor_report_path=args.processor_report,
            train_jsonl=args.train_jsonl,
            sft_profile_paths=args.sft_profile,
            output=args.output,
            epochs=args.epochs,
            parameter_count=args.parameter_count,
            empirical_profile_paths=args.empirical_profile or (),
        )
        if not result["passed_static_gate"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "validate-128k-stage":
        result = validate_128k_stage(
            stage=args.stage,
            training_profile_path=args.training_profile,
            sft_profile_path=args.sft_profile,
            prior_gate_path=args.prior_gate,
            output=args.output,
        )
        if not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
