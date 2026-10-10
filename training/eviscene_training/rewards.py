"""Deterministic reward composition for EviScene Agent RL.

The default path consumes post-rollout correctness, audit gates, policy step IDs,
and optional deterministic process components.  A frozen semantic artifact may be
attached for offline diagnostics, but it never changes reward or trainability.
"""

from __future__ import annotations

import hashlib
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import canonical_json, load_json, write_json


SEMANTIC_REWARD_SCHEMA_VERSION = "ifv-semantic-reward-v2"
REWARD_LEDGER_SCHEMA_VERSION = "ifv-rl-reward-ledger-v3"
LEGACY_REWARD_LEDGER_SCHEMA_VERSION = "ifv-rl-reward-ledger-v2"
REWARD_PROFILE_SCHEMA_VERSION = "ifv-rl-reward-profile-v1"
GRPO_GROUP_SCHEMA_VERSION = "ifv-standard-grpo-group-v1"

DEFAULT_COMPONENT_WEIGHTS = {
    "evidence_chain_reward": 0.40,
    "discrepancy_alignment_reward": 0.40,
    "stop_quality_reward": 0.20,
}

SUPPORTED_COMPONENT_WEIGHTS = frozenset(DEFAULT_COMPONENT_WEIGHTS)
DEFAULT_CORRECT_REWARD_FLOOR = 0.35


def _mapping(value: Any, *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value


def _number(value: Any, *, location: str, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{location} must be within [0, 1]")
    return result


def _artifact_core(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in artifact.items()
        if key not in {"artifact_id", "created_at"}
    }


def validate_semantic_reward_artifact(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on schema, IDs, metrics, calls, or content hash changes."""

    errors: list[str] = []
    if artifact.get("schema_version") != SEMANTIC_REWARD_SCHEMA_VERSION:
        errors.append("unsupported semantic reward schema_version")
    case_id = str(artifact.get("case_id", "")).strip()
    if not case_id:
        errors.append("case_id is required")
    source_trace = artifact.get("source_trace")
    if not isinstance(source_trace, Mapping):
        errors.append("source_trace is required")
    elif len(str(source_trace.get("sha256", ""))) != 64:
        errors.append("source_trace.sha256 is invalid")
    reward_input = artifact.get("reward_input")
    if not isinstance(reward_input, Mapping):
        errors.append("reward_input is required")
    elif len(str(reward_input.get("sha256", ""))) != 64:
        errors.append("reward_input.sha256 is invalid")
    rollout = artifact.get("rollout")
    if not isinstance(rollout, Mapping):
        errors.append("rollout is required")
    else:
        if not str(rollout.get("episode_id", "")).strip():
            errors.append("rollout.episode_id is required")
        step_ids = rollout.get("policy_step_ids")
        if not isinstance(step_ids, list) or any(
            not str(value).strip() for value in step_ids
        ):
            errors.append("rollout.policy_step_ids must be a list of IDs")
        terminal = str(rollout.get("terminal_policy_step_id", ""))
        if isinstance(step_ids, list) and step_ids:
            if terminal != str(step_ids[-1]):
                errors.append("rollout.terminal_policy_step_id is inconsistent")
        elif terminal:
            errors.append("rollout.terminal_policy_step_id requires a policy step")

    metrics = artifact.get("metrics")
    if not isinstance(metrics, Mapping):
        errors.append("metrics is required")
        metrics = {}
    required_metrics = (
        "verdict_blind_agreement",
        "claim_label_agreement",
        "claim_entailment",
        "evidence_citation_fidelity",
        "evidence_sufficiency",
        "evidence_quality",
        "investigation_progress",
        "search_direction",
        "evidence_use",
        "belief_revision",
        "overall_process_quality",
    )
    for name in required_metrics:
        try:
            _number(metrics.get(name), location=f"metrics.{name}")
        except ValueError as exc:
            errors.append(str(exc))
    invalid_ids = metrics.get("invalid_judge_evidence_ids", [])
    if not isinstance(invalid_ids, list):
        errors.append("metrics.invalid_judge_evidence_ids must be a list")
    invalid_turn_ids = metrics.get("invalid_judge_turn_ids", [])
    if not isinstance(invalid_turn_ids, list):
        errors.append("metrics.invalid_judge_turn_ids must be a list")

    gates = artifact.get("gates")
    if not isinstance(gates, Mapping):
        errors.append("gates is required")
    else:
        for name in (
            "strict_trace_audit_pass",
            "engineering_valid",
            "semantic_audit_pass",
        ):
            if not isinstance(gates.get(name), bool):
                errors.append(f"gates.{name} must be boolean")

    judge = artifact.get("judge")
    if not isinstance(judge, Mapping):
        errors.append("judge is required")
    else:
        calls = judge.get("calls")
        if not isinstance(calls, list) or len(calls) != 1:
            errors.append("judge.calls must contain exactly one trajectory call")
        else:
            for index, call in enumerate(calls):
                if not isinstance(call, Mapping):
                    errors.append(f"judge.calls[{index}] must be an object")
                    continue
                if len(str(call.get("request_sha256", ""))) != 64:
                    errors.append(f"judge.calls[{index}].request_sha256 is invalid")
                if len(str(call.get("response_sha256", ""))) != 64:
                    errors.append(f"judge.calls[{index}].response_sha256 is invalid")
                usage = call.get("usage")
                if not isinstance(usage, Mapping):
                    errors.append(f"judge.calls[{index}].usage is required")
    judgment = artifact.get("trajectory_judgment")
    if not isinstance(judgment, Mapping):
        errors.append("trajectory_judgment is required")
    else:
        if judgment.get("predicted_verdict") not in {"real", "fake", "unclear"}:
            errors.append("trajectory_judgment.predicted_verdict is invalid")
        claim_reviews = judgment.get("claim_reviews")
        if not isinstance(claim_reviews, list) or not claim_reviews:
            errors.append("trajectory_judgment.claim_reviews must be a non-empty list")

    artifact_id = str(artifact.get("artifact_id", ""))
    expected_id = "sha256:" + hashlib.sha256(
        canonical_json(_artifact_core(artifact)).encode("utf-8")
    ).hexdigest()
    if artifact_id != expected_id:
        errors.append("artifact_id does not match semantic artifact content")

    return {
        "schema_version": "ifv-semantic-reward-audit-v1",
        "case_id": case_id,
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "artifact_id": artifact_id,
    }


def load_reward_profile(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        return {
            "schema_version": REWARD_PROFILE_SCHEMA_VERSION,
            "profile_id": "ifv-deterministic-process-v2",
            "correct_reward_floor": DEFAULT_CORRECT_REWARD_FLOOR,
            "weights": dict(DEFAULT_COMPONENT_WEIGHTS),
            "allow_nonfatal_audit_failures": True,
        }
    profile = load_json(path)
    if profile.get("schema_version") != REWARD_PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported reward profile schema_version")
    profile_id = str(profile.get("profile_id", "")).strip()
    if not profile_id:
        raise ValueError("reward profile requires profile_id")
    correct_reward_floor = _number(
        profile.get("correct_reward_floor", DEFAULT_CORRECT_REWARD_FLOOR),
        location="profile.correct_reward_floor",
    )
    allow_nonfatal_audit_failures = profile.get(
        "allow_nonfatal_audit_failures", False
    )
    if not isinstance(allow_nonfatal_audit_failures, bool):
        raise ValueError("profile.allow_nonfatal_audit_failures must be boolean")
    raw_weights = _mapping(profile.get("weights"), location="profile.weights")
    weights: dict[str, float] = {}
    for name, value in raw_weights.items():
        if name not in SUPPORTED_COMPONENT_WEIGHTS:
            raise ValueError(f"unknown reward component weight: {name}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"weight {name} must be numeric")
        numeric = float(value)
        if numeric < 0.0:
            raise ValueError(f"weight {name} cannot be negative")
        weights[str(name)] = numeric
    return {
        **profile,
        "correct_reward_floor": correct_reward_floor,
        "weights": weights,
        "allow_nonfatal_audit_failures": allow_nonfatal_audit_failures,
    }


def _optional_bool(value: Any, *, location: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{location} must be boolean or null")
    return value


def compose_reward_ledger(
    deterministic: Mapping[str, Any],
    *,
    profile: Mapping[str, Any] | None = None,
    semantic_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    semantic_audit: Mapping[str, Any] | None = None
    if semantic_artifact is not None:
        semantic_audit = validate_semantic_reward_artifact(semantic_artifact)
        if not semantic_audit["passed"]:
            raise ValueError(
                "invalid semantic reward artifact: "
                + "; ".join(semantic_audit["errors"])
            )
    profile = dict(profile or load_reward_profile())
    if profile.get("schema_version") != REWARD_PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported reward profile")
    weights = _mapping(profile.get("weights"), location="profile.weights")
    correct_reward_floor = _number(
        profile.get("correct_reward_floor", DEFAULT_CORRECT_REWARD_FLOOR),
        location="profile.correct_reward_floor",
    )
    allow_nonfatal_audit_failures = profile.get(
        "allow_nonfatal_audit_failures", False
    )
    if not isinstance(allow_nonfatal_audit_failures, bool):
        raise ValueError("profile.allow_nonfatal_audit_failures must be boolean")
    rollout = (
        _mapping(
            semantic_artifact.get("rollout"),
            location="semantic_artifact.rollout",
        )
        if semantic_artifact is not None
        else {}
    )

    classification_correct = _optional_bool(
        deterministic.get("classification_correct"),
        location="deterministic.classification_correct",
    )
    provider_fatal = bool(deterministic.get("fatal_engineering_error", False))
    engineering_valid = not provider_fatal
    strict_trace_audit = bool(
        deterministic.get("strict_trace_audit_pass", False)
    )
    hard_trace_audit = _optional_bool(
        deterministic.get("hard_trace_audit_pass"),
        location="deterministic.hard_trace_audit_pass",
    )
    if hard_trace_audit is None:
        # Old post-rollout artifacts only recorded the strict aggregate gate.
        # Fail closed: a strict failure cannot be assumed non-fatal retroactively.
        hard_trace_audit = strict_trace_audit
    training_prohibited = bool(
        deterministic.get("training_prohibited", False)
    )
    raw_process_components = deterministic.get("process_components", {})
    if raw_process_components is None:
        raw_process_components = {}
    process_components = _mapping(
        raw_process_components,
        location="deterministic.process_components",
    )
    components: dict[str, float | None] = {
        "classification_correct": (
            None if classification_correct is None else float(classification_correct)
        ),
        "strict_trace_audit": float(strict_trace_audit),
    }
    for name in SUPPORTED_COMPONENT_WEIGHTS:
        if name not in process_components:
            continue
        components[name] = _number(
            process_components.get(name),
            location=f"deterministic.process_components.{name}",
        )

    fatal_mask = not engineering_valid or provider_fatal
    mask_reason = ""
    if provider_fatal:
        mask_reason = "fatal_environment_or_provider_error"

    weighted_terms: list[tuple[str, float, float]] = []
    for name, raw_weight in weights.items():
        value = components.get(str(name))
        weight = float(raw_weight)
        if value is None or weight <= 0.0:
            continue
        weighted_terms.append((str(name), float(value), weight))
    weight_sum = sum(weight for _, _, weight in weighted_terms)
    quality = (
        sum(value * weight for _, value, weight in weighted_terms)
        / weight_sum
        if weight_sum > 0.0
        else 1.0 if strict_trace_audit else 0.0
    )
    audit_accepted_for_reward = bool(
        strict_trace_audit
        or (allow_nonfatal_audit_failures and hard_trace_audit)
    )
    reward_masked = fatal_mask or not audit_accepted_for_reward
    if reward_masked:
        scalar_reward_value: float | None = None
        if not mask_reason:
            mask_reason = (
                "hard_trace_audit_failed"
                if not hard_trace_audit
                else "strict_trace_audit_failed"
            )
    elif classification_correct is None:
        reward_masked = True
        scalar_reward_value = None
        mask_reason = "classification_correctness_missing"
    elif classification_correct is False:
        scalar_reward_value = 0.0
    else:
        scalar_reward_value = round(
            float(correct_reward_floor)
            + (1.0 - float(correct_reward_floor))
            * max(0.0, min(1.0, quality)),
            8,
        )

    case_id = str(
        deterministic.get("case_id")
        or (
            semantic_artifact.get("case_id")
            if semantic_artifact is not None
            else ""
        )
    )
    episode_id = str(
        deterministic.get("episode_id") or rollout.get("episode_id") or case_id
    )
    step_ids = deterministic.get("step_ids")
    if step_ids is None:
        step_ids = rollout.get("policy_step_ids", [])
    if not isinstance(step_ids, list):
        raise ValueError("deterministic.step_ids must be a list")
    semantic_metrics = (
        dict(
            _mapping(
                semantic_artifact.get("metrics"),
                location="semantic_artifact.metrics",
            )
        )
        if semantic_artifact is not None
        else {}
    )
    semantic_gates = (
        dict(
            _mapping(
                semantic_artifact.get("gates"),
                location="semantic_artifact.gates",
            )
        )
        if semantic_artifact is not None
        else {}
    )
    ledger_core = {
        "schema_version": REWARD_LEDGER_SCHEMA_VERSION,
        "reward_policy": "outcome-dominant-deterministic-v2",
        "case_id": case_id,
        "episode_id": episode_id,
        "source_semantic_artifact_id": (
            semantic_artifact.get("artifact_id")
            if semantic_artifact is not None
            else None
        ),
        "source_trace_sha256": (
            deterministic.get("source_trace_sha256")
            or (
                _mapping(
                    semantic_artifact.get("source_trace"),
                    location="semantic_artifact.source_trace",
                ).get("sha256")
                if semantic_artifact is not None
                else None
            )
        ),
        "profile": {
            "profile_id": profile.get("profile_id"),
            "correct_reward_floor": correct_reward_floor,
            "allow_nonfatal_audit_failures": allow_nonfatal_audit_failures,
            "weights": dict(weights),
            "active_weight_sum": weight_sum,
        },
        "components": components,
        "semantic_metrics": semantic_metrics,
        "diagnostics": {
            "semantic_reward": (
                {
                    "role": "diagnostic_only",
                    "artifact_id": semantic_artifact.get("artifact_id"),
                    "semantic_audit_pass": semantic_gates.get(
                        "semantic_audit_pass"
                    ),
                }
                if semantic_artifact is not None
                else None
            )
        },
        "gates": {
            "engineering_valid": engineering_valid,
            "classification_correct": classification_correct,
            "has_policy_steps": bool(step_ids),
            "training_prohibited": training_prohibited,
            "strict_trace_audit_pass": strict_trace_audit,
            "hard_trace_audit_pass": hard_trace_audit,
            "audit_accepted_for_reward": audit_accepted_for_reward,
            "rl_reward_eligible": bool(
                not reward_masked
                and classification_correct is not None
                and step_ids
                and not training_prohibited
            ),
            "trainable": bool(
                not reward_masked
                and classification_correct is not None
                and step_ids
                and not training_prohibited
            ),
            "eligible_for_positive_buffer": bool(
                not fatal_mask
                and strict_trace_audit
                and step_ids
                and classification_correct is True
                and not training_prohibited
            ),
        },
        "fatal_mask": {
            "masked": reward_masked,
            "reason": mask_reason,
        },
        "scalar_reward": scalar_reward_value,
        "step_ids": [str(value) for value in step_ids],
        "teacher_usage": _teacher_usage(semantic_artifact),
    }
    ledger_id = "sha256:" + hashlib.sha256(
        canonical_json(ledger_core).encode("utf-8")
    ).hexdigest()
    return {
        **ledger_core,
        "ledger_id": ledger_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _teacher_usage(
    artifact: Mapping[str, Any] | None,
) -> dict[str, int]:
    if artifact is None:
        return {
            "call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "thought_tokens": 0,
        }
    judge = _mapping(artifact.get("judge"), location="artifact.judge")
    calls = judge.get("calls", [])
    totals = {"call_count": 0, "input_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
    if not isinstance(calls, list):
        return totals
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        usage = call.get("usage")
        if not isinstance(usage, Mapping):
            continue
        totals["call_count"] += 1
        for name in ("input_tokens", "output_tokens", "thought_tokens"):
            totals[name] += int(usage.get(name, 0) or 0)
    return totals


def validate_reward_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if ledger.get("schema_version") not in {
        REWARD_LEDGER_SCHEMA_VERSION,
        LEGACY_REWARD_LEDGER_SCHEMA_VERSION,
    }:
        errors.append("unsupported reward ledger schema_version")
    scalar = ledger.get("scalar_reward")
    fatal_mask = _mapping(ledger.get("fatal_mask"), location="fatal_mask")
    masked = bool(fatal_mask.get("masked"))
    if masked and scalar is not None:
        errors.append("fatal-masked ledger must have scalar_reward=null")
    if not masked:
        try:
            _number(scalar, location="scalar_reward")
        except ValueError as exc:
            errors.append(str(exc))
    core = {
        str(key): value
        for key, value in ledger.items()
        if key not in {"ledger_id", "created_at"}
    }
    expected = "sha256:" + hashlib.sha256(
        canonical_json(core).encode("utf-8")
    ).hexdigest()
    if ledger.get("ledger_id") != expected:
        errors.append("ledger_id does not match reward ledger content")
    return {
        "schema_version": "ifv-rl-reward-ledger-audit-v1",
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "ledger_id": ledger.get("ledger_id"),
    }


def export_framework_reward(
    ledger: Mapping[str, Any],
    *,
    framework: str,
) -> dict[str, Any]:
    audit = validate_reward_ledger(ledger)
    if not audit["passed"]:
        raise ValueError("invalid reward ledger: " + "; ".join(audit["errors"]))
    framework = framework.casefold()
    fatal_mask = _mapping(ledger.get("fatal_mask"), location="fatal_mask")
    masked = bool(fatal_mask.get("masked"))
    reward = ledger.get("scalar_reward")
    step_ids = [str(value) for value in ledger.get("step_ids", [])]
    terminal_step_id = step_ids[-1] if step_ids else ""
    skip_update = masked or not step_ids
    mask_reason = str(fatal_mask.get("reason", ""))
    if not mask_reason and not step_ids:
        mask_reason = "no_trainable_policy_steps"
    common = {
        "episode_id": ledger.get("episode_id"),
        "reward": reward,
        "skip_update": skip_update,
        "mask_reason": mask_reason,
        "ledger_id": ledger.get("ledger_id"),
        "components": ledger.get("components"),
    }
    if framework == "rllm":
        return {
            "schema_version": "ifv-rllm-reward-record-v1",
            **common,
            "assignment": "terminal_step",
            "terminal_step_id": terminal_step_id,
            "step_rewards": [
                {
                    "step_id": step_id,
                    "reward": reward if step_id == terminal_step_id else 0.0,
                }
                for step_id in step_ids
            ],
        }
    if framework == "verl":
        return {
            "schema_version": "ifv-verl-reward-record-v1",
            "data_source": "ifv_discrepancy_agent_v4",
            **common,
            "reward_assignment": {
                "mode": "terminal_token",
                "terminal_step_id": terminal_step_id,
            },
            "extra_info": {
                "case_id": ledger.get("case_id"),
                "reward_policy": ledger.get("reward_policy"),
            },
        }
    raise ValueError("framework must be rllm or verl")


def build_standard_grpo_groups(
    ledgers: Sequence[Mapping[str, Any]],
    rollout_members: Sequence[Mapping[str, Any]],
    *,
    minimum_valid_members: int = 2,
) -> list[dict[str, Any]]:
    """Join audited episode rewards into standard same-prompt GRPO groups.

    This is a data contract only. It does not implement an advantage estimator:
    rLLM/veRL receives one scalar per complete episode and performs standard GRPO.
    """

    if minimum_valid_members < 2:
        raise ValueError("minimum_valid_members must be at least 2")
    ledger_by_episode: dict[str, Mapping[str, Any]] = {}
    for ledger in ledgers:
        audit = validate_reward_ledger(ledger)
        if not audit["passed"]:
            raise ValueError(
                "invalid reward ledger: " + "; ".join(audit["errors"])
            )
        episode_id = str(ledger.get("episode_id", "")).strip()
        if not episode_id or episode_id in ledger_by_episode:
            raise ValueError("reward ledgers require unique episode_id values")
        ledger_by_episode[episode_id] = ledger

    members_by_group: dict[str, list[Mapping[str, Any]]] = {}
    for member in rollout_members:
        if member.get("schema_version") != "ifv-rollout-group-member-v1":
            raise ValueError("unsupported rollout group member schema")
        group_id = str(member.get("prompt_group_id", "")).strip()
        episode_id = str(member.get("episode_id", "")).strip()
        if not group_id or not episode_id:
            raise ValueError("rollout member requires prompt_group_id and episode_id")
        members_by_group.setdefault(group_id, []).append(member)

    groups: list[dict[str, Any]] = []
    for group_id in sorted(members_by_group):
        source_members = sorted(
            members_by_group[group_id],
            key=lambda item: int(item.get("rollout_index", 0) or 0),
        )
        case_ids = {str(item.get("case_id", "")) for item in source_members}
        group_sizes = {int(item.get("group_size", 0) or 0) for item in source_members}
        if len(case_ids) != 1 or len(group_sizes) != 1:
            raise ValueError("one prompt group must have one case_id and group_size")
        if next(iter(group_sizes)) != len(source_members):
            raise ValueError("rollout group is incomplete")
        members: list[dict[str, Any]] = []
        rewards: list[float] = []
        for source in source_members:
            episode_id = str(source["episode_id"])
            ledger = ledger_by_episode.get(episode_id)
            reward = None if ledger is None else ledger.get("scalar_reward")
            gates = {} if ledger is None else _mapping(
                ledger.get("gates"), location="ledger.gates"
            )
            source_rl_eligible = source.get(
                "rl_reward_eligible",
                source.get("training_eligible", True),
            )
            valid = bool(
                ledger is not None
                and gates.get("trainable")
                and source_rl_eligible is not False
                and source.get("training_prohibited", False) is not True
                and isinstance(reward, (int, float))
            )
            if valid:
                rewards.append(float(reward))
            members.append(
                {
                    "episode_id": episode_id,
                    "rollout_index": int(source.get("rollout_index", 0) or 0),
                    "sampling_seed": int(source.get("sampling_seed", 0) or 0),
                    "reward": float(reward) if valid else None,
                    "policy_step_ids": (
                        [str(value) for value in ledger.get("step_ids", [])]
                        if valid and ledger is not None
                        else []
                    ),
                    "training_eligible": valid,
                    "ledger_id": None if ledger is None else ledger.get("ledger_id"),
                }
            )
        reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        trainable = len(rewards) >= minimum_valid_members and reward_std > 0.0
        skip_reason = None
        if len(rewards) < minimum_valid_members:
            skip_reason = (
                "training_prohibited_source"
                if all(
                    item.get("training_prohibited", False) is True
                    for item in source_members
                )
                else "insufficient_valid_members"
            )
        elif reward_std == 0.0:
            skip_reason = "zero_reward_variance"
        groups.append(
            {
                "schema_version": GRPO_GROUP_SCHEMA_VERSION,
                "prompt_group_id": group_id,
                "case_id": next(iter(case_ids)),
                "group_size": len(source_members),
                "valid_member_count": len(rewards),
                "reward_mean": round(statistics.mean(rewards), 8) if rewards else None,
                "reward_std": round(reward_std, 8) if rewards else None,
                "trainable": trainable,
                "skip_reason": skip_reason,
                "members": members,
            }
        )
    return groups


def build_and_write_ledger(
    *,
    deterministic_path: Path,
    output_path: Path,
    profile_path: Path | None = None,
    semantic_artifact_path: Path | None = None,
) -> dict[str, Any]:
    deterministic = load_json(deterministic_path)
    semantic_artifact = (
        load_json(semantic_artifact_path)
        if semantic_artifact_path is not None
        else None
    )
    profile = load_reward_profile(profile_path)
    ledger = compose_reward_ledger(
        deterministic,
        profile=profile,
        semantic_artifact=semantic_artifact,
    )
    write_json(output_path, ledger)
    return ledger


def build_ledgers_from_run_artifacts(
    *,
    deterministic_rows: Sequence[Mapping[str, Any]],
    semantic_artifacts: Sequence[Mapping[str, Any]] = (),
    profile: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build one reward ledger per deterministic post-rollout row."""

    semantic_by_episode: dict[str, Mapping[str, Any]] = {}
    for artifact in semantic_artifacts:
        rollout = _mapping(artifact.get("rollout"), location="artifact.rollout")
        episode_id = str(rollout.get("episode_id", "")).strip()
        if not episode_id or episode_id in semantic_by_episode:
            raise ValueError("semantic artifacts require unique episode_id values")
        semantic_by_episode[episode_id] = artifact
    ledgers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for deterministic in deterministic_rows:
        episode_id = str(deterministic.get("episode_id", "")).strip()
        if not episode_id or episode_id in seen:
            raise ValueError("deterministic rows require unique episode_id values")
        seen.add(episode_id)
        ledgers.append(
            compose_reward_ledger(
                deterministic,
                profile=profile,
                semantic_artifact=semantic_by_episode.get(episode_id),
            )
        )
    unused_semantic = sorted(set(semantic_by_episode) - seen)
    if unused_semantic:
        raise ValueError(
            "semantic artifacts lack deterministic rows: "
            + ", ".join(unused_semantic[:10])
        )
    return ledgers
