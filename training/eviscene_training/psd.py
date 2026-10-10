"""Offline contracts and target builders for EviScene Privileged Self-Distillation.

This module intentionally does not generate hints or call a model.  It creates
the auditable boundary between a privileged repair process and a future PSD
trainer: the exported target contains token IDs and teacher distributions, never
the hint text or private repair reference.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import (canonical_json, iter_jsonl, load_jsonl, require_new_or_empty,
                 sha256_file, write_json, write_jsonl)
from .artifact_receipts import artifact_identity
from .psd_modality import require_text_only_psd
from .psd_media import media_digest
from .psd_capture_semantics import RAW_POLICY_LOGPROBS


PSD_HINT_AUDIT_SCHEMA_VERSION = "ifv-psd-hint-audit-v1"
PSD_TARGET_SCHEMA_VERSION = "ifv-psd-target-v2"
PSD_MANIFEST_SCHEMA_VERSION = "ifv-psd-target-manifest-v2"
PSD_TOPK_CACHE_SCHEMA_VERSION = "ifv-psd-teacher-topk-cache-v2"
PSD_TOPK_MATERIALIZATION_SCHEMA_VERSION = "ifv-psd-topk-materialization-v2"

REPAIR_TIERS = frozenset(
    {
        "causal_episode_pass",
        "local_pass_downstream",
        "unrepairable",
        "engineering_error",
    }
)
TRAINABLE_HINT_LEVELS = frozenset({1, 2, 3})

_BINARY_LABEL_RE = re.compile(
    r"(?<![\w-])(real|fake|true|false)(?![\w-])|真实|伪造|真图|假图",
    re.IGNORECASE,
)
_INTERNAL_TERM_RE = re.compile(
    r"\b(?:ground[\s_-]*truth|private[\s_-]*gold|expected[\s_-]*verdict|"
    r"reference[\s_-]*fact|acceptable[\s_-]*evidence)\b",
    re.IGNORECASE,
)
_PRIVATE_CONTAINER_KEYS = frozenset(
    {
        "private",
        "private_gold",
        "private_reference",
        "ground_truth",
        "oracle",
        "repair_oracle",
        "expected",
        "reference_facts",
        "acceptable_evidence",
    }
)
_PRIVATE_VALUE_KEYS = frozenset(
    {
        "expected_verdict",
        "gold_verdict",
        "expected_action",
        "expected_calls",
        "exact_action",
        "exact_query",
        "expected_query",
        "source_url",
        "source_urls",
        "evidence_id",
        "evidence_ids",
        "fact_id",
        "fact_ids",
        "claim_id",
        "claim_ids",
        "image_sha256",
        "exact_text",
        "verified_value",
    }
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_strings(child)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for child in value:
            yield from _iter_strings(child)


def _private_values(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Collect values that a repair hint must not disclose."""

    values: list[tuple[str, str]] = []

    def walk(value: Any, path: str, inherited_private: bool = False) -> None:
        if not isinstance(value, Mapping):
            return
        for raw_key, child in value.items():
            key = str(raw_key)
            key_lower = key.casefold()
            child_path = f"{path}.{key}" if path else key
            private = inherited_private or key_lower in _PRIVATE_CONTAINER_KEYS
            if private or key_lower in _PRIVATE_VALUE_KEYS:
                values.extend((child_path, item) for item in _iter_strings(child))
            if isinstance(child, Mapping):
                walk(child, child_path, private)

    walk(row, "")
    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path, value in values:
        normalised = _normalise(value)
        if len(normalised) < 4:
            continue
        identity = (path, normalised)
        if identity not in seen:
            seen.add(identity)
            unique.append((path, value))
    return unique


def _expected_tool_names(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    action = row.get("expected_action") or _mapping(
        row.get("private_reference")
    ).get("expected_action")
    names: list[tuple[str, str]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                child_path = f"{path}.{key}" if path else key
                if key.casefold() in {"name", "tool", "tool_name", "function"}:
                    if isinstance(child, str) and len(child.strip()) >= 4:
                        names.append((child_path, child.strip()))
                walk(child, child_path)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(action, "expected_action")
    return names


def audit_hint(
    row: Mapping[str, Any],
    *,
    hint: str | None = None,
    hint_level: int | None = None,
) -> dict[str, Any]:
    """Fail closed if a repair hint exposes private answer information.

    L1/L2 cannot name an expected tool. L3 may name a tool class, but still
    cannot repeat a private query, URL, value, Evidence ID, or verdict.
    """

    rendered_hint = _text(hint if hint is not None else row.get("hint"))
    raw_level = hint_level if hint_level is not None else row.get("hint_level", 1)
    try:
        level = int(raw_level)
    except (TypeError, ValueError):
        level = 0

    errors: list[str] = []
    leak_paths: list[str] = []
    if not rendered_hint:
        errors.append("hint_empty")
    if len(rendered_hint) > 1200:
        errors.append("hint_too_long")
    if level not in {1, 2, 3, 4}:
        errors.append("hint_level_invalid")
    if _BINARY_LABEL_RE.search(rendered_hint):
        errors.append("binary_verdict_leak")
    if _INTERNAL_TERM_RE.search(rendered_hint):
        errors.append("internal_term_leak")

    normalised_hint = _normalise(rendered_hint)
    for path, value in _private_values(row):
        if _normalise(value) in normalised_hint:
            leak_paths.append(path)
    if level <= 2:
        for path, name in _expected_tool_names(row):
            if _normalise(name) in normalised_hint:
                leak_paths.append(path)
    if leak_paths:
        errors.append("private_value_leak")

    return {
        "schema_version": PSD_HINT_AUDIT_SCHEMA_VERSION,
        "passed": not errors,
        "hint_level": level,
        "hint_chars": len(rendered_hint),
        "hint_sha256": hashlib.sha256(rendered_hint.encode("utf-8")).hexdigest(),
        "errors": sorted(set(errors)),
        "leak_paths": sorted(set(leak_paths)),
    }


def _int_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{field}[{index}] must be an integer")
        try:
            result.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}[{index}] must be an integer") from exc
    return result


def _prompt_ids(row: Mapping[str, Any], key: str) -> list[int]:
    direct = row.get(key)
    if direct is not None:
        return _int_list(direct, field=key)
    parent = "student" if key == "student_prompt_ids" else "teacher"
    return _int_list(_mapping(row.get(parent)).get("prompt_ids"), field=key)


def _completion_ids(row: Mapping[str, Any]) -> list[int]:
    if row.get("completion_ids") is not None:
        return _int_list(row.get("completion_ids"), field="completion_ids")
    return _int_list(
        _mapping(row.get("completion")).get("token_ids"),
        field="completion_ids",
    )


def _topk_entry(value: Any) -> tuple[int, float]:
    if isinstance(value, Mapping):
        token_id = value.get("token_id")
        probability = value.get("probability", value.get("prob"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        token_id, probability = value
    else:
        raise ValueError("top-k entry must be [token_id, probability] or an object")
    if isinstance(token_id, bool):
        raise ValueError("top-k token ID must be numeric")
    try:
        token = int(token_id)
        probability_value = float(probability)
    except (TypeError, ValueError) as exc:
        raise ValueError("top-k entry contains an invalid token/probability") from exc
    if not math.isfinite(probability_value) or probability_value < 0.0:
        raise ValueError("top-k probability must be finite and non-negative")
    return token, probability_value


def validate_topk_by_position(
    completion_ids: Sequence[int],
    topk_by_position: Any,
    *,
    topk: int,
) -> None:
    if topk < 1:
        raise ValueError("topk must be positive")
    if not isinstance(topk_by_position, list):
        raise ValueError("topk_by_position must be a list")
    if len(topk_by_position) != len(completion_ids):
        raise ValueError("topk_by_position length must match completion_ids")
    for index, raw_entries in enumerate(topk_by_position):
        if not isinstance(raw_entries, list) or len(raw_entries) != topk:
            raise ValueError(
                f"topk_by_position[{index}] must contain exactly {topk} entries"
            )
        entries = [_topk_entry(item) for item in raw_entries]
        if len({token for token, _ in entries}) != topk:
            raise ValueError(f"topk_by_position[{index}] has duplicate tokens")
        probability_sum = sum(probability for _, probability in entries)
        if not math.isclose(probability_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                f"topk_by_position[{index}] probabilities must sum to one"
            )


def _assert_trainable_source(row: Mapping[str, Any]) -> None:
    split = _text(
        row.get("split") or _mapping(row.get("source_metadata")).get("split")
    ).casefold()
    if split in {"test", "holdout"}:
        raise ValueError(f"PSD source split is forbidden: {split}")
    if bool(row.get("training_prohibited")):
        raise ValueError("PSD source is training_prohibited")
    if bool(row.get("engineering_error")) or _text(row.get("status")) == "engineering_error":
        raise ValueError("engineering_error cannot become a PSD target")


def _repair_tier(row: Mapping[str, Any]) -> str:
    tier = _text(row.get("repair_tier") or row.get("tier"))
    if tier:
        return tier
    verification = _mapping(row.get("verification"))
    if verification.get("hinted_episode_pass") is True:
        return "causal_episode_pass"
    if verification.get("hinted_local_pass") is True:
        return "local_pass_downstream"
    return ""


def _target_id(
    *,
    kind: str,
    case_id: str,
    episode_id: str,
    step_id: str,
    completion_ids: Sequence[int],
) -> str:
    payload = {
        "kind": kind,
        "case_id": case_id,
        "episode_id": episode_id,
        "step_id": step_id,
        "completion_sha256": hashlib.sha256(
            canonical_json(list(completion_ids)).encode("utf-8")
        ).hexdigest(),
    }
    return "psd:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _topk_payload(
    row: Mapping[str, Any],
    completion_ids: Sequence[int],
    *,
    topk: int,
) -> tuple[str, list[Any]]:
    value = row.get("teacher_topk_by_position")
    if value is None:
        return "pending_topk", []
    validate_topk_by_position(completion_ids, value, topk=topk)
    return "complete", value


def _validated_model_roles(value: Any) -> dict[str, Any]:
    """Validate the three PSD roles and return a detached role record."""

    model_roles = _mapping(value)
    constructor = _mapping(model_roles.get("hint_constructor"))
    teacher = _mapping(model_roles.get("frozen_self_teacher"))
    student = _mapping(model_roles.get("trainable_student"))
    if constructor.get("supplies_training_distribution") is not False:
        raise ValueError("hint constructor cannot supply PSD training targets")
    if teacher.get("supplies_training_distribution") is not True:
        raise ValueError("frozen self-teacher must supply PSD training targets")
    if teacher.get("sees_hint") is not True:
        raise ValueError("frozen self-teacher must see the verified hint")
    if student.get("sees_hint") is not False:
        raise ValueError("trainable student must not see the hint")
    teacher_provider = _text(teacher.get("provider")).casefold()
    student_provider = _text(student.get("provider")).casefold()
    if not teacher_provider or teacher_provider != student_provider:
        raise ValueError("PSD self-teacher/student provider mismatch")
    teacher_model = _text(teacher.get("model"))
    student_model = _text(student.get("model"))
    if not teacher_model or teacher_model != student_model:
        raise ValueError("PSD self-teacher/student policy mismatch")
    teacher_checkpoint = _text(teacher.get("round_start_checkpoint"))
    student_checkpoint = _text(student.get("initial_checkpoint"))
    if not teacher_checkpoint or teacher_checkpoint != student_checkpoint:
        raise ValueError("PSD self-teacher/student checkpoint mismatch")
    teacher_manifest_sha256 = _text(
        teacher.get("checkpoint_manifest_sha256")
    ).casefold()
    student_manifest_sha256 = _text(
        student.get("checkpoint_manifest_sha256")
    ).casefold()
    if (
        not re.fullmatch(r"[0-9a-f]{64}", teacher_manifest_sha256)
        or teacher_manifest_sha256 != student_manifest_sha256
    ):
        raise ValueError("PSD self-teacher/student checkpoint manifest mismatch")
    return {
        "hint_constructor": dict(constructor),
        "frozen_self_teacher": dict(teacher),
        "trainable_student": dict(student),
    }


def summarize_round_model_roles(values: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Bind one training policy while retaining non-training constructor provenance."""
    records = [_validated_model_roles(value) for value in values]
    if not records:
        return None
    policies = {
        canonical_json({"frozen_self_teacher": row["frozen_self_teacher"],
            "trainable_student": row["trainable_student"]})
        for row in records
    }
    if len(policies) != 1:
        raise ValueError("PSD rows do not share one frozen teacher/student policy")
    constructors = {canonical_json(row["hint_constructor"]): row["hint_constructor"] for row in records}
    if len(constructors) == 1:
        constructor = dict(next(iter(constructors.values())))
    else:
        constructor = {"provider": "multiple", "model": "multiple-non-training-constructors",
            "supplies_training_distribution": False,
            "variants": [constructors[key] for key in sorted(constructors)]}
    first = records[0]
    return {"hint_constructor": constructor,
        "frozen_self_teacher": first["frozen_self_teacher"],
        "trainable_student": first["trainable_student"]}


def _teacher_identity(target: Mapping[str, Any]) -> dict[str, str]:
    roles = _validated_model_roles(target.get("model_roles"))
    teacher = _mapping(roles["frozen_self_teacher"])
    return {
        "provider": _text(teacher.get("provider")).casefold(),
        "model": _text(teacher.get("model")),
        "checkpoint": _text(teacher.get("round_start_checkpoint")),
        "checkpoint_manifest_sha256": _text(
            teacher.get("checkpoint_manifest_sha256")
        ).casefold(),
    }


def build_repair_target(
    row: Mapping[str, Any],
    *,
    topk: int = 20,
    allow_local_only: bool = False,
) -> dict[str, Any]:
    _assert_trainable_source(row)
    if row.get("accepted") is False:
        raise ValueError("repair row is not accepted")
    tier = _repair_tier(row)
    if tier not in REPAIR_TIERS:
        raise ValueError("repair row needs a valid repair_tier")
    if tier != "causal_episode_pass" and not allow_local_only:
        raise ValueError(f"repair tier is not enabled for primary PSD: {tier}")

    hint_record = _mapping(row.get("hint_record"))
    hint = _text(hint_record.get("text") or row.get("hint"))
    raw_hint_level = hint_record.get("level") or row.get("hint_level", 1)
    hint_audit = audit_hint(row, hint=hint, hint_level=int(raw_hint_level))
    if not hint_audit["passed"]:
        raise ValueError("hint audit failed: " + ",".join(hint_audit["errors"]))
    if int(hint_audit["hint_level"]) not in TRAINABLE_HINT_LEVELS:
        raise ValueError("only L1/L2/L3 hints can create PSD targets")

    student_prompt_ids = _prompt_ids(row, "student_prompt_ids")
    teacher_prompt_ids = _prompt_ids(row, "teacher_prompt_ids")
    completion_ids = _completion_ids(row)
    require_text_only_psd(row, student_prompt_ids, teacher_prompt_ids, completion_ids)
    if student_prompt_ids == teacher_prompt_ids:
        raise ValueError("teacher prompt must differ from student prompt")

    case_id = _text(row.get("case_id"))
    episode_id = _text(row.get("episode_id"))
    repair_site = _mapping(row.get("repair_site"))
    step_id = _text(row.get("repair_step_id") or row.get("step_id") or repair_site.get("step_id"))
    if not case_id or not episode_id or not step_id:
        raise ValueError("repair row requires case_id, episode_id and repair step ID")
    verification = _mapping(row.get("verification"))
    if tier == "causal_episode_pass":
        required = (
            "source_rollout_failed",
            "hinted_local_pass",
            "hinted_episode_pass",
            "hinted_strict_trace_audit_pass",
        )
        missing = [name for name in required if verification.get(name) is not True]
        if missing:
            raise ValueError(
                "causal_episode_pass requires verified PSD repair fields: "
                + ",".join(missing)
            )
    model_roles = _validated_model_roles(row.get("model_roles"))

    row_weight = float(row.get("row_weight", 1.0))
    if not math.isfinite(row_weight) or row_weight <= 0:
        raise ValueError("row_weight must be finite and positive")
    target_status, teacher_topk = _topk_payload(row, completion_ids, topk=topk)
    return {
        "schema_version": PSD_TARGET_SCHEMA_VERSION,
        "target_id": _target_id(
            kind="repair",
            case_id=case_id,
            episode_id=episode_id,
            step_id=step_id,
            completion_ids=completion_ids,
        ),
        "kind": "repair",
        "target_status": target_status,
        "case_id": case_id,
        "episode_id": episode_id,
        "repair_site": {
            "step_id": step_id,
            "stage": _text(row.get("stage") or repair_site.get("stage")),
            "example_type": _text(
                row.get("example_type") or repair_site.get("example_type")
            ),
        },
        "repair_tier": tier,
        "hint_level": int(hint_audit["hint_level"]),
        "hint_sha256": hint_audit["hint_sha256"],
        "student_prompt_ids": student_prompt_ids,
        "teacher_prompt_ids": teacher_prompt_ids,
        **({"psd_media": dict(row["psd_media"])} if row.get("psd_media") else {}),
        "completion_ids": completion_ids,
        "teacher_topk_by_position": teacher_topk,
        "teacher_logprob_semantics": row.get("teacher_logprob_semantics", ""),
        "row_weight": row_weight,
        "model_roles": dict(model_roles),
        "verification": {
            "source_rollout_failed": verification.get("source_rollout_failed"),
            "hinted_local_pass": verification.get("hinted_local_pass"),
            "hinted_episode_pass": verification.get("hinted_episode_pass"),
            "hinted_strict_trace_audit_pass": verification.get(
                "hinted_strict_trace_audit_pass"
            ),
        },
        "source": {
            "source_run_id": _text(row.get("source_run_id")),
            "runtime_commit": _text(row.get("runtime_commit")),
            "source_trace_sha256": _text(row.get("source_trace_sha256")),
            **({"source_task_review": row["source_task_review"]} if row.get("source_task_review") else {}),
            "psd_round_index": row.get("psd_round_index"),
            "psd_rollout_gate_sha256": _text(
                row.get("psd_rollout_gate_sha256")
            ),
        },
    }


def _preservation_steps(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_steps = row.get("preservation_steps")
    if raw_steps is None:
        raw_steps = _mapping(row.get("preserve_rollout")).get("steps")
    if raw_steps is None:
        raw_steps = [row]
    if not isinstance(raw_steps, list):
        raise ValueError("preservation_steps must be a list")
    return [step for step in raw_steps if isinstance(step, Mapping)]


def build_preservation_targets(
    row: Mapping[str, Any],
    *,
    topk: int = 20,
    model_roles: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    _assert_trainable_source(row)
    if row.get("verified_full_task") is not True:
        raise ValueError("preservation row is not a verified policy pass")
    if row.get("strict_trace_audit_pass") is not True:
        raise ValueError("preservation row did not pass strict trace audit")
    bound_roles = _validated_model_roles(model_roles or row.get("model_roles"))
    row_roles = row.get("model_roles")
    if row_roles is not None and canonical_json(bound_roles) != canonical_json(
        _validated_model_roles(row_roles)
    ):
        raise ValueError("preservation row model roles differ from PSD round")
    steps = _preservation_steps(row)
    if not steps:
        raise ValueError("preservation row has no usable steps")

    case_id = _text(row.get("case_id"))
    episode_id = _text(row.get("episode_id"))
    if not case_id or not episode_id:
        raise ValueError("preservation row requires case_id and episode_id")
    per_target_weight = float(row.get("row_weight", 1.0))
    if not math.isfinite(per_target_weight) or per_target_weight <= 0:
        raise ValueError("preservation row_weight must be finite and positive")

    targets: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        merged = {**row, **step}
        student_prompt_ids = _prompt_ids(merged, "student_prompt_ids")
        completion_ids = _completion_ids(merged)
        require_text_only_psd(merged, student_prompt_ids, completion_ids)
        target_status, teacher_topk = _topk_payload(
            merged, completion_ids, topk=topk
        )
        step_id = _text(
            merged.get("step_id")
            or merged.get("preservation_step_id")
            or f"{episode_id}:preserve:{index}"
        )
        targets.append(
            {
                "schema_version": PSD_TARGET_SCHEMA_VERSION,
                "target_id": _target_id(
                    kind="preserve",
                    case_id=case_id,
                    episode_id=episode_id,
                    step_id=step_id,
                    completion_ids=completion_ids,
                ),
                "kind": "preserve",
                "target_status": target_status,
                "case_id": case_id,
                "episode_id": episode_id,
                "repair_site": {"step_id": step_id, "stage": "", "example_type": ""},
                "repair_tier": "preservation",
                "hint_level": 0,
                "hint_sha256": "",
                "student_prompt_ids": student_prompt_ids,
                "teacher_prompt_ids": student_prompt_ids,
                **({"psd_media": dict(merged["psd_media"])} if merged.get("psd_media") else {}),
                "completion_ids": completion_ids,
                "teacher_topk_by_position": teacher_topk,
                "teacher_logprob_semantics": merged.get("teacher_logprob_semantics", ""),
                # Match the published PSD recipe: every retained preservation
                # assistant step is one target and receives the configured
                # per-target weight.  Do not divide an episode's weight across
                # its steps or rebalance aggregate repair/preservation mass.
                "row_weight": per_target_weight,
                "model_roles": bound_roles,
                "verification": {
                    "local_pass": True,
                    "full_episode_pass": True,
                    "strict_trace_audit_pass": True,
                },
                "source": {
                    "source_run_id": _text(row.get("source_run_id")),
                    "runtime_commit": _text(row.get("runtime_commit")),
                    "source_trace_sha256": _text(row.get("source_trace_sha256")),
                    **({"source_task_review": row["source_task_review"]} if row.get("source_task_review") else {}),
                    "psd_round_index": row.get("psd_round_index"),
                    "psd_rollout_gate_sha256": _text(
                        row.get("psd_rollout_gate_sha256")
                    ),
                },
            }
        )
    return targets


def audit_psd_hints(input_path: Path, output_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(load_jsonl(input_path)):
        rows.append(
            {
                "row_index": index,
                "case_id": _text(row.get("case_id")),
                "episode_id": _text(row.get("episode_id")),
                "repair_step_id": _text(row.get("repair_step_id") or row.get("step_id")),
                **audit_hint(row),
            }
        )
    write_jsonl(output_path, rows)
    return {
        "schema_version": PSD_HINT_AUDIT_SCHEMA_VERSION,
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(rows),
        "passed": sum(item["passed"] for item in rows),
        "rejected": sum(not item["passed"] for item in rows),
    }


def build_psd_target_package(
    *,
    repairs_path: Path,
    preservation_path: Path | None,
    output_dir: Path,
    topk: int = 20,
    allow_local_only: bool = False,
) -> dict[str, Any]:
    require_new_or_empty(output_dir)
    repair_target_count = 0
    preservation_target_count = 0
    rejection_count = 0
    pending_topk = 0
    target_ids: set[str] = set()
    role_records: list[Mapping[str, Any]] = []
    repair_tiers: Counter[str] = Counter()
    repair_output = output_dir / "repair_targets.jsonl"
    preservation_output = output_dir / "preservation_targets.jsonl"
    targets_output = output_dir / "targets.jsonl"
    rejections_output = output_dir / "rejections.jsonl"

    with (repair_output.open("w", encoding="utf-8") as repair_handle,
          preservation_output.open("w", encoding="utf-8") as preservation_handle,
          targets_output.open("w", encoding="utf-8") as targets_handle,
          rejections_output.open("w", encoding="utf-8") as rejection_handle):
        for index, row in enumerate(iter_jsonl(repairs_path)):
            try:
                target = build_repair_target(row, topk=topk,
                    allow_local_only=allow_local_only)
            except ValueError as exc:
                rejection = {"kind": "repair", "row_index": index,
                    "case_id": _text(row.get("case_id")),
                    "episode_id": _text(row.get("episode_id")), "reason": str(exc)}
                rejection_handle.write(canonical_json(rejection) + "\n")
                rejection_count += 1
                continue
            target_id = target["target_id"]
            if target_id in target_ids:
                raise ValueError("PSD target IDs must be unique")
            target_ids.add(target_id)
            role_records.append(target["model_roles"])
            rendered = canonical_json(target) + "\n"
            repair_handle.write(rendered)
            targets_handle.write(rendered)
            repair_target_count += 1
            pending_topk += target["target_status"] == "pending_topk"
            repair_tiers[target["repair_tier"]] += 1

        try:
            round_model_roles = summarize_round_model_roles(role_records)
        except ValueError as error:
            raise ValueError(
                "all PSD repairs in one package must use the same round-start policy") from error
        if preservation_path is not None:
            for index, row in enumerate(iter_jsonl(preservation_path)):
                try:
                    row_targets = build_preservation_targets(row, topk=topk,
                        model_roles=round_model_roles)
                except ValueError as exc:
                    rejection = {"kind": "preserve", "row_index": index,
                        "case_id": _text(row.get("case_id")),
                        "episode_id": _text(row.get("episode_id")), "reason": str(exc)}
                    rejection_handle.write(canonical_json(rejection) + "\n")
                    rejection_count += 1
                    continue
                row_ids = [target["target_id"] for target in row_targets]
                if (len(row_ids) != len(set(row_ids))
                        or any(target_id in target_ids for target_id in row_ids)):
                    raise ValueError("PSD target IDs must be unique")
                target_ids.update(row_ids)
                for target in row_targets:
                    rendered = canonical_json(target) + "\n"
                    preservation_handle.write(rendered)
                    targets_handle.write(rendered)
                    preservation_target_count += 1
                    pending_topk += target["target_status"] == "pending_topk"

    target_count = repair_target_count + preservation_target_count
    manifest = {
        "schema_version": PSD_MANIFEST_SCHEMA_VERSION,
        "topk": topk,
        "allow_local_only": allow_local_only,
        "source": {
            "repairs": str(repairs_path),
            "repairs_sha256": sha256_file(repairs_path),
            "preservation": str(preservation_path) if preservation_path else None,
            "preservation_sha256": (
                sha256_file(preservation_path) if preservation_path else None
            ),
        },
        "counts": {
            "repair_targets": repair_target_count,
            "preservation_targets": preservation_target_count,
            "targets": target_count,
            "pending_topk": pending_topk,
            "complete_topk": target_count - pending_topk,
            "rejections": rejection_count,
        },
        "repair_tiers": dict(sorted(repair_tiers.items())),
        "artifacts": {
            "repair_targets": "repair_targets.jsonl",
            "preservation_targets": "preservation_targets.jsonl",
            "targets": "targets.jsonl",
            "rejections": "rejections.jsonl",
        },
        "status": (
            "blocked_missing_source_kind"
            if target_count and (not repair_target_count or not preservation_target_count)
            else "ready_for_topk_cache"
            if target_count and pending_topk
            else "ready_for_training"
            if target_count
            else "empty"
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(
        canonical_json(list(token_ids)).encode("utf-8")
    ).hexdigest()


def _cache_topk(row: Mapping[str, Any]) -> Any:
    value = row.get("teacher_topk_by_position")
    return value if value is not None else row.get("topk_by_position")


def materialize_psd_topk_cache(
    *,
    targets_path: Path,
    cache_path: Path,
    output_dir: Path,
    topk: int = 20,
) -> dict[str, Any]:
    """Attach a verified teacher top-k cache to a PSD target package.

    The cache is intentionally keyed by target ID *and* token hashes.  A cache
    row generated for a similar state, a different template, or a different
    completion is unusable.  If any target cannot be verified, no
    ``targets.jsonl`` training artifact is emitted.
    """

    require_new_or_empty(output_dir)
    target_rows = load_jsonl(targets_path)
    if not target_rows:
        raise ValueError("PSD target input is empty")

    cache_by_target: dict[str, Mapping[str, Any]] = {}
    cache_rejections: list[dict[str, Any]] = []
    for row_index, row in enumerate(load_jsonl(cache_path)):
        target_id = _text(row.get("target_id"))
        if not target_id:
            cache_rejections.append(
                {
                    "kind": "cache_row",
                    "row_index": row_index,
                    "reason": "cache_row_missing_target_id",
                }
            )
            continue
        if target_id in cache_by_target:
            cache_rejections.append(
                {
                    "kind": "cache_row",
                    "row_index": row_index,
                    "target_id": target_id,
                    "reason": "duplicate_cache_target_id",
                }
            )
            continue
        cache_by_target[target_id] = row

    resolved: list[dict[str, Any]] = []
    target_rejections: list[dict[str, Any]] = []
    consumed_target_ids: set[str] = set()
    target_ids: set[str] = set()
    for row_index, target in enumerate(target_rows):
        target_id = _text(target.get("target_id"))
        if not target_id:
            target_rejections.append(
                {
                    "kind": "target",
                    "row_index": row_index,
                    "reason": "target_missing_target_id",
                }
            )
            continue
        if target_id in target_ids:
            target_rejections.append(
                {
                    "kind": "target",
                    "row_index": row_index,
                    "target_id": target_id,
                    "reason": "duplicate_target_id",
                }
            )
            continue
        target_ids.add(target_id)
        try:
            if _text(target.get("schema_version")) != PSD_TARGET_SCHEMA_VERSION:
                raise ValueError("target_schema_invalid")
            teacher_prompt_ids = _prompt_ids(target, "teacher_prompt_ids")
            completion_ids = _completion_ids(target)
            expected_teacher = _teacher_identity(target)
            cache_row = cache_by_target.get(target_id)
            if cache_row is None:
                raise ValueError("cache_entry_missing")
            if _text(cache_row.get("schema_version")) != PSD_TOPK_CACHE_SCHEMA_VERSION:
                raise ValueError("cache_schema_invalid")
            teacher_provider = _text(cache_row.get("teacher_provider")).casefold()
            teacher_model = _text(
                cache_row.get("teacher_model") or cache_row.get("model")
            )
            teacher_checkpoint = _text(cache_row.get("teacher_checkpoint"))
            teacher_checkpoint_manifest_sha256 = _text(
                cache_row.get("teacher_checkpoint_manifest_sha256")
            ).casefold()
            if teacher_provider != expected_teacher["provider"]:
                raise ValueError("cache_teacher_provider_mismatch")
            if teacher_model != expected_teacher["model"]:
                raise ValueError("cache_teacher_model_mismatch")
            if teacher_checkpoint != expected_teacher["checkpoint"]:
                raise ValueError("cache_teacher_checkpoint_mismatch")
            if teacher_checkpoint_manifest_sha256 != expected_teacher[
                "checkpoint_manifest_sha256"
            ]:
                raise ValueError("cache_teacher_checkpoint_manifest_mismatch")
            if _text(cache_row.get("teacher_prompt_sha256")) != _token_ids_sha256(
                teacher_prompt_ids
            ):
                raise ValueError("cache_teacher_prompt_sha256_mismatch")
            if _text(cache_row.get("completion_sha256")) != _token_ids_sha256(
                completion_ids
            ):
                raise ValueError("cache_completion_sha256_mismatch")
            cache_topk = _cache_topk(cache_row)
            if cache_row.get("teacher_logprob_semantics") != RAW_POLICY_LOGPROBS:
                raise ValueError("cache_teacher_logprob_semantics_unattested")
            if target.get("psd_media") and cache_row.get("media_sha256") != media_digest(target["psd_media"]):
                raise ValueError("cache_media_sha256_mismatch")
            validate_topk_by_position(completion_ids, cache_topk, topk=topk)
        except ValueError as exc:
            target_rejections.append(
                {
                    "kind": "target",
                    "row_index": row_index,
                    "target_id": target_id,
                    "reason": str(exc),
                }
            )
            continue

        consumed_target_ids.add(target_id)
        resolved_target = dict(target)
        resolved_target["target_status"] = "complete"
        resolved_target["teacher_topk_by_position"] = cache_topk
        resolved_target["teacher_logprob_semantics"] = RAW_POLICY_LOGPROBS
        resolved_target["teacher"] = {
            "provider": teacher_provider,
            "model": teacher_model,
            "checkpoint": teacher_checkpoint,
            "checkpoint_manifest_sha256": teacher_checkpoint_manifest_sha256,
            "teacher_prompt_sha256": _token_ids_sha256(teacher_prompt_ids),
            "completion_sha256": _token_ids_sha256(completion_ids),
        }
        resolved.append(resolved_target)

    cache_orphans = [
        {
            "target_id": target_id,
            "reason": "cache_target_not_present_in_input",
        }
        for target_id in sorted(set(cache_by_target) - consumed_target_ids)
    ]
    rejections = cache_rejections + target_rejections
    all_targets_resolved = len(resolved) == len(target_rows) and not rejections

    write_jsonl(output_dir / "resolved_targets.jsonl", resolved)
    write_jsonl(output_dir / "topk_cache_rejections.jsonl", rejections)
    write_jsonl(output_dir / "topk_cache_orphans.jsonl", cache_orphans)
    artifacts = {
        "resolved_targets": "resolved_targets.jsonl",
        "topk_cache_rejections": "topk_cache_rejections.jsonl",
        "topk_cache_orphans": "topk_cache_orphans.jsonl",
    }
    if all_targets_resolved:
        repair_targets = [
            row for row in resolved if _text(row.get("kind")) == "repair"
        ]
        preservation_targets = [
            row for row in resolved if _text(row.get("kind")) == "preserve"
        ]
        write_jsonl(output_dir / "repair_targets.jsonl", repair_targets)
        write_jsonl(
            output_dir / "preservation_targets.jsonl",
            preservation_targets,
        )
        write_jsonl(output_dir / "targets.jsonl", resolved)
        artifacts.update(
            {
                "repair_targets": "repair_targets.jsonl",
                "preservation_targets": "preservation_targets.jsonl",
                "targets": "targets.jsonl",
            }
        )

    manifest = {
        "schema_version": PSD_TOPK_MATERIALIZATION_SCHEMA_VERSION,
        "topk": topk,
        "source": {
            "targets": str(targets_path),
            "targets_identity": artifact_identity(targets_path),
            "cache": str(cache_path),
            "cache_identity": artifact_identity(cache_path),
        },
        "counts": {
            "input_targets": len(target_rows),
            "resolved_targets": len(resolved),
            "cache_rejections": len(cache_rejections),
            "target_rejections": len(target_rejections),
            "cache_orphans": len(cache_orphans),
        },
        "artifacts": artifacts,
        "status": (
            "ready_for_training"
            if all_targets_resolved
            else "blocked_topk_cache"
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
