"""Fail-closed preflight for materialized sparse top-K PSD datums."""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .artifact_receipts import artifact_identity
from .io import iter_jsonl, load_json, sha256_file
from .psd_datums import (
    PSD_SPARSE_DATUM_MANIFEST_SCHEMA_VERSION,
    PSD_SPARSE_DATUM_SCHEMA_VERSION,
    PSD_COMPACT_DATUM_SCHEMA_VERSION,
    PSD_WEIGHTING_POLICY,
)


PSD_TRAINING_INPUT_GATE_SCHEMA_VERSION = "ifv-psd-training-input-gate-v1"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _integer(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _datum_error(row: Mapping[str, Any], *, topk: int, max_context: int) -> str:
    from .psd_capture_semantics import RAW_POLICY_LOGPROBS
    if row.get("teacher_logprob_semantics") != RAW_POLICY_LOGPROBS:
        return "datum_teacher_logprob_semantics_unattested"
    if _text(row.get("schema_version")) not in {PSD_SPARSE_DATUM_SCHEMA_VERSION, PSD_COMPACT_DATUM_SCHEMA_VERSION}:
        return "datum_schema_invalid"
    if _text(row.get("kind")) not in {"repair", "preserve"}:
        return "datum_kind_invalid"
    if _integer(row.get("topk")) != topk:
        return "datum_topk_mismatch"
    input_ids = row.get("input_ids")
    compact = row.get("schema_version") == PSD_COMPACT_DATUM_SCHEMA_VERSION
    target_tokens = row.get("sparse_target_tokens" if compact else "target_tokens")
    weights = row.get("sparse_weights" if compact else "weights")
    if not isinstance(input_ids, list) or not input_ids:
        return "datum_input_ids_invalid"
    if any(
        isinstance(token, bool) or not isinstance(token, int)
        for token in input_ids
    ):
        return "datum_input_ids_invalid"
    if len(input_ids) > max_context:
        return "datum_context_exceeded"
    try:
        from .psd_modality import require_text_only_psd
        require_text_only_psd(row, input_ids)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return f"datum_media_invalid:{exc}"
    positions = row.get("loss_positions")
    completion_tokens = row.get("completion_tokens")
    if (type(completion_tokens) is not int or not 0 < completion_tokens <= len(input_ids)
            or positions != list(range(len(input_ids) - completion_tokens, len(input_ids)))):
        return "datum_causal_prediction_positions_invalid"
    if compact and ("target_tokens" in row or "weights" in row):
        return "datum_ambiguous_target_layout"
    expected_length = len(positions) if compact else len(input_ids)
    if not isinstance(target_tokens, list) or len(target_tokens) != expected_length:
        return "datum_target_shape_invalid"
    if not isinstance(weights, list) or len(weights) != expected_length:
        return "datum_weight_shape_invalid"
    try:
        row_weight = float(row.get("row_weight"))
        source_row_weight = float(row.get("source_row_weight"))
    except (TypeError, ValueError):
        return "datum_row_weight_invalid"
    if (
        not math.isfinite(row_weight)
        or row_weight <= 0
        or not math.isfinite(source_row_weight)
        or source_row_weight <= 0
    ):
        return "datum_row_weight_invalid"
    if not math.isclose(
        row_weight,
        source_row_weight,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        return "datum_aggregate_source_rebalancing_detected"
    active = False
    active_positions = []
    for index, (tokens, values) in enumerate(zip(target_tokens, weights)):
        if not isinstance(tokens, list) or len(tokens) != topk:
            return "datum_target_topk_shape_invalid"
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in tokens
        ):
            return "datum_target_token_invalid"
        if not isinstance(values, list) or len(values) != topk:
            return "datum_weight_topk_shape_invalid"
        try:
            numeric = [float(value) for value in values]
        except (TypeError, ValueError):
            return "datum_weight_non_numeric"
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            return "datum_weight_invalid"
        position_mass = sum(numeric)
        if position_mass > 0:
            active_positions.append(positions[index] if compact else index)
            if len(set(tokens)) != topk:
                return "datum_duplicate_topk_token"
        if position_mass > 0 and not math.isclose(
            position_mass,
            row_weight,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            return "datum_active_position_mass_mismatch"
        active = active or any(value > 0 for value in numeric)
    if not active:
        return "datum_has_no_active_target"
    if active_positions != positions:
        return "datum_active_positions_do_not_match_completion"
    return ""


def verify_psd_training_input(
    *,
    datums_path: Path,
    manifest_path: Path,
    expected_topk: int,
    max_context: int,
) -> dict[str, Any]:
    errors: list[str] = []
    datums_path = datums_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if expected_topk < 1:
        raise ValueError("expected_topk must be positive")
    if max_context < 2:
        raise ValueError("max_context must be at least two")

    manifest: Mapping[str, Any] = {}
    if not datums_path.is_file():
        errors.append("datums_file_missing")
    if not manifest_path.is_file():
        errors.append("manifest_file_missing")
    if not errors:
        try:
            manifest = load_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("manifest_json_invalid")
    if manifest:
        source = _mapping(manifest.get("source"))
        targets = Path(_text(source.get("targets")))
        schema = manifest.get("schema_version")
        if schema == PSD_SPARSE_DATUM_MANIFEST_SCHEMA_VERSION:
            try:
                if artifact_identity(targets) != source.get("targets_identity"):
                    errors.append("manifest_source_targets_changed_or_missing")
            except (OSError, ValueError):
                errors.append("manifest_source_targets_changed_or_missing")
        elif not targets.is_file() or sha256_file(targets) != source.get("targets_sha256"):
            # Legacy manifests retain their old fail-closed hash contract.
            errors.append("manifest_source_targets_changed_or_missing")
        if (
            schema not in {PSD_SPARSE_DATUM_MANIFEST_SCHEMA_VERSION,
                "ifv-psd-sparse-topk-manifest-v4", "ifv-psd-sparse-topk-manifest-v3"}
        ):
            errors.append("manifest_schema_invalid")
        if manifest.get("status") != "ready_for_trainer":
            errors.append("manifest_not_ready_for_trainer")
        if _integer(manifest.get("topk")) != expected_topk:
            errors.append("manifest_topk_mismatch")
        if manifest.get("weighting_policy") != PSD_WEIGHTING_POLICY:
            errors.append("manifest_weighting_policy_invalid")
        if manifest.get("aggregate_source_rebalancing") is not False:
            errors.append("manifest_aggregate_source_rebalancing_enabled")
        if manifest.get("require_both_kinds") is not True:
            errors.append("manifest_both_source_kinds_not_required")
        if manifest.get("missing_source_kinds") not in ([], None):
            errors.append("manifest_source_kind_missing")
        manifest_max_context = _integer(manifest.get("max_sequence_length"))
        if not 2 <= manifest_max_context <= max_context:
            errors.append("manifest_context_limit_incompatible")
        recorded_datums = _text(_mapping(manifest.get("artifacts")).get("datums"))
        if not recorded_datums:
            errors.append("manifest_datums_artifact_missing")
        elif (manifest_path.parent / recorded_datums).resolve() != datums_path:
            errors.append("manifest_datums_path_mismatch")
        if schema == PSD_SPARSE_DATUM_MANIFEST_SCHEMA_VERSION:
            recorded_identity = _mapping(manifest.get("artifact_identity")).get("datums")
            try:
                if not recorded_identity or artifact_identity(datums_path) != recorded_identity:
                    errors.append("manifest_datums_identity_mismatch")
            except (OSError, ValueError):
                errors.append("manifest_datums_identity_mismatch")
        else:
            recorded_sha256 = _text(_mapping(manifest.get("artifact_sha256")).get("datums"))
            if not recorded_sha256:
                errors.append("manifest_datums_sha256_missing")
            elif datums_path.is_file() and recorded_sha256 != sha256_file(datums_path):
                errors.append("manifest_datums_sha256_mismatch")

    identities: set[str] = set()
    kind_counts: Counter[str] = Counter()
    effective_mass: Counter[str] = Counter()
    max_observed_context = 0
    datum_rejections: list[dict[str, Any]] = []
    row_count = 0
    if datums_path.is_file():
        try:
            for index, row in enumerate(iter_jsonl(datums_path)):
                row_count += 1
                error = _datum_error(row, topk=expected_topk, max_context=max_context)
                target_id = _text(row.get("target_id"))
                if not target_id:
                    error = error or "datum_target_id_missing"
                elif target_id in identities:
                    error = error or "datum_target_id_duplicate"
                identities.add(target_id)
                if error:
                    datum_rejections.append(
                        {"row_index": index, "target_id": target_id, "reason": error}
                    )
                    continue
                kind = _text(row.get("kind"))
                kind_counts[kind] += 1
                effective_mass[kind] += float(row["row_weight"])
                max_observed_context = max(max_observed_context, len(row["input_ids"]))
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("datums_jsonl_invalid")

    if not row_count:
        errors.append("datums_empty")
    if datum_rejections:
        errors.append("datum_contract_rejections")
    if set(kind_counts) != {"repair", "preserve"}:
        errors.append("repair_and_preserve_datums_required")
    expected_count = _integer(
        _mapping(manifest.get("counts")).get("candidate_datums")
    )
    if manifest and expected_count != row_count:
        errors.append("manifest_datum_count_mismatch")

    return {
        "schema_version": PSD_TRAINING_INPUT_GATE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "error_count": len(errors),
        "errors": list(dict.fromkeys(errors)),
        "datums": {
            "path": str(datums_path),
            "identity": artifact_identity(datums_path) if datums_path.is_file() else None,
            "rows": row_count,
            "by_kind": dict(sorted(kind_counts.items())),
            "effective_row_mass_by_kind": dict(sorted(effective_mass.items())),
            "max_observed_input_tokens": max_observed_context,
            "lengths": _mapping(manifest.get("lengths")),
        },
        "manifest": {
            "path": str(manifest_path),
            "identity": artifact_identity(manifest_path) if manifest_path.is_file() else None,
            "schema_version": manifest.get("schema_version"),
            "status": manifest.get("status"),
        },
        "expected_topk": expected_topk,
        "max_context": max_context,
        "datum_rejections": datum_rejections,
    }
