"""Build framework-neutral sparse top-K PSD cross-entropy datums."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .artifact_receipts import artifact_identity
from .io import (
    canonical_json,
    iter_jsonl,
    require_new_or_empty,
    write_json,
)
from .psd import validate_topk_by_position
from .psd_modality import require_text_only_psd


PSD_SPARSE_DATUM_SCHEMA_VERSION = "ifv-psd-sparse-topk-datum-v3"
PSD_COMPACT_DATUM_SCHEMA_VERSION = "ifv-psd-sparse-topk-datum-v4"
PSD_SPARSE_DATUM_MANIFEST_SCHEMA_VERSION = "ifv-psd-sparse-topk-manifest-v5"
PSD_WEIGHTING_POLICY = "per_target"


def _trainer_json_media(media: Mapping[str, Any]) -> dict[str, Any]:
    """Keep mixed legacy/compact media rows compatible with Arrow JSON.

    Hugging Face infers nested struct fields from the first JSON block. Legacy
    tensor bindings precede compact image bindings in a mixed PSD bank, so the
    later-only ``image_paths`` field otherwise makes dataset loading fail. The
    legacy loader ignores this compatibility field and still consumes the
    exact frozen tensor artifact from ``path``.
    """
    result = dict(media)
    if (result.get("schema_version") == "ifv-psd-media-v1"
            and "image_paths" not in result):
        result["image_paths"] = [str(result["path"])]
    return result


def compact_datum(datum):
    """Store only supervised positions; long prompt zero matrices stay off disk."""
    result = {key: value for key, value in datum.items() if key not in {"weights", "target_tokens"}}
    result.update(schema_version=PSD_COMPACT_DATUM_SCHEMA_VERSION,
        sparse_target_tokens=[datum["target_tokens"][index] for index in datum["loss_positions"]],
        sparse_weights=[datum["weights"][index] for index in datum["loss_positions"]])
    return result


def expand_datum_targets(datum):
    """Expand one compact row immediately before collation, not the full bank."""
    if "sparse_target_tokens" not in datum and "sparse_weights" not in datum:
        return datum.get("target_tokens"), datum.get("weights")
    positions, topk = datum.get("loss_positions"), datum.get("topk")
    length = len(datum["input_ids"])
    if (type(topk) is not int or topk < 1 or not isinstance(positions, list) or not positions
            or any(type(index) is not int or not 0 <= index < length for index in positions)
            or positions != sorted(set(positions))):
        raise ValueError("invalid compact PSD prediction positions")
    compact_targets, compact_weights = datum.get("sparse_target_tokens"), datum.get("sparse_weights")
    if (not isinstance(compact_targets, list) or not isinstance(compact_weights, list)
            or len(compact_targets) != len(positions) or len(compact_weights) != len(positions)
            or "target_tokens" in datum or "weights" in datum):
        raise ValueError("invalid or ambiguous compact PSD target arrays")
    targets = [[0] * topk for _ in range(length)]
    weights = [[0.0] * topk for _ in range(length)]
    for position, tokens, values in zip(positions, compact_targets, compact_weights, strict=True):
        targets[position], weights[position] = tokens, values
    return targets, weights


def _text(value: Any) -> str:
    return str(value or "").strip()


def _required_int_list(value: Any, *, field: str) -> list[int]:
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


def _positive_weight(value: Any, *, field: str) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return weight


def _topk_entry(value: Any) -> tuple[int, float]:
    if isinstance(value, Mapping):
        token_id = value.get("token_id")
        probability = value.get("probability", value.get("prob"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        token_id, probability = value
    else:
        raise ValueError("top-k entry has invalid shape")
    if isinstance(token_id, bool):
        raise ValueError("top-k token ID must be numeric")
    try:
        return int(token_id), float(probability)
    except (TypeError, ValueError) as exc:
        raise ValueError("top-k entry contains invalid values") from exc


def build_sparse_topk_datum(
    target: Mapping[str, Any],
    *,
    topk: int,
    effective_row_weight: float | None = None,
    max_sequence_length: int = 131_072,
) -> dict[str, Any]:
    """Project one complete PSD target onto causal prediction positions.

    ``input_ids`` excludes the final completion token, matching next-token
    training.  Only positions predicting completion tokens receive non-zero
    weights; prompt and tool-observation tokens are never direct loss targets.
    """

    if _text(target.get("target_status")) != "complete":
        raise ValueError("target_status must be complete")
    from .psd_capture_semantics import RAW_POLICY_LOGPROBS
    if target.get("teacher_logprob_semantics") != RAW_POLICY_LOGPROBS:
        raise ValueError("teacher_logprob_semantics_unattested: re-score exact original tokens")
    target_id = _text(target.get("target_id"))
    if not target_id:
        raise ValueError("target_id_missing")
    kind = _text(target.get("kind"))
    if kind not in {"repair", "preserve"}:
        raise ValueError(f"unsupported target kind: {kind}")

    prompt_ids = _required_int_list(
        target.get("student_prompt_ids"),
        field="student_prompt_ids",
    )
    completion_ids = _required_int_list(
        target.get("completion_ids"),
        field="completion_ids",
    )
    combined = [*prompt_ids, *completion_ids]
    require_text_only_psd(target, combined)
    if len(combined) > max_sequence_length:
        raise ValueError(
            f"sequence_length_exceeds_limit:{len(combined)}>{max_sequence_length}"
        )
    if len(combined) < 2:
        raise ValueError("sequence must contain at least two tokens")

    distributions = target.get("teacher_topk_by_position")
    validate_topk_by_position(completion_ids, distributions, topk=topk)
    source_row_weight = _positive_weight(
        target.get("row_weight", 1.0),
        field="row_weight",
    )
    row_weight = _positive_weight(
        source_row_weight
        if effective_row_weight is None
        else effective_row_weight,
        field="effective_row_weight",
    )

    input_ids = combined[:-1]
    target_tokens = [[0] * topk for _ in input_ids]
    weights = [[0.0] * topk for _ in input_ids]
    loss_positions: list[int] = []
    for completion_offset, raw_entries in enumerate(distributions):
        prediction_position = len(prompt_ids) + completion_offset - 1
        if prediction_position < 0 or prediction_position >= len(input_ids):
            raise ValueError(
                f"completion prediction position out of range: {prediction_position}"
            )
        entries = [_topk_entry(entry) for entry in raw_entries]
        for rank, (token_id, probability) in enumerate(entries):
            target_tokens[prediction_position][rank] = token_id
            weights[prediction_position][rank] = probability * row_weight
        loss_positions.append(prediction_position)

    return {
        "schema_version": PSD_SPARSE_DATUM_SCHEMA_VERSION,
        "target_id": target_id,
        "kind": kind,
        "topk": topk,
        "teacher_logprob_semantics": RAW_POLICY_LOGPROBS,
        "input_ids": input_ids,
        **({"psd_media": _trainer_json_media(target["psd_media"])}
           if target.get("psd_media") else {}),
        "target_tokens": target_tokens,
        "weights": weights,
        "loss_positions": loss_positions,
        "source_row_weight": source_row_weight,
        "row_weight": row_weight,
        "sequence_tokens": len(combined),
        "input_tokens": len(input_ids),
        "completion_tokens": len(completion_ids),
    }


def _length_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    ordered = sorted(values)

    def percentile(percent: int) -> int:
        rank = max(0, math.ceil((percent / 100) * len(ordered)) - 1)
        return ordered[rank]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": ordered[-1],
    }


def _context_buckets(values: list[int]) -> dict[str, int]:
    bounds = (8_192, 16_384, 32_768, 65_536, 131_072)
    result = {
        f"le_{bound}": sum(value <= bound for value in values)
        for bound in bounds
    }
    result["gt_131072"] = sum(value > bounds[-1] for value in values)
    return result


def build_sparse_topk_package(
    *,
    targets_path: Path,
    output_dir: Path,
    topk: int = 20,
    max_sequence_length: int = 131_072,
    require_both_kinds: bool = True,
) -> dict[str, Any]:
    """Create a fail-closed sparse top-K training interface package."""

    if topk < 1:
        raise ValueError("topk must be positive")
    if max_sequence_length < 2:
        raise ValueError("max_sequence_length must be at least two")
    require_new_or_empty(output_dir)
    target_ids: set[str] = set()
    input_kinds: set[str] = set()
    input_target_count = 0
    datum_count = 0
    rejection_count = 0
    effective_mass = Counter()
    loss_positions = Counter()
    target_kind_counts = Counter()
    input_lengths: list[int] = []
    sequence_lengths: list[int] = []
    completion_lengths: list[int] = []
    candidate_path = output_dir / "candidate_datums.jsonl"
    rejection_path = output_dir / "rejections.jsonl"
    with (candidate_path.open("w", encoding="utf-8") as candidate_handle,
          rejection_path.open("w", encoding="utf-8") as rejection_handle):
        for row_index, target in enumerate(iter_jsonl(targets_path)):
            input_target_count += 1
            target_id = _text(target.get("target_id"))
            if not target_id:
                raise ValueError("all targets must have target_id")
            if target_id in target_ids:
                raise ValueError("PSD target IDs must be unique")
            target_ids.add(target_id)
            kind = _text(target.get("kind"))
            input_kinds.add(kind)
            source_weight = target.get("row_weight", 1.0)
            try:
                datum = compact_datum(build_sparse_topk_datum(target, topk=topk,
                    effective_row_weight=float(source_weight),
                    max_sequence_length=max_sequence_length))
            except (KeyError, TypeError, ValueError) as exc:
                rejection = {"row_index": row_index, "target_id": target_id,
                    "kind": kind, "reason": str(exc)}
                rejection_handle.write(canonical_json(rejection) + "\n")
                rejection_count += 1
                continue
            candidate_handle.write(canonical_json(datum) + "\n")
            datum_count += 1
            target_kind_counts[datum["kind"]] += 1
            effective_mass[datum["kind"]] += float(datum["row_weight"])
            loss_positions[datum["kind"]] += len(datum["loss_positions"])
            input_lengths.append(int(datum["input_tokens"]))
            sequence_lengths.append(int(datum["sequence_tokens"]))
            completion_lengths.append(int(datum["completion_tokens"]))
    if not input_target_count:
        raise ValueError("PSD target input is empty")
    missing_kinds = sorted({"repair", "preserve"} - input_kinds)
    source_kind_gate = not require_both_kinds or not missing_kinds
    ready = datum_count == input_target_count and not rejection_count and source_kind_gate
    artifacts = {
        "candidate_datums": "candidate_datums.jsonl",
        "rejections": "rejections.jsonl",
    }
    if ready:
        # Two artifact names are useful for diagnostics, not two physical copies.
        import os
        try:
            os.link(output_dir / "candidate_datums.jsonl", output_dir / "datums.jsonl")
        except OSError:
            import shutil
            shutil.copyfile(output_dir / "candidate_datums.jsonl", output_dir / "datums.jsonl")
        artifacts["datums"] = "datums.jsonl"
    manifest = {
        "schema_version": PSD_SPARSE_DATUM_MANIFEST_SCHEMA_VERSION,
        "source": {
            "targets": str(targets_path),
            "targets_identity": artifact_identity(targets_path),
        },
        "topk": topk,
        "max_sequence_length": max_sequence_length,
        "weighting_policy": PSD_WEIGHTING_POLICY,
        "target_layout": "completion_positions_only",
        "aggregate_source_rebalancing": False,
        "require_both_kinds": require_both_kinds,
        "missing_source_kinds": missing_kinds,
        "counts": {
            "input_targets": input_target_count,
            "candidate_datums": datum_count,
            "rejections": rejection_count,
            "loss_positions": sum(loss_positions.values()),
        },
        "targets_by_kind": dict(sorted(target_kind_counts.items())),
        "loss_positions_by_kind": dict(sorted(loss_positions.items())),
        "effective_row_mass_by_kind": dict(sorted(effective_mass.items())),
        "lengths": {
            "input_tokens": _length_summary(input_lengths),
            "sequence_tokens": _length_summary(sequence_lengths),
            "completion_tokens": _length_summary(completion_lengths),
            "input_context_buckets": _context_buckets(input_lengths),
        },
        "artifacts": artifacts,
        "artifact_identity": {
            "candidate_datums": artifact_identity(output_dir / "candidate_datums.jsonl"),
            "datums": artifact_identity(output_dir / "datums.jsonl") if ready else None,
        },
        "status": (
            "ready_for_trainer"
            if ready
            else "blocked_invalid_target"
            if rejection_count or datum_count != input_target_count
            else "blocked_missing_source_kind"
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
