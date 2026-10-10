"""Build and load evaluator-private gold sidecars for unified datasets."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from eviscene.trajectory.sft_eligibility import build_sft_target


EVALUATOR_PRIVATE_GOLD_SCHEMA_VERSION = "ifv-evaluator-private-gold-v1"
PRIVATE_GOLD_FIELDS = (
    "target_claim",
    "primary_claim",
    "claim_atom",
    "decisive_visual_atom",
    "automatic_qa",
    "evidence",
    "source_url",
    "assignment_id",
    "generation_prompt_id",
    "construction_subroute",
    "target_subtype",
    "target_capability_cell",
    "factual_status",
)
_IDENTITY_KEYS = (
    "case_id",
    "unified_case_id",
    "archive_source_version_id",
    "candidate_id",
    "assignment_id",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _present(value: Any) -> bool:
    return value not in (None, "", [], {})


def stable_case_id(row: Mapping[str, Any]) -> str:
    """Return the runtime-compatible primary identity for a dataset row."""

    for key in (
        "case_id",
        "unified_case_id",
        "archive_source_version_id",
        "candidate_id",
    ):
        value = _text(row.get(key))
        if value:
            return value
    raise ValueError("dataset row lacks case_id/archive_source_version_id/candidate_id")


def _aliases(row: Mapping[str, Any], *, canonical_case_id: str) -> list[str]:
    values = [canonical_case_id]
    values.extend(_text(row.get(key)) for key in _IDENTITY_KEYS)
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def private_gold_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index canonical records and only unambiguous compatibility aliases."""

    result: dict[str, dict[str, Any]] = {}
    alias_rows: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        canonical = _text(row.get("case_id"))
        if not canonical:
            canonical = stable_case_id(row)
            row["case_id"] = canonical
        previous = result.get(canonical)
        if previous is not None and previous != row:
            raise ValueError(f"duplicate private-gold case_id: {canonical!r}")
        result[canonical] = row
        aliases = row.get("case_id_aliases")
        if not isinstance(aliases, list):
            aliases = _aliases(row, canonical_case_id=canonical)
        for value in aliases:
            text = _text(value)
            if text:
                alias_rows.setdefault(text, []).append(row)

    for alias, matches in alias_rows.items():
        if alias not in result and len(matches) == 1:
            result[alias] = matches[0]
    return result


def build_evaluator_private_gold_records(
    *,
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    archive_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge archive-only gold into test/train manifests and validate all rows."""

    expected_splits = {"test", "train"}
    unexpected = set(split_rows) - expected_splits
    if unexpected:
        raise ValueError(f"unsupported splits: {sorted(unexpected)}")
    missing = expected_splits - set(split_rows)
    if missing:
        raise ValueError(f"missing required splits: {sorted(missing)}")

    archive_index: dict[str, dict[str, Any]] = {}
    for raw in archive_rows:
        row = dict(raw)
        key = _text(row.get("archive_source_version_id"))
        if not key:
            continue
        if key in archive_index and archive_index[key] != row:
            raise ValueError(f"duplicate archive_source_version_id: {key!r}")
        archive_index[key] = row

    records: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    split_stats: dict[str, dict[str, int]] = {}
    for split in ("test", "train"):
        stats: Counter[str] = Counter()
        for raw in split_rows[split]:
            manifest_row = dict(raw)
            case_id = stable_case_id(manifest_row)
            if case_id in seen_case_ids:
                raise ValueError(f"train/test duplicate stable case_id: {case_id!r}")
            seen_case_ids.add(case_id)
            archive_key = _text(manifest_row.get("archive_source_version_id"))
            archive_row = archive_index.get(archive_key)
            merged = dict(manifest_row)
            if archive_row is not None:
                stats["archive_matched"] += 1
                for field in PRIVATE_GOLD_FIELDS:
                    if not _present(merged.get(field)) and _present(
                        archive_row.get(field)
                    ):
                        merged[field] = archive_row[field]
                        stats[f"filled_{field}"] += 1
            elif archive_key:
                stats["archive_unavailable"] += 1
            else:
                stats["no_archive_identity"] += 1

            merged["case_id"] = case_id
            aliases = _aliases(merged, canonical_case_id=case_id)
            try:
                private_target = build_sft_target(merged)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{split}:{case_id} cannot build evaluator-private target: {exc}"
                ) from exc
            image_fact = private_target.get("image_fact")
            if not isinstance(image_fact, Mapping) or not any(
                _text(image_fact.get(key))
                for key in (
                    "statement",
                    "subject",
                    "event_or_context",
                    "relation",
                    "depicted_value",
                )
            ):
                raise ValueError(
                    f"{split}:{case_id} lacks a usable evaluator-private fact"
                )
            record = {
                "schema_version": EVALUATOR_PRIVATE_GOLD_SCHEMA_VERSION,
                "case_id": case_id,
                "split": split,
                "case_id_aliases": aliases,
                "archive_source_version_id": _text(
                    merged.get("archive_source_version_id")
                ),
                "candidate_id": _text(merged.get("candidate_id")),
                "assignment_id": _text(merged.get("assignment_id")),
                "factual_status": _text(merged.get("factual_status")),
                "target_claim": merged.get("target_claim"),
                "primary_claim": merged.get("primary_claim"),
                "claim_atom": merged.get("claim_atom"),
                "decisive_visual_atom": merged.get("decisive_visual_atom"),
                "automatic_qa": merged.get("automatic_qa"),
                "evidence": merged.get("evidence"),
                "source_url": merged.get("source_url"),
                "generation_prompt_id": merged.get("generation_prompt_id"),
                "construction_subroute": merged.get("construction_subroute"),
                "target_subtype": merged.get("target_subtype"),
                "target_capability_cell": merged.get("target_capability_cell"),
                "private_target": private_target,
                "provenance": {
                    "manifest_gold_complete_before_archive": all(
                        _present(manifest_row.get(field))
                        for field in PRIVATE_GOLD_FIELDS
                        if field in {
                            "factual_status",
                            "target_claim",
                            "claim_atom",
                            "decisive_visual_atom",
                            "evidence",
                        }
                    ),
                    "archive_source_version_id": archive_key,
                    "archive_match_used": archive_row is not None,
                },
            }
            records.append(record)
            stats["records"] += 1
        split_stats[split] = dict(stats)

    # Constructing the index verifies canonical uniqueness and identifies aliases
    # that must not be used by evaluators because they are ambiguous.
    indexed = private_gold_index(records)
    canonical_count = len(records)
    alias_count = len(indexed) - canonical_count
    summary = {
        "schema_version": EVALUATOR_PRIVATE_GOLD_SCHEMA_VERSION,
        "record_count": canonical_count,
        "split_counts": {
            split: sum(item["split"] == split for item in records)
            for split in ("test", "train")
        },
        "split_stats": split_stats,
        "archive_rows_available": len(archive_index),
        "canonical_case_ids": canonical_count,
        "unambiguous_compatibility_aliases": alias_count,
        "all_records_have_private_target": True,
    }
    return records, summary


def case_alias_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Return only aliases safe for an evaluator to resolve automatically."""

    indexed = private_gold_index(records)
    canonical = {
        _text(row.get("case_id"))
        for row in records
        if _text(row.get("case_id"))
    }
    rows: list[dict[str, str]] = []
    for alias, record in sorted(indexed.items()):
        case_id = _text(record.get("case_id"))
        if alias == case_id or alias not in canonical:
            rows.append({"alias": alias, "case_id": case_id})
    return rows
