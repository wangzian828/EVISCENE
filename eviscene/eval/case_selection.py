from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from eviscene.eval.run_artifacts import load_jsonl_objects


def case_list(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    case_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("--case-list values must be unique")
    return case_ids


def select_samples(
    samples: list[Dict[str, Any]],
    *,
    requested_case_ids: list[str] | None,
    limit: int | None,
    shard_count: int = 1,
    shard_index: int = 0,
) -> list[Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        case_id = str(sample.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("runtime input row lacks case_id")
        if case_id in indexed:
            raise ValueError(f"runtime input contains duplicate case_id {case_id}")
        indexed[case_id] = sample
    if requested_case_ids:
        requested = [str(item).strip() for item in requested_case_ids]
        if any(not item for item in requested):
            raise ValueError("requested case_id values must be non-empty")
        if len(requested) != len(set(requested)):
            raise ValueError("requested case_id values must be unique")
        missing = [case_id for case_id in requested if case_id not in indexed]
        if missing:
            raise ValueError(
                "requested case_id values are absent from release: "
                + ", ".join(missing)
            )
        samples = [indexed[case_id] for case_id in requested]
    if shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must be in [0, shard-count)")
    if shard_count > 1:
        samples = [
            sample
            for index, sample in enumerate(
                sorted(samples, key=lambda item: str(item["case_id"]))
            )
            if index % shard_count == shard_index
        ]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1 when supplied")
        samples = samples[:limit]
    return samples


def metadata_index(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl_objects(path):
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError(f"metadata row lacks non-empty case_id: {path}")
        if case_id in indexed:
            raise ValueError(f"duplicate metadata row for case_id {case_id}")
        indexed[case_id] = dict(row)
    return indexed
