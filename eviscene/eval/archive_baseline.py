"""Lightweight evaluator-side metrics for archive-backed baseline rollouts.

This module is intentionally separate from SFT/process scoring.  It reads the
private labels in a historical archive only after an Agent rollout has
completed, and writes baseline accuracy artifacts.  Archive labels never enter
the runtime case projection or any provider request.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from eviscene.eval.run_artifacts import write_json, write_jsonl


ARCHIVE_BASELINE_METRICS_SCHEMA_VERSION = "ifv-archive-baseline-metrics-v1"
ARCHIVE_BASELINE_CASE_SCHEMA_VERSION = "ifv-archive-baseline-case-v1"
_VALID_VERDICTS = frozenset({"real", "fake"})
_STATUS_TO_VERDICT = {"supported": "real", "refuted": "fake"}


def _runtime_case_id(row: Mapping[str, Any]) -> str:
    value = str(
        row.get("archive_source_version_id")
        or row.get("candidate_id")
        or row.get("assignment_id")
        or ""
    ).strip()
    if not value:
        raise ValueError("archive candidate row lacks a runtime case identity")
    return value


def _load_archive_summary(candidate_path: Path) -> Dict[str, Any]:
    summary_path = candidate_path.parent / "archive-summary.json"
    if not summary_path.is_file():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"archive summary must be an object: {summary_path}")
    return payload


def load_archive_baseline_gold(candidate_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load only evaluator-side label metadata keyed by runtime case ID."""

    path = candidate_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"archive candidate file does not exist: {path}")

    indexed: Dict[str, Dict[str, Any]] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"archive candidate line {line_number} must be an object")
        case_id = _runtime_case_id(row)
        if case_id in indexed:
            raise ValueError(f"duplicate archive runtime case ID: {case_id}")
        factual_status = str(row.get("factual_status") or "").strip()
        expected_verdict = _STATUS_TO_VERDICT.get(factual_status)
        if expected_verdict is None:
            raise ValueError(
                f"archive candidate line {line_number} has unsupported "
                f"factual_status={factual_status!r}"
            )
        indexed[case_id] = {
            "case_id": case_id,
            "candidate_id": str(row.get("candidate_id") or "").strip() or None,
            "factual_status": factual_status,
            "gold_verdict": expected_verdict,
            "production_class": row.get("production_class"),
            "target_subtype": row.get("target_subtype"),
            "target_capability_cell": row.get("target_capability_cell"),
        }
    if not indexed:
        raise ValueError(f"archive candidate file is empty: {path}")
    return indexed


def _prediction(row: Mapping[str, Any]) -> str | None:
    verdict = str(row.get("verdict") or "").strip()
    status = str(row.get("status") or "").strip()
    termination = str(row.get("termination") or "").strip()
    error = str(row.get("error") or "").strip()
    if (
        verdict not in _VALID_VERDICTS
        or status != "success"
        or termination != "success"
        or error
    ):
        return None
    return verdict


def _engineering_error(row: Mapping[str, Any]) -> bool:
    return bool(
        str(row.get("error") or "").strip()
        or str(row.get("status") or "").strip() == "error"
        or str(row.get("termination") or "").strip() == "error"
        or str(row.get("verdict") or "").strip() == "error"
    )


def score_archive_baseline(
    run_results: Iterable[Mapping[str, Any]],
    candidate_path: Path,
    *,
    run_id: str | None = None,
    git_commit: str | None = None,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Score complete or partially completed baseline results against archive labels."""

    gold_index = load_archive_baseline_gold(candidate_path)
    case_metrics: list[Dict[str, Any]] = []
    seen: set[str] = set()

    for row in run_results:
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError("run result lacks case_id")
        if case_id in seen:
            raise ValueError(f"duplicate run result case_id: {case_id}")
        seen.add(case_id)
        gold = gold_index.get(case_id)
        if gold is None:
            raise ValueError(f"run result case_id is absent from archive: {case_id}")
        prediction = _prediction(row)
        engineering_error = _engineering_error(row)
        valid_prediction = prediction in _VALID_VERDICTS
        correct = bool(valid_prediction and prediction == gold["gold_verdict"])
        case_metrics.append(
            {
                "schema_version": ARCHIVE_BASELINE_CASE_SCHEMA_VERSION,
                "case_id": case_id,
                "candidate_id": gold["candidate_id"],
                "factual_status": gold["factual_status"],
                "gold_verdict": gold["gold_verdict"],
                "production_class": gold["production_class"],
                "target_subtype": gold["target_subtype"],
                "target_capability_cell": gold["target_capability_cell"],
                "prediction": prediction,
                "status": row.get("status"),
                "termination": row.get("termination"),
                "engineering_error": engineering_error,
                "valid_prediction": valid_prediction,
                "correct": correct,
                "episode_id": row.get("episode_id"),
                "trace_path": row.get("trace_path"),
                "time_taken": row.get("time_taken"),
                "total_tool_calls": row.get("total_tool_calls"),
                "llm_api_calls": row.get("llm_api_calls"),
                "error": row.get("error"),
            }
        )

    total = len(case_metrics)
    valid_count = sum(bool(row["valid_prediction"]) for row in case_metrics)
    correct_count = sum(bool(row["correct"]) for row in case_metrics)
    engineering_errors = sum(
        bool(row["engineering_error"]) for row in case_metrics
    )
    invalid_count = total - valid_count

    def rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 6) if denominator else 0.0

    predicted = Counter(
        str(row["prediction"] or "invalid") for row in case_metrics
    )
    gold = Counter(str(row["gold_verdict"]) for row in case_metrics)
    by_class: Dict[str, Dict[str, Any]] = {}
    for row in case_metrics:
        key = str(row["production_class"] or "unknown")
        bucket = by_class.setdefault(
            key,
            {"total": 0, "correct": 0, "valid_predictions": 0},
        )
        bucket["total"] += 1
        bucket["correct"] += int(bool(row["correct"]))
        bucket["valid_predictions"] += int(bool(row["valid_prediction"]))
    for bucket in by_class.values():
        bucket["raw_accuracy"] = rate(bucket["correct"], bucket["total"])
        bucket["valid_only_accuracy"] = rate(
            bucket["correct"],
            bucket["valid_predictions"],
        )

    summary: Dict[str, Any] = {
        "schema_version": ARCHIVE_BASELINE_METRICS_SCHEMA_VERSION,
        "run_id": run_id,
        "git_commit": git_commit,
        "archive_id": _load_archive_summary(candidate_path).get("archive_id"),
        "candidate_file": str(candidate_path.resolve()),
        "total_cases": total,
        "valid_predictions": valid_count,
        "invalid_or_missing_predictions": invalid_count,
        "correct": correct_count,
        "incorrect_valid_predictions": valid_count - correct_count,
        "engineering_errors": engineering_errors,
        "raw_accuracy": rate(correct_count, total),
        "valid_only_accuracy": rate(correct_count, valid_count),
        "engineering_error_rate": rate(engineering_errors, total),
        "invalid_prediction_rate": rate(invalid_count, total),
        "gold_distribution": dict(gold),
        "prediction_distribution": dict(predicted),
        "by_production_class": by_class,
        "case_metrics": "baseline_case_metrics.jsonl",
    }
    return summary, case_metrics


def write_archive_baseline_metrics(
    run_dir: Path,
    candidate_path: Path,
    run_results: Iterable[Mapping[str, Any]],
    *,
    run_id: str | None = None,
    git_commit: str | None = None,
) -> Dict[str, Any]:
    """Write baseline metrics beside a lightweight archive-backed run."""

    summary, case_metrics = score_archive_baseline(
        run_results,
        candidate_path,
        run_id=run_id,
        git_commit=git_commit,
    )
    run_dir = run_dir.expanduser().resolve()
    write_jsonl(run_dir / "baseline_case_metrics.jsonl", case_metrics)
    write_json(run_dir / "baseline_metrics.json", summary)
    return summary
