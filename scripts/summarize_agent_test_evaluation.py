#!/usr/bin/env python3
"""Summarize one or more evaluator-only Agent private-gold audits."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eviscene.eval.private_gold_metrics import (
    agent_private_gold_category,
    agent_private_gold_category_counts,
)


SCHEMA_VERSION = "ifv-agent-test-evaluation-summary-v1"
VALID_VERDICTS = frozenset({"real", "fake"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _case_id(row: Mapping[str, Any]) -> str:
    value = str(
        row.get("unified_case_id")
        or row.get("archive_source_version_id")
        or row.get("case_id")
        or ""
    ).strip()
    if not value:
        raise ValueError("row lacks a stable case ID")
    return value


def _parse_input(value: str) -> tuple[str, Path]:
    name, separator, rendered_path = value.partition("=")
    if not separator or not name.strip() or not rendered_path.strip():
        raise argparse.ArgumentTypeError(
            "--input must be NAME=PATH, for example old100=/tmp/audit-results.jsonl"
        )
    return name.strip(), Path(rendered_path.strip()).expanduser().resolve()


def _metric_payload(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [
        row
        for row in rows
        if row.get("status") == "completed"
        and str(row.get("gold_verdict") or "").lower() in VALID_VERDICTS
        and str(row.get("candidate_verdict") or "").lower() in VALID_VERDICTS
    ]
    matrix = {
        expected: {predicted: 0 for predicted in sorted(VALID_VERDICTS)}
        for expected in sorted(VALID_VERDICTS)
    }
    for row in completed:
        expected = str(row["gold_verdict"]).lower()
        predicted = str(row["candidate_verdict"]).lower()
        matrix[expected][predicted] += 1

    total = sum(sum(predictions.values()) for predictions in matrix.values())
    correct = sum(matrix[label][label] for label in matrix)
    recall = {
        label: (
            matrix[label][label] / sum(matrix[label].values())
            if sum(matrix[label].values())
            else None
        )
        for label in matrix
    }
    available_recalls = [value for value in recall.values() if value is not None]
    return {
        "evaluated": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "balanced_accuracy": (
            sum(available_recalls) / len(available_recalls)
            if available_recalls
            else None
        ),
        "recall_by_gold_verdict": recall,
        "confusion_gold_rows_candidate_columns": matrix,
    }


def _route_breakdown(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("construction_subroute") or "unknown")].append(row)
    output: dict[str, dict[str, Any]] = {}
    for route, route_rows in sorted(groups.items()):
        metrics = _metric_payload(route_rows)
        categories = agent_private_gold_category_counts(route_rows)
        output[route] = {
            "selected": len(route_rows),
            "completed": sum(row.get("status") == "completed" for row in route_rows),
            "engineering_or_judge_errors": sum(
                row.get("status") != "completed" for row in route_rows
            ),
            **metrics,
            "three_way_private_gold_categories": categories,
        }
    return output


def summarize_agent_test_evaluation(
    *,
    inputs: list[tuple[str, Path]],
    test_manifest: Path,
    output_dir: Path,
    expected_case_count: int | None = None,
) -> dict[str, Any]:
    """Join audit records with test metadata and calculate reported metrics."""

    test_manifest = test_manifest.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    manifest_rows = _read_jsonl(test_manifest)
    manifest_by_case: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        case_id = _case_id(row)
        if case_id in manifest_by_case:
            raise ValueError(f"duplicate case ID in test manifest: {case_id}")
        manifest_by_case[case_id] = row

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_counts: dict[str, int] = {}
    for source_name, path in inputs:
        if source_name in source_counts:
            raise ValueError(f"duplicate input name: {source_name}")
        source_rows = _read_jsonl(path)
        source_counts[source_name] = len(source_rows)
        for row in source_rows:
            case_id = _case_id(row)
            if case_id in seen:
                raise ValueError(f"case appears in multiple audit inputs: {case_id}")
            seen.add(case_id)
            manifest_row = manifest_by_case.get(case_id)
            if manifest_row is None:
                raise ValueError(f"audit case is absent from test manifest: {case_id}")
            combined = dict(row)
            combined["case_id"] = case_id
            combined["audit_source"] = source_name
            for field in (
                "construction_subroute",
                "production_class",
                "target_route",
                "capability_cell",
                "factual_status",
            ):
                combined[field] = manifest_row.get(field)
            combined["private_gold_category"] = agent_private_gold_category(combined)
            merged.append(combined)
    if expected_case_count is not None and len(merged) != expected_case_count:
        raise ValueError(
            f"expected {expected_case_count} audit rows, found {len(merged)}"
        )

    merged.sort(key=lambda row: row["case_id"])
    status_counts = Counter(str(row.get("status") or "unknown") for row in merged)
    reason_quality = Counter(
        str(row.get("reason_quality") or "unknown")
        for row in merged
        if row.get("status") == "completed"
    )
    fact_alignment = Counter(
        str(row.get("fact_alignment") or "unknown")
        for row in merged
        if row.get("status") == "completed"
    )
    failure_modes = Counter(
        str(mode)
        for row in merged
        if row.get("status") == "completed"
        for mode in (row.get("failure_modes") or [])
        if str(mode).strip()
    )
    wrong_rows = [
        row
        for row in merged
        if row.get("private_gold_category") == "wrong_verdict"
    ]
    insufficient_rows = [
        row
        for row in merged
        if row.get("private_gold_category") == "correct_verdict_insufficient_evidence"
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "test_manifest": str(test_manifest),
        "inputs": {name: str(path) for name, path in inputs},
        "input_row_counts": source_counts,
        "selected": len(merged),
        "status_counts": dict(status_counts),
        "engineering_or_judge_errors": sum(
            row.get("status") != "completed" for row in merged
        ),
        "binary_metrics": _metric_payload(merged),
        "three_way_private_gold_categories": agent_private_gold_category_counts(
            merged
        ),
        "by_construction_subroute": _route_breakdown(merged),
        "failure_analysis": {
            "wrong_verdict_count": len(wrong_rows),
            "correct_but_insufficient_evidence_count": len(insufficient_rows),
            "wrong_verdict_by_subroute": dict(
                Counter(
                    str(row.get("construction_subroute") or "unknown")
                    for row in wrong_rows
                )
            ),
            "wrong_verdict_fact_alignment": dict(
                Counter(str(row.get("fact_alignment") or "unknown") for row in wrong_rows)
            ),
            "insufficient_evidence_reason_quality": dict(
                Counter(
                    str(row.get("reason_quality") or "unknown")
                    for row in insufficient_rows
                )
            ),
            "all_completed_reason_quality": dict(reason_quality),
            "all_completed_fact_alignment": dict(fact_alignment),
            "judge_failure_modes": dict(failure_modes),
        },
    }
    _write_jsonl(output_dir / "combined-audit-results.jsonl", merged)
    result["combined_audit_results"] = str(
        output_dir / "combined-audit-results.jsonl"
    )
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=_parse_input)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-case-count", type=int)
    args = parser.parse_args()
    if args.expected_case_count is not None and args.expected_case_count < 1:
        parser.error("--expected-case-count must be positive")
    result = summarize_agent_test_evaluation(
        inputs=args.input,
        test_manifest=Path(args.test_manifest),
        output_dir=Path(args.output_dir),
        expected_case_count=args.expected_case_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
