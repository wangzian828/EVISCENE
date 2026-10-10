#!/usr/bin/env python3
"""Run a recoverable evaluator-only Agent rollout from a public test release.

This wrapper deliberately reuses the durable engineering-retry collector used
for teacher rollout.  It differs in two important ways:

* the input must be a ``development_subset`` release marked
  ``training_prohibited=true``;
* it never runs SFT eligibility, reward scoring, or training-data staging.

Every successful trace is retained and linked into ``rollouts/test/merged``.
Only case IDs without a terminal success are retried.  ``agent-results.jsonl``
is the normalized, one-row-per-case input for the post-hoc private-gold judge.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.trajectory.run_teacher_rollout_autopilot import (
    _read_json,
    _read_jsonl,
    _run_engineering_retries,
    _write_json,
    _write_jsonl,
)
from eviscene.eval.public_release import load_public_release
from eviscene.eval.run_artifacts import load_jsonl_objects


SCHEMA_VERSION = "ifv-agent-test-rollout-v1"
VALID_VERDICTS = frozenset({"real", "fake"})


def _read_release_case_ids(benchmark: Path) -> list[str]:
    rows = load_jsonl_objects(benchmark)
    case_ids = [str(row.get("case_id") or "").strip() for row in rows]
    if not case_ids or any(not case_id for case_id in case_ids):
        raise ValueError("test release contains an empty case_id")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("test release contains duplicate case_id values")
    return case_ids


def _test_release_is_training_prohibited(release_root: Path) -> None:
    manifest_path = release_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("release_stage") != "development_subset":
        raise ValueError(
            "test rollout only accepts release_stage=development_subset; "
            "refusing a teacher/train release"
        )
    if manifest.get("training_prohibited") is not True:
        raise ValueError(
            "test rollout release must set training_prohibited=true"
        )


def _build_agent_results(
    *,
    merged_dir: Path,
    target_case_ids: list[str],
) -> list[dict[str, Any]]:
    provenance_path = merged_dir / "trace-provenance.jsonl"
    provenance_rows = _read_jsonl(provenance_path)
    by_case: dict[str, Mapping[str, Any]] = {}
    for row in provenance_rows:
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError(f"trace provenance row lacks case_id: {provenance_path}")
        if case_id in by_case:
            raise ValueError(f"duplicate merged trace provenance: {case_id}")
        by_case[case_id] = row

    target = set(target_case_ids)
    unexpected = sorted(set(by_case) - target)
    if unexpected:
        raise ValueError(f"merged trace has unexpected case ID: {unexpected[0]}")

    results: list[dict[str, Any]] = []
    for case_id in target_case_ids:
        provenance = by_case.get(case_id)
        if provenance is None:
            results.append(
                {
                    "case_id": case_id,
                    "status": "error",
                    "error": "engineering attempts exhausted without terminal success",
                    "training_prohibited": True,
                }
            )
            continue
        trace_name = Path(str(provenance.get("source_trace") or "")).name
        trace_path = merged_dir / "traces" / trace_name
        if not trace_path.is_file():
            raise FileNotFoundError(f"merged trace is missing: {trace_path}")
        trace = _read_json(trace_path)
        verdict = str(trace.get("verdict") or "").lower()
        termination = str(trace.get("termination") or "")
        if termination != "success" or verdict not in VALID_VERDICTS:
            raise ValueError(
                "merged trace is not terminal success: "
                f"{case_id} termination={termination!r} verdict={verdict!r}"
            )
        trace_case_id = str(trace.get("case_id") or "").strip()
        if trace_case_id != case_id:
            raise ValueError(
                f"merged trace case mismatch: result={case_id}, trace={trace_case_id}"
            )
        results.append(
            {
                "case_id": case_id,
                "status": "success",
                "trace_path": str(trace_path.relative_to(merged_dir).as_posix()),
                "verdict": verdict,
                "confidence": trace.get("confidence"),
                "termination": termination,
                "source_attempt": provenance.get("source_attempt"),
                "source_trace": provenance.get("source_trace"),
                "training_prohibited": True,
            }
        )
    return results


def run_agent_test_rollout(
    *,
    benchmark: Path,
    output_dir: Path,
    profile: str,
    concurrency: int,
    base_seed: int,
    timeout: float,
    maximum_attempts: int,
) -> dict[str, Any]:
    """Run test cases with durable engineering recovery only."""

    benchmark = benchmark.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    release = load_public_release(benchmark)
    _test_release_is_training_prohibited(release.root)
    target_case_ids = _read_release_case_ids(benchmark)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output_dir}")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if maximum_attempts < 1:
        raise ValueError("maximum_attempts must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        output_dir / "run-config.json",
        {
            "schema_version": SCHEMA_VERSION,
            "benchmark": str(benchmark),
            "release_id": release.release_id,
            "release_stage": release.release_stage,
            "training_prohibited": True,
            "profile": profile,
            "concurrency": concurrency,
            "base_seed": base_seed,
            "timeout": timeout,
            "maximum_attempts": maximum_attempts,
            "target_case_count": len(target_case_ids),
        },
    )
    (output_dir / "target-case-list.txt").write_text(
        "".join(f"{case_id}\n" for case_id in target_case_ids),
        encoding="utf-8",
    )
    merged_dir, merged_manifest = _run_engineering_retries(
        group_name="test",
        benchmark=benchmark,
        pipeline_dir=output_dir,
        target_ids=target_case_ids,
        profile=profile,
        rollout_concurrency=concurrency,
        base_seed=base_seed,
        timeout=timeout,
        maximum_attempts=maximum_attempts,
        candidates_per_case=1,
    )
    results = _build_agent_results(
        merged_dir=merged_dir,
        target_case_ids=target_case_ids,
    )
    _write_jsonl(output_dir / "agent-results.jsonl", results)
    result = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": str(benchmark),
        "release_id": release.release_id,
        "training_prohibited": True,
        "target_case_count": len(target_case_ids),
        "successful_case_count": sum(
            row.get("status") == "success" for row in results
        ),
        "engineering_error_count": sum(
            row.get("status") != "success" for row in results
        ),
        "merged_dir": str(merged_dir),
        "merged_manifest": str(merged_dir / "run_manifest.json"),
        "agent_results": str(output_dir / "agent-results.jsonl"),
        "retry_manifest": merged_manifest,
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an isolated Agent test release with engineering-only retries."
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default="teacher-gemini")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=1729)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--maximum-attempts", type=int, default=4)
    args = parser.parse_args()
    result = run_agent_test_rollout(
        benchmark=Path(args.benchmark),
        output_dir=Path(args.output_dir),
        profile=args.profile,
        concurrency=args.concurrency,
        base_seed=args.base_seed,
        timeout=args.timeout,
        maximum_attempts=args.maximum_attempts,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
