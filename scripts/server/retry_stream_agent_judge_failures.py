"""Retry explicitly rejected streaming-judge requests exactly once.

The original attempt receipts remain immutable.  This helper only creates an
``attempt-02`` intent/result pair for case directories that have an explicit
``failed.json`` receipt, no accepted ``result.json``, and an explicit HTTP 429
or 503 provider rejection. Existing successful judge results are never sent again.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
from pathlib import Path
import time
from typing import Any

from scripts.audit_agent_private_gold import _audit_one
from scripts.server.stream_agent_judges import (
    StreamingJudge,
    atomic_json,
    compact_record,
    stat_identity,
    terminal_judge,
)
from eviscene.integrations.gemini import GeminiInteractionsClient


RETRYABLE_ERROR_TYPES = {"GeminiInteractionsHTTPError"}
RETRYABLE_REJECTIONS = ("HTTP 429", "HTTP 503")


def explicitly_rejected(failure: dict[str, Any]) -> bool:
    return (
        failure.get("error_type") in RETRYABLE_ERROR_TYPES
        and any(code in str(failure.get("error") or "") for code in RETRYABLE_REJECTIONS)
    )


def retryable_failure(directory: Path) -> dict[str, Any]:
    """Return the frozen attempt-01 failure after enforcing replay safety."""
    if (directory / "result.json").exists():
        raise ValueError(f"case already has an accepted result: {directory}")
    required = [
        directory / "attempt-01.intent.json",
        directory / "attempt-01.result.json",
        directory / "failed.json",
    ]
    if any(not path.is_file() for path in required):
        raise ValueError(f"incomplete attempt-01 receipts: {directory}")
    failure = json.loads((directory / "failed.json").read_text(encoding="utf-8"))
    if failure.get("state") != "judge_attempt_failed":
        raise ValueError(f"unexpected failure state: {directory}")
    if not explicitly_rejected(failure):
        raise ValueError(f"only explicit HTTP 429/503 rejections may be retried: {directory}")
    return failure


def unresolved_failed_directories(judge: StreamingJudge) -> list[Path]:
    root = judge.output / "cases"
    if not root.exists():
        return []
    return sorted(
        path.parent
        for path in root.glob("*/failed.json")
        if not (path.parent / "result.json").exists()
    )


def retryable_rejection_directories(directories: list[Path]) -> list[Path]:
    """Select confirmed 429/503 rejections; timeouts may be ambiguous."""
    result: list[Path] = []
    for directory in directories:
        failure = json.loads((directory / "failed.json").read_text(encoding="utf-8"))
        if explicitly_rejected(failure):
            result.append(directory)
    return result


class TailRepair:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.judge = StreamingJudge(args)
        discovered = self.judge.discover()
        expected_agent_successes = (
            args.expected_agent_successes
            if args.expected_agent_successes is not None
            else args.expected_count
        )
        if len(discovered) != expected_agent_successes:
            raise ValueError(
                f"expected {expected_agent_successes} successful Agent traces, found {len(discovered)}"
            )
        self.all_failures = unresolved_failed_directories(self.judge)
        if (
            args.expected_total_failures is not None
            and len(self.all_failures) != args.expected_total_failures
        ):
            raise ValueError(
                f"expected {args.expected_total_failures} total judge failures, found {len(self.all_failures)}"
            )
        self.failures = retryable_rejection_directories(self.all_failures)
        if len(self.failures) != args.expected_failures:
            raise ValueError(
                f"expected {args.expected_failures} unresolved judge failures, found {len(self.failures)}"
            )
        self.failure_rows: dict[str, dict[str, Any]] = {}
        for directory in self.failures:
            failure = retryable_failure(directory)
            case_id = str(failure.get("case_id") or "")
            if self.judge.case_dir(case_id) != directory:
                raise ValueError(f"failure receipt case identity mismatch: {directory}")
            self.failure_rows[case_id] = failure
        binding = {
            "schema_version": "ifv-streaming-agent-judge-tail-repair-v2",
            # StreamingJudge validates and rewrites an identical small binding
            # during initialization.  Bind to its contents, not its resulting
            # mtime, so a credential-free preflight cannot invalidate a safe
            # retry before any provider request was sent.
            "judge_binding": json.loads(
                (self.judge.output / "binding.json").read_text(encoding="utf-8")
            ),
            "source_model": args.source_model,
            "judge_model": args.judge_model,
            "thinking_level": args.thinking_level,
            "max_output_tokens": args.max_output_tokens,
            "expected_failures": args.expected_failures,
            "expected_agent_successes": expected_agent_successes,
            "expected_total_failures": args.expected_total_failures,
            "excluded_ambiguous_or_other_failures": len(self.all_failures) - len(self.failures),
            "case_ids": sorted(self.failure_rows),
            "large_payload_hashing": False,
        }
        binding_path = self.judge.output / "tail-repair-binding-v2.json"
        if binding_path.exists() and json.loads(binding_path.read_text(encoding="utf-8")) != binding:
            raise ValueError("judge tail-repair binding changed")
        atomic_json(binding_path, binding)
        self.semaphore = asyncio.Semaphore(args.concurrency)

    async def retry_one(self, client: GeminiInteractionsClient, case_id: str) -> str:
        directory = self.judge.case_dir(case_id)
        result_path = directory / "result.json"
        if result_path.exists():
            return "already_completed"
        intent_path = directory / "attempt-02.intent.json"
        response_path = directory / "attempt-02.result.json"
        retry_failed_path = directory / "attempt-02.failed.json"
        if response_path.exists():
            record = json.loads(response_path.read_text(encoding="utf-8"))
            if terminal_judge(record):
                atomic_json(result_path, record)
                return "reconciled_completed"
            if not retry_failed_path.exists():
                atomic_json(
                    retry_failed_path,
                    {
                        "case_id": case_id,
                        "state": "judge_retry_failed",
                        "error_type": record.get("error_type"),
                        "error": record.get("error"),
                        "replay_automatic": False,
                    },
                )
            return "retry_failed"
        if retry_failed_path.exists():
            return "retry_failed"
        if intent_path.exists():
            atomic_json(
                directory / "attempt-02.ambiguous.json",
                {
                    "case_id": case_id,
                    "state": "ambiguous_prior_dispatch",
                    "action": "do_not_replay_without_reconciliation",
                },
            )
            return "ambiguous"

        row, run_dir = self.judge.success_rows[case_id]
        atomic_json(
            intent_path,
            {
                "case_id": case_id,
                "state": "in_flight",
                "source_trace": stat_identity(Path(row["source_trace_path"])),
                "prior_failure": stat_identity(directory / "failed.json"),
                "created_at": time.time(),
            },
        )
        record = await _audit_one(
            row,
            self.judge.gold.get(case_id),
            None,
            run_dir=run_dir,
            manifest_root=self.judge.manifest_root,
            client=client,
            judge_model=self.args.judge_model,
            response_format=self.judge.response_format,
            generation_config=self.judge.generation_config,
            semaphore=self.semaphore,
        )
        compact = compact_record(record, source_trace=Path(row["source_trace_path"]))
        atomic_json(response_path, compact)
        if terminal_judge(compact):
            atomic_json(result_path, compact)
            return "completed"
        atomic_json(
            retry_failed_path,
            {
                "case_id": case_id,
                "state": "judge_retry_failed",
                "error_type": compact.get("error_type"),
                "error": compact.get("error"),
                "replay_automatic": False,
            },
        )
        return "retry_failed"

    def finalize(self, outcomes: dict[str, str]) -> None:
        records: list[dict[str, Any]] = []
        for case_id in self.judge.expected:
            path = self.judge.case_dir(case_id) / "result.json"
            if path.is_file():
                records.append(json.loads(path.read_text(encoding="utf-8")))
        final_path = self.judge.output / "judge-final.jsonl.gz"
        with gzip.open(final_path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        unresolved = unresolved_failed_directories(self.judge)
        ambiguous = [
            path
            for path in (self.judge.output / "cases").glob("*/ambiguous.json")
            if not (path.parent / "result.json").exists()
        ]
        complete = (
            not unresolved
            and not ambiguous
            and len(records) == len(self.judge.expected) == len(self.judge.success_rows)
        )
        state = {
            "schema_version": "ifv-streaming-agent-judge-state-v1",
            "phase": "judge_complete" if complete else "judge_tail_retry_incomplete",
            "agent_successes_seen": len(self.judge.success_rows),
            "judge_completed": len(records),
            "judge_failed": len(unresolved),
            "judge_ambiguous": len(ambiguous),
            "active": 0,
            "expected_runnable": len(self.judge.expected),
            "formal_denominator": self.args.formal_denominator,
            "final_records": len(records),
            "agent_missing_or_failed": len(self.judge.expected) - len(self.judge.success_rows),
            "historical_failed_attempts_preserved": len(self.failures),
            "ambiguous_or_other_prior_failures_preserved": len(self.all_failures) - len(self.failures),
            "tail_retry_outcomes": outcomes,
            "large_payload_hashing": False,
        }
        atomic_json(self.judge.output / "state.json", state)
        atomic_json(self.judge.output / "summary.json", state)
        atomic_json(self.judge.output / "tail-repair-state.json", state)

    async def run(self) -> None:
        async with GeminiInteractionsClient(timeout=240, max_retries=2) as client:
            tasks = {
                case_id: asyncio.create_task(self.retry_one(client, case_id))
                for case_id in sorted(self.failure_rows)
            }
            outcomes = {
                case_id: await task
                for case_id, task in tasks.items()
            }
        self.finalize(outcomes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rollout-root", "benchmark", "manifest", "gold", "output", "inference-summary"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--expected-count", type=int, default=1526)
    parser.add_argument("--formal-denominator", type=int, default=1527)
    parser.add_argument("--judge-model", default="gemini-3.7-flash")
    parser.add_argument("--thinking-level", default="low")
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--expected-failures", type=int, required=True)
    parser.add_argument("--expected-total-failures", type=int)
    parser.add_argument("--expected-agent-successes", type=int)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= args.expected_failures:
        parser.error("concurrency must be within the bounded failed-case count")
    asyncio.run(TailRepair(args).run())


if __name__ == "__main__":
    main()
