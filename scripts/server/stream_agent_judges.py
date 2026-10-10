"""Judge terminal Agent cases as soon as each trace is durably available.

This is a case-granular producer/consumer bridge.  It tails only small
``run_results.jsonl`` ledgers, never scans or hashes trace/model payloads, and
submits a successful case without waiting for the rest of its rollout cohort.
Every judge case has an intent receipt before dispatch and one compact result
receipt afterwards, so restarts cannot silently duplicate a paid request.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from eviscene.eval.evaluator_private_gold import private_gold_index
from eviscene.eval.private_gold_judge_contract import (
    PRIVATE_GOLD_JUDGE_PROMPT,
    PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA,
)
from eviscene.integrations.gemini import GeminiInteractionsClient
from scripts.audit_agent_private_gold import _audit_one, _read_jsonl


TERMINAL_INFERENCE_PHASES = {
    "inference_complete",
    "engineering_retry_budget_exhausted",
    "smoke_failed_requires_fix",
}
DROP_FROM_RECEIPT = {
    "candidate_output",
    "candidate_answer",
    "private_gold",
    "judge_output_text",
    "judge_native_thought",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def stat_identity(path: Path) -> dict[str, object]:
    value = path.stat()
    return {
        "path": str(path.resolve()),
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
    }


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def case_token(index: int, case_id: str) -> str:
    suffix = hashlib.sha256(case_id.encode()).hexdigest()[:16]
    return f"c{index:04d}-{suffix}"


def compact_record(record: Mapping[str, Any], *, source_trace: Path) -> dict[str, Any]:
    compact = {key: value for key, value in record.items() if key not in DROP_FROM_RECEIPT}
    compact["source_trace"] = stat_identity(source_trace)
    compact["large_payload_hashing"] = False
    return compact


def terminal_judge(record: Mapping[str, Any]) -> bool:
    if record.get("status") == "not_auditable":
        return True
    if record.get("status") != "completed" or record.get("judge_json_parse_error"):
        return False
    properties = PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA["properties"]
    return all(
        record.get(key) in properties[key]["enum"]
        for key in ("quality_bucket", "fact_alignment", "reason_quality")
    )


class LedgerTail:
    """Incrementally read complete JSON lines without rereading old bytes."""

    def __init__(self) -> None:
        self.offsets: dict[Path, int] = {}
        self.buffers: dict[Path, bytes] = {}

    def read_new(self, path: Path) -> list[dict[str, Any]]:
        offset = self.offsets.get(path, 0)
        size = path.stat().st_size
        if size < offset:
            raise RuntimeError(f"run-results ledger shrank: {path}")
        with path.open("rb") as handle:
            handle.seek(offset)
            # Read exactly the stat-observed extent.  If the producer appends
            # between stat() and read(), those bytes belong to the next poll;
            # otherwise advancing only to ``size`` would replay them.
            chunk = handle.read(size - offset)
        self.offsets[path] = size
        payload = self.buffers.pop(path, b"") + chunk
        if payload and not payload.endswith(b"\n"):
            complete, partial = payload.rsplit(b"\n", 1) if b"\n" in payload else (b"", payload)
            self.buffers[path] = partial
        else:
            complete = payload
        return [json.loads(line) for line in complete.splitlines() if line.strip()]


class StreamingJudge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        benchmark_rows = rows(args.benchmark)
        self.expected = [str(row["case_id"]) for row in benchmark_rows]
        if len(self.expected) != args.expected_count or len(set(self.expected)) != len(self.expected):
            raise ValueError("unexpected or duplicate frozen evaluation cases")
        self.index = {case: number for number, case in enumerate(self.expected, 1)}
        self.gold = private_gold_index(_read_jsonl(args.gold))
        self.manifest_root = args.manifest.resolve().parent
        self.tail = LedgerTail()
        self.queued: set[str] = set()
        # Discovery may outrun the provider.  Keep undispatched cases in
        # memory so their paid-request intents are written only when a worker
        # slot is actually available.  A restart reconstructs this queue from
        # the durable Agent ledgers.
        self.pending: dict[str, tuple[dict[str, Any], Path]] = {}
        self.success_rows: dict[str, tuple[dict[str, Any], Path]] = {}
        self.tasks: set[asyncio.Task[None]] = set()
        self.semaphore = asyncio.Semaphore(args.concurrency)
        self.response_format = {
            "type": "text",
            "mime_type": "application/json",
            "schema": PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA,
        }
        self.generation_config = {
            "max_output_tokens": args.max_output_tokens,
            "thinking_level": args.thinking_level,
            "thinking_summaries": "auto",
        }
        binding = {
            "schema_version": "ifv-streaming-agent-judge-binding-v1",
            "rollout_root": str(args.rollout_root.resolve()),
            "benchmark": stat_identity(args.benchmark),
            "manifest": stat_identity(args.manifest),
            "gold": stat_identity(args.gold),
            "inference_summary": str(args.inference_summary.resolve()),
            "source_model": args.source_model,
            "judge_model": args.judge_model,
            "thinking_level": args.thinking_level,
            "max_output_tokens": args.max_output_tokens,
            "formal_denominator": args.formal_denominator,
            "large_payload_hashing": False,
        }
        binding_path = self.output / "binding.json"
        if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
            raise ValueError("streaming judge binding changed")
        atomic_json(binding_path, binding)
        (self.output / "prompt.txt").write_text(
            PRIVATE_GOLD_JUDGE_PROMPT + "\n", encoding="utf-8"
        )

    def case_dir(self, case_id: str) -> Path:
        return self.output / "cases" / case_token(self.index[case_id], case_id)

    def completed_ids(self) -> set[str]:
        completed: set[str] = set()
        root = self.output / "cases"
        if not root.exists():
            return completed
        for path in root.glob("*/result.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            case_id = str(record.get("case_id") or "")
            if case_id not in self.index or case_id in completed:
                raise ValueError("invalid duplicate or unknown judge receipt")
            completed.add(case_id)
        return completed

    def discover(self) -> list[tuple[str, dict[str, Any], Path]]:
        discovered: list[tuple[str, dict[str, Any], Path]] = []
        for ledger in sorted(self.args.rollout_root.glob("*/run_results.jsonl")):
            run_dir = ledger.parent
            for row in self.tail.read_new(ledger):
                case_id = str(row.get("case_id") or "")
                if case_id not in self.index:
                    raise ValueError(f"unknown rollout case: {case_id}")
                if not (
                    row.get("status") == "success"
                    and row.get("termination") == "success"
                    and row.get("verdict") in {"real", "fake"}
                    and row.get("trace_path")
                ):
                    continue
                if case_id in self.success_rows:
                    raise ValueError(f"successful case was sampled twice: {case_id}")
                trace = (run_dir / str(row["trace_path"])).resolve()
                trace.relative_to(run_dir.resolve())
                if not trace.is_file():
                    raise FileNotFoundError(trace)
                frozen = dict(row)
                frozen["source_trace_path"] = str(trace)
                frozen["model"] = self.args.source_model
                self.success_rows[case_id] = (frozen, run_dir)
                discovered.append((case_id, frozen, run_dir))
        return discovered

    async def judge_one(
        self,
        client: GeminiInteractionsClient,
        case_id: str,
        row: dict[str, Any],
        run_dir: Path,
    ) -> None:
        directory = self.case_dir(case_id)
        result_path = directory / "result.json"
        if result_path.exists():
            return
        intent_path = directory / "attempt-01.intent.json"
        response_path = directory / "attempt-01.result.json"
        if response_path.exists() or (directory / "failed.json").exists():
            # A terminal provider response already exists.  Reconciliation is
            # explicit; a process restart must never turn it into a second
            # paid request.
            return
        if intent_path.exists() and not response_path.exists():
            atomic_json(
                directory / "ambiguous.json",
                {
                    "case_id": case_id,
                    "state": "ambiguous_prior_dispatch",
                    "action": "do_not_replay_without_reconciliation",
                },
            )
            return
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(
            intent_path,
            {
                "case_id": case_id,
                "state": "in_flight",
                "source_trace": stat_identity(Path(row["source_trace_path"])),
                "created_at": time.time(),
            },
        )
        record = await _audit_one(
            row,
            self.gold.get(case_id),
            None,
            run_dir=run_dir,
            manifest_root=self.manifest_root,
            client=client,
            judge_model=self.args.judge_model,
            response_format=self.response_format,
            generation_config=self.generation_config,
            semaphore=self.semaphore,
        )
        compact = compact_record(record, source_trace=Path(row["source_trace_path"]))
        atomic_json(response_path, compact)
        if terminal_judge(compact):
            atomic_json(result_path, compact)
        else:
            atomic_json(
                directory / "failed.json",
                {
                    "case_id": case_id,
                    "state": "judge_attempt_failed",
                    "error_type": compact.get("error_type"),
                    "error": compact.get("error"),
                    "replay_automatic": False,
                },
            )

    def status(self, *, phase: str) -> dict[str, Any]:
        completed = self.completed_ids()
        failures = list((self.output / "cases").glob("*/failed.json")) if (self.output / "cases").exists() else []
        ambiguous = list((self.output / "cases").glob("*/ambiguous.json")) if (self.output / "cases").exists() else []
        return {
            "schema_version": "ifv-streaming-agent-judge-state-v1",
            "phase": phase,
            "agent_successes_seen": len(self.success_rows),
            "judge_completed": len(completed),
            "judge_failed": len(failures),
            "judge_ambiguous": len(ambiguous),
            "active": len(self.tasks),
            "pending_dispatch": len(self.pending),
            "expected_runnable": len(self.expected),
            "formal_denominator": self.args.formal_denominator,
            "large_payload_hashing": False,
        }

    def inference_done(self) -> bool:
        if not self.args.inference_summary.is_file():
            return False
        summary = json.loads(self.args.inference_summary.read_text(encoding="utf-8"))
        return summary.get("phase") in TERMINAL_INFERENCE_PHASES

    def write_final(self) -> None:
        records: list[dict[str, Any]] = []
        for case_id in self.expected:
            path = self.case_dir(case_id) / "result.json"
            if path.is_file():
                records.append(json.loads(path.read_text(encoding="utf-8")))
        final_path = self.output / "judge-final.jsonl.gz"
        with gzip.open(final_path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        state = self.status(
            phase=(
                "judge_complete"
                if len(records) == len(self.success_rows)
                else "judge_incomplete_requires_reconciliation"
            )
        )
        state["final_records"] = len(records)
        state["agent_missing_or_failed"] = len(self.expected) - len(self.success_rows)
        atomic_json(self.output / "summary.json", state)

    def remember_discovered(
        self,
        discovered: list[tuple[str, dict[str, Any], Path]],
        completed: set[str],
    ) -> None:
        for case_id, row, run_dir in discovered:
            if case_id not in completed and case_id not in self.queued:
                self.pending[case_id] = (row, run_dir)

    def dispatch_ready(self, client: GeminiInteractionsClient) -> None:
        while self.pending and len(self.tasks) < self.args.concurrency:
            case_id = next(iter(self.pending))
            row, run_dir = self.pending.pop(case_id)
            self.queued.add(case_id)
            self.tasks.add(asyncio.create_task(self.judge_one(client, case_id, row, run_dir)))

    async def run(self) -> None:
        completed = self.completed_ids()
        async with GeminiInteractionsClient(timeout=240, max_retries=2) as client:
            while True:
                self.remember_discovered(self.discover(), completed)
                finished = {task for task in self.tasks if task.done()}
                for task in finished:
                    self.tasks.remove(task)
                    task.result()
                self.dispatch_ready(client)
                atomic_json(self.output / "state.json", self.status(phase="streaming"))
                if self.inference_done() and not self.tasks and not self.pending:
                    # One final tail pass closes the race between the inference
                    # summary rename and the last run-results flush.
                    self.remember_discovered(self.discover(), completed)
                    self.dispatch_ready(client)
                    if not self.tasks and not self.pending:
                        break
                await asyncio.sleep(self.args.poll_seconds)
        self.write_final()


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
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    asyncio.run(StreamingJudge(args).run())


if __name__ == "__main__":
    main()
