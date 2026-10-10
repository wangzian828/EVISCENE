"""Audit completed Agent traces against private construction gold.

Unlike direct QA, this audit sees the persisted final report plus the actual
Evidence/basis selected by the Agent.  It never feeds private gold back into a
rollout and it never treats an Agent prose claim as evidence by itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

from eviscene.eval.agent_private_gold import (
    AGENT_PRIVATE_GOLD_PROJECTION_SCHEMA_VERSION,
    agent_candidate_answer,
    build_agent_private_gold_candidate,
)
from eviscene.eval.evaluator_private_gold import private_gold_index
from eviscene.eval.private_gold_judge_contract import (
    PRIVATE_GOLD_JUDGE_PROMPT,
    PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA,
)
from eviscene.eval.private_gold_metrics import (
    agent_private_gold_audit_summary,
    agent_private_gold_category_counts,
    annotate_agent_private_gold_category,
)
from eviscene.integrations.gemini import GeminiInteractionsClient, extract_text
from scripts.audit_direct_qa_baseline import (
    _build_private_gold_rows,
    _private_gold,
    _read_jsonl,
    _write_json,
)
from scripts.run_direct_qa_baseline import (
    _image_input,
    _parse_json_object,
    _resolve_image_path,
    extract_native_thought,
)


def _case_id(row: Mapping[str, Any]) -> str:
    return str(row.get("case_id") or "").strip()


def _source_result_status(result: Mapping[str, Any]) -> str:
    """Normalize the two persisted Agent result conventions.

    ``run_cases`` writes terminal runtime state as ``termination`` while some
    higher-level collectors write ``status``.  Both describe the same source
    rollout outcome for this post-hoc audit.
    """

    return str(result.get("status") or result.get("termination") or "").strip()


def _source_result_is_successful(result: Mapping[str, Any]) -> bool:
    return _source_result_status(result) in {"success", "completed"}


def _trace_path(run_dir: Path, row: Mapping[str, Any]) -> Path:
    source_trace = str(row.get("source_trace_path") or "").strip()
    raw = source_trace or str(row.get("trace_path") or "").strip()
    if not raw:
        raise ValueError("run result does not name a trace_path")
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (run_dir / path).resolve()
    if not source_trace:
        try:
            resolved.relative_to(run_dir)
        except ValueError:
            raise ValueError(f"trace_path escapes run directory: {raw}") from None
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not a JSON object")
    return dict(value)


def _default_sidecar_for_manifest(manifest: Path) -> Path:
    split = "test" if "test" in manifest.name.casefold() else "train"
    return (
        manifest.parent
        / "evaluator_private"
        / "private-gold-v1"
        / f"{split}-private-gold.jsonl"
    )


def _latest_report_records(
    records: list[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = _case_id(record)
        if case_id:
            latest[case_id] = dict(record)
    return latest


def _overlay_report_sidecar(
    trace: Mapping[str, Any],
    report_record: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    """Overlay a post-hoc report in memory without changing the raw trace."""

    if not report_record:
        return dict(trace), "missing"
    report = report_record.get("report")
    if not isinstance(report, Mapping):
        return dict(trace), "invalid"
    projected = dict(trace)
    judgment_value = projected.get("judgment")
    judgment = (
        dict(judgment_value)
        if isinstance(judgment_value, Mapping)
        else {}
    )
    if not isinstance(judgment.get("fact_check_report"), Mapping):
        judgment["fact_check_report"] = dict(report)
    citations = report_record.get("evidence_citations")
    if (
        not isinstance(judgment.get("evidence_citations"), list)
        and isinstance(citations, list)
    ):
        judgment["evidence_citations"] = list(citations)
    projected["judgment"] = judgment
    return projected, "applied"


async def _audit_one(
    result: Mapping[str, Any],
    gold_row: Mapping[str, Any] | None,
    report_record: Mapping[str, Any] | None,
    *,
    run_dir: Path,
    manifest_root: Path,
    client: GeminiInteractionsClient,
    judge_model: str,
    response_format: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    case_id = _case_id(result)
    started = time.monotonic()
    audit: dict[str, Any] = {
        "case_id": case_id,
        "source_status": _source_result_status(result),
        "source_model": result.get("model"),
        "judge_model": judge_model,
        "audit_mode": "agent_trace_evidence",
        "status": "error",
    }
    try:
        if gold_row is None:
            raise ValueError("private gold row is missing")
        if not _source_result_is_successful(result):
            raise ValueError("Agent source result is not successful")
        trace_path = _trace_path(run_dir, result)
        raw_trace = _read_json(trace_path)
        trace, report_sidecar_status = _overlay_report_sidecar(
            raw_trace,
            report_record,
        )
        gold = _private_gold(gold_row)
        candidate = build_agent_private_gold_candidate(trace)
        candidate_answer = agent_candidate_answer(candidate)
        audit.update(
            {
                "source_trace_path": str(trace_path),
                "gold_verdict": gold.get("expected_verdict"),
                "candidate_verdict": candidate_answer.get("verdict"),
                "verdict_matches_gold": bool(
                    gold.get("expected_verdict")
                    and candidate_answer.get("verdict")
                    == gold.get("expected_verdict")
                ),
                "candidate_output": candidate,
                "candidate_answer": candidate_answer,
                "report_sidecar_status": report_sidecar_status,
                "report_sidecar_case_id": (
                    _case_id(report_record) if report_record else None
                ),
                "private_gold_auditable": gold.get("auditable") is True,
                "private_gold": gold,
            }
        )
        if not gold.get("auditable"):
            audit.update(
                {
                    "status": "not_auditable",
                    "quality_bucket": "not_auditable",
                    "private_gold_fact_match": "not_auditable",
                    "fact_alignment": "not_auditable",
                    "reason_quality": "not_auditable",
                    "explanation": "private gold lacks a usable factual target",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            )
            return audit

        image_path = _resolve_image_path(
            {"image_path": trace.get("image_path")},
            manifest_root,
        )
        payload_text = json.dumps(
            {
                "private_gold": gold,
                "candidate_material": {
                    "mode": "agent_trace",
                    "candidate_answer": candidate_answer,
                    "supporting_trace_material": candidate,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        request_input: list[dict[str, Any]] = [
            _image_input(image_path),
            {
                "type": "text",
                "text": PRIVATE_GOLD_JUDGE_PROMPT
                + "\n\nAUDIT INPUT:\n"
                + payload_text,
            },
        ]
        async with semaphore:
            payload = await client.create(
                model=judge_model,
                input=request_input,
                response_format=response_format,
                generation_config=generation_config,
                store=True,
            )
        output_text = extract_text(payload)
        parsed, parse_error = _parse_json_object(output_text)
        native_thought = extract_native_thought(payload)
        if parsed is not None:
            audit.update(
                {
                    "quality_bucket": parsed.get("quality_bucket"),
                    "fact_alignment": parsed.get("fact_alignment"),
                    "private_gold_fact_match": parsed.get("fact_alignment"),
                    "reason_quality": parsed.get("reason_quality"),
                    "failure_modes": parsed.get("failure_modes"),
                    "explanation": parsed.get("explanation"),
                }
            )
        audit.update(
            {
                "status": "completed",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "interaction_id": payload.get("id"),
                "interaction_status": payload.get("status"),
                "judge_output_text": output_text,
                "judge_output": parsed,
                "judge_json_parse_error": parse_error,
                "judge_native_thought": native_thought,
                "judge_native_thought_chars": len(native_thought),
                "judge_native_thought_tokens": int(
                    (payload.get("usage") or {}).get(
                        "total_thought_tokens",
                        (payload.get("usage") or {}).get("thought_tokens", 0),
                    )
                    or 0
                ),
                "usage": payload.get("usage"),
            }
        )
    except Exception as exc:
        audit.update(
            {
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return audit


async def _run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else None
    )
    private_gold_sidecar = (
        Path(args.private_gold_sidecar).expanduser().resolve()
        if args.private_gold_sidecar
        else None
    )
    if private_gold_sidecar is None and manifest is not None:
        candidate_sidecar = _default_sidecar_for_manifest(manifest)
        if candidate_sidecar.is_file():
            private_gold_sidecar = candidate_sidecar
    archive_root = (
        Path(args.archive_root).expanduser().resolve()
        if args.archive_root
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if private_gold_sidecar is not None:
        private_gold_rows = _read_jsonl(private_gold_sidecar)
        # Agent trace image paths are rooted at the unified dataset.  The
        # report/audit output directory is never an image root.
        manifest_root = (
            manifest.parent
            if manifest is not None
            else private_gold_sidecar.parent.parent.parent
        )
    else:
        if manifest is None:
            raise ValueError(
                "provide --private-gold-sidecar or --manifest with --archive-root"
            )
        manifest_rows = _read_jsonl(manifest)
        private_gold_rows = _build_private_gold_rows(
            manifest_rows,
            archive_root=archive_root,
            manifest_root=manifest.parent,
        )
        manifest_root = manifest.parent
    gold_by_case = private_gold_index(private_gold_rows)
    results_path = (
        Path(args.results_path).expanduser().resolve()
        if args.results_path
        else run_dir / "run_results.jsonl"
    )
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    results = _read_jsonl(results_path)
    if args.limit > 0:
        results = results[: args.limit]
    if args.case_id:
        selected = set(args.case_id)
        results = [row for row in results if _case_id(row) in selected]
    report_by_case: dict[str, dict[str, Any]] = {}
    report_sidecar_path = (
        Path(args.report_sidecar).expanduser().resolve()
        if args.report_sidecar
        else None
    )
    if report_sidecar_path is not None:
        report_by_case = _latest_report_records(
            _read_jsonl(report_sidecar_path)
        )

    response_format = {
        "type": "text",
        "mime_type": "application/json",
        "schema": PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA,
    }
    generation_config = {
        "max_output_tokens": args.max_output_tokens,
        "thinking_level": args.thinking_level,
        "thinking_summaries": "auto",
    }
    (output_dir / "prompt.txt").write_text(
        PRIVATE_GOLD_JUDGE_PROMPT + "\n",
        encoding="utf-8",
    )
    _write_json(
        output_dir / "run-config.json",
        {
            "schema_version": "ifv-private-gold-audit-config-v2",
            "run_dir": str(run_dir),
            "results_path": str(results_path),
            "manifest": str(manifest) if manifest else None,
            "private_gold_sidecar": (
                str(private_gold_sidecar) if private_gold_sidecar else None
            ),
            "report_sidecar": (
                str(report_sidecar_path) if report_sidecar_path else None
            ),
            "archive_root": str(archive_root) if archive_root else None,
            "judge_model": args.judge_model,
            "thinking_level": args.thinking_level,
            "max_output_tokens": args.max_output_tokens,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "concurrency": args.concurrency,
            "selected_results": len(results),
            "judge_input": [
                "image",
                "private_construction_gold",
                "candidate_answer",
                "complete_semantic_agent_output",
                "complete_tool_action_history",
                "complete_tool_observations",
                "structured_trace_material_when_present",
            ],
            "agent_projection_schema_version": (
                AGENT_PRIVATE_GOLD_PROJECTION_SCHEMA_VERSION
            ),
            "agent_projection_exclusions": [
                "embedded_image_or_binary_payloads",
                "raw_html",
                "repeated_policy_inputs",
                "hidden_native_reasoning_artifacts",
                "transport_metadata",
            ],
        },
    )

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    output_path = output_dir / "audit-results.jsonl"
    records: list[dict[str, Any]] = []
    async with GeminiInteractionsClient(
        timeout=args.timeout,
        max_retries=args.max_retries,
    ) as client:
        tasks = [
            asyncio.create_task(
                _audit_one(
                    result,
                    gold_by_case.get(_case_id(result)),
                    report_by_case.get(_case_id(result)),
                    run_dir=run_dir,
                    manifest_root=manifest_root,
                    client=client,
                    judge_model=args.judge_model,
                    response_format=response_format,
                    generation_config=generation_config,
                    semaphore=semaphore,
                )
            )
            for result in results
        ]
        with output_path.open("w", encoding="utf-8") as handle:
            for task in asyncio.as_completed(tasks):
                record = annotate_agent_private_gold_category(await task)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    json.dumps(
                        {
                            "case_id": record.get("case_id"),
                            "status": record.get("status"),
                            "quality_bucket": record.get("quality_bucket"),
                            "fact_alignment": record.get("fact_alignment"),
                            "reason_quality": record.get("reason_quality"),
                            "verdict_matches_gold": record.get(
                                "verdict_matches_gold"
                            ),
                            "error_type": record.get("error_type"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    completed = [row for row in records if row.get("status") == "completed"]
    summary = {
        "schema_version": "ifv-private-gold-audit-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "manifest": str(manifest) if manifest else None,
        "private_gold_sidecar": (
            str(private_gold_sidecar) if private_gold_sidecar else None
        ),
        "archive_root": str(archive_root) if archive_root else None,
        "judge_model": args.judge_model,
        "thinking_level": args.thinking_level,
        "concurrency": args.concurrency,
        "counts": {
            "selected": len(results),
            "completed": len(completed),
            "errors": len(records) - len(completed),
            "private_gold_auditable": sum(
                row.get("private_gold_auditable") is True for row in records
            ),
            "verdict_matches_gold": sum(
                row.get("verdict_matches_gold") is True for row in records
            ),
            "judge_thought_present": sum(
                row.get("judge_native_thought_chars", 0) > 0
                for row in completed
            ),
        },
        "quality_buckets": dict(
            Counter(str(row.get("quality_bucket") or "unknown") for row in completed)
        ),
        "fact_match": dict(
            Counter(
                str(row.get("fact_alignment") or "unknown")
                for row in completed
            )
        ),
        "reason_quality": dict(
            Counter(
                str(row.get("reason_quality") or "unknown")
                for row in completed
            )
        ),
        "private_gold_categories": agent_private_gold_category_counts(records),
        "private_gold_audit": agent_private_gold_audit_summary(records),
        "private_gold_category_definition": {
            "correct_point_with_strong_evidence": (
                "verdict_matches_gold=true and "
                "reason_quality=decisive_and_grounded"
            ),
            "correct_verdict_insufficient_evidence": (
                "verdict_matches_gold=true and "
                "reason_quality!=decisive_and_grounded"
            ),
            "wrong_verdict": "verdict_matches_gold=false",
        },
        "audit_results": str(output_path),
        "prompt_file": str(output_dir / "prompt.txt"),
        "config_file": str(output_dir / "run-config.json"),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit completed Agent traces with private construction gold."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--archive-root")
    parser.add_argument(
        "--private-gold-sidecar",
        help="Evaluator-private private-gold.jsonl; preferred for unified data.",
    )
    parser.add_argument(
        "--report-sidecar",
        help=(
            "Post-hoc fact-check reports.jsonl keyed by case_id; used only "
            "as an in-memory overlay for historical traces."
        ),
    )
    parser.add_argument(
        "--results-path",
        help=(
            "Optional source result JSONL. Supports a replacement ledger with "
            "source_trace_path, rather than only <run-dir>/run_results.jsonl."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", default="gemini-3.1-pro-preview")
    parser.add_argument(
        "--thinking-level",
        choices=("low", "medium", "high"),
        default="high",
    )
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
