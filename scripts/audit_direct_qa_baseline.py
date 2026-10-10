"""Audit direct-QA answers against private construction gold.

This is a post-hoc audit.  The direct-QA model never receives the private gold.
The frozen judge receives the private gold and the already-produced answer, and
assesses whether the core fact and reason actually support the verdict.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

from eviscene.integrations.gemini import GeminiInteractionsClient, extract_text
from eviscene.eval.private_gold_metrics import (
    annotate_private_gold_category,
    private_gold_audit_summary,
    private_gold_category_counts,
)
from eviscene.eval.evaluator_private_gold import private_gold_index
from eviscene.eval.private_gold_judge_contract import (
    PRIVATE_GOLD_JUDGE_PROMPT,
    PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA,
)

from scripts.run_direct_qa_baseline import (
    _case_id as _case_id,
    _gold_verdict,
    _image_input,
    _parse_json_object,
    _resolve_image_path,
    extract_native_thought,
)


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

RETRYABLE_AUDIT_ERROR_TYPES = {
    "GeminiInteractionsHTTPError",
    "GeminiInteractionsResponseError",
    "ReadTimeout",
    "SSLError",
    "TimeoutError",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _latest_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = str(record.get("case_id") or "").strip()
        if case_id:
            latest[case_id] = dict(record)
    return latest


def _audit_record_is_terminal(record: Mapping[str, Any]) -> bool:
    status = str(record.get("status") or "").strip()
    if status == "not_auditable":
        return True
    if status == "completed":
        # A provider response can arrive successfully while still failing
        # the structured judge contract.  Such a row must be retried rather
        # than being mistaken for a completed quality judgment.
        return str(record.get("quality_bucket") or "").strip() in {
            "strong",
            "usable",
            "rejected",
            "not_auditable",
        }
    if (
        str(record.get("error_type") or "").strip() == "ValueError"
        and "source result is not completed"
        in str(record.get("error") or "")
    ):
        return True
    return False


def _archive_candidate_key(row: Mapping[str, Any]) -> str:
    return str(
        row.get("archive_source_version_id")
        or row.get("candidate_id")
        or row.get("case_id")
        or ""
    ).strip()


def _load_archive_private_gold(
    archive_root: Path,
) -> dict[str, dict[str, Any]]:
    """Load evaluator-private construction rows from an immutable archive."""

    root = archive_root.expanduser().resolve()
    candidate_path = root / "human-review-candidates.jsonl"
    if not candidate_path.is_file():
        raise FileNotFoundError(
            f"archive private-gold file does not exist: {candidate_path}"
        )
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(_read_jsonl(candidate_path), start=1):
        key = _archive_candidate_key(row)
        if not key:
            raise ValueError(
                f"{candidate_path}:{line_number} has no archive identity"
            )
        if key in indexed:
            raise ValueError(f"duplicate archive private-gold key: {key}")
        indexed[key] = row
    if not indexed:
        raise ValueError(f"archive private-gold file is empty: {candidate_path}")
    return indexed


def _load_qa_fallback_gold(
    row: Mapping[str, Any],
    *,
    manifest_root: Path,
) -> dict[str, Any]:
    """Read the target atom from the per-image QA artifact when available.

    This is only a last-resort audit fallback.  The historical construction
    archive remains authoritative because QA does not contain source evidence
    or the complete claim atom.
    """

    value = str(row.get("unified_qa_path") or "").strip()
    if not value:
        return {}
    path = Path(value)
    if path.is_absolute():
        resolved = path
    else:
        resolved = (manifest_root / path).resolve()
        try:
            resolved.relative_to(manifest_root.resolve())
        except ValueError:
            return {}
    if not resolved.is_file():
        return {}
    try:
        qa = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(qa, Mapping):
        return {}
    semantic = qa.get("semantic_checks")
    if not isinstance(semantic, Mapping):
        semantic = {}
    target_claim = str(qa.get("target_claim") or "").strip()
    target_atom = str(
        qa.get("target_visual_atom")
        or semantic.get("target_visual_atom_observation")
        or ""
    ).strip()
    if not target_claim and not target_atom:
        return {}
    return {
        "target_claim": target_claim,
        "decisive_visual_atom": target_atom,
        "automatic_qa": dict(semantic),
        "_private_gold_source": "unified_qa_fallback",
    }


def _build_private_gold_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    archive_root: Path | None,
    manifest_root: Path,
) -> list[dict[str, Any]]:
    """Attach private gold only to the post-hoc evaluator-side rows.

    The public unified manifest intentionally omits construction details for
    many records.  Merge the matching immutable archive row by
    ``archive_source_version_id`` before judging.  No merged row is sent to
    the direct-QA model.
    """

    archive_index = (
        _load_archive_private_gold(archive_root)
        if archive_root is not None
        else {}
    )
    merged_rows: list[dict[str, Any]] = []
    for source_row in manifest_rows:
        row = dict(source_row)
        archive_key = str(row.get("archive_source_version_id") or "").strip()
        archive_row = archive_index.get(archive_key)
        if archive_row is not None:
            for field in PRIVATE_GOLD_FIELDS:
                current = row.get(field)
                if current in (None, "", [], {}):
                    if field in archive_row:
                        row[field] = archive_row[field]
            row["_private_gold_source"] = "historical_archive"
        else:
            qa_fallback = _load_qa_fallback_gold(
                row,
                manifest_root=manifest_root,
            )
            for field, value in qa_fallback.items():
                if row.get(field) in (None, "", [], {}):
                    row[field] = value
            if qa_fallback:
                row["_private_gold_source"] = "unified_qa_fallback"
        merged_rows.append(row)
    return merged_rows


def _private_gold(row: Mapping[str, Any]) -> dict[str, Any]:
    claim_atom = row.get("claim_atom")
    if not isinstance(claim_atom, Mapping):
        claim_atom = {}
    evidence = row.get("evidence")
    if not isinstance(evidence, Mapping):
        evidence = {}
    binding = evidence.get("binding")
    if not isinstance(binding, Mapping):
        binding = {}
    automatic_qa = row.get("automatic_qa")
    if not isinstance(automatic_qa, Mapping):
        automatic_qa = {}

    target_claim = str(row.get("target_claim") or "").strip()
    decisive_visual_atom = str(
        row.get("decisive_visual_atom")
        or automatic_qa.get("target_visual_atom_observation")
        or automatic_qa.get("decisive_fact_observation")
        or ""
    ).strip()
    source_fact = str(
        binding.get("event_scope_span")
        or binding.get("source_evidence_span")
        or binding.get("candidate_exact_span")
        or ""
    ).strip()
    expected_verdict = _gold_verdict(row)
    gold = {
        "expected_verdict": expected_verdict,
        "target_claim": target_claim,
        "decisive_visual_atom": decisive_visual_atom,
        "claim_atom": {
            key: value
            for key, value in claim_atom.items()
            if key in {
                "subject",
                "event_or_context",
                "depicted_value",
                "relation_slot",
            }
        },
        "source_fact": source_fact,
        "construction_subroute": str(row.get("construction_subroute") or ""),
        "target_subtype": str(
            row.get("target_subtype") or row.get("target_capability_cell") or ""
        ),
    }
    gold["auditable"] = bool(
        expected_verdict
        and (
            target_claim
            or decisive_visual_atom
            or gold["claim_atom"]
            or source_fact
        )
    )
    gold["source"] = str(row.get("_private_gold_source") or "manifest").strip()
    return gold


def _candidate_output(result: Mapping[str, Any]) -> dict[str, Any]:
    output = result.get("model_output")
    if isinstance(output, Mapping):
        return {
            "core_fact": str(output.get("core_fact") or ""),
            "verdict": str(output.get("verdict") or ""),
            "reason": str(output.get("reason") or ""),
        }
    return {
        "core_fact": "",
        "verdict": str(result.get("predicted_verdict") or ""),
        "reason": "",
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


async def _audit_one(
    result: Mapping[str, Any],
    gold_row: Mapping[str, Any] | None,
    *,
    client: GeminiInteractionsClient,
    judge_model: str,
    judge_prompt: str,
    response_format: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    manifest_root: Path,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    case_id = str(result.get("case_id") or "").strip()
    started = time.monotonic()
    audit: dict[str, Any] = {
        "case_id": case_id,
        "source_image_path": result.get("image_path"),
        "source_model": result.get("model"),
        "source_status": result.get("status"),
        "judge_model": judge_model,
        "audit_mode": "direct_qa_image_only",
        "trace_evidence_grounded": "not_available",
        "status": "error",
    }
    try:
        if gold_row is None:
            raise ValueError("private gold row is missing")
        gold = _private_gold(gold_row)
        candidate = _candidate_output(result)
        audit.update(
            {
                "gold_verdict": gold.get("expected_verdict"),
                "candidate_verdict": candidate.get("verdict"),
                "verdict_matches_gold": bool(
                    gold.get("expected_verdict")
                    and candidate.get("verdict") == gold.get("expected_verdict")
                ),
                "candidate_output": candidate,
                "private_gold_auditable": gold.get("auditable") is True,
                "private_gold": gold,
            }
        )
        if result.get("status") != "completed":
            raise ValueError("direct-QA source result is not completed")
        if not gold.get("auditable"):
            audit.update(
                {
                    "status": "not_auditable",
                    "quality_bucket": "not_auditable",
                    "private_gold_fact_match": "not_auditable",
                    "reason_quality": "not_auditable",
                    "explanation": "private gold lacks a usable factual target",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            )
            return audit

        image_path = _resolve_image_path(
            {"image_path": result.get("image_path")},
            manifest_root,
        )
        payload_text = json.dumps(
            {
                "private_gold": gold,
                "candidate_material": {
                    "mode": "direct_qa_image_only",
                    "candidate_answer": candidate,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        request_input: list[dict[str, Any]] = [
            _image_input(image_path),
            {
                "type": "text",
                "text": judge_prompt + "\n\nAUDIT INPUT:\n" + payload_text,
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
        candidate_sidecar = (
            manifest.parent
            / "evaluator_private"
            / "private-gold-v1"
            / "test-private-gold.jsonl"
        )
        if candidate_sidecar.is_file():
            private_gold_sidecar = candidate_sidecar
    archive_root = (
        Path(args.archive_root).expanduser().resolve()
        if args.archive_root
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_results = run_dir / "results.jsonl"
    if not source_results.is_file():
        raise FileNotFoundError(source_results)

    if private_gold_sidecar is not None:
        private_gold_rows = _read_jsonl(private_gold_sidecar)
        # The source result's image_path is relative to the unified dataset,
        # not to the QA output directory.  Keep an explicit manifest as the
        # authoritative image root; when callers provide only the standard
        # evaluator-private sidecar, derive its sibling dataset root.
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
    # Recovery appends outcomes; select one result per case so earlier errors
    # do not duplicate judge requests or final counts.
    results = list(_latest_records(_read_jsonl(source_results)).values())
    if args.limit > 0:
        results = results[: args.limit]
    if args.case_id:
        selected = set(args.case_id)
        results = [row for row in results if row.get("case_id") in selected]

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
        judge_prompt_for_file(args.prompt_file) + "\n",
        encoding="utf-8",
    )
    _write_json(
        output_dir / "run-config.json",
        {
            "schema_version": "ifv-direct-qa-private-audit-config-v1",
            "run_dir": str(run_dir),
            "source_results": str(source_results),
            "manifest": str(manifest) if manifest else None,
            "private_gold_sidecar": (
                str(private_gold_sidecar) if private_gold_sidecar else None
            ),
            "archive_root": str(archive_root) if archive_root else None,
            "judge_model": args.judge_model,
            "thinking_level": args.thinking_level,
            "thinking_summaries": "auto",
            "max_output_tokens": args.max_output_tokens,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "concurrency": args.concurrency,
            "selected_results": len(results),
            "judge_input": [
                "image",
                "private_construction_gold",
                "candidate_core_fact_verdict_reason",
            ],
            "evidence_context": "not_available_for_direct_qa",
        },
    )

    prompt = judge_prompt_for_file(args.prompt_file)
    output_path = output_dir / "audit-results.jsonl"
    existing_records = _read_jsonl(output_path) if output_path.is_file() else []
    existing_latest = _latest_records(existing_records)
    results = [
        result
        for result in results
        if not _audit_record_is_terminal(
            existing_latest.get(str(result.get("case_id") or ""), {})
        )
    ]
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    records: list[dict[str, Any]] = list(existing_records)
    async with GeminiInteractionsClient(
        timeout=args.timeout,
        max_retries=args.max_retries,
    ) as client:
        tasks = [
            asyncio.create_task(
                _audit_one(
                    result,
                    gold_by_case.get(str(result.get("case_id") or "")),
                    client=client,
                    judge_model=args.judge_model,
                    judge_prompt=prompt,
                    response_format=response_format,
                    generation_config=generation_config,
                    manifest_root=manifest_root,
                    semaphore=semaphore,
                )
            )
            for result in results
        ]
        with output_path.open("a", encoding="utf-8") as handle:
            for task in asyncio.as_completed(tasks):
                record = annotate_private_gold_category(await task)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    json.dumps(
                        {
                            "case_id": record.get("case_id"),
                            "status": record.get("status"),
                            "quality_bucket": record.get("quality_bucket"),
                            "reason_quality": record.get("reason_quality"),
                            "verdict_matches_gold": record.get(
                                "verdict_matches_gold"
                            ),
                            "private_gold_category": record.get(
                                "private_gold_category"
                            ),
                            "error_type": record.get("error_type"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    latest_records = list(_latest_records(records).values())
    completed = [
        row for row in latest_records if row.get("status") == "completed"
    ]

    def judge_value(row: Mapping[str, Any], key: str) -> str:
        direct = row.get(key)
        if direct is not None:
            return str(direct)
        nested = row.get("judge_output")
        if isinstance(nested, Mapping):
            return str(nested.get(key, "unknown"))
        return "unknown"

    quality_counts = Counter(
        judge_value(row, "quality_bucket")
        for row in completed
    )
    reason_counts = Counter(
        judge_value(row, "reason_quality")
        for row in completed
    )
    summary = {
        "schema_version": "ifv-direct-qa-private-audit-v1",
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
            "selected": len(latest_records),
            "selected_this_invocation": len(results),
            "completed": len(completed),
            "errors": len(latest_records) - len(completed),
            "private_gold_auditable": sum(
                row.get("private_gold_auditable") is True
                for row in latest_records
            ),
            "verdict_matches_gold": sum(
                row.get("verdict_matches_gold") is True
                for row in latest_records
            ),
            "judge_thought_present": sum(
                row.get("judge_native_thought_chars", 0) > 0 for row in completed
            ),
        },
        "quality_buckets": dict(quality_counts),
        "reason_quality": dict(reason_counts),
        "fact_match": dict(
            Counter(
                str(row.get("private_gold_fact_match") or "unknown")
                for row in completed
            )
        ),
        "trace_evidence_grounded": {
            "not_available": len(completed),
        },
        "private_gold_categories": private_gold_category_counts(latest_records),
        "private_gold_audit": private_gold_audit_summary(latest_records),
        "private_gold_category_definition": {
            "correct_point_with_strong_evidence": (
                "verdict_matches_gold=true and quality_bucket=strong"
            ),
            "correct_verdict_insufficient_evidence": (
                "verdict_matches_gold=true and quality_bucket!=strong"
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


def judge_prompt_for_file(path: str | None) -> str:
    if path:
        value = Path(path).expanduser().read_text(encoding="utf-8").strip()
        if not value:
            raise ValueError("judge prompt file is empty")
        return value
    return PRIVATE_GOLD_JUDGE_PROMPT


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit direct-QA answers with private construction gold."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument(
        "--archive-root",
        help=(
            "immutable archive containing human-review-candidates.jsonl; "
            "used to recover sparse private gold after rollout"
        ),
    )
    parser.add_argument(
        "--private-gold-sidecar",
        help="Evaluator-private private-gold.jsonl; preferred for unified data.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-file")
    parser.add_argument(
        "--judge-model",
        default="gemini-3.1-pro-preview",
    )
    parser.add_argument(
        "--thinking-level",
        choices=("low", "medium", "high"),
        default="high",
    )
    parser.add_argument("--max-output-tokens", type=int, default=8192)
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
