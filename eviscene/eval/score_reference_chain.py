"""Score saved runtime traces against evaluator-private reference chains."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from eviscene.orchestrator.llm_backend import APIBackend
from eviscene.trajectory.reference_chain import (
    LLMReferenceEvidenceMatcher,
    score_reference_chain_trace,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate how completely saved traces recover the frozen decisive "
            "reference chain."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--gold",
        default=None,
        help="Optional evaluator_private/gold.jsonl override.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to <run-dir>/reference_chain_metrics.jsonl.",
    )
    parser.add_argument(
        "--semantic-judge",
        action="store_true",
        help=(
            "Use an LLM only for qualified evidence unresolved by deterministic "
            "matching."
        ),
    )
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--wire-api",
        default=None,
        choices=["interactions", "responses", "chat_completions"],
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--max-llm-candidates-per-fact",
        type=int,
        default=3,
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"expected JSONL object row: {path}")
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rendered = "\n".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        for row in rows
    )
    pending = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(rendered + ("\n" if rendered else ""), encoding="utf-8")
    pending.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_gold_path(
    run_dir: Path,
    manifest: Mapping[str, Any],
    override: str | None,
) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(path)
    benchmark = manifest.get("benchmark")
    if isinstance(benchmark, Mapping):
        descriptor = benchmark.get("evaluation_gold")
        if isinstance(descriptor, Mapping):
            configured = str(descriptor.get("path", "")).strip()
            if configured and Path(configured).is_file():
                return Path(configured).resolve()
        benchmark_path = str(benchmark.get("path", "")).strip()
        if benchmark_path:
            candidate = (
                Path(benchmark_path).resolve().parent.parent
                / "evaluator_private"
                / "gold.jsonl"
            )
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "could not resolve evaluator-private gold from run manifest; pass --gold"
    )


def _private_index(
    rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("gold row lacks case_id")
        if case_id in indexed:
            raise ValueError(f"duplicate gold case_id: {case_id}")
        indexed[case_id] = dict(row)
    return indexed


def _mean_metrics(rows: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    collected: Dict[str, List[float]] = {}
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                collected.setdefault(str(key), []).append(float(value))
    return {
        key: round(statistics.mean(values), 6)
        for key, values in sorted(collected.items())
        if values
    }


async def _run(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest = _load_json(run_dir / "run_manifest.json")
    gold_path = _resolve_gold_path(run_dir, manifest, args.gold)
    gold_index = _private_index(_load_jsonl(gold_path))
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / "reference_chain_metrics.jsonl"
    )

    agent_config = manifest.get("agent")
    if not isinstance(agent_config, Mapping):
        agent_config = {}
    provider = str(args.provider or agent_config.get("provider") or "gemini")
    model = str(args.model or agent_config.get("model") or "gemini-3.7-flash")
    wire_api = args.wire_api or agent_config.get("llm_wire_api")
    backend: APIBackend | None = None
    matcher: LLMReferenceEvidenceMatcher | None = None
    if args.semantic_judge:
        backend = APIBackend(
            provider=provider,
            model_name=model,
            wire_api=wire_api,
            temperature=0.0,
            max_tokens=500,
            timeout=args.timeout,
        )
        matcher = LLMReferenceEvidenceMatcher(
            backend,
            provider=provider,
            model=model,
        )

    score_metadata = {
        "evaluation_gold": {
            "path": str(gold_path),
            "sha256": _sha256(gold_path),
        },
        "source_run_id": manifest.get("run_id"),
        "source_runtime_commit": manifest.get("git_commit"),
    }
    rows: List[Dict[str, Any]] = []
    try:
        for case_id, gold in gold_index.items():
            trace_path = run_dir / "traces" / f"{case_id}.json"
            if not trace_path.is_file():
                rows.append(
                    {
                        "schema_version": "ifv-reference-chain-metrics-v2",
                        "case_id": case_id,
                        "score_metadata": score_metadata,
                        "engineering_error": True,
                        "error": "canonical trace is missing",
                        "metrics": {
                            "fact_recovery_recall": 0.0,
                            "chain_recovery_recall": 0.0,
                            "evidence_recovery_recall": 0.0,
                            "basis_reference_precision": 0.0,
                        },
                    }
                )
                continue
            trace = _load_json(trace_path)
            rows.append(
                await score_reference_chain_trace(
                    trace,
                    gold,
                    semantic_matcher=matcher,
                    score_metadata=score_metadata,
                    max_llm_candidates_per_fact=(
                        args.max_llm_candidates_per_fact
                    ),
                )
            )
    finally:
        if backend is not None:
            await backend.aclose()

    _write_jsonl(output_path, rows)
    summary = {
        "run_dir": str(run_dir),
        "output": str(output_path),
        "case_count": len(rows),
        "semantic_judge_enabled": bool(args.semantic_judge),
        "mean_metrics": _mean_metrics(rows),
        "semantic_judge": dict(matcher.metadata) if matcher else None,
        "matcher_error_count": sum(
            len(
                row.get("semantic_matcher", {}).get("errors", [])
                if isinstance(row.get("semantic_matcher"), Mapping)
                else []
            )
            for row in rows
        ),
    }
    return summary


def main() -> None:
    args = _parse_args()
    summary = asyncio.run(_run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if int(summary.get("matcher_error_count", 0) or 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
