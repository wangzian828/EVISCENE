"""Score completed teacher traces against private structured SFT targets."""

from __future__ import annotations

from eviscene.environment import get_env

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping

from scripts.audit_real_trace import audit_trace
from scripts.trajectory.stage_accepted_teacher_release import (
    SCHEMA_VERSION as TEACHER_STORAGE_SCHEMA_VERSION,
    stage_release,
)
from eviscene.eval.score_semantic_reward import _resolve_image_path
from eviscene.orchestrator.llm_backend import APIBackend
from eviscene.orchestrator.source_access import SourceAccessPolicy
from eviscene.trajectory.semantic_reward import sha256_file
from eviscene.trajectory.sft_eligibility import (
    SFT_ELIGIBILITY_SCHEMA_VERSION,
    SFTEligibilityJudge,
    build_sft_eligibility_artifact,
    build_sft_eligibility_input,
    classify_sft_audit_failures,
    sft_eligibility_cache_key,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--gold",
        type=Path,
        help="Selected private-gold rows for this run.",
    )
    parser.add_argument(
        "--private-gold-sidecar",
        type=Path,
        help="Complete evaluator-private sidecar; preferred for a split.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument(
        "--storage-dir",
        type=Path,
        help=(
            "Automatic structured teacher storage destination. Defaults to a "
            "run-specific directory under the data root's generated/sft tree."
        ),
    )
    parser.add_argument(
        "--provider",
        default=get_env("EVISCENE_SFT_ELIGIBILITY_PROVIDER", "gemini"),
        choices=["gemini", "qwen", "qwen_local", "lmdeploy"],
    )
    parser.add_argument(
        "--model",
        default=get_env("EVISCENE_SFT_ELIGIBILITY_MODEL", "gemini-3.7-flash"),
    )
    parser.add_argument(
        "--base-url",
        default=get_env("EVISCENE_SFT_ELIGIBILITY_BASE_URL"),
        help="Optional judge endpoint override, especially for qwen_local.",
    )
    parser.add_argument(
        "--api-key-env",
        default=get_env("EVISCENE_SFT_ELIGIBILITY_API_KEY_ENV"),
        help=(
            "Optional environment-variable name containing the judge API key. "
            "The variable name is recorded; the credential value is not."
        ),
    )
    parser.add_argument(
        "--wire-api",
        default=get_env("EVISCENE_SFT_ELIGIBILITY_WIRE_API"),
        choices=["chat_completions", "responses", "interactions"],
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=(
            get_env("EVISCENE_SFT_ELIGIBILITY_ENABLE_THINKING", "")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
            if get_env("EVISCENE_SFT_ELIGIBILITY_ENABLE_THINKING") is not None
            else None
        ),
        help="Enable native Qwen thinking for the judge when supported.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--provider-retries",
        type=int,
        default=int(get_env("SFT_ELIGIBILITY_PROVIDER_RETRIES", "3")),
        help="Retries for one frozen-judge provider request.",
    )
    parser.add_argument(
        "--trace-retries",
        type=int,
        default=int(get_env("SFT_ELIGIBILITY_TRACE_RETRIES", "12")),
        help="Retries for a trace that failed outside the provider request retry loop.",
    )
    parser.add_argument(
        "--trace-retry-delay",
        type=float,
        default=float(get_env("SFT_ELIGIBILITY_TRACE_RETRY_DELAY", "30")),
        help="Seconds to wait before retrying failed traces.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum simultaneous frozen-judge provider calls.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _json_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl_index(path: Path) -> Dict[str, Dict[str, Any]]:
    """Index private targets without treating legacy aliases as global IDs.

    ``case_id`` is the canonical identity used by rollout traces.  Some
    historical/generated pools reuse ``candidate_id`` or ``assignment_id``
    across distinct canonical cases, so those fields are compatibility
    aliases only: an alias is indexed when it is unambiguous, and an
    ambiguous alias is deliberately omitted.  Failing the whole audit on
    such an alias would prevent every canonical case from being scored.
    """
    result: Dict[str, Dict[str, Any]] = {}
    canonical_ids: set[str] = set()
    alias_rows: Dict[str, list[Dict[str, Any]]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must be an object")
        aliases = [
            str(row.get(key, "")).strip()
            for key in ("case_id", "candidate_id", "assignment_id")
        ]
        aliases = [value for value in aliases if value]
        if not aliases:
            raise ValueError(
                f"{path}:{line_number} lacks case_id/candidate_id/assignment_id"
            )
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            if case_id in canonical_ids:
                raise ValueError(f"duplicate private target case_id: {case_id!r}")
            canonical_ids.add(case_id)
            result[case_id] = row

        for alias in aliases:
            alias_rows.setdefault(alias, []).append(row)

    # Add only aliases that identify exactly one row.  Canonical case IDs
    # always win if a legacy alias happens to have the same spelling.
    for alias, matching_rows in alias_rows.items():
        if alias in result or len(matching_rows) != 1:
            continue
        result[alias] = matching_rows[0]
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def _api_key_from_env(variable_name: str | None) -> str | None:
    name = str(variable_name or "").strip()
    if not name:
        return None
    value = get_env(name, "").strip()
    if not value:
        raise ValueError(
            f"SFT judge API key environment variable is empty: {name}"
        )
    return value


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
    )
    pending.replace(path)


def _load_cache(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    value = _json_object(path)
    if value.get("schema_version") != SFT_ELIGIBILITY_SCHEMA_VERSION:
        raise ValueError(f"unsupported SFT eligibility cache entry: {path}")
    return value


def _trace_case_id(trace: Mapping[str, Any]) -> str:
    state = trace.get("state")
    runtime_case = (
        state.get("runtime_case")
        if isinstance(state, Mapping)
        and isinstance(state.get("runtime_case"), Mapping)
        else {}
    )
    return str(runtime_case.get("case_id", "")).strip()


def _batch_error_row(trace_path: Path, exc: Exception) -> Dict[str, Any]:
    """Return durable diagnostics without turning a provider failure into data."""

    case_id = ""
    try:
        trace = _json_object(trace_path)
        case_id = _trace_case_id(trace)
    except Exception:
        pass
    return {
        "status": "error",
        "case_id": case_id,
        "trace": str(trace_path),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _default_storage_dir(run_dir: Path, eligibility_dir: Path) -> Path:
    """Keep independently scored eligibility versions in separate storage roots."""

    resolved = run_dir.expanduser().resolve()
    resolved_eligibility = eligibility_dir.expanduser().resolve()
    runs_root = next(
        (ancestor for ancestor in (resolved, *resolved.parents) if ancestor.name == "runs"),
        None,
    )
    if runs_root is None:
        return resolved / "sft-storage" / resolved_eligibility.name
    data_root = runs_root.parent
    storage_name = f"{resolved.name}--{resolved_eligibility.name}"
    return data_root / "generated" / "sft" / storage_name


def _existing_storage_manifest(
    storage_dir: Path,
    *,
    run_dir: Path,
    eligibility_dir: Path,
) -> Dict[str, Any] | None:
    manifest_path = storage_dir / "accepted_release_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _json_object(manifest_path)
    if manifest.get("schema_version") != TEACHER_STORAGE_SCHEMA_VERSION:
        return None
    sources = manifest.get("sources") or []
    if len(sources) != 1:
        return None
    source = sources[0]
    if (
        str(source.get("run_dir", "")) == str(run_dir)
        and str(source.get("eligibility_dir", "")) == str(eligibility_dir)
    ):
        return manifest
    raise FileExistsError(
        "storage directory already belongs to another run: "
        f"{storage_dir}"
    )


async def _run(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    manifest = _json_object(run_dir / "run_manifest.json")
    if manifest.get("status") not in {"completed", "completed_with_errors"}:
        raise RuntimeError(
            "private gold may be loaded only after the rollout run reaches a "
            "terminal completed status"
        )
    source_policy = manifest.get("source_access_policy")
    source_policy_active = bool(
        isinstance(source_policy, Mapping) and source_policy.get("active") is True
    )
    source_policy_object = None
    if source_policy_active:
        policy_path = str(source_policy.get("path") or "").strip()
        if not policy_path:
            raise ValueError(
                "active source_access_policy in run manifest lacks policy path"
            )
        source_policy_object = SourceAccessPolicy.load(policy_path)
    trace_paths = sorted((run_dir / "traces").glob("*.json"))
    if not trace_paths:
        raise FileNotFoundError(f"no canonical traces under {run_dir / 'traces'}")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.provider_retries < 0:
        raise ValueError("--provider-retries must be non-negative")
    if args.trace_retries < 0:
        raise ValueError("--trace-retries must be non-negative")
    if args.trace_retry_delay < 0:
        raise ValueError("--trace-retry-delay must be non-negative")

    # The completed-manifest check intentionally precedes this private read.
    private_gold_sidecar = getattr(args, "private_gold_sidecar", None)
    if args.gold is not None and private_gold_sidecar is not None:
        raise ValueError("--gold and --private-gold-sidecar are mutually exclusive")
    if private_gold_sidecar is not None:
        gold_path = private_gold_sidecar.expanduser().resolve()
    elif args.gold is not None:
        gold_path = args.gold.expanduser().resolve()
    else:
        raise ValueError("one of --gold or --private-gold-sidecar is required")
    gold = _jsonl_index(gold_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (args.cache_dir or output_dir / "cache").expanduser().resolve()
    image_root = args.image_root.expanduser().resolve() if args.image_root else None
    judge_base_url = getattr(args, "base_url", None)
    judge_api_key = _api_key_from_env(getattr(args, "api_key_env", None))
    judge_wire_api = getattr(args, "wire_api", None)
    judge_enable_thinking = getattr(args, "enable_thinking", None)

    backend = APIBackend(
        provider=args.provider,
        model_name=args.model,
        api_key=judge_api_key,
        base_url=judge_base_url,
        wire_api=judge_wire_api,
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.provider_retries,
    )
    judge = SFTEligibilityJudge(
        backend,
        provider=args.provider,
        model=args.model,
        max_tokens=args.max_tokens,
        enable_thinking=judge_enable_thinking,
    )
    judge_semaphore = asyncio.Semaphore(args.concurrency)

    async def score_trace(trace_path: Path) -> Dict[str, Any]:
        trace = _json_object(trace_path)
        case_id = _trace_case_id(trace)
        if case_id not in gold:
            raise ValueError(f"private gold lacks trace case_id {case_id!r}")
        image_path = _resolve_image_path(
            trace,
            explicit_image=None,
            image_root=image_root,
        )
        if image_path is None:
            raise FileNotFoundError(
                f"cannot resolve original image for teacher trace: {trace_path}"
            )
        packet = build_sft_eligibility_input(
            trace,
            gold[case_id],
            image_path=image_path,
        )
        trace_sha = sha256_file(trace_path)
        report = audit_trace(
            trace_path,
            enforce_source_access_policy=source_policy_active,
            source_access_policy=source_policy_object,
        )
        failures = report.failures(strict_scheduler=True)
        warnings = report.warnings(strict_scheduler=True)
        fatal_audit_errors, _ = classify_sft_audit_failures(
            [asdict(item) for item in failures]
        )
        engineering_valid = bool(
            str(trace.get("termination", "")) == "success"
            and str(trace.get("verdict", "")) in {"real", "fake"}
        )
        cache_key = sft_eligibility_cache_key(
            trace_sha256=trace_sha,
            packet=packet,
            provider=args.provider,
            model=args.model,
            generation_version=judge.generation_identity,
        )
        cache_path = cache_dir / cache_key[:2] / f"{cache_key}.json"
        artifact = None if args.force else _load_cache(cache_path)
        from_cache = artifact is not None
        if artifact is None:
            judgment = None
            judge_audit = None
            if engineering_valid and not fatal_audit_errors:
                async with judge_semaphore:
                    judgment, judge_audit = await judge.judge(
                        packet,
                        image_path=image_path,
                    )
            artifact = build_sft_eligibility_artifact(
                trace=trace,
                trace_sha256=trace_sha,
                packet=packet,
                judgment=judgment,
                judge_audit=judge_audit,
                strict_trace_audit_pass=not failures,
                strict_trace_audit_failures=[asdict(item) for item in failures],
                strict_trace_audit_warnings=[asdict(item) for item in warnings],
            )
            _write_json(cache_path, artifact)
        episode_id = str(artifact.get("episode_id", ""))
        artifact_path = output_dir / f"{episode_id}.sft_eligibility.json"
        _write_json(artifact_path, artifact)
        return {
            "status": "success",
            "case_id": case_id,
            "episode_id": episode_id,
            "artifact": str(artifact_path),
            "artifact_id": artifact.get("artifact_id"),
            "sft_eligibility_pass": artifact.get("gates", {}).get(
                "sft_eligibility_pass"
            ),
            "target_scope": artifact.get("metrics", {}).get(
                "target_scope"
            ),
            "decision_support": artifact.get("metrics", {}).get(
                "decision_support"
            ),
            "retrieval_quality": artifact.get("metrics", {}).get(
                "retrieval_quality"
            ),
            "decisive_observation_ids": artifact.get("metrics", {}).get(
                "decisive_observation_ids"
            ),
            "fatal_errors": artifact.get("metrics", {}).get(
                "fatal_errors"
            ),
            "warnings": artifact.get("metrics", {}).get("warnings"),
            "from_cache": from_cache,
        }

    rows_by_trace: dict[Path, Dict[str, Any]] = {}
    failures: list[Dict[str, Any]] = []

    async def score_trace_isolated(trace_path: Path) -> Dict[str, Any]:
        """Keep one provider/network failure from cancelling sibling tasks."""

        try:
            return await score_trace(trace_path)
        except Exception as exc:
            return _batch_error_row(trace_path, exc)

    pending = list(trace_paths)
    error_history: list[Dict[str, Any]] = []
    retry_round = 0
    try:
        while pending:
            results = await asyncio.gather(
                *(score_trace_isolated(path) for path in pending)
            )
            next_pending: list[Path] = []
            round_failures: list[Dict[str, Any]] = []
            for trace_path, result in zip(pending, results):
                if result.get("status") == "error":
                    error = dict(result)
                    error["retry_round"] = retry_round
                    round_failures.append(error)
                    error_history.append(error)
                    next_pending.append(trace_path)
                else:
                    rows_by_trace[trace_path] = result

            progress = {
                "schema_version": "ifv-sft-eligibility-progress-v1",
                "run_dir": str(run_dir),
                "episode_count": len(trace_paths),
                "completed_count": len(rows_by_trace),
                "pending_count": len(next_pending),
                "retry_round": retry_round,
                "failed_case_ids": [
                    row["case_id"] for row in round_failures if row.get("case_id")
                ],
            }
            _write_json(output_dir / "sft_eligibility_progress.json", progress)
            _write_jsonl(output_dir / "sft_eligibility_errors.jsonl", error_history)

            if not next_pending:
                break
            if retry_round >= args.trace_retries:
                failures = round_failures
                raise RuntimeError(
                    "SFT eligibility audit incomplete after trace retries: "
                    f"{len(failures)} trace(s) remain; "
                    f"{len(rows_by_trace)}/{len(trace_paths)} completed."
                )
            retry_round += 1
            if args.trace_retry_delay:
                await asyncio.sleep(args.trace_retry_delay)
            pending = next_pending
    finally:
        await backend.aclose()

    rows = [rows_by_trace[path] for path in trace_paths]

    accepted = [row for row in rows if row["sft_eligibility_pass"] is True]
    _write_jsonl(output_dir / "accepted_episodes.jsonl", accepted)
    storage_dir = (
        args.storage_dir.expanduser().resolve()
        if args.storage_dir
        else _default_storage_dir(run_dir, output_dir)
    )
    storage_manifest = _existing_storage_manifest(
        storage_dir,
        run_dir=run_dir,
        eligibility_dir=output_dir,
    )
    if storage_manifest is None:
        storage_manifest = stage_release(
            [(run_dir, output_dir, None)],
            storage_dir,
            minimum_accepted_cases=0,
        )
    summary = {
        "schema_version": "ifv-sft-eligibility-summary-v2",
        "run_dir": str(run_dir),
        "private_gold": {"path": str(gold_path), "sha256": sha256_file(gold_path)},
        "provider": args.provider,
        "model": args.model,
        "judge_config": {
            "base_url": judge_base_url,
            "wire_api": judge_wire_api,
            "enable_thinking": judge_enable_thinking,
        },
        "episode_count": len(rows),
        "passed_count": len(accepted),
        "storage": {
            "run_dir": str(storage_dir),
            "accepted_case_count": storage_manifest.get("accepted_case_count", 0),
            "rejected_case_count": storage_manifest.get("rejected_case_count", 0),
            "manifest": str(
                storage_dir / "accepted_release_manifest.json"
            ),
        },
        "rows": rows,
    }
    _write_json(output_dir / "sft_eligibility_summary.json", summary)
    run_manifest = _json_object(run_dir / "run_manifest.json")
    run_manifest["sft_storage"] = summary["storage"]
    _write_json(run_dir / "run_manifest.json", run_manifest)
    return summary


def main() -> None:
    summary = asyncio.run(_run(_parse_args()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
