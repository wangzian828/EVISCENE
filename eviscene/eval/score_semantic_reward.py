"""Create frozen, gold-free semantic reward artifacts from canonical traces."""

from __future__ import annotations

from eviscene.environment import get_env

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping

from scripts.audit_real_trace import audit_trace
from eviscene.orchestrator.source_access import SourceAccessPolicy
from eviscene.orchestrator.llm_backend import APIBackend
from eviscene.trajectory.semantic_reward import (
    SemanticRewardCache,
    SemanticRewardJudge,
    build_semantic_reward_artifact,
    build_semantic_reward_input,
    sha256_file,
    sha256_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score completed canonical v4 traces with a frozen Gemini semantic "
            "judge. This command never reads evaluator-private gold."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path)
    source.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--allow-missing-image", action="store_true")
    parser.add_argument("--provider", default="gemini", choices=["gemini"])
    parser.add_argument(
        "--model",
        default=get_env("EVISCENE_SEMANTIC_JUDGE_MODEL", "gemini-3.7-flash"),
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _trace_paths(args: argparse.Namespace) -> list[Path]:
    if args.trace:
        path = args.trace.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return [path]
    run_dir = args.run_dir.expanduser().resolve()
    paths = sorted((run_dir / "traces").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no canonical traces under {run_dir / 'traces'}")
    return paths


def _manifest_image_root(run_dir: Path) -> Path | None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _load_json(manifest_path)
    benchmark = manifest.get("benchmark")
    if not isinstance(benchmark, Mapping):
        return None
    configured = str(benchmark.get("path", "")).strip()
    if not configured:
        return None
    path = Path(configured).expanduser()
    if path.is_file():
        return path.parent
    return path if path.is_dir() else None


def _resolve_image_path(
    trace: Mapping[str, Any],
    *,
    explicit_image: Path | None,
    image_root: Path | None,
) -> Path | None:
    if explicit_image is not None:
        path = explicit_image.expanduser().resolve()
        return path if path.is_file() else None
    state = trace.get("state") if isinstance(trace.get("state"), Mapping) else {}
    runtime_case = state.get("runtime_case") if isinstance(state, Mapping) else {}
    raw_path = (
        runtime_case.get("image_path", "")
        if isinstance(runtime_case, Mapping)
        else ""
    )
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if image_root is not None and str(raw_path).strip():
        candidate = (image_root / str(raw_path)).resolve()
        if candidate.is_file():
            return candidate
    return None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


async def _run(args: argparse.Namespace) -> Dict[str, Any]:
    trace_paths = _trace_paths(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_image_root = _manifest_image_root(args.run_dir.expanduser().resolve()) if args.run_dir else None
    image_root = (args.image_root or run_image_root)
    if image_root is not None:
        image_root = image_root.expanduser().resolve()
    cache = SemanticRewardCache(
        (args.cache_dir or output_dir / "cache").expanduser().resolve()
    )
    source_policy_active = True
    source_policy_object = None
    if args.run_dir:
        run_manifest = _load_json(
            args.run_dir.expanduser().resolve() / "run_manifest.json"
        )
        source_policy = run_manifest.get("source_access_policy")
        source_policy_active = bool(
            isinstance(source_policy, Mapping) and source_policy.get("active") is True
        )
        if source_policy_active:
            policy_path = str(source_policy.get("path") or "").strip()
            if not policy_path:
                raise ValueError(
                    "active source_access_policy in run manifest lacks policy path"
                )
            source_policy_object = SourceAccessPolicy.load(policy_path)
    backend = APIBackend(
        provider=args.provider,
        model_name=args.model,
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    judge = SemanticRewardJudge(
        backend,
        provider=args.provider,
        model=args.model,
        max_tokens=args.max_tokens,
    )
    rows: list[Dict[str, Any]] = []
    try:
        for trace_path in trace_paths:
            trace = _load_json(trace_path)
            trace_sha = sha256_file(trace_path)
            recorded_verdict = str(trace.get("verdict", "")).strip()
            if recorded_verdict not in {"real", "fake"}:
                rows.append(
                    {
                        "case_id": str(trace.get("case_id") or trace_path.stem),
                        "episode_id": str(trace.get("image_id") or trace_path.stem),
                        "artifact": None,
                        "artifact_id": None,
                        "semantic_audit_pass": False,
                        "from_cache": False,
                        "excluded_reason": "non_binary_verdict",
                    }
                )
                continue
            image_path = _resolve_image_path(
                trace,
                explicit_image=args.image,
                image_root=image_root,
            )
            if image_path is None and not args.allow_missing_image:
                raise FileNotFoundError(
                    f"cannot resolve original image for canonical trace: {trace_path}"
                )
            packet = build_semantic_reward_input(trace, image_path=image_path)
            cache_key = cache.key(
                trace_sha256=trace_sha,
                reward_input_sha256=sha256_json(packet),
                provider=args.provider,
                model=args.model,
                generation_version=judge.generation_identity,
            )
            artifact = None if args.force else cache.load(cache_key)
            from_cache = artifact is not None
            if artifact is None:
                report = audit_trace(
                    trace_path,
                    enforce_source_access_policy=source_policy_active,
                    source_access_policy=source_policy_object,
                )
                failures = report.failures(strict_scheduler=True)
                judgment, judge_audit = await judge.judge(
                    packet,
                    image_path=image_path,
                )
                artifact = build_semantic_reward_artifact(
                    trace=trace,
                    trace_sha256=trace_sha,
                    packet=packet,
                    judgment=judgment,
                    judge_audit=judge_audit,
                    strict_trace_audit_pass=not failures,
                    strict_trace_audit_failures=[asdict(item) for item in failures],
                )
                cache.store(cache_key, artifact)
            case_id = str(artifact.get("case_id", ""))
            episode_id = str(
                artifact.get("rollout", {}).get("episode_id", "")
            )
            if not case_id or not episode_id:
                raise ValueError(f"semantic artifact has no case_id: {trace_path}")
            output_path = output_dir / f"{episode_id}.semantic_reward.json"
            _write_json(output_path, artifact)
            rows.append(
                {
                    "case_id": case_id,
                    "episode_id": episode_id,
                    "artifact": str(output_path),
                    "artifact_id": artifact.get("artifact_id"),
                    "semantic_audit_pass": artifact.get("gates", {}).get("semantic_audit_pass"),
                    "from_cache": from_cache,
                }
            )
    finally:
        await backend.aclose()
    summary = {
        "schema_version": "ifv-semantic-reward-summary-v2",
        "case_count": len({str(row.get("case_id", "")) for row in rows}),
        "episode_count": len(rows),
        "passed_count": sum(
            1 for row in rows if row.get("semantic_audit_pass") is True
        ),
        "rows": rows,
    }
    _write_json(output_dir / "semantic_reward_summary.json", summary)
    return summary


def main() -> None:
    args = _parse_args()
    summary = asyncio.run(_run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
