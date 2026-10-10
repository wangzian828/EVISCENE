"""Build a split-safe full-trajectory SFT dataset from completed runs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eviscene.trajectory.schema import (
    DatasetPerceptionExample,
    DatasetTrajectorySFTExample,
    PerceptionExample,
    TrajectorySFTExample,
 )
from eviscene.trajectory.exporter import export_trajectory_sft_example


SPLITS = ("train", "validation", "test")
ACCEPTED_RELEASE_SCHEMAS = frozenset(
    {
        "ifv-accepted-teacher-release-v2",
        "ifv-accepted-teacher-release-v3",
        "ifv-accepted-teacher-release-v4",
    }
)
# Use bytes/3 for conservative admission; final token counts require
# the actual model processor, especially near the context limit.
UTF8_BYTES_PER_APPROX_TOKEN = 3
DEFAULT_SHORT_TRAJECTORY_MAX_TOKENS = 131072
DETERMINISTIC_FATAL_TEACHER_REASONS = frozenset(
    {
        "incorrect_result",
        "engineering_error",
        "legacy_core_ownership",
    }
)


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").split("\n"),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    lines = [
        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def _export_trace_file_job(
    job: tuple[str, Path, Path | None, Dict[str, Any]],
) -> tuple[str, TrajectorySFTExample | None, str]:
    """Export one trace while keeping raw trace data out of the coordinator."""

    (
        episode_id,
        trace_path,
        runtime_store_path,
        source_metadata,
    ) = job
    try:
        trace = _load_json(trace_path)
        if runtime_store_path is not None:
            state = dict(_mapping(trace.get("state")))
            runtime_store = dict(_mapping(state.get("runtime_store")))
            runtime_store["runtime_path"] = str(runtime_store_path)
            state["runtime_store"] = runtime_store
            trace["state"] = state
        return (
            episode_id,
            export_trajectory_sft_example(
                trace,
                source_metadata=source_metadata,
            ),
            "",
        )
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        if not isinstance(exc, ValueError):
            message = f"{type(exc).__name__}: {message}"
        return episode_id, None, message


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> List[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _collect_runtime_ids(trace: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).endswith("_id") and isinstance(child, str):
                    if child.strip():
                        result.add(child.strip())
                elif str(key).endswith("_ids") and isinstance(child, list):
                    result.update(
                        str(item).strip()
                        for item in child
                        if str(item).strip()
                    )
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(trace)
    return result


def _cross_case_source_families(trace: Mapping[str, Any]) -> List[str]:
    """Return only source identities that can indicate cross-case leakage.

    ``source_family`` has two runtime meanings: a content digest when fetched
    content is available, and a registered-domain fallback otherwise.  A
    shared content digest can make two cases near-duplicates; merely visiting
    the same host (for example Wikipedia) cannot.  Domain families remain
    useful inside one investigation for source-diversity accounting, but must
    not merge unrelated cases into one train/validation component.
    """
    state = _mapping(trace.get("state"))
    investigation = _mapping(state.get("investigation_state"))
    families = {
        str(item.get("source_family", "")).strip()
        for item in _rows(investigation.get("evidence"))
        if str(item.get("source_family", "")).strip().startswith("content:")
    }
    return sorted(families)


def _trace_case_id(trace: Mapping[str, Any], episode_id: str) -> str:
    state = _mapping(trace.get("state"))
    runtime_case = _mapping(state.get("runtime_case"))
    return str(
        runtime_case.get("case_id")
        or trace.get("case_id")
        or trace.get("image_id")
        or episode_id
    ).strip()


def _load_fixed_case_split(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = _load_jsonl(path.expanduser().resolve())
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        split = str(row.get("split", "")).strip()
        group_id = str(row.get("split_group_id", "")).strip()
        if not case_id or split not in {"train", "validation"} or not group_id:
            raise ValueError(
                "fixed case split rows require case_id, train/validation split, "
                "and split_group_id"
            )
        if case_id in result:
            raise ValueError(f"duplicate fixed case split case_id: {case_id}")
        result[case_id] = dict(row)
    if not result:
        raise ValueError("fixed case split is empty")
    return result


def _load_gate_artifacts(
    root: Path,
    *,
    suffix: str,
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.expanduser().resolve().rglob(f"*{suffix}")):
        payload = _load_json(path)
        episode_id = str(
            payload.get("episode_id")
            or _mapping(payload.get("rollout")).get("episode_id", "")
        ).strip()
        if not episode_id:
            raise ValueError(f"gate artifact lacks episode_id: {path}")
        if episode_id in result:
            raise ValueError(f"duplicate gate artifact episode_id: {episode_id}")
        result[episode_id] = payload
    return result


def _load_canonical_accepted_release(
    path: Path,
) -> tuple[Path, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load the one canonical source of frozen teacher examples.

    The accepted release already contains the post-judge policy export.  A
    frozen dataset export must consume those rows instead of reconstructing
    policy examples from the original rollout directories, whose historical
    exports may have used a stricter pre-judge verdict-chain gate.
    """

    root = path.expanduser().resolve()
    manifest_path = root / "accepted_release_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") not in ACCEPTED_RELEASE_SCHEMAS:
        raise ValueError(
            "accepted-release has unsupported schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if not (root / "run_manifest.json").is_file():
        raise ValueError("accepted-release is missing run_manifest.json")
    eligibility_dir = root / "eligibility"
    if not eligibility_dir.is_dir():
        raise ValueError("accepted-release is missing eligibility/")

    selected_rows = _load_jsonl(root / "selected_episodes.jsonl")
    selected_by_episode: Dict[str, Dict[str, Any]] = {}
    selected_cases: set[str] = set()
    for row in selected_rows:
        episode_id = str(row.get("episode_id", "")).strip()
        case_id = str(row.get("case_id", "")).strip()
        trace_sha256 = str(row.get("source_trace_sha256", "")).strip()
        if not episode_id or not case_id or not trace_sha256:
            raise ValueError(
                "accepted-release selected episode requires "
                "case_id, episode_id and source_trace_sha256"
            )
        if episode_id in selected_by_episode:
            raise ValueError(
                f"accepted-release has duplicate episode_id: {episode_id}"
            )
        if case_id in selected_cases:
            raise ValueError(
                f"accepted-release has duplicate selected case_id: {case_id}"
            )
        if row.get("deterministic_hard_gate_pass") is not True:
            raise ValueError(
                "accepted-release contains a selected episode without a "
                f"passing deterministic gate: {episode_id}"
            )
        selected_by_episode[episode_id] = row
        selected_cases.add(case_id)

    expected_count = int(manifest.get("accepted_case_count", -1))
    if expected_count != len(selected_cases):
        raise ValueError(
            "accepted-release accepted_case_count does not match "
            f"selected episodes: {expected_count} != {len(selected_cases)}"
        )
    if not selected_by_episode:
        raise ValueError("accepted-release selected_episodes.jsonl is empty")
    return root, selected_by_episode, manifest


def _accepted_release_runtime_store(
    root: Path,
    row: Mapping[str, Any],
    *,
    required: bool,
) -> Path | None:
    raw = str(row.get("runtime_store_path", "")).strip()
    episode_id = str(row.get("episode_id", "")).strip()
    if not raw:
        if required:
            raise ValueError(
                "accepted-release v4 selected episode has no runtime store: "
                f"{episode_id}"
            )
        return None
    root = root.resolve()
    runtime_store = (root / raw).resolve()
    if runtime_store != root and root not in runtime_store.parents:
        raise ValueError(
            "accepted-release runtime store escapes the release root: "
            f"{episode_id}"
        )
    if not runtime_store.is_dir():
        raise ValueError(
            "accepted-release runtime store is missing: "
            f"{episode_id}: {runtime_store}"
        )
    if not any((runtime_store / "context").glob("*.json")):
        raise ValueError(
            "accepted-release runtime store has no context manifests: "
            f"{episode_id}"
        )
    if not (runtime_store / "artifacts" / "sha256").is_dir():
        raise ValueError(
            "accepted-release runtime store has no artifact store: "
            f"{episode_id}"
        )
    expected_files = int(row.get("runtime_store_file_count", 0) or 0)
    if required and expected_files < 1:
        raise ValueError(
            "accepted-release v4 selected episode has invalid runtime store "
            f"metadata: {episode_id}"
        )
    return runtime_store


def _deterministic_teacher_quality(score: Mapping[str, Any]) -> Dict[str, Any]:
    if not score:
        return {
            "hard_gate_pass": False,
            "fatal_reasons": ["missing_training_quality_score"],
            "red_flags": [],
        }
    reasons = [
        str(item)
        for item in score.get("training_exclusion_reasons", []) or []
        if str(item).strip()
    ]
    fatal_reasons = [
        reason
        for reason in reasons
        if reason in DETERMINISTIC_FATAL_TEACHER_REASONS
    ]
    if score.get("training_eligible") is False and not reasons:
        fatal_reasons.append("teacher_quality_gate_failed")
    return {
        "hard_gate_pass": not fatal_reasons,
        "fatal_reasons": list(dict.fromkeys(fatal_reasons)),
        "red_flags": [
            reason
            for reason in reasons
            if reason not in DETERMINISTIC_FATAL_TEACHER_REASONS
        ],
    }


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[max(root_left, root_right)] = min(root_left, root_right)


def _assign_split(
    group_id: str,
    *,
    train_ratio: float,
    validation_ratio: float,
    seed: str,
) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def export_dataset(
    run_dirs: Sequence[Path] | None,
    output_dir: Path,
    *,
    accepted_release_path: Path | None = None,
    case_split_path: Path | None = None,
    eligibility_dir: Path | None = None,
    semantic_reward_dir: Path | None = None,
    require_frozen_gates: bool = False,
    minimum_accepted_cases: int | None = None,
    require_all_validation_cases: bool = True,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: str = "ifv-policy-v1",
    short_max_tokens: int = DEFAULT_SHORT_TRAJECTORY_MAX_TOKENS,
    export_concurrency: int = 1,
) -> Dict[str, Any]:
    accepted_release: Path | None = None
    accepted_release_rows: Dict[str, Dict[str, Any]] = {}
    accepted_release_manifest: Dict[str, Any] = {}
    if accepted_release_path is not None:
        if run_dirs:
            raise ValueError(
                "accepted-release mode cannot be combined with --run-dir"
            )
        (
            accepted_release,
            accepted_release_rows,
            accepted_release_manifest,
        ) = (
            _load_canonical_accepted_release(accepted_release_path)
        )
        run_dirs = [accepted_release]
        canonical_eligibility_dir = accepted_release / "eligibility"
        if eligibility_dir is not None and (
            eligibility_dir.expanduser().resolve() != canonical_eligibility_dir
        ):
            raise ValueError(
                "accepted-release mode owns its eligibility directory; "
                "do not pass a second --eligibility-dir"
            )
        eligibility_dir = canonical_eligibility_dir
        require_frozen_gates = True
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    frozen_inputs = (case_split_path, eligibility_dir, semantic_reward_dir)
    if any(value is not None for value in frozen_inputs):
        require_frozen_gates = True
    if require_frozen_gates and case_split_path is None:
        raise ValueError("frozen SFT export requires --case-split")
    if require_frozen_gates and eligibility_dir is None:
        raise ValueError("frozen SFT export requires --eligibility-dir")
    if minimum_accepted_cases is not None and minimum_accepted_cases < 0:
        raise ValueError("minimum_accepted_cases must be non-negative")
    if require_frozen_gates and minimum_accepted_cases is None:
        minimum_accepted_cases = 40
    if train_ratio <= 0 or validation_ratio < 0:
        raise ValueError("split ratios must be non-negative")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be below 1")
    if short_max_tokens < 1:
        raise ValueError("short_max_tokens must be positive")
    if export_concurrency < 1:
        raise ValueError("export_concurrency must be at least 1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"dataset output directory must be new or empty: {output_dir}"
        )

    fixed_split = (
        _load_fixed_case_split(case_split_path)
        if case_split_path is not None
        else None
    )
    eligibility_by_episode = (
        _load_gate_artifacts(eligibility_dir, suffix=".sft_eligibility.json")
        if eligibility_dir is not None
        else {}
    )
    semantic_by_episode = (
        _load_gate_artifacts(semantic_reward_dir, suffix=".semantic_reward.json")
        if semantic_reward_dir is not None
        else {}
    )
    trajectory_by_episode: Dict[str, TrajectorySFTExample] = {}
    perception_by_episode: Dict[str, PerceptionExample] = {}
    action_only_by_episode: Dict[str, Dict[str, Any]] = {}
    episode_metadata: Dict[str, Dict[str, Any]] = {}
    export_jobs: Dict[
        str, tuple[str, Path, Path | None, Dict[str, Any]]
    ] = {}
    score_by_episode: Dict[str, Dict[str, Any]] = {}
    source_runs: List[Dict[str, Any]] = []
    for raw_run_dir in run_dirs:
        run_dir = raw_run_dir.expanduser().resolve()
        manifest = _load_json(run_dir / "run_manifest.json")
        source_runs.append(
            {
                "run_id": manifest.get("run_id"),
                "path": str(run_dir),
                "git_commit": manifest.get("git_commit"),
                "release_id": _mapping(manifest.get("benchmark")).get(
                    "release_id"
                ),
            }
        )
        for row in _load_jsonl(run_dir / "trajectory_scores.jsonl"):
            score_key = str(
                row.get("episode_id") or row.get("case_id", "")
            ).strip()
            if score_key:
                score_by_episode[score_key] = dict(row)
        for row in _load_jsonl(run_dir / "perception_trajectories.jsonl"):
            example = PerceptionExample.model_validate(row)
            if example.episode_id in perception_by_episode:
                raise ValueError(
                    "duplicate perception episode across source runs: "
                    f"{example.episode_id}"
                )
            perception_by_episode[example.episode_id] = example
        for trace_path in sorted((run_dir / "traces").glob("*.json")):
            trace = _load_json(trace_path)
            episode_id = str(
                trace.get("image_id") or trace_path.stem
            ).strip()
            case_id = _trace_case_id(trace, episode_id)
            trace_sha256 = _sha256(trace_path)
            score = score_by_episode.get(episode_id, {})
            deterministic_quality = _deterministic_teacher_quality(score)
            accepted_release_row = (
                accepted_release_rows.get(episode_id)
                if accepted_release is not None
                else None
            )
            runtime_store_path: Path | None = None
            if accepted_release is not None:
                if accepted_release_row is None:
                    raise ValueError(
                        "accepted-release trace is not listed as selected: "
                        f"{episode_id}"
                    )
                if str(
                    accepted_release_row.get("source_trace_sha256", "")
                ) != trace_sha256:
                    raise ValueError(
                        "accepted-release trace SHA-256 mismatch: "
                        f"{episode_id}"
                    )
                runtime_store_path = _accepted_release_runtime_store(
                    accepted_release,
                    accepted_release_row,
                    required=(
                        accepted_release_manifest.get("schema_version")
                        == "ifv-accepted-teacher-release-v4"
                        and str(
                            accepted_release_row.get(
                                "decision_policy_version",
                                "",
                            )
                        )
                        == "unified-react-v1"
                    ),
                )
                buckets = accepted_release_row.get(
                    "training_buckets",
                    ["reasoning_sft"],
                )
                if (
                    isinstance(buckets, list)
                    and "reasoning_sft" not in buckets
                ):
                    # Action-only traces are intentionally excluded from the
                    # reasoning policy corpus, but their independent
                    # perception target remains valid training data.
                    action_only_by_episode[episode_id] = {
                        "case_id": case_id,
                        "source_run_id": manifest.get("run_id"),
                        "source_trace": str(trace_path),
                        "runtime_store_path": (
                            str(runtime_store_path)
                            if runtime_store_path is not None
                            else None
                        ),
                    }
                    continue
                score = {
                    "episode_id": episode_id,
                    "total": float(
                        accepted_release_row.get("teacher_score", 0.0)
                        or 0.0
                    ),
                    "training_eligible": True,
                    "training_exclusion_reasons": [],
                }
                deterministic_quality = {
                    "hard_gate_pass": True,
                    "fatal_reasons": [
                        str(item)
                        for item in accepted_release_row.get(
                            "deterministic_fatal_reasons", []
                        )
                        or []
                    ],
                    "red_flags": [
                        str(item)
                        for item in accepted_release_row.get(
                            "deterministic_red_flags", []
                        )
                        or []
                    ],
                }
            training_eligible = bool(score.get("training_eligible", False))
            eligibility = eligibility_by_episode.get(episode_id, {})
            semantic = semantic_by_episode.get(episode_id, {})
            if semantic:
                semantic_episode_id = str(
                    _mapping(semantic.get("rollout")).get(
                        "episode_id",
                        semantic.get("episode_id", ""),
                    )
                )
                if (
                    str(semantic.get("case_id", "")) != case_id
                    or semantic_episode_id != episode_id
                    or str(
                        _mapping(semantic.get("source_trace")).get(
                            "sha256", ""
                        )
                    )
                    != trace_sha256
                ):
                    raise ValueError(
                        "semantic diagnostic artifact does not match trace: "
                        f"{episode_id}"
                    )
            if require_frozen_gates:
                eligibility_gates = _mapping(eligibility.get("gates"))
                split_row = (fixed_split or {}).get(case_id, {})
                runtime_case = _mapping(
                    _mapping(trace.get("state")).get("runtime_case")
                )
                structured_gate_pass = bool(
                    str(eligibility.get("case_id", "")) == case_id
                    and str(eligibility.get("episode_id", "")) == episode_id
                    and eligibility_gates.get("engineering_valid") is True
                    and eligibility_gates.get("sft_eligibility_pass") is True
                    and str(
                        _mapping(eligibility.get("source_trace")).get(
                            "sha256", ""
                        )
                    )
                    == trace_sha256
                    and str(split_row.get("image_sha256", ""))
                    == str(runtime_case.get("image_sha256", ""))
                )
                training_eligible = bool(
                    structured_gate_pass
                    and deterministic_quality["hard_gate_pass"]
                )
            exclusion_reasons = [
                str(item)
                for item in score.get(
                    "training_exclusion_reasons",
                    ["missing_training_quality_score"],
                )
                or []
            ]
            if not score:
                exclusion_reasons = ["missing_training_quality_score"]
            if require_frozen_gates and not training_eligible:
                frozen_reasons = []
                if not structured_gate_pass:
                    frozen_reasons.append("structured_teacher_gate_failed")
                frozen_reasons.extend(deterministic_quality["fatal_reasons"])
                exclusion_reasons = [
                    *frozen_reasons,
                    *exclusion_reasons,
                ]
            elif require_frozen_gates:
                exclusion_reasons = []
            episode_metadata[episode_id] = {
                "episode_id": episode_id,
                "case_id": case_id,
                "source_run_id": manifest.get("run_id"),
                "source_trace": str(trace_path),
                "source_family_keys": _cross_case_source_families(trace),
                "runtime_ids": sorted(_collect_runtime_ids(trace)),
                "teacher_score": float(
                    score.get("total", 0.0)
                    or 0.0
                ),
                "training_eligible": training_eligible,
                "training_exclusion_reasons": exclusion_reasons,
                "deterministic_hard_gate_pass": deterministic_quality[
                    "hard_gate_pass"
                ],
                "deterministic_fatal_reasons": deterministic_quality[
                    "fatal_reasons"
                ],
                "deterministic_red_flags": deterministic_quality["red_flags"],
                "structured_eligibility_pass": eligibility.get("gates", {}).get(
                    "sft_eligibility_pass"
                ),
                "semantic_audit_pass": semantic.get("gates", {}).get(
                    "semantic_audit_pass"
                ),
                "sft_eligibility_artifact_id": eligibility.get("artifact_id"),
                "semantic_reward_artifact_id": semantic.get("artifact_id"),
                "decision_policy_version": str(
                    trace.get("decision_policy_version")
                    or _mapping(trace.get("state")).get(
                        "decision_policy_version",
                        "",
                    )
                    or ""
                ),
                "runtime_store_path": (
                    str(runtime_store_path)
                    if runtime_store_path is not None
                    else None
                ),
            }
            export_jobs[episode_id] = (
                episode_id,
                trace_path,
                runtime_store_path,
                {
                    "source_run_id": manifest.get("run_id"),
                    "runtime_commit": manifest.get("git_commit"),
                    "release_id": _mapping(manifest.get("benchmark")).get(
                        "release_id"
                    ),
                    "runtime_contract_version": _mapping(
                        manifest.get("benchmark")
                    ).get("runtime_contract_version", ""),
                    "process_reference_protocol_version": manifest.get(
                        "process_reference_protocol_version",
                        "",
                    ),
                },
            )

    ordered_jobs = [
        export_jobs[episode_id]
        for episode_id in sorted(export_jobs)
    ]
    executor: ThreadPoolExecutor | None = None
    if export_concurrency == 1 or len(ordered_jobs) <= 1:
        export_results = (
            _export_trace_file_job(job)
            for job in ordered_jobs
        )
        for episode_id, trajectory, export_error in export_results:
            metadata = episode_metadata[episode_id]
            metadata["trajectory_export_error"] = export_error
            if trajectory is None:
                continue
            trajectory_by_episode[episode_id] = trajectory
            metadata["trajectory_token_count_estimate"] = (
                trajectory.token_count_estimate
            )
            metadata["tool_call_count"] = trajectory.tool_call_count
    else:
        executor = ThreadPoolExecutor(
            max_workers=min(export_concurrency, len(ordered_jobs)),
            thread_name_prefix="trajectory-export",
        )
        futures = {
            executor.submit(_export_trace_file_job, job): job[0]
            for job in ordered_jobs
        }
        try:
            for future in as_completed(futures):
                episode_id, trajectory, export_error = future.result()
                metadata = episode_metadata[episode_id]
                metadata["trajectory_export_error"] = export_error
                if trajectory is None:
                    continue
                trajectory_by_episode[episode_id] = trajectory
                metadata["trajectory_token_count_estimate"] = (
                    trajectory.token_count_estimate
                )
                metadata["tool_call_count"] = trajectory.tool_call_count
        finally:
            executor.shutdown(wait=True)

    all_episodes = sorted(episode_metadata)
    policy_versions = {
        str(metadata["decision_policy_version"])
        for metadata in episode_metadata.values()
    }
    if len(policy_versions) > 1:
        raise ValueError(
            "SFT dataset cannot mix decision_policy_version values: "
            + ", ".join(sorted(policy_versions))
        )
    decision_policy_version = next(iter(policy_versions), "")

    if require_frozen_gates:
        missing_split = sorted(
            {
                str(metadata["case_id"])
                for metadata in episode_metadata.values()
                if str(metadata["case_id"]) not in (fixed_split or {})
            }
        )
        if missing_split:
            raise ValueError(
                "fixed case split lacks teacher cases: "
                + ", ".join(missing_split[:10])
            )
    if require_frozen_gates:
        for episode_id in all_episodes:
            metadata = episode_metadata[episode_id]
            if episode_id not in trajectory_by_episode:
                metadata["training_eligible"] = False
                metadata["training_exclusion_reasons"] = list(
                    dict.fromkeys(
                        [
                            "no_exportable_trajectory_sft",
                            *metadata["training_exclusion_reasons"],
                        ]
                    )
                )
        eligible_by_case: Dict[str, List[str]] = defaultdict(list)
        for episode_id in all_episodes:
            metadata = episode_metadata[episode_id]
            if metadata["training_eligible"]:
                eligible_by_case[str(metadata["case_id"])].append(episode_id)
        selected_by_case: Dict[str, str] = {}
        for case_id, candidates in eligible_by_case.items():
            selected_by_case[case_id] = max(
                candidates,
                key=lambda episode_id: (
                    -len(episode_metadata[episode_id]["deterministic_red_flags"]),
                    float(episode_metadata[episode_id]["teacher_score"]),
                    episode_id,
                ),
            )
        for episode_id in all_episodes:
            metadata = episode_metadata[episode_id]
            if (
                metadata["training_eligible"]
                and selected_by_case.get(str(metadata["case_id"])) != episode_id
            ):
                metadata["training_eligible"] = False
                metadata["training_exclusion_reasons"] = list(
                    dict.fromkeys(
                        [
                            "lower_quality_rollout_for_case",
                            *metadata["training_exclusion_reasons"],
                        ]
                    )
                )
        episodes = sorted(selected_by_case.values())
        accepted_case_ids = {
            str(episode_metadata[episode_id]["case_id"])
            for episode_id in episodes
        }
        if minimum_accepted_cases is not None and (
            len(accepted_case_ids) < minimum_accepted_cases
        ):
            raise ValueError(
                "frozen SFT export accepted too few cases: "
                f"{len(accepted_case_ids)} < {minimum_accepted_cases}"
            )
        if require_all_validation_cases:
            policy_case_ids = {
                str(episode_metadata[episode_id]["case_id"])
                for episode_id in all_episodes
            }
            expected_validation = {
                case_id
                for case_id, row in (fixed_split or {}).items()
                if (
                    row.get("split") == "validation"
                    and case_id in policy_case_ids
                )
            }
            missing_validation = sorted(expected_validation - accepted_case_ids)
            if missing_validation:
                raise ValueError(
                    "frozen SFT export lacks accepted validation cases: "
                    + ", ".join(missing_validation[:10])
                )
    else:
        episodes = [
            episode_id
            for episode_id in all_episodes
            if episode_metadata[episode_id]["training_eligible"]
            and episode_id in trajectory_by_episode
        ]
    excluded_episode_rows = [
        {
            **episode_metadata[episode_id],
            "trajectory_sft_rows": int(episode_id in trajectory_by_episode),
            "perception_example_count": int(
                episode_id in perception_by_episode
            ),
        }
        for episode_id in all_episodes
        if episode_id not in episodes
    ]
    union = _UnionFind(episodes)
    family_episodes: Dict[str, List[str]] = defaultdict(list)
    for episode_id in episodes:
        for family in episode_metadata[episode_id]["source_family_keys"]:
            family_episodes[family].append(episode_id)
    for members in family_episodes.values():
        for member in members[1:]:
            union.union(members[0], member)

    components: Dict[str, List[str]] = defaultdict(list)
    for episode_id in episodes:
        components[union.find(episode_id)].append(episode_id)
    split_by_episode: Dict[str, str] = {}
    group_by_episode: Dict[str, str] = {}
    for members in components.values():
        if fixed_split is not None:
            member_case_splits = {
                str(fixed_split[str(episode_metadata[item]["case_id"])]["split"])
                for item in members
            }
            if len(member_case_splits) != 1:
                raise ValueError(
                    "fixed case split separates one source-family group: "
                    + ", ".join(sorted(member_case_splits))
                )
            for episode_id in members:
                fixed_row = fixed_split[
                    str(episode_metadata[episode_id]["case_id"])
                ]
                split_by_episode[episode_id] = str(fixed_row["split"])
                group_by_episode[episode_id] = str(
                    fixed_row["split_group_id"]
                )
            continue
        group_payload = {
            "episodes": sorted(members),
            "families": sorted(
                {
                    family
                    for episode_id in members
                    for family in episode_metadata[episode_id][
                        "source_family_keys"
                    ]
                }
            ),
        }
        group_id = hashlib.sha256(
            json.dumps(
                group_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        split = _assign_split(
            group_id,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            seed=seed,
        )
        for episode_id in members:
            split_by_episode[episode_id] = split
            group_by_episode[episode_id] = group_id

    split_rows: Dict[str, List[Dict[str, Any]]] = {
        split: [] for split in SPLITS
    }
    perception_split_rows: Dict[str, List[Dict[str, Any]]] = {
        split: [] for split in SPLITS
    }
    long_holdout_rows: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []
    accepted_episode_rows: List[Dict[str, Any]] = []
    for episode_id in episodes:
        metadata = episode_metadata[episode_id]
        split = split_by_episode[episode_id]
        trajectory = trajectory_by_episode[episode_id]
        approximate_tokens = (
            trajectory.token_count_estimate + UTF8_BYTES_PER_APPROX_TOKEN - 1
        ) // UTF8_BYTES_PER_APPROX_TOKEN
        is_short = approximate_tokens <= short_max_tokens
        accepted_row = {
            **metadata,
            "split": split,
            "split_group_id": group_by_episode[episode_id],
            "sft_route": "short_sft" if is_short else "long_holdout",
            "short_max_tokens": short_max_tokens,
            "trajectory_approx_tokens": approximate_tokens,
        }
        metadata_rows.append(accepted_row)
        accepted_episode_rows.append(
            {
                key: accepted_row.get(key)
                for key in (
                    "case_id",
                    "episode_id",
                    "split",
                    "split_group_id",
                    "source_run_id",
                    "source_trace",
                    "sft_eligibility_artifact_id",
                    "semantic_reward_artifact_id",
                    "deterministic_hard_gate_pass",
                    "deterministic_fatal_reasons",
                    "deterministic_red_flags",
                    "teacher_score",
                    "trajectory_token_count_estimate",
                    "tool_call_count",
                    "sft_route",
                )
            }
        )
        dataset_trajectory = DatasetTrajectorySFTExample(
            **trajectory.model_dump(mode="json"),
            dataset_version=(
                "ifv-trajectory-sft-dataset-v3"
                if decision_policy_version == "unified-react-v1"
                else "ifv-trajectory-sft-dataset-v2"
            ),
            split=split,
            split_group_id=group_by_episode[episode_id],
            source_family_keys=metadata["source_family_keys"],
            teacher_score=metadata["teacher_score"],
        ).model_dump(mode="json")
        if is_short:
            split_rows[split].append(dataset_trajectory)
        else:
            long_holdout_rows.append(
                {
                    **dataset_trajectory,
                    "holdout_reason": "trajectory_exceeds_short_token_budget",
                    "short_max_tokens": short_max_tokens,
                }
            )
        perception_example = perception_by_episode.get(episode_id)
        if perception_example is not None and is_short:
            dataset_perception = DatasetPerceptionExample(
                **perception_example.model_dump(mode="json"),
                split=split,
                split_group_id=group_by_episode[episode_id],
            )
            perception_split_rows[split].append(
                dataset_perception.model_dump(mode="json")
            )

    # Perception is an independent image-only task. Do not make it depend on
    # the policy trace having readable provider thought. In particular, an
    # accepted action-only trace is excluded from reasoning SFT but still has
    # a valid PerceptionReport target.
    action_only_perception_count = 0
    for episode_id, action_only in action_only_by_episode.items():
        perception_example = perception_by_episode.get(episode_id)
        if perception_example is None:
            continue
        case_id = str(action_only["case_id"])
        if fixed_split is not None:
            fixed_row = fixed_split.get(case_id)
            if fixed_row is None:
                raise ValueError(
                    "fixed case split lacks action-only perception case: "
                    f"{case_id}"
                )
            split = str(fixed_row["split"])
            split_group_id = str(fixed_row["split_group_id"])
        else:
            # Canonical accepted-release exports require a frozen split. Keep
            # this invariant explicit instead of inventing another split.
            raise ValueError(
                "action-only perception export requires a frozen case split"
            )
        dataset_perception = DatasetPerceptionExample(
            **perception_example.model_dump(mode="json"),
            split=split,
            split_group_id=split_group_id,
        )
        perception_split_rows[split].append(
            dataset_perception.model_dump(mode="json")
        )
        action_only_perception_count += 1

    for split, rows in split_rows.items():
        rows.sort(key=lambda item: item["episode_id"])
        _write_jsonl(output_dir / f"{split}.jsonl", rows)
        perception_rows = perception_split_rows[split]
        perception_rows.sort(key=lambda item: item["episode_id"])
        _write_jsonl(
            output_dir / f"perception.{split}.jsonl",
            perception_rows,
        )
    long_holdout_rows.sort(key=lambda item: item["episode_id"])
    _write_jsonl(output_dir / "long_holdout.jsonl", long_holdout_rows)
    metadata_rows.sort(key=lambda item: item["episode_id"])
    _write_jsonl(output_dir / "episode_metadata.jsonl", metadata_rows)
    accepted_episode_rows.sort(key=lambda item: item["episode_id"])
    _write_jsonl(output_dir / "accepted_episodes.jsonl", accepted_episode_rows)
    excluded_episode_rows.sort(key=lambda item: item["episode_id"])
    _write_jsonl(
        output_dir / "excluded_episode_metadata.jsonl",
        excluded_episode_rows,
    )
    action_only_rows: List[Dict[str, Any]] = []
    if accepted_release is not None:
        action_only_rows = _load_jsonl(accepted_release / "action_only.jsonl")
    _write_jsonl(output_dir / "action_only.jsonl", action_only_rows)
    selected_release_case_count = (
        len(accepted_release_rows)
        if accepted_release is not None
        else len(episodes)
    )
    manifest = {
        "schema_version": "ifv-trajectory-sft-dataset-manifest-v1",
        "dataset_version": (
            "ifv-trajectory-sft-dataset-v3"
            if decision_policy_version == "unified-react-v1"
            else "ifv-trajectory-sft-dataset-v2"
        ),
        "trajectory_version": (
            "ifv-trajectory-sft-v3"
        ),
        "decision_policy_version": decision_policy_version,
        "perception_version": "ifv-perception-v1",
        "legacy_step_policy_export": "disabled",
        "split_mode": (
            "frozen_teacher_sft" if require_frozen_gates else "derived"
        ),
        "seed": seed,
        "short_max_tokens": short_max_tokens,
        "short_max_tokens_unit": "approximate_model_tokens",
        "token_estimate_unit": "utf8_bytes",
        "split_ratios": (
            None
            if require_frozen_gates
            else {
                "train": train_ratio,
                "validation": validation_ratio,
                "test": 1 - train_ratio - validation_ratio,
            }
        ),
        "source_runs": source_runs,
        "frozen_inputs": (
            {
                "case_split": {
                    "path": str(case_split_path.expanduser().resolve()),
                    "sha256": _sha256(case_split_path.expanduser().resolve()),
                },
                "eligibility_dir": str(eligibility_dir.expanduser().resolve()),
                "semantic_reward_dir": str(
                    semantic_reward_dir.expanduser().resolve()
                )
                if semantic_reward_dir is not None
                else None,
                "accepted_release": (
                    {
                        "path": str(accepted_release),
                        "manifest_sha256": _sha256(
                            accepted_release
                            / "accepted_release_manifest.json"
                        ),
                        "schema_version": accepted_release_manifest.get(
                            "schema_version"
                        ),
                        "runtime_store_archive": accepted_release_manifest.get(
                            "runtime_store_archive"
                        ),
                    }
                    if accepted_release is not None
                    else None
                ),
            }
            if require_frozen_gates
            else None
        ),
        "source_mode": (
            "canonical_accepted_release"
            if accepted_release is not None
            else "rollout_run"
        ),
        "export_concurrency": export_concurrency,
        "episode_count": sum(len(rows) for rows in split_rows.values()),
        "accepted_episode_count": len(episodes),
        "selected_release_case_count": selected_release_case_count,
        "action_only_episode_count": len(action_only_rows),
        "action_only_perception_count": action_only_perception_count,
        "long_holdout_episode_count": len(long_holdout_rows),
        "accepted_case_count": len(
            {str(episode_metadata[episode_id]["case_id"]) for episode_id in episodes}
        ),
        "short_sft_case_count": len(
            {
                str(row["case_id"])
                for rows in split_rows.values()
                for row in rows
            }
        ),
        "candidate_episode_count": len(all_episodes),
        "excluded_episode_count": len(excluded_episode_rows),
        "group_count": len(components),
        "example_counts": {
            split: len(rows) for split, rows in split_rows.items()
        },
        "perception_example_counts": {
            split: len(rows)
            for split, rows in perception_split_rows.items()
        },
        "artifacts": {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
            "perception_train": "perception.train.jsonl",
            "perception_validation": "perception.validation.jsonl",
            "perception_test": "perception.test.jsonl",
            "episode_metadata": "episode_metadata.jsonl",
            "accepted_episodes": "accepted_episodes.jsonl",
            "excluded_episode_metadata": (
                "excluded_episode_metadata.jsonl"
            ),
            "long_holdout": "long_holdout.jsonl",
            "action_only": "action_only.jsonl",
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path)
    parser.add_argument("--accepted-release", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-split", type=Path)
    parser.add_argument("--eligibility-dir", type=Path)
    parser.add_argument("--semantic-reward-dir", type=Path)
    parser.add_argument("--minimum-accepted-cases", type=int)
    parser.add_argument("--allow-missing-validation-cases", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", default="ifv-policy-v1")
    parser.add_argument(
        "--short-max-tokens",
        type=int,
        default=DEFAULT_SHORT_TRAJECTORY_MAX_TOKENS,
        help=(
            "Maximum estimated tokens for short full-trajectory SFT rows; "
            "longer accepted episodes are kept in long_holdout.jsonl."
        ),
    )
    parser.add_argument(
        "--export-concurrency",
        type=int,
        default=1,
        help=(
            "number of concurrent trace-to-SFT exporter workers; output "
            "ordering remains stable by episode_id"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.run_dir) == bool(args.accepted_release):
        raise SystemExit(
            "provide exactly one of --run-dir or --accepted-release"
        )
    result = export_dataset(
        args.run_dir,
        args.output_dir,
        accepted_release_path=args.accepted_release,
        case_split_path=args.case_split,
        eligibility_dir=args.eligibility_dir,
        semantic_reward_dir=args.semantic_reward_dir,
        minimum_accepted_cases=args.minimum_accepted_cases,
        require_all_validation_cases=not args.allow_missing_validation_cases,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        short_max_tokens=args.short_max_tokens,
        export_concurrency=args.export_concurrency,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
