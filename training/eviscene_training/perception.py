from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import assert_model_visible
from .io import (
    canonical_json,
    load_json,
    load_jsonl,
    require_new_or_empty,
    sha256_file,
    write_json,
    write_jsonl,
)


OUTPUT_VERSION = "ifv-ms-swift-perception-v2"
SUPPORTED_ACCEPTED_DATASET_VERSIONS = {
    "ifv-policy-dataset-v2",
    "ifv-trajectory-sft-dataset-v1",
    "ifv-trajectory-sft-dataset-v2",
    "ifv-trajectory-sft-dataset-v3",
}
PERCEPTION_INSTRUCTION = """<image>
Report only literal, visible image content as one JSON object matching the
PerceptionReport contract. Include scene_description, image_type, entities with
normalized bounding boxes, positioned text regions, and uncertainty. Do not use
web knowledge, evaluator labels, or hidden benchmark context."""
PERCEPTION_SYSTEM = (
    "You are the Image Factual Verifier perception model. "
    "Return only the requested visible-content JSON."
)


def _scores(run_dir: Path) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("case_id", "")): row
        for row in load_jsonl(run_dir / "trajectory_scores.jsonl")
        if str(row.get("case_id", ""))
    }


def perception_row(trace: Mapping[str, Any]) -> dict[str, Any]:
    state = trace.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("trace.state must be an object")
    report = state.get("perception")
    if not isinstance(report, Mapping):
        raise ValueError("trace has no PerceptionReport")
    assert_model_visible(report, location="perception")
    image_path = Path(str(trace.get("image_path") or state.get("image_path") or ""))
    if not image_path.is_file():
        raise FileNotFoundError(f"trace image is unavailable: {image_path}")
    return {
        "messages": [
            {"role": "system", "content": PERCEPTION_SYSTEM},
            {"role": "user", "content": PERCEPTION_INSTRUCTION},
            {
                "role": "assistant",
                "content": canonical_json(report),
            },
        ],
        "images": [str(image_path.resolve())],
    }


def accepted_perception_row(
    source: Mapping[str, Any],
    *,
    expected_split: str,
) -> dict[str, Any]:
    """Convert one frozen accepted perception example to ms-swift format."""

    if str(source.get("dataset_version", "")) not in (
        SUPPORTED_ACCEPTED_DATASET_VERSIONS
    ):
        raise ValueError("accepted perception row has an unsupported dataset version")
    if str(source.get("split", "")) != expected_split:
        raise ValueError("accepted perception row split does not match its file")
    report = source.get("perception_report")
    if not isinstance(report, Mapping):
        raise ValueError("accepted perception row has no PerceptionReport")
    assert_model_visible(report, location="perception_report")
    image_path = Path(str(source.get("image_path", ""))).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"accepted perception image is unavailable: {image_path}")
    expected_sha = str(source.get("image_sha256", ""))
    if not expected_sha or sha256_file(image_path) != expected_sha:
        raise ValueError("accepted perception image SHA-256 does not match")
    instruction = str(source.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("accepted perception row has no instruction")
    if "<image>" not in instruction:
        instruction = f"<image>\n{instruction}"
    return {
        "messages": [
            {"role": "system", "content": PERCEPTION_SYSTEM},
            {"role": "user", "content": instruction},
            {
                "role": "assistant",
                "content": canonical_json(report),
            },
        ],
        "images": [str(image_path)],
    }


def convert_accepted_perception_dataset(
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Convert only frozen, three-gate-accepted perception examples."""

    source_manifest = load_json(input_dir / "manifest.json")
    if source_manifest.get("dataset_version") not in SUPPORTED_ACCEPTED_DATASET_VERSIONS:
        raise ValueError(
            "accepted perception adapter requires "
            f"one of {sorted(SUPPORTED_ACCEPTED_DATASET_VERSIONS)}"
        )
    if source_manifest.get("split_mode") != "frozen_teacher_sft":
        raise ValueError("accepted perception adapter requires a frozen teacher dataset")
    require_new_or_empty(output_dir)
    artifacts: dict[str, dict[str, Any]] = {}
    index_rows: list[dict[str, Any]] = []
    row_id = 0
    for split in ("train", "validation", "test"):
        converted_rows: list[dict[str, Any]] = []
        for source_index, source in enumerate(
            load_jsonl(input_dir / f"perception.{split}.jsonl")
        ):
            converted_rows.append(
                accepted_perception_row(source, expected_split=split)
            )
            index_rows.append(
                {
                    "row_id": row_id,
                    "split": split,
                    "source_index": source_index,
                    "episode_id": source.get("episode_id"),
                    "source_run_id": source.get("source_run_id"),
                    "runtime_commit": source.get("runtime_commit"),
                    "image_sha256": source.get("image_sha256"),
                }
            )
            row_id += 1
        output_path = output_dir / f"{split}.jsonl"
        write_jsonl(output_path, converted_rows)
        artifacts[split] = {
            "path": output_path.name,
            "rows": len(converted_rows),
            "sha256": sha256_file(output_path),
        }
    index_path = output_dir / "index.jsonl"
    write_jsonl(index_path, index_rows)
    artifacts["index"] = {
        "path": index_path.name,
        "rows": len(index_rows),
        "sha256": sha256_file(index_path),
    }
    manifest = {
        "schema_version": "ifv-ms-swift-dataset-manifest-v1",
        "dataset_version": OUTPUT_VERSION,
        "framework": {"name": "ms-swift", "version": "4.4.2"},
        "source": {
            "dataset_version": source_manifest["dataset_version"],
            "manifest_sha256": sha256_file(input_dir / "manifest.json"),
        },
        "example_count": row_id,
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def convert_perception_runs(
    run_dirs: Sequence[Path],
    output_dir: Path,
    *,
    split_map_path: Path,
) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    require_new_or_empty(output_dir)
    split_map = {
        str(row.get("episode_id", "")): str(row.get("split", ""))
        for row in load_jsonl(split_map_path)
        if str(row.get("episode_id", ""))
    }
    if not split_map:
        raise ValueError(f"split map is empty: {split_map_path}")
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    index_rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()

    for raw_run_dir in run_dirs:
        run_dir = raw_run_dir.expanduser().resolve()
        run_manifest = load_json(run_dir / "run_manifest.json")
        scores = _scores(run_dir)
        for trace_path in sorted((run_dir / "traces").glob("*.json")):
            trace = load_json(trace_path)
            case_id = str(trace.get("image_id") or trace_path.stem)
            score = scores.get(case_id)
            if not score:
                exclusions["missing_training_quality_score"] += 1
                continue
            if not bool(score.get("training_eligible", False)):
                reasons = score.get("training_exclusion_reasons") or [
                    "training_quality_gate_failed"
                ]
                for reason in reasons:
                    exclusions[str(reason)] += 1
                continue
            split = split_map.get(case_id)
            if split not in rows_by_split:
                exclusions["missing_or_invalid_split"] += 1
                continue
            row = perception_row(trace)
            rows_by_split[split].append(row)
            index_rows.append(
                {
                    "row_id": len(index_rows),
                    "case_id": case_id,
                    "split": split,
                    "source_run_id": run_manifest.get("run_id"),
                    "runtime_commit": run_manifest.get("git_commit"),
                    "trace_sha256": sha256_file(trace_path),
                    "image_sha256": sha256_file(Path(row["images"][0])),
                }
            )

    index_path = output_dir / "index.jsonl"
    write_jsonl(index_path, index_rows)
    artifacts: dict[str, dict[str, Any]] = {}
    for split, rows in rows_by_split.items():
        dataset_path = output_dir / f"{split}.jsonl"
        write_jsonl(dataset_path, rows)
        artifacts[split] = {
            "path": dataset_path.name,
            "rows": len(rows),
            "sha256": sha256_file(dataset_path),
        }
    artifacts["index"] = {
        "path": index_path.name,
        "rows": len(index_rows),
        "sha256": sha256_file(index_path),
    }
    manifest = {
        "schema_version": "ifv-ms-swift-dataset-manifest-v1",
        "dataset_version": OUTPUT_VERSION,
        "framework": {"name": "ms-swift", "version": "4.4.2"},
        "format_contract": {
            "name": "ms-swift-qwen-vision",
            "version": "v2",
            "message_roles": ["system", "user", "assistant"],
            "preserves_native_think": False,
            "assistant_loss_metadata": "omitted; inferred by ms-swift template",
        },
        "source_split_map_sha256": sha256_file(split_map_path),
        "example_count": sum(len(rows) for rows in rows_by_split.values()),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
