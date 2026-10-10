"""Provider-neutral perception teacher export from canonical traces."""

from __future__ import annotations

from typing import Any, Mapping

from eviscene.orchestrator.state import PerceptionReport
from eviscene.trajectory.exporter import _assert_no_private_data
from eviscene.trajectory.schema import PerceptionExample

PERCEPTION_INSTRUCTION = (
    "Report only literal, visible image content as one JSON object matching the "
    "PerceptionReport contract. Include scene_description, image_type, entities "
    "with normalized bounding boxes, and positioned text regions. Do not use web "
    "knowledge, evaluator labels, or hidden benchmark context."
)

def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}

def export_perception_example(
    trace: Mapping[str, Any],
    *,
    source_metadata: Mapping[str, Any] | None = None,
 ) -> PerceptionExample:
    """Export only the public image identity and canonical PerceptionReport."""

    state = _mapping(trace.get("state"))
    if str(trace.get("input_mode") or state.get("input_mode") or "") != "image_only":
        raise ValueError("perception exporter accepts image-only traces only")
    runtime_case = _mapping(state.get("runtime_case"))
    episode_id = str(
        trace.get("image_id") or state.get("image_id") or runtime_case.get("case_id") or ""
    ).strip()
    if not episode_id:
        raise ValueError("canonical trace requires image_id")
    image_path = str(
        runtime_case.get("image_path")
        or trace.get("image_path")
        or state.get("image_path")
        or ""
    ).strip()
    image_sha256 = str(runtime_case.get("image_sha256") or "").strip()
    if not image_path or not image_sha256:
        raise ValueError("canonical trace requires public image path and sha256")

    report = PerceptionReport.model_validate(_mapping(state.get("perception")))
    report_payload = report.model_dump(mode="json")
    _assert_no_private_data(report_payload)
    source_metadata = source_metadata or {}
    return PerceptionExample(
        episode_id=episode_id,
        source_run_id=str(source_metadata.get("source_run_id", "")),
        runtime_commit=str(source_metadata.get("runtime_commit", "")),
        release_id=str(source_metadata.get("release_id", "")),
        runtime_contract_version=str(
            source_metadata.get("runtime_contract_version", "")
        ),
        image_path=image_path,
        image_sha256=image_sha256,
        instruction=PERCEPTION_INSTRUCTION,
        perception_report=report_payload,
    )
