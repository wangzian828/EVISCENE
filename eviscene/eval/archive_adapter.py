"""Read-only adapter for historical human-review candidate archives.

The benchmark-pipeline archive is intentionally not rewritten into a runtime
release.  This adapter projects only the public runtime fields required by the
Agent and keeps all construction, label, and evidence fields out of the
runtime rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ARCHIVE_ADAPTER_SCHEMA_VERSION = "ifv-hrc-archive-runtime-adapter-v1"
CANDIDATE_FILE_NAME = "human-review-candidates.jsonl"
SUMMARY_FILE_NAME = "archive-summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(row: Mapping[str, Any], key: str, *, line_number: int) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(
            f"{CANDIDATE_FILE_NAME} line {line_number} requires non-empty {key}"
        )
    if key == "candidate_id" and len(value) > 200:
        raise ValueError(
            f"{CANDIDATE_FILE_NAME} line {line_number} candidate_id exceeds "
            "the runtime case_id limit of 200 characters"
        )
    return value


def _runtime_case_id(
    candidate: Mapping[str, Any],
    *,
    line_number: int,
) -> str:
    candidate_id = _required_text(
        candidate,
        "candidate_id",
        line_number=line_number,
    )
    version_id = str(candidate.get("archive_source_version_id") or "").strip()
    case_id = version_id or candidate_id
    if len(case_id) > 200:
        raise ValueError(
            f"{CANDIDATE_FILE_NAME} line {line_number} archive identity exceeds "
            "the runtime case_id limit of 200 characters"
        )
    return case_id


def _resolve_image(
    root: Path,
    value: str,
    *,
    line_number: int,
    require_file: bool = True,
) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(
            f"{CANDIDATE_FILE_NAME} line {line_number} image path must be relative"
        )
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{CANDIDATE_FILE_NAME} line {line_number} image path escapes archive"
        ) from exc
    if require_file and not resolved.is_file():
        raise FileNotFoundError(
            f"{CANDIDATE_FILE_NAME} line {line_number} image does not exist: "
            f"{resolved}"
        )
    return resolved


@dataclass(frozen=True)
class ArchiveRuntimeInput:
    """Minimal, Agent-visible projection of one immutable archive."""

    root: Path
    candidate_path: Path
    archive_id: str
    rows: List[Dict[str, str]]

    @property
    def case_count(self) -> int:
        return len(self.rows)

    @property
    def schema_version(self) -> str:
        return ARCHIVE_ADAPTER_SCHEMA_VERSION


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name} line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{path.name} line {line_number} must be a JSON object"
                )
            yield line_number, payload


def materialize_archive_runtime_rows(
    rows: Iterable[Mapping[str, Any]],
) -> List[Dict[str, str]]:
    """Add per-image digests only after a selected archive subset is known.

    Historical archives can contain thousands of images.  Callers that first
    select a small case list should not repeatedly hash every image in the
    archive before running one batch.  This function preserves the runtime
    contract by checking each selected image and attaching its exact digest
    immediately before it becomes an ``ImageOnlyRuntimeCase``.
    """

    materialized: List[Dict[str, str]] = []
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        image_value = str(row.get("image_path") or "").strip()
        if not case_id or not image_value:
            raise ValueError(
                "archive runtime row requires non-empty case_id and image_path"
            )
        image_path = Path(image_value).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(
                f"selected archive image does not exist: {image_path}"
            )
        materialized.append(
            {
                "case_id": case_id,
                "image_path": str(image_path),
                "image_sha256": _sha256(image_path),
            }
        )
    return materialized


def load_archive_runtime_input(
    archive_root: Path,
    *,
    materialize_image_hashes: bool = True,
) -> ArchiveRuntimeInput:
    """Load candidate rows without modifying or reserializing the archive.

    Only the stable archive identity and ``archive_image_path`` are read into the
    runtime projection.  Fields such as ``factual_status``, ``claim_atom``,
    ``evidence``, and source/construction metadata are deliberately not copied.

    ``materialize_image_hashes=False`` is intended for a batch runner that first
    selects a subset by case ID, then calls ``materialize_archive_runtime_rows``.
    The default retains the complete runtime-row contract for direct callers.
    """

    root = archive_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"archive root does not exist: {root}")

    candidate_path = root / CANDIDATE_FILE_NAME
    if not candidate_path.is_file():
        raise FileNotFoundError(f"missing archive candidate file: {candidate_path}")

    archive_id = ""
    summary_path = root / SUMMARY_FILE_NAME
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{SUMMARY_FILE_NAME} is not valid JSON") from exc
        if not isinstance(summary, Mapping):
            raise ValueError(f"{SUMMARY_FILE_NAME} must be a JSON object")
        archive_id = str(summary.get("archive_id") or "").strip()

    rows: List[Dict[str, str]] = []
    seen_case_ids: set[str] = set()
    for line_number, candidate in _iter_jsonl(candidate_path):
        case_id = _runtime_case_id(candidate, line_number=line_number)
        image_value = _required_text(
            candidate,
            "archive_image_path",
            line_number=line_number,
        )
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate archive candidate_id: {case_id}")
        seen_case_ids.add(case_id)
        image_path = _resolve_image(
            root,
            image_value,
            line_number=line_number,
            require_file=materialize_image_hashes,
        )
        runtime_row = {
            "case_id": case_id,
            "image_path": str(image_path),
        }
        if materialize_image_hashes:
            runtime_row["image_sha256"] = _sha256(image_path)
        rows.append(runtime_row)

    if not rows:
        raise ValueError(f"archive candidate file is empty: {candidate_path}")

    return ArchiveRuntimeInput(
        root=root,
        candidate_path=candidate_path,
        archive_id=archive_id or root.name,
        rows=rows,
    )
