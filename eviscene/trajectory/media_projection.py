"""Project runtime-request images into portable trajectory SFT rows."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass
class TrajectoryMediaProjection:
    """Ordered portable media plus diagnostics for one exported episode."""

    initial_images: list[str] = field(default_factory=list)
    images_after_step: dict[int, list[str]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def images(self) -> list[str]:
        result = list(self.initial_images)
        for step_index in sorted(self.images_after_step):
            result.extend(self.images_after_step[step_index])
        return result


def project_trajectory_media(
    trace: Mapping[str, Any],
    *,
    candidate_steps: Sequence[Mapping[str, Any]],
    fallback_image_path: str,
) -> TrajectoryMediaProjection:
    """Recover the exact images seen by policy requests.

    The first request supplies the initial user image. Images newly present in
    the request after tool step ``N`` are attached to that step's
    ``tool_response``. Content hashes suppress repeated original-image
    reinjection without dropping genuinely new crops or candidate images.
    """

    projection = TrajectoryMediaProjection()
    runtime_root = _runtime_root(trace)
    seen_hashes: set[str] = set()

    request_media: list[list[tuple[str, str]]] = []
    for step_index, step in enumerate(candidate_steps):
        request_id = _context_request_id(step)
        if runtime_root is None or not request_id:
            request_media.append([])
            continue
        media, missing = _request_media(runtime_root, request_id)
        request_media.append(media)
        projection.missing.extend(
            f"step[{step_index}] request {request_id}: {item}" for item in missing
        )

    if request_media:
        projection.initial_images.extend(_new_data_urls(request_media[0], seen_hashes))

    if not projection.initial_images:
        fallback = _fallback_data_url(fallback_image_path)
        if fallback is not None:
            digest, data_url = fallback
            seen_hashes.add(digest)
            projection.initial_images.append(data_url)
        else:
            projection.missing.append(
                f"initial image is unavailable: {fallback_image_path}"
            )

    for next_step_index in range(1, len(request_media)):
        previous_step_index = next_step_index - 1
        if str(candidate_steps[previous_step_index].get("action_type", "")) != (
            "tool_call"
        ):
            continue
        newly_visible = _new_data_urls(
            request_media[next_step_index],
            seen_hashes,
        )
        if newly_visible:
            projection.images_after_step[previous_step_index] = newly_visible

    return projection


def project_attested_request_media(
    *,
    candidate_steps: Sequence[tuple[int, Mapping[str, Any], str]],
    request_bindings: Sequence[Mapping[str, Any]],
    sidecar_root: Path,
    expected_trace_id: str,
    digest_cache: dict[Path, str] | None = None,
) -> TrajectoryMediaProjection:
    """Project a frozen request-image sidecar into one canonical episode.

    Each sidecar row describes the complete ordered image sequence supplied to
    one provider request.  Requests in these Agent traces are cumulative, so a
    later request must begin with the exact sequence from the preceding
    trainable request.  Only the newly appended suffix is attached after the
    preceding tool response.  Comparing ordered prefixes, rather than globally
    deduplicating hashes, preserves intentional repeated image slots.
    """

    root = sidecar_root.expanduser().resolve()
    media_root = (root / "media").resolve()
    if not media_root.is_dir():
        raise ValueError(f"attested media directory is missing: {media_root}")
    cache = digest_cache if digest_cache is not None else {}
    by_request: dict[str, Mapping[str, Any]] = {}
    for row in request_bindings:
        trace_id = str(row.get("trace_id", "")).strip()
        request_id = str(row.get("context_request_id", "")).strip()
        if trace_id != expected_trace_id or not request_id:
            raise ValueError("request-image binding identity mismatch")
        if request_id in by_request:
            raise ValueError(f"duplicate request-image binding: {request_id}")
        by_request[request_id] = row

    request_sequences: list[list[str]] = []
    request_digests: list[list[str]] = []
    for raw_step_index, step, _ in candidate_steps:
        metadata = step.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        request_id = str(metadata.get("context_request_id", "")).strip()
        recovered_parent = (
            str(metadata.get("sft_policy_input_provenance", ""))
            == "rejected_parent_request"
        )
        if not request_id and recovered_parent:
            request_id = str(metadata.get("parent_context_request_id", "")).strip()
        binding = by_request.get(request_id)
        if binding is None:
            raise ValueError(
                "trainable policy request lacks an attested image binding: "
                f"step={raw_step_index} request={request_id!r}"
            )
        if not recovered_parent:
            bound_steps = [
                int(value) for value in binding.get("trace_step_indices", [])
            ]
            if raw_step_index not in bound_steps:
                raise ValueError(
                    "request-image binding points at a different trace step: "
                    f"step={raw_step_index} request={request_id}"
                )
        paths, digests = _attested_request_images(
            binding,
            root=root,
            media_root=media_root,
            digest_cache=cache,
        )
        request_sequences.append(paths)
        request_digests.append(digests)

    if not request_sequences or not request_sequences[0]:
        raise ValueError("first trainable request has no attested image")
    projection = TrajectoryMediaProjection(initial_images=list(request_sequences[0]))
    previous_paths = request_sequences[0]
    previous_digests = request_digests[0]
    for candidate_index in range(1, len(request_sequences)):
        current_paths = request_sequences[candidate_index]
        current_digests = request_digests[candidate_index]
        if current_digests[: len(previous_digests)] != previous_digests:
            raise ValueError(
                "attested request images are not a cumulative ordered prefix: "
                f"candidate={candidate_index}"
            )
        preceding_step = candidate_steps[candidate_index - 1][1]
        if len(current_paths) > len(previous_paths):
            if str(preceding_step.get("action_type", "")) != "tool_call":
                raise ValueError(
                    "new request images do not follow a tool-call observation"
                )
            projection.images_after_step[candidate_index - 1] = current_paths[
                len(previous_paths) :
            ]
        previous_paths = current_paths
        previous_digests = current_digests
    return projection


def _attested_request_images(
    binding: Mapping[str, Any],
    *,
    root: Path,
    media_root: Path,
    digest_cache: dict[Path, str],
) -> tuple[list[str], list[str]]:
    raw_images = binding.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError("request-image binding has no ordered images")
    images = sorted(
        (item for item in raw_images if isinstance(item, Mapping)),
        key=lambda item: int(item.get("image_slot_index", -1)),
    )
    slots = [int(item.get("image_slot_index", -1)) for item in images]
    if slots != list(range(len(images))):
        raise ValueError("request-image slots must be contiguous and zero-based")
    paths: list[str] = []
    digests: list[str] = []
    for image in images:
        expected = str(image.get("sha256", "")).strip().casefold()
        relative = Path(str(image.get("file", "")))
        path = (root / relative).resolve()
        if (
            len(expected) != 64
            or any(value not in "0123456789abcdef" for value in expected)
            or relative.is_absolute()
            or path == media_root
            or media_root not in path.parents
            or not path.is_file()
        ):
            raise ValueError(
                "attested request image escapes, is missing, or lacks a hash"
            )
        actual = digest_cache.get(path)
        if actual is None:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            digest_cache[path] = actual
        if actual != expected or path.stem.casefold() != expected:
            raise ValueError(f"attested request image hash mismatch: {relative}")
        paths.append(str(path))
        digests.append(expected)
    return paths, digests


def image_markers(count: int) -> str:
    """Return one marker per projected image, ready for message appending."""

    return "\n".join("<image>" for _ in range(max(0, int(count))))


def _runtime_root(trace: Mapping[str, Any]) -> Path | None:
    state = trace.get("state")
    if not isinstance(state, Mapping):
        return None
    runtime_store = state.get("runtime_store")
    if not isinstance(runtime_store, Mapping):
        return None
    raw = str(runtime_store.get("runtime_path", "")).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def _context_request_id(step: Mapping[str, Any]) -> str:
    metadata = step.get("metadata")
    if not isinstance(metadata, Mapping):
        return ""
    return str(metadata.get("context_request_id", "")).strip()


def _request_media(
    runtime_root: Path,
    request_id: str,
) -> tuple[list[tuple[str, str]], list[str]]:
    manifest_path = runtime_root / "context" / f"{request_id}.json"
    if not manifest_path.is_file():
        return [], [f"manifest missing at {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"manifest unreadable: {exc}"]

    artifact_root = (runtime_root / "artifacts" / "sha256").resolve()
    result: list[tuple[str, str]] = []
    missing: list[str] = []
    for context_item in manifest.get("context_items", []) or []:
        if not isinstance(context_item, Mapping):
            continue
        if str(context_item.get("kind", "")) != "input_payload":
            continue
        for descriptor in context_item.get("media", []) or []:
            if not isinstance(descriptor, Mapping):
                continue
            recovered, error = _descriptor_data_url(
                artifact_root,
                descriptor,
            )
            if recovered is not None:
                result.append(recovered)
            elif error:
                missing.append(error)
    return result, missing


def _descriptor_data_url(
    artifact_root: Path,
    descriptor: Mapping[str, Any],
) -> tuple[tuple[str, str] | None, str]:
    relative = str(descriptor.get("artifact_path", "")).strip()
    expected_hash = str(descriptor.get("sha256", "")).strip().lower()
    media_type = str(descriptor.get("media_type", "application/octet-stream")).strip()
    if not relative:
        return None, "media descriptor has no artifact_path"
    path = (artifact_root / relative).resolve()
    if artifact_root != path and artifact_root not in path.parents:
        return None, f"media path escapes artifact store: {relative}"
    try:
        content = path.read_bytes()
    except OSError as exc:
        return None, f"media artifact unreadable ({relative}): {exc}"
    actual_hash = hashlib.sha256(content).hexdigest()
    if expected_hash and actual_hash != expected_hash:
        return None, (
            f"media hash mismatch ({relative}): "
            f"expected {expected_hash}, got {actual_hash}"
        )
    return (
        actual_hash,
        f"data:{media_type};base64," + base64.b64encode(content).decode("ascii"),
    ), ""


def _fallback_data_url(image_path: str) -> tuple[str, str] | None:
    path = Path(str(image_path)).expanduser()
    if not path.is_file():
        return None
    try:
        content = path.read_bytes()
    except OSError:
        return None
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(path.suffix.casefold(), "application/octet-stream")
    digest = hashlib.sha256(content).hexdigest()
    return (
        digest,
        f"data:{media_type};base64," + base64.b64encode(content).decode("ascii"),
    )


def _new_data_urls(
    media: Sequence[tuple[str, str]],
    seen_hashes: set[str],
) -> list[str]:
    result: list[str] = []
    for digest, data_url in media:
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        result.append(data_url)
    return result
