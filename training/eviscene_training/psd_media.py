"""Immutable image tensors shared by PSD teacher scoring and student training."""
from __future__ import annotations

import base64
import hashlib
import io
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import canonical_json, sha256_file

SCHEMA = "ifv-psd-media-v3"
COMPACT_PREPROCESSED_SCHEMA = "ifv-psd-media-v2"
LEGACY_SCHEMA = "ifv-psd-media-v1"
IMAGE_TOKEN = 248056


def media_digest(media: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(media)).encode()).hexdigest()


def media_from_archive(step, *, processor_path: str, output_dir: Path, prompt_ids):
    from . import _repo_import  # noqa: F401
    from eviscene.orchestrator.runtime_events import reconstruct_archived_request
    import json

    path = Path(processor_path)
    if (path / "adapter_config.json").is_file():
        path = Path(json.loads((path / "adapter_config.json").read_text())["base_model_name_or_path"])
    processor = _load_processor(str(path))
    request = reconstruct_archived_request(step["runtime_store_path"], step["context_request_id"])
    return bind_media(request.get("input_payload", request), processor=processor,
                      output_dir=output_dir, prompt_ids=prompt_ids, processor_id=str(path))


@lru_cache(maxsize=4)
def _load_processor(path: str):
    """Load the exact image processor bound by the media record."""
    from transformers import AutoProcessor
    import json

    resolved = Path(path)
    if (resolved / "adapter_config.json").is_file():
        resolved = Path(json.loads(
            (resolved / "adapter_config.json").read_text(encoding="utf-8")
        )["base_model_name_or_path"])
    processor = AutoProcessor.from_pretrained(str(resolved), local_files_only=True)
    if not hasattr(processor, "image_processor"):
        raise ValueError(f"PSD processor has no image_processor: {resolved}")
    return processor


def _decode_images(blobs: Sequence[bytes]):
    from PIL import Image

    images = []
    for blob in blobs:
        with Image.open(io.BytesIO(blob)) as image:
            images.append(image.convert("RGB"))
    return images


def _encode_images(blobs: Sequence[bytes], processor: Any) -> dict[str, Any]:
    encoded = processor.image_processor(
        images=_decode_images(blobs), return_tensors="pt"
    )
    return {
        name: encoded[name].detach().cpu().contiguous()
        for name in ("pixel_values", "image_grid_thw")
    }


def image_bytes(request: Any) -> list[bytes]:
    """Read only image blocks, in order; repeated images remain repeated.

    A reconstructed runtime request contains archived data URLs. Remote URLs
    are deliberately not refetched: their contents may have changed.
    """
    result = []

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, Mapping):
            kind = value.get("type")
            if kind == "image_url":
                url = value.get("image_url", {}).get("url", "")
                if not url.startswith("data:image/") or ";base64," not in url:
                    raise ValueError("PSD requires archived image bytes, not remote URLs")
                result.append(base64.b64decode(url.split(",", 1)[1], validate=True))
            elif kind == "image":
                result.append(base64.b64decode(value["data"], validate=True))
            elif kind in {"video", "video_url", "input_audio"}:
                raise ValueError("PSD media adapter currently supports images only")
            else:
                for item in value.values():
                    visit(item)

    visit(request)
    return result


def proposer_image_inputs(value):
    """Replace encoded images in JSON prose with references and native blocks.

    Unlike student/teacher conditioning, the reviewer can receive each unique
    image once while the JSON references retain its exact order and repeats.
    """
    blocks = {}

    def visit(item):
        if isinstance(item, list):
            return [visit(child) for child in item]
        if not isinstance(item, Mapping):
            return item
        if item.get("type") in {"image_url", "image"}:
            blob = image_bytes(item)[0]
            digest = hashlib.sha256(blob).hexdigest()
            if item["type"] == "image_url":
                url = item["image_url"]["url"]
            else:
                mime = item.get("mime_type") or "image/jpeg"
                url = f"data:{mime};base64,{base64.b64encode(blob).decode()}"
            blocks[digest] = {"type": "image_url", "image_url": {"url": url}}
            return {"type": "archived_image_reference", "image_sha256": digest}
        if item.get("type") in {"video", "video_url", "input_audio"}:
            raise ValueError("PSD proposer currently supports images only")
        return {key: visit(child) for key, child in item.items()}

    cleaned = visit(value)
    native = []
    for digest, block in blocks.items():
        native.extend([{"type": "text", "text": "Archived image SHA-256: " + digest}, block])
    return cleaned, native


def bind_media(
    request: Mapping[str, Any], *, processor: Any, output_dir: Path,
    prompt_ids: Sequence[int], processor_id: str,
) -> dict[str, Any]:
    blobs = image_bytes(request)
    if not blobs:
        if IMAGE_TOKEN in prompt_ids:
            raise ValueError("visual prompt has no archived images")
        return {}
    merge_size = int(processor.image_processor.merge_size)
    runs = _image_run_lengths(prompt_ids)
    if len(runs) != len(blobs):
        raise ValueError(
            f"PSD image placeholder/image count mismatch: {len(runs)} != {len(blobs)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    image_sha256 = []
    for blob in blobs:
        digest = hashlib.sha256(blob).hexdigest()
        path = output_dir / f"{digest}.image"
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(blob)
            temporary.replace(path)
        elif sha256_file(path) != digest:
            raise ValueError("PSD compact image artifact hash mismatch")
        image_paths.append(str(path.resolve()))
        image_sha256.append(digest)
    provenance = {
        "schema_version": SCHEMA,
        "processor_id": processor_id,
        "processor_config": processor.image_processor.to_dict(),
        "image_sha256": image_sha256,
        "image_paths": image_paths,
        "merge_size": merge_size,
    }
    return provenance


def validate_image_runs(ids: Sequence[int], grids: Sequence[Sequence[int]], merge: int):
    runs = _image_run_lengths(ids)
    if merge <= 0 or any(len(grid) != 3 or any(int(x) <= 0 for x in grid) for grid in grids):
        raise ValueError("invalid image grids")
    expected = [int(t) * int(h) * int(w) // (merge * merge) for t, h, w in grids]
    if runs != expected:
        raise ValueError(f"PSD image placeholder/grid mismatch: {runs} != {expected}")


def _image_run_lengths(ids: Sequence[int]) -> list[int]:
    runs = []
    count = 0
    for token in [*ids, -1]:
        if token == IMAGE_TOKEN:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    return runs


def validate_media(media: Mapping[str, Any], ids: Sequence[int]) -> None:
    if media.get("schema_version") == LEGACY_SCHEMA:
        path = Path(media["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("PSD media tensor artifact missing")
        if sha256_file(path) != media.get("sha256"):
            raise ValueError("PSD media tensor hash mismatch")
        grids = media["image_grid_thw"]
        if len(grids) != len(media["image_sha256"]):
            raise ValueError("PSD media image count mismatch")
        validate_image_runs(ids, grids, int(media["merge_size"]))
        return
    if media.get("schema_version") not in {SCHEMA, COMPACT_PREPROCESSED_SCHEMA}:
        raise ValueError("invalid PSD media schema")
    paths = media.get("image_paths")
    digests = media.get("image_sha256")
    if not isinstance(paths, list) or not isinstance(digests, list) or len(paths) != len(digests):
        raise ValueError("PSD compact image bindings are invalid")
    for raw_path, digest in zip(paths, digests):
        path = Path(raw_path)
        if not path.is_absolute() or not path.is_file():
            raise ValueError("PSD compact image artifact missing")
        if sha256_file(path) != digest:
            raise ValueError("PSD compact image artifact hash mismatch")
    if media.get("schema_version") == COMPACT_PREPROCESSED_SCHEMA:
        grids = media["image_grid_thw"]
        if len(grids) != len(digests):
            raise ValueError("PSD media image count mismatch")
        validate_image_runs(ids, grids, int(media["merge_size"]))
    elif len(_image_run_lengths(ids)) != len(digests):
        raise ValueError("PSD image placeholder/image count mismatch")


def load_media(media: Mapping[str, Any], ids: Sequence[int]) -> dict[str, Any]:
    import torch

    validate_media(media, ids)
    schema = media.get("schema_version")
    if schema == LEGACY_SCHEMA:
        tensors = torch.load(media["path"], map_location="cpu", weights_only=True)
    else:
        processor = _load_processor(str(media["processor_id"]))
        # The binding is persisted through JSON. Tuples become lists and
        # IntEnum values become integers, so compare the canonical serialized
        # meaning rather than Python container/value implementation types.
        if canonical_json(processor.image_processor.to_dict()) != canonical_json(
            media.get("processor_config")
        ):
            raise ValueError("PSD image processor config changed")
        blobs = [Path(path).read_bytes() for path in media["image_paths"]]
        tensors = _encode_images(blobs, processor)
    if set(tensors) != {"pixel_values", "image_grid_thw"}:
        raise ValueError("unexpected PSD media tensor fields")
    grids = tensors["image_grid_thw"].tolist()
    validate_image_runs(ids, grids, int(media["merge_size"]))
    if schema == COMPACT_PREPROCESSED_SCHEMA and grids != media["image_grid_thw"]:
        raise ValueError("PSD media tensor grid mismatch")
    if not torch.isfinite(tensors["pixel_values"]).all():
        raise ValueError("nonfinite PSD image pixels")
    pixels_sha = hashlib.sha256(tensors["pixel_values"].float().numpy().tobytes()).hexdigest()
    if schema in {LEGACY_SCHEMA, COMPACT_PREPROCESSED_SCHEMA} and pixels_sha != media.get("pixel_values_sha256"):
        raise ValueError("PSD pixel content differs from processor output")
    return tensors
