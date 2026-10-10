"""Crash-safe local checkpoints for expensive PSD continuation calls."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import gzip
import hashlib
import json
import os
import tempfile

from .io import canonical_json, load_json
from .psd_gemini_judge import _atomic_json
from .psd_repair import _sha


DIRECT_GZIP_BYTES = 1024 * 1024


def _write_gzip_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            with gzip.GzipFile(fileobj=target, mode="wb", compresslevel=1, mtime=0) as stream:
                stream.write(raw)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_bound(path: Path, *, identity, payload):
    document = {"identity": identity, "payload": payload, "payload_sha256": _sha(payload)}
    raw = canonical_json(document).encode("utf-8")
    if len(raw) < DIRECT_GZIP_BYTES:
        _atomic_json(path, document)
        return
    digest = hashlib.sha256(raw).hexdigest()
    archive = path.with_name(f"{path.stem}-{digest[:16]}.json.gz")
    if archive.exists():
        with gzip.open(archive, "rb") as stream:
            current = stream.read(len(raw) + 1)
        if current != raw:
            raise ValueError("Compressed PSD cache archive collision")
    else:
        _write_gzip_atomic(archive, raw)
    _atomic_json(path, {"schema_version": "ifv-psd-bound-gzip-v1",
        "identity": identity, "payload_sha256": document["payload_sha256"],
        "archive": archive.name, "uncompressed_bytes": len(raw),
        "uncompressed_sha256": digest})


def bound_artifact_paths(path: Path) -> list[Path]:
    """Return the tiny binding plus its sole payload archive, when present."""
    paths = [path]
    saved = load_json(path)
    if saved.get("schema_version") == "ifv-psd-bound-gzip-v1":
        name = saved.get("archive", "")
        if not name or Path(name).name != name:
            raise ValueError("Invalid compressed PSD cache reference")
        paths.append(path.parent / name)
    return paths


def load_bound(path: Path, *, identity):
    saved = load_json(path)
    if saved.get('schema_version') == 'ifv-psd-bound-gzip-v1':
        name = saved.get('archive', '')
        size = saved.get('uncompressed_bytes')
        if (not name or Path(name).name != name or not name.endswith('.json.gz')
                or type(size) is not int or not 0 < size <= 1024**3):
            raise ValueError('Invalid compressed PSD cache reference')
        archive = path.parent/name
        if archive.is_symlink() or archive.resolve().parent != path.resolve().parent:
            raise ValueError('Compressed PSD cache escaped its slot')
        with gzip.open(archive, 'rb') as stream:
            raw = stream.read(size+1)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != saved.get('uncompressed_sha256'):
            raise ValueError('Compressed PSD cache bytes changed')
        original = json.loads(raw)
        if (saved.get('identity') != original.get('identity')
                or saved.get('payload_sha256') != original.get('payload_sha256')):
            raise ValueError('Compressed PSD cache binding changed')
        saved = original
    if saved.get("identity") != identity or saved.get("payload_sha256") != _sha(saved.get("payload")):
        raise ValueError("PSD continuation checkpoint binding changed")
    return saved["payload"]


def continuation_from_payload(payload):
    from .psd_repair_runtime import ContinuationResult
    from eviscene.orchestrator.stage_runner import StageStep
    payload = dict(payload)
    return ContinuationResult(
        teacher_steps=[StageStep(**step) for step in payload.pop("teacher_steps")],
        student_steps=[StageStep(**step) for step in payload.pop("student_steps")],
        teacher_history=[], student_history=[], **payload)


def continuation_payload(result):
    return {"teacher_steps": [asdict(step) for step in result.teacher_steps],
               "student_steps": [asdict(step) for step in result.student_steps],
               "hint": result.hint, "teacher_complete": result.teacher_complete,
               "student_complete": result.student_complete, "stop_reason": result.stop_reason,
               "teacher_episode_trace": result.teacher_episode_trace,
               "local_targets": result.local_targets}


async def cached_continuation(path: Path, *, identity, generate):
    if path.exists():
        return continuation_from_payload(load_bound(path, identity=identity))
    result = await generate()
    payload = continuation_payload(result)
    # Raw history is already immutable in the runtime archive. Do not duplicate
    # all base64 images in a second continuation snapshot.
    save_bound(path, identity=identity, payload=payload)
    return result
