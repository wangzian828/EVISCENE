"""O(1) identity receipts for immutable artifacts.

The lightweight PSD pipeline never reads a large payload merely to hash it.
Stages bind the file's filesystem identity and reject any later identity change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import tempfile

from .io import canonical_json, load_json


SCHEMA = "ifv-immutable-artifact-stat-receipt-v1"


def artifact_identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular non-symlink file: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def freeze_artifact(path: Path, receipt: Path) -> dict[str, Any]:
    """Atomically record ``path`` without reading its payload."""
    identity = artifact_identity(path)
    if receipt.exists():
        saved = load_json(receipt)
        if saved.get("schema_version") != SCHEMA or saved.get("identity") != identity:
            raise ValueError(f"immutable artifact identity changed: {path}")
        return saved
    record = {
        "schema_version": SCHEMA,
        "identity": identity,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{receipt.name}.", suffix=".tmp", dir=receipt.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, receipt)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return record


def verify_artifact(receipt: Path) -> dict[str, Any]:
    """Verify a frozen artifact without rereading its payload."""
    saved = load_json(receipt)
    if saved.get("schema_version") != SCHEMA:
        raise ValueError(f"unknown artifact receipt schema: {receipt}")
    identity = saved.get("identity")
    if not isinstance(identity, dict) or artifact_identity(Path(identity.get("path", ""))) != identity:
        raise ValueError(f"immutable artifact identity changed: {receipt}")
    return saved
