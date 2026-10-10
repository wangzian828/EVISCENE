from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

from eviscene.redaction import sanitize_for_persistence


ROLLOUT_MEMBER_SCHEMA_VERSION = "ifv-rollout-group-member-v1"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_descriptor(path: Path | None) -> Dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def load_json_object(path: Path, *, name: str = "JSON file") -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return payload


def load_jsonl_objects(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_no} must be an object: {path}")
        rows.append(row)
    return rows


def load_benchmark(path: Path) -> list[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl_objects(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Benchmark must be a JSON list or JSONL file.")
    if not all(isinstance(item, dict) for item in payload):
        raise TypeError("Benchmark JSON list must contain objects only.")
    return list(payload)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    write_text(
        path,
        json.dumps(
            sanitize_for_persistence(payload),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    serialized = [
        json.dumps(
            sanitize_for_persistence(row),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for row in rows
    ]
    write_text(path, "\n".join(serialized) + ("\n" if serialized else ""))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(text, encoding="utf-8")
    pending.replace(path)


def private_index(
    rows: list[Dict[str, Any]],
    *,
    key: str,
    name: str,
) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rendered = next(
            (
                str(row.get(candidate_key, "")).strip()
                for candidate_key in (key, "candidate_id", "assignment_id")
                if str(row.get(candidate_key, "")).strip()
            ),
            "",
        )
        if not rendered:
            raise ValueError(
                f"{name} row lacks non-empty {key}/candidate_id/assignment_id"
            )
        if rendered in indexed:
            raise ValueError(f"duplicate {name} row for {rendered}")
        indexed[rendered] = row
    return indexed
