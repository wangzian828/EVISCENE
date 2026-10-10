"""Durable runtime events, content-addressed artifacts, and context manifests."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from eviscene.redaction import sanitize_for_persistence


CONTEXT_LEDGER_SCHEMA_VERSION = "ifv-context-ledger-v1"
RUNTIME_EVENT_SCHEMA_VERSION = "ifv-runtime-events-v1"
CANONICAL_TRACE_SCHEMA_VERSION = "ifv-canonical-trace-v5"

_CURRENT_CASE_RUNTIME_STORE: ContextVar[Optional["CaseRuntimeStore"]] = ContextVar(
    "eviscene_current_case_runtime_store",
    default=None,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write JSON through an fsynced sibling and atomic replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".tmp-{uuid.uuid4().hex[:8]}")
    persisted = sanitize_for_persistence(payload)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(persisted, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _safe_name(value: str) -> str:
    text = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value)
    ).strip("._")
    if not text:
        return "unknown"
    if len(text) <= 36:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"{text[:24]}-{digest}"


def _recovery_normalize(value: Any) -> Any:
    """Normalize tool arguments into a stable, retry-safe cache key."""

    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            # These are model/runtime annotations, not part of the external
            # request. They can differ when the same action is replayed.
            if str(key).startswith("__"):
                continue
            normalized[str(key)] = _recovery_normalize(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_recovery_normalize(item) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        # Model-produced strings are often long claims, evidence passages, or
        # search queries.  ``Path.is_file()`` can raise ``OSError: [Errno 36]
        # File name too long`` for those strings instead of simply returning
        # False.  A recovery key must treat such values as ordinary text.
        try:
            is_file = candidate.is_file()
        except OSError:
            is_file = False
        if is_file:
            hasher = hashlib.sha256()
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            return {
                "__file_name__": candidate.name,
                "__file_sha256__": hasher.hexdigest(),
            }
        return value.strip()
    return value


def tool_recovery_key(
    *,
    stage: str,
    tool_name: str,
    tool_args: Mapping[str, Any],
) -> str:
    """Return the stable identity of one external tool request."""

    return sha256_bytes(
        stable_json_bytes(
            {
                "stage": str(stage),
                "tool_name": str(tool_name),
                "tool_args": _recovery_normalize(tool_args),
            }
        )
    )


def bind_case_runtime_store(store: Optional["CaseRuntimeStore"]):
    """Bind one case store to the current async task and return its reset token."""

    return _CURRENT_CASE_RUNTIME_STORE.set(store)


def reset_case_runtime_store(token: Any) -> None:
    _CURRENT_CASE_RUNTIME_STORE.reset(token)


def current_case_runtime_store() -> Optional["CaseRuntimeStore"]:
    return _CURRENT_CASE_RUNTIME_STORE.get()


class ContentAddressedArtifactStore:
    """Immutable sha256-addressed byte storage with small JSON descriptors."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def put_bytes(
        self,
        content: bytes,
        *,
        media_type: str = "application/octet-stream",
        suffix: str = ".bin",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        digest = sha256_bytes(content)
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        path = self.root / digest[:2] / f"{digest}{safe_suffix}"
        descriptor_path = path.with_suffix(path.suffix + ".meta.json")
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:8]}")
                with temporary.open("wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            if not descriptor_path.exists():
                atomic_write_json(
                    descriptor_path,
                    {
                        "sha256": digest,
                        "media_type": media_type,
                        "byte_count": len(content),
                        "artifact_path": path.relative_to(self.root).as_posix(),
                        "created_at": utc_now_iso(),
                        "metadata": dict(metadata or {}),
                    },
                )
        return {
            "sha256": digest,
            "media_type": media_type,
            "byte_count": len(content),
            "artifact_path": path.relative_to(self.root).as_posix(),
        }

    def put_text(
        self,
        value: str,
        *,
        media_type: str = "text/plain; charset=utf-8",
        suffix: str = ".txt",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.put_bytes(
            str(value).encode("utf-8"),
            media_type=media_type,
            suffix=suffix,
            metadata=metadata,
        )

    def read_bytes(self, descriptor: Mapping[str, Any]) -> bytes:
        relative = str(descriptor.get("artifact_path", "")).strip()
        path = (self.root / relative).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("artifact path escapes the content-addressed store")
        content = path.read_bytes()
        expected = str(descriptor.get("sha256", "")).strip()
        if expected and sha256_bytes(content) != expected:
            raise ValueError("artifact hash does not match its descriptor")
        return content


class CaseRuntimeStore:
    """One case-attempt event stream and its immutable artifact namespace."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        case_id: str,
        attempt_id: Optional[str] = None,
        resume_from: Optional[str | Path] = None,
    ) -> None:
        self.case_id = str(case_id)
        self.resume_from = (
            Path(resume_from).expanduser().resolve()
            if resume_from is not None
            else None
        )
        self.attempt_id = attempt_id or (
            datetime.now(timezone.utc).strftime("%H%M%S")
            + "-"
            + uuid.uuid4().hex[:6]
        )
        self.root = (
            Path(output_root)
            / "runtime"
            / _safe_name(self.case_id)
            / _safe_name(self.attempt_id)
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.snapshots_dir = self.root / "snapshots"
        self.archive_index_path = self.root / "archive" / "items.jsonl"
        self.artifacts = ContentAddressedArtifactStore(self.root / "artifacts" / "sha256")
        self._event_lock = threading.Lock()
        self._sequence = self._last_sequence()
        self._recovery_index: Optional[Dict[str, Dict[str, Any]]] = None
        self.context_ledger = ContextLedger(self)
        self.append_event(
            "case_attempt_started",
            {
                "case_id": self.case_id,
                "attempt_id": self.attempt_id,
            },
        )

    def reuse_tool_result(
        self,
        *,
        stage: str,
        tool_name: str,
        tool_args: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Reuse one successful tool result from a prior attempt.

        A replay only accepts an archived result whose request identity matches
        exactly and whose persisted payload still has ``status=success``.
        Failed, partial, or torn results are deliberately never reused.
        """

        if self.resume_from is None:
            return None
        key = tool_recovery_key(
            stage=stage,
            tool_name=tool_name,
            tool_args=tool_args,
        )
        if self._recovery_index is None:
            self._recovery_index = self._build_recovery_index()
        candidate = self._recovery_index.get(key)
        if candidate is None:
            return None

        source_root = Path(str(candidate["source_root"]))
        artifact = candidate.get("artifact")
        if not isinstance(artifact, Mapping):
            return None
        try:
            raw = ContentAddressedArtifactStore(
                source_root / "artifacts" / "sha256"
            ).read_bytes(artifact).decode("utf-8")
            payload = json.loads(raw)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, Mapping) or payload.get("status") != "success":
            return None

        self.append_event(
            "tool_result_reused",
            {
                "stage": str(stage),
                "tool_name": str(tool_name),
                "recovery_key": key,
                "source_attempt_id": candidate.get("attempt_id"),
                "source_runtime_path": source_root.as_posix(),
                "source_memory_id": candidate.get("memory_id"),
                "artifact": dict(artifact),
            },
        )
        return {
            "result": raw,
            "recovery_key": key,
            "source_attempt_id": candidate.get("attempt_id"),
            "source_runtime_path": source_root.as_posix(),
            "source_memory_id": candidate.get("memory_id"),
            "artifact": dict(artifact),
        }

    def _build_recovery_index(self) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for source_root in self._recovery_attempt_roots():
            archive_path = source_root / "archive" / "items.jsonl"
            if not archive_path.exists():
                continue
            for raw in archive_path.read_bytes().splitlines():
                try:
                    item = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(item, Mapping):
                    continue
                tool_name = str(item.get("tool_name", "")).strip()
                stage = str(item.get("stage", "")).strip()
                tool_args = item.get("tool_args")
                artifact = item.get("artifact")
                if not tool_name or not stage or not isinstance(tool_args, Mapping):
                    continue
                if not isinstance(artifact, Mapping):
                    continue
                key = str(item.get("recovery_key", "")).strip() or tool_recovery_key(
                    stage=stage,
                    tool_name=tool_name,
                    tool_args=tool_args,
                )
                # Newer attempts win if the same request was archived more
                # than once across retries.
                index[key] = {
                    "attempt_id": item.get("attempt_id"),
                    "source_root": source_root.as_posix(),
                    "memory_id": item.get("memory_id"),
                    "artifact": dict(artifact),
                }
        return index

    def _recovery_attempt_roots(self) -> list[Path]:
        if self.resume_from is None:
            return []
        source = self.resume_from
        candidates: list[Path] = []
        if (source / "events.jsonl").is_file():
            candidates.append(source)
        for base in (
            source,
            source / "runtime",
            source / "traces" / "runtime",
        ):
            case_root = base / _safe_name(self.case_id)
            if case_root.is_dir():
                candidates.extend(
                    path
                    for path in case_root.iterdir()
                    if path.is_dir() and (path / "events.jsonl").is_file()
                )
        unique = {
            path.resolve()
            for path in candidates
            if path.resolve() != self.root.resolve()
        }
        return sorted(
            unique,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    @property
    def descriptor(self) -> Dict[str, Any]:
        return {
            "schema_version": RUNTIME_EVENT_SCHEMA_VERSION,
            "case_id": self.case_id,
            "attempt_id": self.attempt_id,
            "runtime_path": self.root.as_posix(),
            "events_path": self.events_path.as_posix(),
        }

    def append_event(self, event_type: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Append one fsynced JSON line; readers ignore a torn final line."""

        with self._event_lock:
            self._sequence += 1
            event = sanitize_for_persistence(
                {
                    "schema_version": RUNTIME_EVENT_SCHEMA_VERSION,
                    "sequence": self._sequence,
                    "event_id": f"evt-{self._sequence:08d}",
                    "event_type": str(event_type),
                    "recorded_at": utc_now_iso(),
                    "case_id": self.case_id,
                    "attempt_id": self.attempt_id,
                    "payload": dict(payload),
                }
            )
            line = stable_json_bytes(event) + b"\n"
            with self.events_path.open("ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            return dict(event)

    def archive_tool_result(
        self,
        *,
        stage: str,
        action_index: int,
        tool_name: str,
        tool_args: Mapping[str, Any],
        tool_result: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        descriptor = self.artifacts.put_text(
            tool_result,
            media_type="application/json; charset=utf-8",
            suffix=".json",
            metadata={
                "kind": "tool_result",
                "stage": stage,
                "action_index": action_index,
                "tool_name": tool_name,
            },
        )
        memory_id = "memory-" + sha256_bytes(
            stable_json_bytes(
                {
                    "stage": stage,
                    "action_index": action_index,
                    "tool_name": tool_name,
                    "tool_args": dict(tool_args),
                    "artifact_sha256": descriptor["sha256"],
                }
            )
        )[:20]
        archive_item = {
            "schema_version": "ifv-investigation-archive-v1",
            "memory_id": memory_id,
            "case_id": self.case_id,
            "attempt_id": self.attempt_id,
            "stage": stage,
            "action_index": action_index,
            "tool_name": tool_name,
            "tool_args": dict(tool_args),
            "recovery_key": tool_recovery_key(
                stage=stage,
                tool_name=tool_name,
                tool_args=tool_args,
            ),
            "artifact": descriptor,
            "metadata": dict(metadata or {}),
            "created_at": utc_now_iso(),
        }
        self._append_archive_item(archive_item)
        self.append_event(
            "tool_result_archived",
            {
                "memory_id": memory_id,
                "stage": stage,
                "action_index": action_index,
                "tool_name": tool_name,
                "tool_args": dict(tool_args),
                "artifact": descriptor,
                "metadata": dict(metadata or {}),
            },
        )
        return {**descriptor, "memory_id": memory_id}

    def recall_archive(
        self,
        *,
        query: str = "",
        filters: Optional[Mapping[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Return bounded archive candidates; this never upgrades Evidence status."""

        normalized_filters = dict(filters or {})
        terms = _search_terms(query)
        candidates: list[tuple[int, Dict[str, Any]]] = []
        for item in self.read_archive_items():
            if not _archive_item_matches(item, normalized_filters):
                continue
            raw = self.artifacts.read_bytes(item.get("artifact", {})).decode(
                "utf-8", errors="replace"
            )
            searchable = " ".join(
                [
                    str(item.get("tool_name", "")),
                    json.dumps(item.get("tool_args", {}), ensure_ascii=False),
                    raw,
                ]
            ).casefold()
            score = sum(searchable.count(term) for term in terms)
            if terms and score <= 0:
                continue
            candidate = {
                "memory_id": item.get("memory_id"),
                "tool_name": item.get("tool_name"),
                "action_index": item.get("action_index"),
                "tool_args": item.get("tool_args", {}),
                "artifact_sha256": item.get("artifact", {}).get("sha256", ""),
                "preview": raw[:600],
                "score": score,
            }
            candidates.append((score, candidate))
        candidates.sort(
            key=lambda row: (
                row[0],
                int(row[1].get("action_index", 0) or 0),
            ),
            reverse=True,
        )
        selected = [item for _, item in candidates[: max(1, min(int(top_k), 12))]]
        event = self.append_event(
            "archive_recall",
            {
                "query": query,
                "filters": normalized_filters,
                "candidate_ids": [item["memory_id"] for item in selected],
                "selected_ids": [],
                "read_spans": [],
                "used_in_request_id": None,
                "decision_impact": "none_recorded",
            },
        )
        return {
            "status": "success",
            "recall_id": event["event_id"],
            "candidates": selected,
            "candidate_count": len(selected),
        }

    def read_archive_item(
        self,
        memory_id: str,
        *,
        offset: int = 0,
        length: int = 6000,
        include_raw: bool = True,
    ) -> Dict[str, Any]:
        item = next(
            (
                row
                for row in self.read_archive_items()
                if str(row.get("memory_id", "")) == str(memory_id)
            ),
            None,
        )
        if item is None:
            return {"status": "error", "error": f"unknown memory_id: {memory_id}"}
        raw = self.artifacts.read_bytes(item.get("artifact", {})).decode(
            "utf-8", errors="replace"
        )
        safe_offset = max(0, min(int(offset), len(raw)))
        safe_length = max(1, min(int(length), 24000))
        end = min(len(raw), safe_offset + safe_length)
        self.append_event(
            "archive_exact_read",
            {
                "memory_id": memory_id,
                "selected_ids": [memory_id],
                "read_spans": [f"{memory_id}:{safe_offset}-{end}"],
                "used_in_request_id": None,
                "decision_impact": "none_recorded",
            },
        )
        return {
            "status": "success",
            "memory_id": memory_id,
            "tool_name": item.get("tool_name"),
            "tool_args": item.get("tool_args", {}),
            "artifact": item.get("artifact", {}),
            "offset": safe_offset,
            "end": end,
            "total_chars": len(raw),
            "has_more": end < len(raw),
            "content": raw[safe_offset:end] if include_raw else "",
        }

    def read_archive_items(self) -> list[Dict[str, Any]]:
        if not self.archive_index_path.exists():
            return []
        ordered_ids: list[str] = []
        by_id: Dict[str, Dict[str, Any]] = {}
        for raw in self.archive_index_path.read_bytes().splitlines():
            try:
                item = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(item, dict):
                continue
            memory_id = str(item.get("memory_id", "")).strip()
            if not memory_id:
                continue
            if memory_id not in by_id:
                ordered_ids.append(memory_id)
                by_id[memory_id] = item
        return [by_id[memory_id] for memory_id in ordered_ids]

    def _append_archive_item(self, item: Mapping[str, Any]) -> None:
        self.archive_index_path.parent.mkdir(parents=True, exist_ok=True)
        line = stable_json_bytes(sanitize_for_persistence(dict(item))) + b"\n"
        with self._event_lock:
            with self.archive_index_path.open("ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    def write_snapshot(self, snapshot_type: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        sequence = self._sequence + 1
        snapshot_id = f"snapshot-{sequence:08d}"
        path = self.snapshots_dir / f"{snapshot_id}.json"
        body = {
            "schema_version": RUNTIME_EVENT_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "snapshot_type": snapshot_type,
            "case_id": self.case_id,
            "attempt_id": self.attempt_id,
            "created_at": utc_now_iso(),
            "payload": dict(payload),
        }
        atomic_write_json(path, body)
        self.append_event(
            "snapshot_committed",
            {
                "snapshot_id": snapshot_id,
                "snapshot_type": snapshot_type,
                "snapshot_path": path.relative_to(self.root).as_posix(),
            },
        )
        return {
            "snapshot_id": snapshot_id,
            "snapshot_type": snapshot_type,
            "snapshot_path": path.relative_to(self.root).as_posix(),
        }

    def read_events(self) -> list[Dict[str, Any]]:
        if not self.events_path.exists():
            return []
        events: list[Dict[str, Any]] = []
        for raw in self.events_path.read_bytes().splitlines():
            try:
                item = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(item, dict):
                events.append(item)
        return events

    def _last_sequence(self) -> int:
        if not self.events_path.exists():
            return 0
        last = 0
        for item in self.read_events():
            try:
                last = max(last, int(item.get("sequence", 0) or 0))
            except (TypeError, ValueError):
                continue
        return last


class ContextLedger:
    """Record the exact logical composition and provider usage of model requests."""

    def __init__(self, store: CaseRuntimeStore) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._counter = 0
        self._open: Dict[str, Dict[str, Any]] = {}

    def begin_request(
        self,
        *,
        stage: str,
        lifecycle_kind: str,
        system_instruction: Any,
        input_payload: Any,
        tools: Any = None,
        response_format: Any = None,
        generation_config: Any = None,
        max_output_tokens: Optional[int] = None,
        previous_interaction_id: Optional[str] = None,
        parent_request_id: Optional[str] = None,
        model: str = "",
        prompt_version: str = "",
    ) -> str:
        with self._lock:
            self._counter += 1
            request_id = f"req-{self._counter:06d}"
        components = [
            ("system_prompt", system_instruction, "stage_instruction"),
            ("input_payload", input_payload, "stage_input"),
            ("tool_schema", tools or [], "available_tools"),
            ("response_format", response_format, "output_contract"),
            ("generation_config", generation_config or {}, "generation_config"),
        ]
        items: list[Dict[str, Any]] = []
        explicit_chars = 0
        serialized_input_chars = 0
        media_bytes = 0
        component_sizes: Dict[str, Dict[str, Any]] = {}
        for index, (kind, value, reason) in enumerate(components, start=1):
            if value is None or value == [] or value == {} or value == "":
                continue
            persisted, media = self._externalize_media(value)
            serialized_actual = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            serialized_persisted = json.dumps(
                persisted,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            # Base64 image bytes are transported as media, not ordinary text
            # tokens.  Estimate text from the externalized representation while
            # retaining the raw serialized size and exact request hash.
            explicit_chars += len(serialized_persisted)
            serialized_input_chars += len(serialized_actual)
            component_media_bytes = sum(
                int(item.get("byte_count", 0) or 0) for item in media
            )
            media_bytes += component_media_bytes
            component_sizes[kind] = {
                "char_count": len(serialized_persisted),
                "serialized_input_chars": len(serialized_actual),
                "token_count_estimate": max(
                    1,
                    (len(serialized_persisted) + 3) // 4,
                ),
                "media_bytes": component_media_bytes,
                "content_hash": sha256_bytes(
                    serialized_actual.encode("utf-8")
                ),
            }
            descriptor = self.store.artifacts.put_bytes(
                stable_json_bytes(persisted),
                media_type="application/json; charset=utf-8",
                suffix=".json",
                metadata={
                    "kind": "context_item",
                    "request_id": request_id,
                    "context_kind": kind,
                },
            )
            items.append(
                {
                    "context_item_id": f"{request_id}-ctx-{index:03d}",
                    "request_id": request_id,
                    "position": index,
                    "kind": kind,
                    "source_ids": [],
                    "content_hash": sha256_bytes(serialized_actual.encode("utf-8")),
                    "char_count": len(serialized_persisted),
                    "serialized_input_chars": len(serialized_actual),
                    "token_count_estimate": max(1, (len(serialized_persisted) + 3) // 4),
                    "priority": "recent",
                    "inclusion_reason": reason,
                    "retrieval_mode": "explicit",
                    "compressed": False,
                    "artifact": descriptor,
                    "media": media,
                }
            )
        manifest = {
            "schema_version": CONTEXT_LEDGER_SCHEMA_VERSION,
            "request_id": request_id,
            "logical_request_id": request_id,
            "case_id": self.store.case_id,
            "attempt_id": self.store.attempt_id,
            "stage": stage,
            "lifecycle_kind": lifecycle_kind,
            "prompt_version": prompt_version,
            "model": model,
            "parent_interaction_id": previous_interaction_id,
            "parent_request_id": parent_request_id,
            "explicit_input_chars": explicit_chars,
            "explicit_input_tokens_estimate": max(1, (explicit_chars + 3) // 4),
            "serialized_input_chars": serialized_input_chars,
            "media_bytes": media_bytes,
            "component_sizes": component_sizes,
            "request_snapshot_sha256": sha256_bytes(
                stable_json_bytes(
                    {
                        "system_instruction": system_instruction,
                        "input_payload": input_payload,
                        "tools": tools or [],
                        "response_format": response_format,
                        "generation_config": generation_config or {},
                        "max_output_tokens": max_output_tokens,
                        "previous_interaction_id": previous_interaction_id,
                    }
                )
            ),
            "provider_input_tokens": None,
            "provider_output_tokens": None,
            "provider_thought_tokens": None,
            "retry_metadata": None,
            "response_content_chars": None,
            "response_reasoning_chars": None,
            "reasoning_artifact": None,
            "max_output_tokens": max_output_tokens,
            "image_count": sum(len(item.get("media", [])) for item in items),
            "status": "started",
            "started_at": utc_now_iso(),
            "completed_at": None,
            "interaction_id": None,
            "context_items": items,
        }
        with self._lock:
            self._open[request_id] = manifest
        self.store.append_event("context_request_started", manifest)
        return request_id

    def complete_request(
        self,
        request_id: str,
        *,
        usage: Optional[Mapping[str, Any]] = None,
        interaction_id: Optional[str] = None,
        status: str = "completed",
        error: str = "",
        response_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            manifest = self._open.pop(request_id, None)
        if manifest is None:
            raise KeyError(f"unknown context request {request_id}")
        normalized_usage = dict(usage or {})
        normalized_response = dict(response_metadata or {})
        manifest.update(
            {
                "provider_input_tokens": _usage_int(
                    normalized_usage,
                    "total_input_tokens",
                    "input_tokens",
                    "prompt_tokens",
                ),
                "provider_output_tokens": _usage_int(
                    normalized_usage,
                    "total_output_tokens",
                    "output_tokens",
                    "completion_tokens",
                ),
                "provider_thought_tokens": _usage_int(
                    normalized_usage,
                    "total_thought_tokens",
                    "thought_tokens",
                ),
                "retry_metadata": normalized_response.get("retry_metadata"),
                "response_content_chars": normalized_response.get(
                    "response_content_chars"
                ),
                "response_reasoning_chars": normalized_response.get(
                    "response_reasoning_chars"
                ),
                "reasoning_artifact": normalized_response.get(
                    "reasoning_artifact"
                ),
                "interaction_id": interaction_id,
                "status": status,
                "error": error,
                "completed_at": utc_now_iso(),
            }
        )
        manifest_path = self.store.root / "context" / f"{request_id}.json"
        atomic_write_json(manifest_path, manifest)
        self.store.append_event(
            "context_request_completed",
            {
                "request_id": request_id,
                "status": status,
                "interaction_id": interaction_id,
                "manifest_path": manifest_path.relative_to(self.store.root).as_posix(),
                "provider_input_tokens": manifest["provider_input_tokens"],
                "provider_output_tokens": manifest["provider_output_tokens"],
                "provider_thought_tokens": manifest["provider_thought_tokens"],
                "retry_metadata": manifest["retry_metadata"],
                "response_content_chars": manifest["response_content_chars"],
                "response_reasoning_chars": manifest[
                    "response_reasoning_chars"
                ],
                "reasoning_artifact": manifest["reasoning_artifact"],
                "error": error,
            },
        )
        return manifest

    def reconstruct_request(self, request_id: str) -> Dict[str, Any]:
        path = self.store.root / "context" / f"{request_id}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        reconstructed: Dict[str, Any] = {}
        for item in manifest.get("context_items", []) or []:
            if not isinstance(item, Mapping):
                continue
            raw = self.store.artifacts.read_bytes(item.get("artifact", {}))
            value = json.loads(raw.decode("utf-8"))
            reconstructed[str(item.get("kind", ""))] = self._restore_media(value)
        return reconstructed


    def _externalize_media(self, value: Any) -> tuple[Any, list[Dict[str, Any]]]:
        media: list[Dict[str, Any]] = []

        def visit(item: Any) -> Any:
            if isinstance(item, list):
                return [visit(child) for child in item]
            if not isinstance(item, Mapping):
                return sanitize_for_persistence(item)
            candidate = dict(item)
            url = ""
            if str(candidate.get("type", "")).lower() == "image_url":
                image_url = candidate.get("image_url")
                if isinstance(image_url, Mapping):
                    url = str(image_url.get("url", ""))
            elif str(candidate.get("type", "")).lower() == "image":
                direct_data = str(candidate.get("data", ""))
                if direct_data and not direct_data.startswith("data:"):
                    try:
                        content = base64.b64decode(direct_data, validate=True)
                    except (ValueError, base64.binascii.Error):
                        pass
                    else:
                        media_type = str(
                            candidate.get("mime_type", "application/octet-stream")
                        )
                        descriptor = self.store.artifacts.put_bytes(
                            content,
                            media_type=media_type,
                            suffix=_media_suffix(media_type),
                            metadata={"kind": "context_media"},
                        )
                        media.append(descriptor)
                        return {
                            "type": "image",
                            "artifact_ref": descriptor,
                            "media_type": media_type,
                            "encoding": "raw_base64",
                        }
                url = direct_data or str(candidate.get("url", ""))
            if url.startswith("data:") and ";base64," in url:
                header, encoded = url.split(",", 1)
                try:
                    content = base64.b64decode(encoded, validate=True)
                except (ValueError, base64.binascii.Error):
                    return sanitize_for_persistence(candidate)
                media_type = header[5:].split(";", 1)[0] or "application/octet-stream"
                suffix = _media_suffix(media_type)
                descriptor = self.store.artifacts.put_bytes(
                    content,
                    media_type=media_type,
                    suffix=suffix,
                    metadata={"kind": "context_media"},
                )
                media.append(descriptor)
                return {
                    "type": candidate.get("type", "image"),
                    "artifact_ref": descriptor,
                    "media_type": media_type,
                }
            return {
                str(key): visit(child)
                for key, child in candidate.items()
            }

        return visit(value), media

    def _restore_media(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._restore_media(item) for item in value]
        if not isinstance(value, Mapping):
            return value
        descriptor = value.get("artifact_ref")
        if isinstance(descriptor, Mapping):
            content = self.store.artifacts.read_bytes(descriptor)
            media_type = str(value.get("media_type", "application/octet-stream"))
            if str(value.get("encoding", "")) == "raw_base64":
                return {
                    "type": value.get("type", "image"),
                    "mime_type": media_type,
                    "data": base64.b64encode(content).decode("ascii"),
                }
            data_url = (
                f"data:{media_type};base64,"
                + base64.b64encode(content).decode("ascii")
            )
            if str(value.get("type", "")).lower() == "image_url":
                return {"type": "image_url", "image_url": {"url": data_url}}
            return {"type": value.get("type", "image"), "data": data_url}
        return {str(key): self._restore_media(item) for key, item in value.items()}


def reconstruct_archived_request(
    runtime_root: str | Path,
    request_id: str,
) -> Dict[str, Any]:
    """Read one archived provider request without creating a new attempt."""

    root = Path(runtime_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"runtime archive does not exist: {root}")
    manifest_path = root / "context" / f"{str(request_id).strip()}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("runtime context manifest must be an object")
    store = ContentAddressedArtifactStore(root / "artifacts" / "sha256")

    def restore(value: Any) -> Any:
        if isinstance(value, list):
            return [restore(item) for item in value]
        if not isinstance(value, Mapping):
            return value
        descriptor = value.get("artifact_ref")
        if isinstance(descriptor, Mapping):
            content = store.read_bytes(descriptor)
            media_type = str(value.get("media_type", "application/octet-stream"))
            if str(value.get("encoding", "")) == "raw_base64":
                return {"type": value.get("type", "image"), "mime_type": media_type,
                        "data": base64.b64encode(content).decode("ascii")}
            data_url = f"data:{media_type};base64," + base64.b64encode(content).decode("ascii")
            if str(value.get("type", "")).lower() == "image_url":
                return {"type": "image_url", "image_url": {"url": data_url}}
            return {"type": value.get("type", "image"), "data": data_url}
        return {str(key): restore(child) for key, child in value.items()}

    result: Dict[str, Any] = {}
    for item in manifest.get("context_items", []) or []:
        if not isinstance(item, Mapping):
            continue
        raw = store.read_bytes(item.get("artifact", {}))
        result[str(item.get("kind", ""))] = restore(json.loads(raw.decode("utf-8")))
    return result


def _usage_int(usage: Mapping[str, Any], *names: str) -> Optional[int]:
    for name in names:
        if name not in usage:
            continue
        try:
            return int(usage.get(name) or 0)
        except (TypeError, ValueError):
            return None
    return None


def _media_suffix(media_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(str(media_type).lower(), ".bin")


def _search_terms(query: str) -> list[str]:
    return list(
        dict.fromkeys(
            token.casefold()
            for token in re.findall(r"[\w\u3400-\u9fff]+", str(query))
            if len(token.strip()) >= 2
        )
    )


def _archive_item_matches(
    item: Mapping[str, Any],
    filters: Mapping[str, Any],
) -> bool:
    if not filters:
        return True
    aliases = {
        "tool": "tool_name",
        "tools": "tool_name",
        "stage": "stage",
        "action_index": "action_index",
    }
    for raw_key, expected in filters.items():
        key = aliases.get(str(raw_key), str(raw_key))
        if expected in (None, "", []):
            continue
        actual: Any = item.get(key)
        if actual is None:
            metadata = item.get("metadata", {})
            if isinstance(metadata, Mapping):
                actual = metadata.get(key)
        if actual is None:
            tool_args = item.get("tool_args", {})
            if isinstance(tool_args, Mapping):
                actual = tool_args.get(key)
                if actual is None and key == "task_id":
                    actual = tool_args.get("__question_id")
        expected_values = expected if isinstance(expected, list) else [expected]
        actual_values = actual if isinstance(actual, list) else [actual]
        if not {
            str(value).casefold() for value in expected_values
        } & {str(value).casefold() for value in actual_values}:
            return False
    return True
