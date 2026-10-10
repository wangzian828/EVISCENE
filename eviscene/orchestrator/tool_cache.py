# -*- coding: utf-8 -*-
"""Simple disk-backed cache for deterministic tool results."""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from weakref import WeakKeyDictionary

from eviscene.orchestrator.tool_result import parse_tool_result
from eviscene.redaction import sanitize_for_persistence


WEB_EVIDENCE_CONTRACT_VERSION = "relation-scope-v3"


_SINGLEFLIGHT_LOCKS: "WeakKeyDictionary[Any, Dict[str, asyncio.Lock]]" = (
    WeakKeyDictionary()
)
_SINGLEFLIGHT_LOCKS_GUARD = threading.Lock()


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _result_is_cacheable(result: str) -> bool:
    try:
        _, succeeded = parse_tool_result(result)
    except Exception:
        return False
    return succeeded


@dataclass
class ToolResultCache:
    """Disk-backed cache keyed by tool name + normalized arguments."""

    cache_dir: str = ".cache/tool_results"
    enabled: bool = False
    ttl_seconds: float = 3600.0
    namespace: str = "tool-contract-v1"
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _file_hashes: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.enabled:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        if not self.enabled:
            return None
        path = self._entry_path(tool_name, args)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        created_at = payload.get("created_at")
        if not isinstance(created_at, (int, float)):
            return None
        if self.ttl_seconds <= 0 or time.time() - float(created_at) > self.ttl_seconds:
            return None
        result = payload.get("result")
        if not isinstance(result, str) or not _result_is_cacheable(result):
            return None
        return result

    def put(self, tool_name: str, args: Dict[str, Any], result: str) -> None:
        if not self.enabled or not _result_is_cacheable(result):
            return
        path = self._entry_path(tool_name, args)
        payload = {
            "namespace": self.namespace,
            "tool_name": tool_name,
            "normalized_args": sanitize_for_persistence(self._normalize_value(args)),
            "result": sanitize_for_persistence(str(result)),
            "created_at": time.time(),
        }
        with self._lock:
            path.write_text(_stable_json_dumps(payload), encoding="utf-8")

    @asynccontextmanager
    async def singleflight(self, tool_name: str, args: Dict[str, Any]):
        """Serialize concurrent production of one cache key.

        The lock is process-local and event-loop-local. Different cache objects
        used by rollout children still coordinate when they share the same event
        loop and cache directory, while separate processes remain independent.
        Cache reads and writes stay the source of truth; this only prevents a
        cache miss stampede.
        """

        if not self.enabled:
            yield
            return

        loop = asyncio.get_running_loop()
        key = str(self._entry_path(tool_name, args))
        with _SINGLEFLIGHT_LOCKS_GUARD:
            loop_locks = _SINGLEFLIGHT_LOCKS.setdefault(loop, {})
            lock = loop_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                loop_locks[key] = lock

        await lock.acquire()
        try:
            yield
        finally:
            lock.release()
            waiters = getattr(lock, "_waiters", None)
            has_waiters = bool(waiters) if waiters is not None else False
            if not lock.locked() and not has_waiters:
                with _SINGLEFLIGHT_LOCKS_GUARD:
                    loop_locks = _SINGLEFLIGHT_LOCKS.get(loop)
                    if loop_locks is not None and loop_locks.get(key) is lock:
                        loop_locks.pop(key, None)

    def _entry_path(self, tool_name: str, args: Dict[str, Any]) -> Path:
        key_payload = {
            "namespace": self.namespace,
            "tool_name": tool_name,
            "args": self._normalize_value(args),
        }
        digest = hashlib.sha256(_stable_json_dumps(key_payload).encode("utf-8")).hexdigest()
        return Path(self.cache_dir) / f"{digest}.json"

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._normalize_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, list):
            return [self._normalize_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._normalize_value(item) for item in value]
        if isinstance(value, str):
            path = Path(value)
            if path.exists() and path.is_file():
                return {
                    "__file__": path.name,
                    "__sha256__": self._hash_file(path),
                }
            return value.strip()
        return value

    def _hash_file(self, path: Path) -> str:
        key = str(path.resolve())
        with self._lock:
            cached = self._file_hashes.get(key)
        if cached is not None:
            return cached

        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        with self._lock:
            self._file_hashes[key] = digest
        return digest
