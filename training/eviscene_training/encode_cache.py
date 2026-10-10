from __future__ import annotations

from eviscene.environment import get_env

import atexit
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "ifv-ms-swift-encode-cache-v1"
METRICS_SCHEMA_VERSION = "ifv-ms-swift-encode-cache-metrics-v1"
REPORT_SCHEMA_VERSION = "ifv-ms-swift-encode-cache-report-v1"
_IMAGE_KEYS = {"image", "images"}
_IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
_PATCHED = False
_METRICS_REGISTERED = False
_METRICS: dict[str, dict[str, Any]] = defaultdict(
    lambda: {
        "requests": 0,
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "write_races": 0,
        "bytes_loaded": 0,
        "bytes_written": 0,
        "load_seconds": 0.0,
        "encode_seconds": 0.0,
        "write_seconds": 0.0,
        "errors": [],
    }
)
_METRICS_DIRS: dict[str, Path] = {}
_CONTRACTS: dict[str, dict[str, Any]] = {}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4096)
def _file_fingerprint_cached(
    path_value: str,
    size: int,
    mtime_ns: int,
) -> dict[str, Any]:
    path = Path(path_value)
    return {
        "kind": "file",
        "sha256": _sha256_file(path),
        "bytes": size,
    }


def _file_fingerprint(path_value: str) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    stat = path.stat()
    return _file_fingerprint_cached(
        str(path),
        stat.st_size,
        stat.st_mtime_ns,
    )


def _image_fingerprint(value: Any) -> Any:
    if isinstance(value, Mapping):
        raw_bytes = value.get("bytes")
        if isinstance(raw_bytes, (bytes, bytearray, memoryview)):
            data = bytes(raw_bytes)
            return {
                "kind": "inline_image",
                "sha256": _sha256_bytes(data),
                "bytes": len(data),
            }
        path = str(value.get("path") or "").strip()
        if path:
            return _file_fingerprint(path)
    if isinstance(value, str):
        path = Path(value).expanduser()
        if path.suffix.casefold() in _IMAGE_SUFFIXES and path.is_file():
            return _file_fingerprint(value)
    try:
        from PIL import Image

        if isinstance(value, Image.Image):
            data = value.tobytes()
            return {
                "kind": "pil_image",
                "sha256": _sha256_bytes(data),
                "bytes": len(data),
                "mode": value.mode,
                "size": list(value.size),
            }
    except ImportError:
        pass
    return _canonicalize(value)


def _canonicalize(value: Any, *, image_context: bool = False) -> Any:
    if image_context:
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray, memoryview),
        ):
            return [_image_fingerprint(item) for item in value]
        return _image_fingerprint(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(
                item,
                image_context=str(key).casefold() in _IMAGE_KEYS,
            )
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        return {
            "kind": "bytes",
            "sha256": _sha256_bytes(data),
            "bytes": len(data),
        }
    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            return {
                "kind": "tensor",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": _sha256_bytes(tensor.numpy().tobytes()),
            }
    except ImportError:
        pass
    return {
        "kind": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _model_asset_hashes(model_id: str) -> dict[str, str]:
    root = Path(model_id).expanduser()
    if not root.is_dir():
        return {}
    result: dict[str, str] = {}
    for name in (
        "chat_template.jinja",
        "config.json",
        "configuration.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
    ):
        path = root / name
        if path.is_file():
            result[name] = _sha256_file(path)
    return result


def _object_digest(value: Any) -> str:
    if value is None:
        return ""
    payload: Any
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            payload = dict(getattr(value, "__dict__", {}))
    elif hasattr(value, "__dict__"):
        payload = dict(value.__dict__)
    else:
        payload = {
            "class": f"{type(value).__module__}.{type(value).__qualname__}"
        }
    return _sha256_bytes(_json_bytes(_canonicalize(payload)))


def _template_contract(encode_func: Callable[..., Any]) -> dict[str, Any]:
    template = getattr(encode_func, "__self__", None)
    processor = getattr(template, "processor", None)
    template_meta = getattr(template, "template_meta", None)
    loss_scale = getattr(template, "loss_scale", None)
    model_id = get_env("EVISCENE_MODEL_ID", "")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "framework": {
            "ms_swift": _package_version("ms-swift"),
            "transformers": _package_version("transformers"),
            "torch": _package_version("torch"),
        },
        "model": {
            "id_or_path": model_id,
            "revision": get_env("EVISCENE_MODEL_REVISION", ""),
            "asset_sha256": _model_asset_hashes(model_id),
        },
        "processor": {
            "revision": get_env("EVISCENE_PROCESSOR_REVISION", ""),
            "class": (
                f"{type(processor).__module__}.{type(processor).__qualname__}"
                if processor is not None
                else ""
            ),
            "config_sha256": _object_digest(processor),
        },
        "template": {
            "class": (
                f"{type(template).__module__}.{type(template).__qualname__}"
                if template is not None
                else ""
            ),
            "version": str(getattr(template, "_version", "")),
            "type": str(getattr(template_meta, "template_type", "")),
            "max_length": getattr(template, "max_length", None),
            "loss_scale": {
                "class": (
                    f"{type(loss_scale).__module__}.{type(loss_scale).__qualname__}"
                    if loss_scale is not None
                    else ""
                ),
                "config_sha256": _object_digest(loss_scale),
            },
            "mode": str(getattr(template, "mode", "")),
            "add_non_thinking_prefix": get_env(
                "EVISCENE_ADD_NON_THINKING_PREFIX",
                "",
            ),
        },
        "limits": {
            "max_length": get_env("EVISCENE_MAX_LENGTH", ""),
            "image_max_token_num": get_env(
                "EVISCENE_IMAGE_MAX_TOKEN_NUM",
                get_env("IMAGE_MAX_TOKEN_NUM", ""),
            ),
            "image_min_token_num": get_env("IMAGE_MIN_TOKEN_NUM", ""),
            "max_pixels": get_env(
                "EVISCENE_MAX_PIXELS",
                get_env("MAX_PIXELS", ""),
            ),
        },
    }
    return contract


@dataclass(frozen=True)
class EncodeCacheConfig:
    root: Path
    metrics_dir: Path
    mode: str = "readwrite"

    @classmethod
    def from_environment(cls) -> "EncodeCacheConfig":
        root_value = get_env("EVISCENE_ENCODE_CACHE_DIR", "").strip()
        if not root_value:
            raise RuntimeError(
                "EVISCENE_ENCODE_CACHE_DIR is required when encoded caching is enabled"
            )
        mode = get_env("EVISCENE_ENCODE_CACHE_MODE", "readwrite").strip()
        if mode not in {"readwrite", "readonly", "refresh"}:
            raise ValueError(f"unsupported EVISCENE_ENCODE_CACHE_MODE: {mode}")
        root = Path(root_value).expanduser().resolve()
        metrics_value = get_env("EVISCENE_ENCODE_CACHE_METRICS_DIR", "").strip()
        metrics_dir = (
            Path(metrics_value).expanduser().resolve()
            if metrics_value
            else root / "metrics"
        )
        return cls(root=root, metrics_dir=metrics_dir, mode=mode)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_metrics_at_exit() -> None:
    for contract_digest, stats in list(_METRICS.items()):
        metrics_dir = _METRICS_DIRS.get(contract_digest)
        if metrics_dir is None:
            continue
        rank = get_env("RANK", "na")
        local_rank = get_env("LOCAL_RANK", "na")
        path = metrics_dir / (
            f"rank-{rank}-local-{local_rank}-pid-{os.getpid()}.json"
        )
        payload = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "contract_digest": contract_digest,
            "contract": _CONTRACTS.get(contract_digest, {}),
            "pid": os.getpid(),
            "rank": rank,
            "local_rank": local_rank,
            **stats,
        }
        try:
            _atomic_write_json(path, payload)
        except Exception:
            pass


def _register_metrics(
    contract_digest: str,
    *,
    metrics_dir: Path,
    contract: Mapping[str, Any],
) -> None:
    global _METRICS_REGISTERED
    _METRICS_DIRS[contract_digest] = metrics_dir
    _CONTRACTS[contract_digest] = dict(contract)
    if not _METRICS_REGISTERED:
        atexit.register(_write_metrics_at_exit)
        _METRICS_REGISTERED = True


def _metric_stats(contract_digest: str) -> dict[str, Any]:
    stats = _METRICS[contract_digest]
    pid = os.getpid()
    if stats.get("_pid") != pid:
        stats.clear()
        stats.update(
            {
                "_pid": pid,
                "requests": 0,
                "hits": 0,
                "misses": 0,
                "writes": 0,
                "write_races": 0,
                "bytes_loaded": 0,
                "bytes_written": 0,
                "load_seconds": 0.0,
                "encode_seconds": 0.0,
                "write_seconds": 0.0,
                "errors": [],
            }
        )
    return stats


class CachedEncodeFunction:
    def __init__(
        self,
        encode_func: Callable[..., Any],
        config: EncodeCacheConfig,
    ) -> None:
        self.encode_func = encode_func
        self.config = config
        self.contract = _template_contract(encode_func)
        self.contract_digest = _sha256_bytes(_json_bytes(self.contract))
        self.contract_root = self.config.root / self.contract_digest
        self.contract_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            self.contract_root / "contract.json",
            self.contract,
        )
        _register_metrics(
            self.contract_digest,
            metrics_dir=self.config.metrics_dir,
            contract=self.contract,
        )
        self._eviscene_cache_wrapped = True

    def _key(self, args: Sequence[Any], kwargs: Mapping[str, Any]) -> str:
        return _sha256_bytes(
            _json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "contract_digest": self.contract_digest,
                    "args": _canonicalize(args),
                    "kwargs": _canonicalize(kwargs),
                }
            )
        )

    def _path(self, key: str) -> Path:
        return self.contract_root / key[:2] / key[2:4] / f"{key}.pt"

    def _load(self, path: Path, key: str) -> Any:
        import torch

        started = time.perf_counter()
        try:
            entry = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"encoded cache entry is unreadable: {path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"encoded cache entry is not an object: {path}")
        if entry.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"encoded cache schema mismatch: {path}")
        if entry.get("key") != key:
            raise RuntimeError(f"encoded cache key mismatch: {path}")
        stats = _metric_stats(self.contract_digest)
        stats["load_seconds"] += time.perf_counter() - started
        stats["bytes_loaded"] += path.stat().st_size
        return entry["payload"]

    def _write(self, path: Path, key: str, payload: Any) -> Any:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=".eviscene-encode-",
            suffix=".pt",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as handle:
                torch.save(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "key": key,
                        "contract_digest": self.contract_digest,
                        "payload": payload,
                    },
                    handle,
                )
            if path.exists():
                _metric_stats(self.contract_digest)["write_races"] += 1
                return self._load(path, key)
            try:
                os.link(temporary, path)
            except FileExistsError:
                _metric_stats(self.contract_digest)["write_races"] += 1
                return self._load(path, key)
            stats = _metric_stats(self.contract_digest)
            stats["writes"] += 1
            stats["bytes_written"] += path.stat().st_size
            return payload
        finally:
            _metric_stats(self.contract_digest)["write_seconds"] += (
                time.perf_counter() - started
            )
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = self._key(args, kwargs)
        path = self._path(key)
        stats = _metric_stats(self.contract_digest)
        stats["requests"] += 1
        if path.is_file() and self.config.mode != "refresh":
            stats["hits"] += 1
            return self._load(path, key)
        if self.config.mode == "readonly":
            raise FileNotFoundError(
                f"encoded cache miss in readonly mode: {path}"
            )
        stats["misses"] += 1
        started = time.perf_counter()
        try:
            payload = self.encode_func(*args, **kwargs)
        except Exception as exc:
            stats["errors"].append(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            stats["encode_seconds"] += time.perf_counter() - started
        return self._write(path, key, payload)


def install_ms_swift_encode_cache() -> bool:
    global _PATCHED
    enabled = get_env("EVISCENE_ENCODE_CACHE_ENABLED", "").strip().casefold()
    if enabled not in {"1", "true", "yes"}:
        return False
    if _PATCHED:
        return True

    from swift.dataset.utils import LazyLLMDataset

    original_init = LazyLLMDataset.__init__

    def patched_init(
        self: Any,
        dataset: Any,
        encode_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not getattr(encode_func, "_eviscene_cache_wrapped", False):
            encode_func = CachedEncodeFunction(
                encode_func,
                EncodeCacheConfig.from_environment(),
            )
        original_init(self, dataset, encode_func, *args, **kwargs)

    LazyLLMDataset.__init__ = patched_init
    LazyLLMDataset._eviscene_encode_cache_patched = True
    _PATCHED = True
    print(
        "[eviscene-encode-cache] enabled "
        f"root={EncodeCacheConfig.from_environment().root}",
        flush=True,
    )
    return True


def aggregate_encode_cache_metrics(
    metrics_dir: Path,
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    metrics_dir = metrics_dir.expanduser().resolve()
    files = sorted(metrics_dir.glob("*.json")) if metrics_dir.is_dir() else []
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if payload.get("schema_version") != METRICS_SCHEMA_VERSION:
            errors.append(f"{path.name}: schema mismatch")
            continue
        rows.append(payload)
    numeric = (
        "requests",
        "hits",
        "misses",
        "writes",
        "write_races",
        "bytes_loaded",
        "bytes_written",
        "load_seconds",
        "encode_seconds",
        "write_seconds",
    )
    totals = {
        name: sum(float(row.get(name, 0) or 0) for row in rows)
        for name in numeric
    }
    for name in (
        "requests",
        "hits",
        "misses",
        "writes",
        "write_races",
        "bytes_loaded",
        "bytes_written",
    ):
        totals[name] = int(totals[name])
    row_errors = [
        str(error)
        for row in rows
        for error in row.get("errors", []) or []
    ]
    errors.extend(row_errors)
    requests = int(totals["requests"])
    hits = int(totals["hits"])
    misses = int(totals["misses"])
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metrics_dir": str(metrics_dir),
        "metric_file_count": len(files),
        "valid_metric_file_count": len(rows),
        "contract_digests": sorted(
            {
                str(row.get("contract_digest", ""))
                for row in rows
                if str(row.get("contract_digest", ""))
            }
        ),
        "totals": totals,
        "hit_rate": hits / requests if requests else 0.0,
        "passed": bool(rows)
        and not errors
        and requests == hits + misses,
        "errors": errors,
    }
    if output is not None:
        _atomic_write_json(output.expanduser().resolve(), report)
    return report
