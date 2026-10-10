from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from eviscene.integrations.search.serper import SerperLensSearchClient
from eviscene.integrations.http_sessions import (
    close_response,
    close_tracked_sessions,
    get_tracked_session,
    init_tracked_sessions,
    configure_provider_session,
)


DEFAULT_TEMP_UPLOAD_URL = "https://litterbox.catbox.moe/resources/internals/api.php"
DEFAULT_ZHIPU_IMAGE_SEARCH_URL = "https://search-svip.bigmodel.cn/api/paas/v4/image_search"


def _get_proxies() -> Optional[Dict[str, str]]:
    proxy = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("http_proxy")
    )
    return {"http": proxy, "https": proxy} if proxy else None


def _is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _extract_response_url(response: requests.Response, upload_url: str) -> str:
    content_type = response.headers.get("Content-Type", "").lower()
    if "application/json" in content_type:
        payload = response.json()
        for key in ("url", "data", "link"):
            value = payload.get(key, "")
            if isinstance(value, str) and _is_http_url(value.strip()):
                return value.strip()
    else:
        text = response.text.strip()
        if _is_http_url(text):
            return text

    raise RuntimeError(f"Upload service did not return a public URL: {upload_url}")


@dataclass
class ImageUploadClient:
    """Upload a local image through one explicitly selected provider."""

    provider: Optional[str] = None
    upload_api_url: Optional[str] = None
    temp_upload_url: str = DEFAULT_TEMP_UPLOAD_URL
    timeout: int = 30
    last_upload_meta: Dict[str, Any] | None = None
    oss_signed_url_expiry_seconds: int = 3600
    upload_cache_ttl_seconds: int = 900

    def __post_init__(self) -> None:
        self.provider = (
            self.provider or os.getenv("IMAGE_UPLOAD_PROVIDER", "oss")
        ).strip().lower()
        if self.provider not in {"oss", "custom", "temp"}:
            raise ValueError(
                "IMAGE_UPLOAD_PROVIDER must be one of: oss, custom, temp."
            )
        if self.upload_api_url is None:
            self.upload_api_url = os.getenv("IMAGE_UPLOAD_API_URL", "").strip() or None
        self.oss_signed_url_expiry_seconds = int(
            os.getenv("OSS_SIGNED_URL_EXPIRY_SECONDS", str(self.oss_signed_url_expiry_seconds)).strip()
        )
        try:
            self.upload_cache_ttl_seconds = max(
                0,
                int(
                    os.getenv(
                        "VISUAL_SEARCH_UPLOAD_CACHE_TTL_SECONDS",
                        str(self.upload_cache_ttl_seconds),
                    ).strip()
                ),
            )
        except ValueError:
            self.upload_cache_ttl_seconds = max(
                0,
                int(self.upload_cache_ttl_seconds),
            )
        self.last_upload_meta = None
        self._thread_local = threading.local()
        self._upload_cache: Dict[str, tuple[float, str, Dict[str, Any]]] = {}
        self._file_hash_cache: Dict[tuple[str, int, int], str] = {}
        self._upload_cache_lock = threading.Lock()
        # oss2 creates its own requests.Session when Bucket() is constructed.
        # Keep one SDK session per worker thread, but own all of them here so
        # Orchestrator.aclose() can release their proxy sockets at case end.
        self._oss_session_local = threading.local()
        self._oss_session_registry_lock = threading.Lock()
        self._oss_sessions: list[Any] = []
        self._oss_sessions_closed = False
        init_tracked_sessions(self)

    def _get_session(self) -> requests.Session:
        return get_tracked_session(self, self._thread_local)

    def close(self) -> None:
        close_tracked_sessions(self)
        with self._oss_session_registry_lock:
            self._oss_sessions_closed = True
            sessions = list(self._oss_sessions)
            self._oss_sessions.clear()
        for session in sessions:
            self._close_oss_session(session)

    @staticmethod
    def _close_oss_session(session: Any) -> None:
        """Close an oss2 session and its nested requests.Session."""

        close = getattr(session, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                return
            return
        nested = getattr(session, "session", None)
        close_nested = getattr(nested, "close", None)
        if callable(close_nested):
            try:
                close_nested()
            except Exception:
                pass

    def _get_oss_session(self) -> Any:
        """Return a thread-owned oss2 session registered for lifecycle cleanup."""

        session = getattr(self._oss_session_local, "session", None)
        if session is not None:
            return session
        try:
            import oss2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("oss2 is required for OSS uploads") from exc

        session = oss2.http.Session()
        # Apply the shared egress adapter to the SDK requests session.
        sdk_session = getattr(session, "session", None)
        if isinstance(sdk_session, requests.Session):
            configure_provider_session(sdk_session)
        with self._oss_session_registry_lock:
            if self._oss_sessions_closed:
                self._close_oss_session(session)
                raise RuntimeError("Image upload client is already closed.")
            self._oss_sessions.append(session)
        self._oss_session_local.session = session
        return session

    def _discard_oss_session(self) -> None:
        """Drop a broken SDK session after an upload-side transport error."""

        session = getattr(self._oss_session_local, "session", None)
        self._oss_session_local.session = None
        if session is None:
            return
        with self._oss_session_registry_lock:
            try:
                self._oss_sessions.remove(session)
            except ValueError:
                pass
        self._close_oss_session(session)

    def upload(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_sha256 = self._file_sha256(path)
        cached = self._get_cached_upload(image_sha256)
        if cached is not None:
            url, metadata = cached
            self.last_upload_meta = {
                **metadata,
                "image_sha256": image_sha256,
                "cache_hit": True,
            }
            return url

        if self.provider == "oss":
            self.validate_configuration()
            url = self._upload_to_oss(path)
            self.last_upload_meta = {
                "source": "oss",
                "used_temp_host": False,
                "upload_api_url": "",
                "image_sha256": image_sha256,
                "cache_hit": False,
            }
            self._cache_upload(image_sha256, url, self.last_upload_meta)
            return url

        if self.provider == "custom":
            self.validate_configuration()
            assert self.upload_api_url is not None
            url = self._upload_via_http(path, self.upload_api_url)
            self.last_upload_meta = {
                "source": "custom_upload_api",
                "used_temp_host": False,
                "upload_api_url": self.upload_api_url,
                "image_sha256": image_sha256,
                "cache_hit": False,
            }
            self._cache_upload(image_sha256, url, self.last_upload_meta)
            return url

        url = self._upload_via_http(path, self.temp_upload_url)
        self.last_upload_meta = {
            "source": "temp_host",
            "used_temp_host": True,
            "upload_api_url": self.temp_upload_url,
            "image_sha256": image_sha256,
            "cache_hit": False,
        }
        self._cache_upload(image_sha256, url, self.last_upload_meta)
        return url

    def _get_cached_upload(
        self,
        image_sha256: str,
    ) -> Optional[tuple[str, Dict[str, Any]]]:
        if self.upload_cache_ttl_seconds <= 0:
            return None
        now = time.time()
        with self._upload_cache_lock:
            item = self._upload_cache.get(image_sha256)
            if item is None:
                return None
            created_at, url, metadata = item
            if now - created_at > self.upload_cache_ttl_seconds:
                self._upload_cache.pop(image_sha256, None)
                return None
            return url, dict(metadata)

    def _cache_upload(
        self,
        image_sha256: str,
        url: str,
        metadata: Dict[str, Any],
    ) -> None:
        if self.upload_cache_ttl_seconds <= 0 or not url:
            return
        with self._upload_cache_lock:
            self._upload_cache[image_sha256] = (
                time.time(),
                url,
                dict(metadata),
            )

    def _file_sha256(self, path: Path) -> str:
        stat = path.stat()
        cache_key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        with self._upload_cache_lock:
            cached = self._file_hash_cache.get(cache_key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        with self._upload_cache_lock:
            self._file_hash_cache[cache_key] = value
        return value

    def validate_configuration(self) -> None:
        if self.provider == "oss" and not self._oss_is_configured():
            raise RuntimeError(
                "IMAGE_UPLOAD_PROVIDER=oss requires OSS_ACCESS_KEY_ID, "
                "OSS_ACCESS_KEY_SECRET, OSS_ENDPOINT, and OSS_BUCKET_NAME."
            )
        if self.provider == "custom" and not self.upload_api_url:
            raise RuntimeError(
                "IMAGE_UPLOAD_PROVIDER=custom requires IMAGE_UPLOAD_API_URL."
            )

    @staticmethod
    def _oss_is_configured() -> bool:
        required = (
            "OSS_ACCESS_KEY_ID",
            "OSS_ACCESS_KEY_SECRET",
            "OSS_ENDPOINT",
            "OSS_BUCKET_NAME",
        )
        return all(os.getenv(item, "").strip() for item in required)

    def _upload_to_oss(self, path: Path) -> str:
        try:
            import oss2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("oss2 is required for OSS uploads") from exc

        access_key_id = os.getenv("OSS_ACCESS_KEY_ID", "").strip()
        access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET", "").strip()
        endpoint = os.getenv("OSS_ENDPOINT", "").strip()
        bucket_name = os.getenv("OSS_BUCKET_NAME", "").strip()
        key_prefix = os.getenv("OSS_KEY_PREFIX", "image-search").strip().strip("/")

        auth = oss2.Auth(access_key_id, access_key_secret)
        bucket = oss2.Bucket(
            auth,
            endpoint,
            bucket_name,
            session=self._get_oss_session(),
        )

        object_name = f"{uuid.uuid4().hex}_{path.name}"
        if key_prefix:
            object_name = f"{key_prefix}/{object_name}"

        try:
            with path.open("rb") as handle:
                result = bucket.put_object(object_name, handle)
        except Exception:
            # oss2 consumes a small error body internally and raises before it
            # returns its response wrapper.  Retire the whole SDK session so a
            # proxy socket from that failed request cannot remain in its pool.
            self._discard_oss_session()
            raise
        self._release_oss_upload_response(result)

        if _env_flag("OSS_USE_SIGNED_URL", True):
            signed_url = bucket.sign_url(
                "GET",
                object_name,
                self.oss_signed_url_expiry_seconds,
                slash_safe=True,
            )
            return str(signed_url)

        endpoint_host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")
        scheme = "https" if endpoint.startswith("https://") or not endpoint.startswith("http://") else "http"
        return f"{scheme}://{bucket_name}.{endpoint_host}/{object_name}"

    @staticmethod
    def _release_oss_upload_response(result: Any) -> None:
        """Release an oss2 ``put_object`` response as soon as upload completes.

        oss2 sends requests with ``stream=True``.  A successful ``put_object``
        result retains that response but exposes only metadata needed by this
        client, so leaving it unread can keep the proxy socket occupied until
        the whole rollout closes its OSS session.  Read the normally empty
        response body, then explicitly close its wrapped requests response.
        Cleanup remains best effort: the upload status has already been
        validated by oss2 before this point.
        """

        response = getattr(result, "resp", None)
        try:
            read = getattr(response, "read", None)
            if callable(read):
                read()
        except Exception:
            pass
        finally:
            raw_response = getattr(response, "response", None)
            close = getattr(raw_response, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _upload_via_http(self, path: Path, upload_url: str) -> str:
        proxies = _get_proxies()
        response = None
        try:
            with path.open("rb") as handle:
                if "catbox" in upload_url or "litterbox" in upload_url:
                    response = self._get_session().post(
                        upload_url,
                        data={"reqtype": "fileupload", "time": "1h"},
                        files={"fileToUpload": (path.name, handle)},
                        timeout=self.timeout,
                        proxies=proxies,
                    )
                else:
                    response = self._get_session().post(
                        upload_url,
                        files={"file": (path.name, handle)},
                        timeout=self.timeout,
                        proxies=proxies,
                    )
            response.raise_for_status()
            return _extract_response_url(response, upload_url)
        finally:
            close_response(response)


@dataclass
class ZhipuImageSearchClient:
    api_key: Optional[str] = None
    endpoint: str = DEFAULT_ZHIPU_IMAGE_SEARCH_URL
    timeout: int = 30
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("ZHIPU_API_KEY", "").strip() or None
        self.endpoint = os.getenv("ZHIPU_IMAGE_SEARCH_URL", self.endpoint).strip() or self.endpoint
        self._thread_local = threading.local()
        init_tracked_sessions(self)

    def _get_session(self) -> requests.Session:
        return get_tracked_session(self, self._thread_local)

    def close(self) -> None:
        close_tracked_sessions(self)

    def search(self, image_url: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("ZHIPU_API_KEY is not set.")
        if not image_url:
            raise ValueError("image_url is required for zhipu image search.")

        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
            "Accept": "*/*",
        }
        payload = {"url": image_url}
        proxies = _get_proxies()

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            response = None
            try:
                response = self._get_session().post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    proxies=proxies,
                )
                response.raise_for_status()
                data = response.json()
                raw_results = data.get("search_result", []) or []
                results: List[Dict[str, Any]] = []
                for idx, item in enumerate(raw_results[:top_k]):
                    link = str(item.get("link", "")).strip()
                    image_hit = str(item.get("image_url", "")).strip()
                    if not link and not image_hit:
                        continue
                    results.append(
                        {
                            "query": image_url,
                            "rank": idx + 1,
                            "title": str(item.get("title", "")).strip() or "image",
                            "url": link,
                            "image_url": image_hit,
                            "thumbnail_url": str(item.get("thumbnail_url", "")).strip(),
                            "source": str(item.get("source", "")).strip(),
                            "snippet": str(item.get("content", "")).strip(),
                            "provider": "zhipu_image_search",
                        }
                    )
                return results
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(1.5 * (attempt + 1))
            finally:
                close_response(response)

        if last_error is not None:
            raise last_error
        raise RuntimeError("ZhipuImageSearchClient failed without a captured exception.")


@dataclass
class VisualReverseSearchClient:
    upload_client: Optional[ImageUploadClient] = None
    serper_lens_client: Optional[SerperLensSearchClient] = None
    zhipu_client: Optional[ZhipuImageSearchClient] = None
    provider: Optional[str] = None

    def __post_init__(self) -> None:
        self.provider = (
            self.provider or os.getenv("VISUAL_SEARCH_PROVIDER", "serper_lens")
        ).strip().lower()
        if self.provider not in {"serper_lens", "zhipu_image_search"}:
            raise ValueError(
                "VISUAL_SEARCH_PROVIDER must be one of: serper_lens, zhipu_image_search."
            )
        if self.upload_client is None:
            self.upload_client = ImageUploadClient()
        if self.serper_lens_client is None:
            self.serper_lens_client = SerperLensSearchClient()
        if self.zhipu_client is None:
            self.zhipu_client = ZhipuImageSearchClient()

    def prepare_image_url(self, image_input: str) -> str:
        if _is_http_url(image_input):
            return image_input
        if image_input.startswith("data:"):
            raise ValueError("Data URLs are not supported for visual reverse search.")
        return self.upload_client.upload(image_input)

    def search(self, image_input: str, *, top_k: int = 5) -> Dict[str, Any]:
        total_t0 = time.perf_counter()
        upload_t0 = time.perf_counter()
        image_url = self.prepare_image_url(image_input)
        upload_duration_ms = round((time.perf_counter() - upload_t0) * 1000, 2)
        upload_meta = dict(self.upload_client.last_upload_meta or {})
        timings: Dict[str, Any] = {
            "upload_ms": upload_duration_ms,
        }

        provider_name = str(self.provider)
        if provider_name == "zhipu_image_search":
            if not self.zhipu_client or not self.zhipu_client.api_key:
                raise RuntimeError(
                    "VISUAL_SEARCH_PROVIDER=zhipu_image_search requires ZHIPU_API_KEY."
                )
            handler = self.zhipu_client.search
        else:
            handler = self._search_with_serper

        provider_t0 = time.perf_counter()
        results = handler(image_url, top_k=top_k)
        provider_duration_ms = round((time.perf_counter() - provider_t0) * 1000, 2)
        timings[f"{provider_name}_ms"] = provider_duration_ms
        return {
            "image_url": image_url,
            "provider": provider_name,
            "results": results,
            "upload": upload_meta,
            "timings": {
                **timings,
                "provider_ms": provider_duration_ms,
                "total_ms": round((time.perf_counter() - total_t0) * 1000, 2),
            },
        }

    def _search_with_serper(self, image_url: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.serper_lens_client.search(image_url=image_url, top_k=top_k)
