"""Baidu general OCR client with in-memory access-token caching."""

from __future__ import annotations

import base64
import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

from eviscene.integrations.http_sessions import (
    close_response,
    close_tracked_sessions,
    get_tracked_session,
    init_tracked_sessions,
)


DEFAULT_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
DEFAULT_OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/general"
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_OCR_READ_TIMEOUT_SECONDS = 120.0
DEFAULT_TOKEN_READ_TIMEOUT_SECONDS = 30.0
# OCR retries are opt-in: a timeout can mean the provider received the image
# but its response was lost, so retrying must be an explicit cost decision.
DEFAULT_MAX_RETRIES = 0
TOKEN_CACHE_SKEW_SECONDS = 60.0

_TOKEN_CACHE: dict[tuple[str, str, str], tuple[str, float]] = {}
_TOKEN_CACHE_LOCK = threading.RLock()


class BaiduOCRError(RuntimeError):
    """A bounded Baidu OCR provider failure."""

    def __init__(
        self,
        message: str,
        *,
        request_count: int = 0,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.request_count = max(0, int(request_count))
        self.retryable = bool(retryable)


@dataclass
class BaiduOCRClient:
    """Call Baidu's general OCR endpoint without persisting credentials."""

    api_key: str = ""
    secret_key: str = ""
    token_url: str = ""
    ocr_url: str = ""
    timeout_seconds: float = DEFAULT_OCR_READ_TIMEOUT_SECONDS
    session: Any = field(default=None, repr=False)
    _session_local: threading.local = field(
        default_factory=threading.local,
        init=False,
        repr=False,
    )

    def _api_key(self) -> str:
        return (self.api_key or os.getenv("BAIDU_OCR_API_KEY", "")).strip()

    def _secret_key(self) -> str:
        return (self.secret_key or os.getenv("BAIDU_OCR_SECRET_KEY", "")).strip()

    def __post_init__(self) -> None:
        self._session_local = threading.local()
        init_tracked_sessions(self)

    def _token_url(self) -> str:
        return (
            self.token_url.strip()
            or os.getenv("BAIDU_OCR_TOKEN_URL", "").strip()
            or DEFAULT_TOKEN_URL
        )

    def _ocr_url(self) -> str:
        return (
            self.ocr_url.strip()
            or os.getenv("BAIDU_OCR_URL", "").strip()
            or DEFAULT_OCR_URL
        )

    @staticmethod
    def _positive_timeout(raw: str, default: float) -> float:
        try:
            return max(1.0, float(raw.strip()))
        except ValueError:
            return default

    def _connect_timeout(self) -> float:
        raw = os.getenv(
            "BAIDU_OCR_CONNECT_TIMEOUT_SECONDS",
            str(DEFAULT_CONNECT_TIMEOUT_SECONDS),
        ).strip()
        return self._positive_timeout(raw, DEFAULT_CONNECT_TIMEOUT_SECONDS)

    def _ocr_read_timeout(self) -> float:
        raw = os.getenv(
            "BAIDU_OCR_READ_TIMEOUT_SECONDS",
            os.getenv("BAIDU_OCR_TIMEOUT_SECONDS", str(self.timeout_seconds)),
        ).strip()
        return self._positive_timeout(raw, DEFAULT_OCR_READ_TIMEOUT_SECONDS)

    def _token_read_timeout(self) -> float:
        raw = os.getenv(
            "BAIDU_OCR_TOKEN_READ_TIMEOUT_SECONDS",
            str(DEFAULT_TOKEN_READ_TIMEOUT_SECONDS),
        ).strip()
        return self._positive_timeout(raw, DEFAULT_TOKEN_READ_TIMEOUT_SECONDS)

    def _token_timeout(self) -> tuple[float, float]:
        return (self._connect_timeout(), self._token_read_timeout())

    def _ocr_timeout(self) -> tuple[float, float]:
        return (self._connect_timeout(), self._ocr_read_timeout())

    @staticmethod
    def _max_retries() -> int:
        raw = os.getenv(
            "BAIDU_OCR_MAX_RETRIES",
            str(DEFAULT_MAX_RETRIES),
        ).strip()
        try:
            return max(0, min(2, int(raw)))
        except ValueError:
            return DEFAULT_MAX_RETRIES

    @staticmethod
    def _retry_backoff(attempt: int) -> float:
        raw = os.getenv("BAIDU_OCR_RETRY_BACKOFF_SECONDS", "1.0").strip()
        try:
            base = max(0.0, float(raw))
        except ValueError:
            base = 1.0
        return min(10.0, base * (2**attempt))

    def _session(self) -> Any:
        if self.session is not None:
            return self.session
        return get_tracked_session(self, self._session_local)

    def close(self) -> None:
        close_tracked_sessions(self)

    @staticmethod
    def _payload(response: Any) -> Dict[str, Any]:
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            status_code = int(getattr(response, "status_code", 0) or 0)
            headers = getattr(response, "headers", {}) or {}
            content_type = str(
                headers.get("Content-Type", headers.get("content-type", ""))
            ).split(";", 1)[0].strip().lower() or "<missing>"
            body = getattr(response, "content", b"")
            if isinstance(body, str):
                body = body.encode("utf-8", errors="replace")
            if not isinstance(body, bytes):
                body = b""
            body_summary = (
                f"body_bytes={len(body)}, "
                f"body_sha256={hashlib.sha256(body).hexdigest()[:16]}"
                if body
                else "body_bytes=0"
            )
            raise BaiduOCRError(
                "Baidu OCR returned non-JSON response "
                f"(http_status={status_code or '<unknown>'}, "
                f"content_type={content_type}, {body_summary})",
                request_count=1,
                retryable=status_code == 429 or status_code >= 500 or status_code == 0,
            ) from exc
        if not isinstance(payload, dict):
            raise BaiduOCRError(
                "Baidu OCR returned a non-object JSON response",
                request_count=1,
            )
        return payload

    @staticmethod
    def _provider_error(
        payload: Dict[str, Any],
        *,
        stage: str,
        status_code: int = 0,
    ) -> BaiduOCRError:
        code = payload.get("error_code", payload.get("error"))
        message = str(
            payload.get("error_msg")
            or payload.get("error_description")
            or "unknown provider error"
        ).strip()
        suffix = f" ({code})" if code not in (None, "") else ""
        return BaiduOCRError(
            f"Baidu {stage} failed{suffix}: {message[:300]}",
            request_count=1,
            retryable=(
                status_code == 429
                or status_code >= 500
                or code in {18, "18", 282000, "282000"}
            ),
        )

    def get_access_token(self) -> tuple[str, bool]:
        """Return ``(token, cache_hit)`` without exposing the token in errors."""

        direct_token = os.getenv("BAIDU_OCR_ACCESS_TOKEN", "").strip()
        if direct_token:
            return direct_token, True

        api_key = self._api_key()
        secret_key = self._secret_key()
        if not api_key or not secret_key:
            raise BaiduOCRError(
                "BAIDU_OCR_API_KEY and BAIDU_OCR_SECRET_KEY are required"
            )

        cache_key = (api_key, secret_key, self._token_url())
        now = time.monotonic()
        with _TOKEN_CACHE_LOCK:
            cached = _TOKEN_CACHE.get(cache_key)
            if cached and cached[1] > now:
                return cached[0], True

            token_request_count = 0
            payload: Dict[str, Any]
            for attempt in range(self._max_retries() + 1):
                try:
                    response = self._session().post(
                        self._token_url(),
                        params={
                            "grant_type": "client_credentials",
                            "client_id": api_key,
                            "client_secret": secret_key,
                        },
                        timeout=self._token_timeout(),
                    )
                    token_request_count += 1
                except (requests.Timeout, requests.ConnectionError) as exc:
                    token_request_count += 1
                    if attempt < self._max_retries():
                        time.sleep(self._retry_backoff(attempt))
                        continue
                    raise BaiduOCRError(
                        f"Baidu token request failed: {type(exc).__name__}",
                        request_count=token_request_count,
                    ) from exc

                try:
                    payload = self._payload(response)
                    status_code = int(getattr(response, "status_code", 0) or 0)
                    response_ok = bool(getattr(response, "ok", False))
                except BaiduOCRError as exc:
                    if exc.retryable and attempt < self._max_retries():
                        time.sleep(self._retry_backoff(attempt))
                        continue
                    raise BaiduOCRError(
                        str(exc),
                        request_count=token_request_count,
                    ) from exc
                finally:
                    close_response(response)
                if not response_ok:
                    provider_error = self._provider_error(
                        payload,
                        stage="token request",
                        status_code=status_code,
                    )
                    if provider_error.retryable and attempt < self._max_retries():
                        time.sleep(self._retry_backoff(attempt))
                        continue
                    raise BaiduOCRError(
                        str(provider_error),
                        request_count=token_request_count,
                    ) from provider_error
                if payload.get("error") or payload.get("error_code"):
                    provider_error = self._provider_error(
                        payload,
                        stage="token request",
                        status_code=status_code,
                    )
                    if provider_error.retryable and attempt < self._max_retries():
                        time.sleep(self._retry_backoff(attempt))
                        continue
                    raise BaiduOCRError(
                        str(provider_error),
                        request_count=token_request_count,
                    ) from provider_error
                break
            else:
                raise BaiduOCRError(
                    "Baidu token request retries exhausted",
                    request_count=token_request_count,
                )

            token = str(payload.get("access_token", "")).strip()
            if not token:
                raise BaiduOCRError(
                    "Baidu token response lacks access_token",
                    request_count=1,
                )
            try:
                expires_in = float(payload.get("expires_in", 0))
            except (TypeError, ValueError):
                expires_in = 0.0
            ttl = max(60.0, expires_in - TOKEN_CACHE_SKEW_SECONDS)
            _TOKEN_CACHE[cache_key] = (token, now + ttl)
            return token, False

    def recognize(
        self,
        image_bytes: bytes,
        *,
        access_token: str,
        language_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload, _ = self.recognize_with_metadata(
            image_bytes,
            access_token=access_token,
            language_type=language_type,
        )
        return payload

    def recognize_with_metadata(
        self,
        image_bytes: bytes,
        *,
        access_token: str,
        language_type: Optional[str] = None,
    ) -> tuple[Dict[str, Any], int]:
        if not image_bytes:
            raise BaiduOCRError("Baidu OCR input image is empty")
        request_count = 0
        for attempt in range(self._max_retries() + 1):
            try:
                response = self._session().post(
                    self._ocr_url(),
                    params={"access_token": access_token},
                    data={
                        "image": base64.b64encode(image_bytes).decode("ascii"),
                        "language_type": (
                            language_type
                            or os.getenv("BAIDU_OCR_LANGUAGE_TYPE", "CHN_ENG")
                        ),
                        "detect_direction": os.getenv(
                            "BAIDU_OCR_DETECT_DIRECTION",
                            "false",
                        ),
                        "detect_language": os.getenv(
                            "BAIDU_OCR_DETECT_LANGUAGE",
                            "false",
                        ),
                        "paragraph": os.getenv("BAIDU_OCR_PARAGRAPH", "false"),
                        "vertexes_location": "true",
                        "probability": "true",
                    },
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    timeout=self._ocr_timeout(),
                )
                request_count += 1
            except (requests.Timeout, requests.ConnectionError) as exc:
                request_count += 1
                if attempt < self._max_retries():
                    time.sleep(self._retry_backoff(attempt))
                    continue
                raise BaiduOCRError(
                    f"Baidu OCR request failed: {type(exc).__name__}",
                    request_count=request_count,
                ) from exc

            try:
                payload = self._payload(response)
                response_ok = bool(getattr(response, "ok", False))
                status_code = int(getattr(response, "status_code", 0) or 0)
            except BaiduOCRError as exc:
                if exc.retryable and attempt < self._max_retries():
                    time.sleep(self._retry_backoff(attempt))
                    continue
                raise BaiduOCRError(str(exc), request_count=request_count) from exc
            finally:
                close_response(response)
            if response_ok and not (
                payload.get("error") or payload.get("error_code")
            ):
                if not isinstance(payload.get("words_result", []), list):
                    raise BaiduOCRError(
                        "Baidu OCR response has invalid words_result",
                        request_count=request_count,
                    )
                return payload, request_count
            provider_error = self._provider_error(
                payload,
                stage="OCR request",
                status_code=status_code,
            )
            if provider_error.retryable and attempt < self._max_retries():
                time.sleep(self._retry_backoff(attempt))
                continue
            raise BaiduOCRError(
                str(provider_error),
                request_count=request_count,
            ) from provider_error
