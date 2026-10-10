"""Async REST client for the Gemini Interactions API."""

from __future__ import annotations

import asyncio
import os
import random
import threading
from collections.abc import Awaitable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Optional

import httpx

from eviscene.integrations.http_transport import provider_httpx_limits


DEFAULT_INTERACTIONS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/interactions"
)
RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_BAD_REQUEST_CODES = frozenset(
    {
        "invalid_request",
        "malformed_tool_call",
    }
)
_MAX_RETRYABLE_BAD_REQUEST_RETRIES = 1
PENDING_INTERACTION_STATUSES = frozenset({"queued", "in_progress"})
FAILED_INTERACTION_STATUSES = frozenset(
    {"failed", "cancelled", "canceled", "expired", "incomplete"}
)


class GeminiInteractionsError(RuntimeError):
    """Base error raised by the Gemini Interactions client."""


class GeminiInteractionsHTTPError(GeminiInteractionsError):
    """Non-success HTTP response with its body retained for diagnostics."""

    def __init__(
        self,
        status_code: int,
        response_body: str,
        url: str,
        *,
        retry_attempts: int = 0,
        retry_delays: Sequence[float] = (),
    ) -> None:
        self.status_code = status_code
        self.response_body = response_body
        self.url = url
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_delays = tuple(max(0.0, float(value)) for value in retry_delays)
        body = response_body.strip() or "<empty response body>"
        retry_summary = ""
        if self.retry_attempts:
            noun = "retry" if self.retry_attempts == 1 else "retries"
            retry_summary = (
                f" after {self.retry_attempts} {noun}"
                f" ({sum(self.retry_delays):.1f}s scheduled backoff)"
            )
        super().__init__(
            f"Gemini Interactions request to {url} failed with HTTP "
            f"{status_code}{retry_summary}. Response body: {body}"
        )


class GeminiInteractionsResponseError(GeminiInteractionsError):
    """Successful HTTP response that is not a JSON object."""


class GeminiRequestGate:
    """Process-wide, thread-safe limit for in-flight Gemini HTTP requests.

    Agent rollouts may run on the main asyncio loop while VLM helpers use a
    persistent background loop.  An asyncio.Semaphore cannot safely coordinate
    those two loops, so the gate uses a bounded threading semaphore and waits
    for it without blocking either event loop.
    """

    def __init__(self, limit: int) -> None:
        parsed_limit = int(limit)
        if parsed_limit < 1:
            raise ValueError("Gemini request gate limit must be at least 1.")
        self.limit = parsed_limit
        self._semaphore = threading.BoundedSemaphore(parsed_limit)

    async def __aenter__(self) -> "GeminiRequestGate":
        while True:
            acquired = await asyncio.to_thread(
                self._semaphore.acquire,
                True,
                0.25,
            )
            if acquired:
                return self
            await asyncio.sleep(0)

    async def __aexit__(self, *_: object) -> None:
        self._semaphore.release()


_PROCESS_REQUEST_GATE: Optional[GeminiRequestGate] = None
_PROCESS_REQUEST_GATE_LOCK = threading.Lock()


def process_gemini_request_gate() -> GeminiRequestGate:
    """Return the process-wide Gemini request gate.

    The environment is intentionally read once, on first use.  The launcher
    sets ``GEMINI_MAX_INFLIGHT_REQUESTS`` before starting a run, so changing it
    halfway through a run cannot silently replace a live gate and overbook the
    provider.
    """

    global _PROCESS_REQUEST_GATE
    if _PROCESS_REQUEST_GATE is None:
        with _PROCESS_REQUEST_GATE_LOCK:
            if _PROCESS_REQUEST_GATE is None:
                raw_limit = os.getenv("GEMINI_MAX_INFLIGHT_REQUESTS", "4")
                try:
                    limit = int(raw_limit)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "GEMINI_MAX_INFLIGHT_REQUESTS must be an integer."
                    ) from exc
                _PROCESS_REQUEST_GATE = GeminiRequestGate(limit)
    return _PROCESS_REQUEST_GATE


class GeminiInteractionsClient:
    """Reusable async client for non-streaming Gemini Interactions requests.

    The API key is deliberately resolved only from ``GEMINI_API_KEY`` or
    ``GOOGLE_API_KEY``. It cannot be passed as a constructor or method argument.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: float | httpx.Timeout = 900.0,
        max_retries: int = 8,
        retry_delay: float = 3.0,
        retry_jitter: float = 2.0,
        retry_max_delay: float = 60.0,
        client: Optional[httpx.AsyncClient] = None,
        request: Optional[
            Callable[..., Awaitable[httpx.Response]]
        ] = None,
        request_gate: Optional[GeminiRequestGate] = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative.")
        if retry_jitter < 0:
            raise ValueError("retry_jitter must be non-negative.")
        if retry_max_delay <= 0:
            raise ValueError("retry_max_delay must be positive.")

        self._api_key = _resolve_api_key()
        configured_url = base_url or os.getenv("GEMINI_INTERACTIONS_URL") or ""
        self.base_url = configured_url.strip().rstrip("/") or DEFAULT_INTERACTIONS_URL
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_jitter = retry_jitter
        self.retry_max_delay = retry_max_delay
        self._sleep = sleep
        self._random_uniform = random_uniform
        if client is not None and request is not None:
            raise ValueError("client and request cannot both be provided.")
        self._client = client or (
            (
                httpx.AsyncClient(
                    timeout=timeout,
                    limits=provider_httpx_limits(),
                )
                if request is None
                else None
            )
        )
        self._request = request or self._client.post  # type: ignore[union-attr]
        self._owns_client = client is None and request is None
        self._request_gate = request_gate or process_gemini_request_gate()
        self.last_retry_metadata: dict[str, Any] = {
            "retry_attempts": 0,
            "retry_delays": [],
            "retry_backoff_seconds": 0.0,
        }

    async def __aenter__(self) -> "GeminiInteractionsClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the internally-created HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    async def create(
        self,
        *,
        model: str,
        input: Any,
        system_instruction: Optional[str] = None,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        previous_interaction_id: Optional[str] = None,
        response_format: Optional[Mapping[str, Any]] = None,
        generation_config: Optional[Mapping[str, Any]] = None,
        user_metadata: Optional[Mapping[str, Any]] = None,
        background: Optional[bool] = False,
        store: Optional[bool] = True,
        stream: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        """Create one interaction and return the raw JSON response object."""

        if not model.strip():
            raise ValueError("model must not be empty.")
        if stream:
            raise ValueError("Streaming responses are not supported by this client.")
        _validate_input_item_kinds(input)

        payload: dict[str, Any] = {
            "model": model,
            "input": input,
            "stream": False,
        }
        if system_instruction is not None:
            payload["system_instruction"] = system_instruction
        if tools is not None:
            payload["tools"] = [dict(tool) for tool in tools]
        if previous_interaction_id is not None:
            payload["previous_interaction_id"] = previous_interaction_id
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        if generation_config is not None:
            payload["generation_config"] = dict(generation_config)
        if user_metadata is not None:
            payload["user_metadata"] = dict(user_metadata)
        if background is not None:
            payload["background"] = background
        if store is not None:
            payload["store"] = store

        if extra:
            names = ", ".join(sorted(extra))
            raise ValueError(f"Unsupported Gemini Interactions request field(s): {names}")
        response = await self._post(payload)
        validate_interaction_response(response, allow_pending=bool(background))
        return response

    @staticmethod
    def extract_text(payload: Mapping[str, Any]) -> str:
        """Extract text from an interaction response."""

        return extract_text(payload)

    @staticmethod
    def extract_images(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Extract image items from an interaction response."""

        return extract_images(payload)

    @staticmethod
    def extract_videos(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Extract video items from an interaction response."""

        return extract_videos(payload)

    @staticmethod
    def extract_function_calls(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Extract function-call items from an interaction response."""

        return extract_function_calls(payload)

    async def generate_image(
        self,
        *,
        model: str,
        prompt: Any,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
        delivery: Optional[str] = None,
        response_format: Optional[Mapping[str, Any]] = None,
        generation_config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create an image-generation interaction."""

        media_format: dict[str, Any] = {
            "type": "image",
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
        }
        if delivery is not None:
            media_format["delivery"] = delivery
        if response_format is not None:
            media_format.update(response_format)
        media_format["type"] = "image"
        return await self.create(
            model=model,
            input=prompt,
            response_format=media_format,
            generation_config=generation_config,
            **kwargs,
        )

    async def generate_video(
        self,
        *,
        model: str,
        prompt: Any,
        aspect_ratio: str = "16:9",
        task: str = "text_to_video",
        delivery: str = "uri",
        response_format: Optional[Mapping[str, Any]] = None,
        generation_config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a video-generation interaction."""

        media_format: dict[str, Any] = {
            "type": "video",
            "delivery": delivery,
            "aspect_ratio": aspect_ratio,
        }
        if response_format is not None:
            media_format.update(response_format)
        media_format["type"] = "video"

        config = dict(generation_config or {})
        video_config = dict(config.get("video_config") or {})
        video_config["task"] = task
        config["video_config"] = video_config
        return await self.create(
            model=model,
            input=prompt,
            response_format=media_format,
            generation_config=config,
            **kwargs,
        )

    async def _post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        retry_delays: list[float] = []
        self.last_retry_metadata = {
            "retry_attempts": 0,
            "retry_delays": [],
            "retry_backoff_seconds": 0.0,
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with self._request_gate:
                    response = await self._request(
                        self.base_url,
                        headers=headers,
                        json=dict(payload),
                    )
            except httpx.TransportError:
                if attempt >= self.max_retries:
                    raise
                retry_delays.append(await self._wait_before_retry(attempt))
                continue

            try:
                if response.status_code in RETRYABLE_HTTP_STATUSES:
                    if attempt < self.max_retries:
                        retry_delays.append(
                            await self._wait_before_retry(attempt, response=response)
                        )
                        continue
                    raise _http_error(
                        response,
                        retry_attempts=attempt,
                        retry_delays=retry_delays,
                    )

                bad_request_retry_limit = min(
                    self.max_retries,
                    _MAX_RETRYABLE_BAD_REQUEST_RETRIES,
                )
                if (
                    _is_retryable_bad_request(response)
                    and attempt < bad_request_retry_limit
                ):
                    retry_delays.append(
                        await self._wait_before_retry(attempt, response=response)
                    )
                    continue

                if not response.is_success:
                    raise _http_error(
                        response,
                        retry_attempts=attempt if retry_delays else 0,
                        retry_delays=retry_delays,
                    )

                try:
                    data = response.json()
                except ValueError as exc:
                    raise GeminiInteractionsResponseError(
                        "Gemini Interactions returned HTTP "
                        f"{response.status_code} with invalid JSON. Response body: "
                        f"{response.text.strip() or '<empty response body>'}"
                    ) from exc
                if not isinstance(data, dict):
                    raise GeminiInteractionsResponseError(
                        "Gemini Interactions returned a non-object JSON payload: "
                        f"{data!r}"
                    )
                self.last_retry_metadata = {
                    "retry_attempts": attempt,
                    "retry_delays": list(retry_delays),
                    "retry_backoff_seconds": round(sum(retry_delays), 3),
                }
                return data
            finally:
                close = getattr(response, "aclose", None)
                if callable(close):
                    await close()

        raise AssertionError("Retry loop exited unexpectedly.")

    async def _wait_before_retry(
        self,
        attempt: int,
        *,
        response: Optional[httpx.Response] = None,
    ) -> float:
        exponential = min(
            self.retry_max_delay,
            self.retry_delay * (2 ** max(0, int(attempt))),
        )
        delay = min(
            self.retry_max_delay,
            exponential + self._random_uniform(0.0, self.retry_jitter),
        )
        if response is not None:
            retry_hint = _response_retry_delay(response)
            if retry_hint is not None:
                delay = min(self.retry_max_delay, max(delay, retry_hint))
        await self._sleep(delay)
        return delay


def extract_text(payload: Mapping[str, Any]) -> str:
    """Join text content from an interaction response in response order."""

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    return "\n".join(
        text
        for item in _iter_content(payload)
        if isinstance((text := item.get("text")), str) and text.strip()
    )


def extract_images(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return image content items without decoding or downloading them."""

    return _extract_content_type(payload, "image")


def extract_videos(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return video content items without downloading them."""

    return _extract_content_type(payload, "video")


def extract_function_calls(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return function-call content items in response order."""

    return [
        dict(item)
        for item in _iter_content(payload)
        if _content_type(item) in {"function_call", "tool_call"}
    ]


def validate_interaction_response(
    payload: Mapping[str, Any],
    *,
    allow_pending: bool = False,
) -> tuple[str, str]:
    """Validate the Interactions response envelope and action state."""

    interaction_id = str(payload.get("id", "")).strip()
    if not interaction_id:
        raise GeminiInteractionsResponseError(
            "Gemini Interactions response did not include an interaction id."
        )

    status = str(payload.get("status", "")).strip().lower()
    if not status:
        raise GeminiInteractionsResponseError(
            f"Gemini interaction {interaction_id} did not include a status."
        )
    if status in FAILED_INTERACTION_STATUSES:
        detail = payload.get("error") or payload.get("incomplete_details") or ""
        if not detail:
            usage = payload.get("usage") or {}
            output = extract_text(payload)
            detail = {
                "usage": usage,
                "output_preview": output[:500],
            }
        raise GeminiInteractionsResponseError(
            f"Gemini interaction {interaction_id} ended with status={status}: {detail}"
        )

    calls = extract_function_calls(payload)
    if calls and status != "requires_action":
        raise GeminiInteractionsResponseError(
            f"Gemini interaction {interaction_id} returned function_call content "
            f"with status={status}; expected status=requires_action."
        )
    if status == "requires_action" and not calls:
        raise GeminiInteractionsResponseError(
            f"Gemini interaction {interaction_id} requires action but did not return "
            "a function_call."
        )

    allowed = {"completed", "requires_action"}
    if allow_pending:
        allowed.update(PENDING_INTERACTION_STATUSES)
    if status not in allowed:
        raise GeminiInteractionsResponseError(
            f"Gemini interaction {interaction_id} returned unsupported status={status}."
        )
    return interaction_id, status


def messages_to_input(messages: Sequence[Mapping[str, Any]]) -> Any:
    """Convert chat-style text/image messages to Interactions input items."""
    if _messages_are_text_only(messages):
        chunks = []
        for message in messages:
            role = str(message.get("role", "user")).upper()
            content = message.get("content", "")
            text = str(content) if isinstance(content, str) else "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text"
            )
            if text.strip():
                chunks.append(f"{role}:\n{text}".strip())
        return "\n\n".join(chunks)

    items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = message.get("content", "")
        if isinstance(content, str):
            if content.strip():
                items.append({"type": "text", "text": f"{role}:\n{content}"})
            continue
        if not isinstance(content, Sequence):
            continue

        role_written = False
        for item in content:
            if not isinstance(item, Mapping):
                continue
            item_type = str(item.get("type", ""))
            if item_type == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    prefix = f"{role}:\n" if not role_written else ""
                    items.append({"type": "text", "text": f"{prefix}{text}"})
                    role_written = True
            elif item_type == "image_url":
                image_url = item.get("image_url", "")
                if isinstance(image_url, Mapping):
                    image_url = image_url.get("url", "")
                image_item = _image_url_to_item(str(image_url))
                if image_item:
                    items.append(image_item)
                    role_written = True
    return items


def _messages_are_text_only(messages: Sequence[Mapping[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            continue
        if not isinstance(content, Sequence):
            return False
        if any(
            not isinstance(item, Mapping) or item.get("type") != "text"
            for item in content
        ):
            return False
    return True


def _image_url_to_item(image_url: str) -> Optional[dict[str, Any]]:
    if not image_url:
        return None
    if image_url.startswith("data:") and ";base64," in image_url:
        header, data = image_url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] or "image/jpeg"
        return {"type": "image", "mime_type": mime_type, "data": data}
    return {"type": "image", "uri": image_url}


def _resolve_api_key() -> str:
    for variable in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        api_key = (os.getenv(variable) or "").strip()
        if api_key:
            return api_key
    raise GeminiInteractionsError(
        "Gemini API key is required in GEMINI_API_KEY or GOOGLE_API_KEY."
    )


def _http_error(
    response: httpx.Response,
    *,
    retry_attempts: int = 0,
    retry_delays: Sequence[float] = (),
) -> GeminiInteractionsHTTPError:
    return GeminiInteractionsHTTPError(
        response.status_code,
        response.text,
        str(response.request.url),
        retry_attempts=retry_attempts,
        retry_delays=retry_delays,
    )


def _response_retry_delay(response: httpx.Response) -> Optional[float]:
    """Return a provider-advised retry delay from headers or google.rpc.RetryInfo."""

    header_delay = _parse_retry_after(response.headers.get("retry-after", ""))
    detail_delay: Optional[float] = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            details = error.get("details")
            if isinstance(details, Sequence) and not isinstance(
                details,
                (str, bytes, bytearray),
            ):
                for detail in details:
                    if not isinstance(detail, Mapping):
                        continue
                    detail_type = str(detail.get("@type", ""))
                    if not detail_type.endswith("google.rpc.RetryInfo"):
                        continue
                    parsed = _parse_duration_seconds(
                        str(detail.get("retryDelay", ""))
                    )
                    if parsed is not None:
                        detail_delay = max(detail_delay or 0.0, parsed)
    delays = [
        value
        for value in (header_delay, detail_delay)
        if value is not None
    ]
    return max(delays) if delays else None


def _is_retryable_bad_request(response: httpx.Response) -> bool:
    """Return whether Gemini marked an HTTP 400 as safe to replay once.

    Interactions can surface provider-side function-generation failures as
    HTTP 400. These are distinct from a deterministic client contract error:
    replaying the identical request may succeed. Keep the exception narrow and
    bounded so unsupported request fields still fail immediately.
    """

    if response.status_code != 400:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    if not isinstance(payload, Mapping):
        return False
    error = payload.get("error")
    if not isinstance(error, Mapping):
        return False
    code = str(error.get("code", "")).strip().lower()
    if code not in _RETRYABLE_BAD_REQUEST_CODES:
        return False
    if code == "invalid_request":
        message = str(error.get("message", "")).strip().lower()
        return message == "request contains an invalid argument."
    return True


def _parse_retry_after(value: str) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        target = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())


def _parse_duration_seconds(value: str) -> Optional[float]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text.endswith("s"):
        text = text[:-1].strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        return None


def _extract_content_type(
    payload: Mapping[str, Any], content_type: str
) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in _iter_content(payload)
        if _content_type(item) == content_type
    ]


def _iter_content(payload: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    for item in _mapping_items(payload.get("content")):
        yield item

    for container_name in ("steps", "output", "outputs"):
        for container in _mapping_items(payload.get(container_name)):
            yield container
            for item in _mapping_items(container.get("content")):
                yield item


def _mapping_items(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        return
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return
    for item in value:
        if isinstance(item, Mapping):
            yield item


def _content_type(item: Mapping[str, Any]) -> str:
    return str(item.get("type", "")).strip().lower().replace("-", "_")


def _validate_input_item_kinds(value: Any) -> None:
    """Reject arrays that mix Interactions Step items with Content items."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return
    item_types = {
        _content_type(item)
        for item in value
        if isinstance(item, Mapping)
    }
    if not item_types:
        return
    step_types = {
        "function_result",
        "function_call",
        "user_input",
        "message",
        "model_output",
        "thought",
    }
    has_steps = bool(item_types & step_types)
    has_content = bool(item_types - step_types)
    if has_steps and has_content:
        raise ValueError(
            "Gemini Interactions input arrays cannot mix Step and Content items."
        )
