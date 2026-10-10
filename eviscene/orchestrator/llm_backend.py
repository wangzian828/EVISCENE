# -*- coding: utf-8 -*-
"""LLM Backend abstraction layer.

Provides a unified async interface for API-based LLM inference.
"""
from __future__ import annotations

from eviscene.environment import get_env

import json
import atexit
import asyncio
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional

import httpx

from eviscene.integrations.gemini import (
    GeminiInteractionsClient,
    extract_text,
    messages_to_input,
    validate_interaction_response,
)
from eviscene.integrations.http_transport import provider_httpx_limits
from eviscene.integrations.llm.openai_compatible import (
    OpenAICompatibleChatClient,
    resolve_model_api_key,
    resolve_model_base_url,
    resolve_model_wire_api,
)
from eviscene.integrations.llm.prefix_cache import validate_case_cache_salt


@dataclass
class LLMResponse:
    """Unified response from any LLM backend."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Optional[Dict[str, Any]] = None


POLICY_TOKEN_CAPTURE_SCHEMA_VERSION = "ifv-policy-token-capture-v1"


def _int_token_ids(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    token_ids: list[int] = []
    for item in value:
        if isinstance(item, bool):
            return []
        try:
            token_ids.append(int(item))
        except (TypeError, ValueError):
            return []
    return token_ids


def _completion_logprob_values(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        for key in ("content", "token_logprobs"):
            if key in value:
                return _completion_logprob_values(value[key])
        return []
    if not isinstance(value, list):
        return []
    result: list[float] = []
    for item in value:
        raw_value = item.get("logprob") if isinstance(item, Mapping) else item
        if isinstance(raw_value, bool):
            return []
        try:
            result.append(float(raw_value))
        except (TypeError, ValueError):
            return []
    return result


def _top_logprob_token_id(value: Mapping[str, Any]) -> int | None:
    for key in ("token_id", "tokenId"):
        raw = value.get(key)
        if isinstance(raw, bool):
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    token = value.get("token")
    if not isinstance(token, str):
        return None
    match = re.fullmatch(r"<?token_id:(\d+)>?", token.strip())
    return int(match.group(1)) if match else None


def _completion_topk_values(value: Any, *, topk: int) -> list[list[dict[str, Any]]]:
    """Normalize numeric per-token top-logprob entries from local Qwen servers."""

    if not isinstance(value, Mapping):
        return []
    content = value.get("content")
    if not isinstance(content, list):
        return []
    result: list[list[dict[str, Any]]] = []
    for position in content:
        if not isinstance(position, Mapping):
            return []
        raw_entries = position.get("top_logprobs")
        if not isinstance(raw_entries, list):
            return []
        entries: list[dict[str, Any]] = []
        for raw_entry in raw_entries[: max(1, int(topk))]:
            if not isinstance(raw_entry, Mapping):
                continue
            token_id = _top_logprob_token_id(raw_entry)
            try:
                logprob = float(raw_entry.get("logprob"))
            except (TypeError, ValueError):
                continue
            if token_id is None or not math.isfinite(logprob):
                continue
            entries.append({"token_id": token_id, "logprob": logprob})
        if len(entries) != max(1, int(topk)):
            return []
        max_logprob = max(item["logprob"] for item in entries)
        weights = [math.exp(item["logprob"] - max_logprob) for item in entries]
        total = sum(weights)
        if not math.isfinite(total) or total <= 0:
            return []
        result.append(
            [
                {"token_id": item["token_id"], "probability": weight / total}
                for item, weight in zip(entries, weights)
            ]
        )
    return result


def extract_policy_token_capture(
    payload: Mapping[str, Any],
    *,
    topk: int = 20,
) -> Dict[str, Any]:
    """Extract a fail-closed numeric token record from a Qwen server response."""

    choices = payload.get("choices")
    choice = (
        choices[0]
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping)
        else {}
    )
    message = choice.get("message") if isinstance(choice, Mapping) else {}
    message = message if isinstance(message, Mapping) else {}
    prompt_token_ids = _int_token_ids(payload.get("prompt_token_ids"))
    completion_token_ids = _int_token_ids(
        message.get("gen_tokens")
        or choice.get("token_ids")
        or message.get("token_ids")
    )
    completion_logprobs = _completion_logprob_values(
        choice.get("logprobs") if isinstance(choice, Mapping) else None
    )
    completion_topk = _completion_topk_values(
        choice.get("logprobs") if isinstance(choice, Mapping) else None,
        topk=topk,
    )
    missing: list[str] = []
    if not prompt_token_ids:
        missing.append("prompt_token_ids")
    if not completion_token_ids:
        missing.append("completion_token_ids")
    if not completion_logprobs:
        missing.append("completion_logprobs")
    elif len(completion_logprobs) != len(completion_token_ids):
        missing.append("completion_logprob_length_mismatch")
    return {
        "schema_version": POLICY_TOKEN_CAPTURE_SCHEMA_VERSION,
        "status": "complete" if not missing else "incomplete",
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": completion_token_ids,
        "completion_logprobs": completion_logprobs,
        "completion_topk_by_position": completion_topk,
        "topk": topk if completion_topk else 0,
        "missing": missing,
    }


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def get_response(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        """Generate a response given a conversation history."""
        ...


@dataclass
class _AsyncClientGeneration:
    client: httpx.AsyncClient
    active_requests: int = 0
    retired: bool = False
    close_started: bool = False


class _AsyncClientPool:
    """Swap failed async clients without interrupting healthy requests.

    A single half-closed keep-alive connection can poison an entire shared
    client.  When one request fails at the transport boundary, retire only
    that client's generation.  Concurrent requests already using it are
    allowed to finish; retries acquire the fresh generation.
    """

    def __init__(
        self,
        client_factory: Callable[[], httpx.AsyncClient],
    ) -> None:
        self._client_factory = client_factory
        self._lock = threading.Lock()
        self._current: Optional[_AsyncClientGeneration] = None
        self._generations: list[_AsyncClientGeneration] = []

    def acquire(self) -> _AsyncClientGeneration:
        with self._lock:
            generation = self._current
            if generation is None or generation.retired:
                generation = _AsyncClientGeneration(self._client_factory())
                self._current = generation
                self._generations.append(generation)
            generation.active_requests += 1
            return generation

    async def retire(self, generation: _AsyncClientGeneration) -> None:
        client_to_close: Optional[httpx.AsyncClient] = None
        with self._lock:
            if not generation.retired:
                generation.retired = True
                if self._current is generation:
                    self._current = None
            if (
                generation.retired
                and generation.active_requests == 0
                and not generation.close_started
            ):
                generation.close_started = True
                client_to_close = generation.client
        if client_to_close is not None:
            await client_to_close.aclose()

    async def release(self, generation: _AsyncClientGeneration) -> None:
        client_to_close: Optional[httpx.AsyncClient] = None
        with self._lock:
            generation.active_requests = max(0, generation.active_requests - 1)
            if (
                generation.retired
                and generation.active_requests == 0
                and not generation.close_started
            ):
                generation.close_started = True
                client_to_close = generation.client
        if client_to_close is not None:
            await client_to_close.aclose()

    async def aclose(self) -> None:
        with self._lock:
            generations = list(self._generations)
            self._current = None
            for generation in generations:
                generation.retired = True
                generation.close_started = True
            self._generations.clear()
        await asyncio.gather(
            *(generation.client.aclose() for generation in generations),
            return_exceptions=True,
        )


class APIBackend(LLMBackend):
    """OpenAI-compatible API backend using async httpx.

    Supports qwen (DashScope), qwen_local (OpenAI-compatible local serving),
    necodex, and standard OpenAI endpoints.
    """

    def __init__(
        self,
        provider: str = "gemini",
        model_name: str = "gemini-3.7-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        wire_api: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        timeout: float = 120.0,
        max_retries: int = 1,
        proxy: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        cache_salt: Optional[str] = None,
    ):
        self.provider = provider.lower().strip()
        if self.provider == "gemini" and api_key is not None:
            raise ValueError(
                "Gemini credentials must come from GEMINI_API_KEY or GOOGLE_API_KEY."
            )
        self.model_name = model_name
        self.wire_api = resolve_model_wire_api(self.provider, wire_api)
        self.api_key = api_key or self._resolve_api_key()
        self.base_url = base_url or self._resolve_base_url()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.proxy = proxy or self._resolve_proxy()
        self.extra_body = extra_body or self._resolve_extra_body()
        self.cache_salt = (
            validate_case_cache_salt(cache_salt)
            if cache_salt is not None
            else None
        )
        self._shared_client: Optional[httpx.AsyncClient] = None
        self._gemini_client_pool: Optional[_AsyncClientPool] = None
        self._client_registered_for_cleanup = False

    def _resolve_api_key(self) -> str:
        if self.provider == "necodex":
            return get_env("NECODEX_API_KEY", "")
        elif self.provider == "gpustack":
            return get_env("GPUSTACK_API_KEY", "")
        if self.provider == "lmdeploy":
            return get_env("LMDEPLOY_API_KEY", "none")
        if self.provider == "qwen_local":
            return get_env("QWEN_LOCAL_API_KEY", "none")
        return resolve_model_api_key(self.provider, None) or ""

    def _resolve_base_url(self) -> str:
        if self.provider == "gemini" and self.wire_api == "interactions":
            return get_env("GEMINI_INTERACTIONS_URL", "https://generativelanguage.googleapis.com/v1beta/interactions")
        if self.provider == "necodex":
            return resolve_model_base_url(self.provider, None, self.wire_api)
        elif self.provider == "gpustack":
            value = get_env("GPUSTACK_BASE_URL", "").strip()
            if not value:
                raise ValueError(
                    "gpustack requires an explicit GPUSTACK_BASE_URL"
                )
            return value
        elif self.provider == "lmdeploy":
            return get_env("LMDEPLOY_BASE_URL", "http://127.0.0.1:8899/v1")
        elif self.provider == "qwen_local":
            return get_env("QWEN_LOCAL_BASE_URL", "http://127.0.0.1:8899/v1")
        return resolve_model_base_url(self.provider, None, self.wire_api) or "https://api.openai.com/v1"

    def _resolve_proxy(self) -> Optional[str]:
        if self.provider == "gpustack":
            return get_env("GPUSTACK_PROXY", "").strip() or None
        if self.provider in {"lmdeploy", "qwen_local"}:
            return None  # Local service, no proxy needed
        # Use HTTP proxy for external API calls
        return get_env("HTTPS_PROXY", get_env("HTTP_PROXY", None))

    def _resolve_extra_body(self) -> Dict[str, Any]:
        if self.provider == "gpustack":
            # Keep this externally hosted Qwen-family endpoint deterministic.
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {}

    def _client_kwargs(self) -> Dict[str, Any]:
        client_kwargs: Dict[str, Any] = {
            "timeout": self.timeout,
            "limits": provider_httpx_limits(),
        }
        if self.proxy:
            client_kwargs["proxy"] = self.proxy
        else:
            client_kwargs["trust_env"] = False
        return client_kwargs

    def _register_client_cleanup(self) -> None:
        if self._client_registered_for_cleanup:
            return
        self._client_registered_for_cleanup = True
        atexit.register(self._close_shared_client_sync)

    def _close_shared_client_sync(self) -> None:
        client = self._shared_client
        self._shared_client = None
        gemini_pool = self._gemini_client_pool
        self._gemini_client_pool = None
        if client is None and gemini_pool is None:
            return
        try:
            async def close_all() -> None:
                if gemini_pool is not None:
                    await gemini_pool.aclose()
                if client is not None:
                    await client.aclose()

            asyncio.run(close_all())
        except Exception:
            pass

    def _get_shared_client(self) -> httpx.AsyncClient:
        if self._shared_client is None:
            self._shared_client = httpx.AsyncClient(**self._client_kwargs())
            self._register_client_cleanup()
        return self._shared_client

    async def aclose(self) -> None:
        """Close the shared HTTP client used by this backend."""
        client = self._shared_client
        self._shared_client = None
        gemini_pool = self._gemini_client_pool
        self._gemini_client_pool = None
        if gemini_pool is not None:
            await gemini_pool.aclose()
        if client is not None:
            await client.aclose()

    def _interactions_base_url(self) -> str:
        if self.base_url.rstrip("/").endswith("/interactions"):
            return self.base_url.rstrip("/")
        return get_env(
            "GEMINI_INTERACTIONS_URL",
            "https://generativelanguage.googleapis.com/v1beta/interactions",
        )

    def _get_gemini_client_pool(self) -> _AsyncClientPool:
        if self._gemini_client_pool is None:
            self._gemini_client_pool = _AsyncClientPool(
                lambda: httpx.AsyncClient(**self._client_kwargs())
            )
        return self._gemini_client_pool

    async def _post_gemini_request(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        json: Any,
    ) -> httpx.Response:
        pool = self._get_gemini_client_pool()
        generation = pool.acquire()
        try:
            return await generation.client.post(
                url,
                headers=headers,
                json=json,
            )
        except httpx.TransportError:
            await pool.retire(generation)
            raise
        finally:
            await pool.release(generation)

    async def get_response(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        """Call the API asynchronously with retry."""
        if self.wire_api == "interactions":
            return await self._get_interactions_response(messages, **kwargs)
        if self.wire_api == "responses":
            return await self._get_responses_response(messages, **kwargs)
        if self.wire_api == "chat_completions":
            return await self._get_chat_completions_response(messages, **kwargs)
        raise RuntimeError(f"Unsupported wire API at runtime: {self.wire_api}")

    async def _get_chat_completions_response(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build request body
        body: Dict[str, Any] = {
            "model": kwargs.get("model", self.model_name),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if self.cache_salt is not None:
            if self.provider not in {"qwen_local", "lmdeploy"}:
                raise ValueError(
                    "cache_salt is only supported for local Qwen-compatible serving"
                )
            body["cache_salt"] = self.cache_salt
        tools = kwargs.get("tools")
        if tools:
            body["tools"] = self._openai_tool_schemas(tools)
            body["tool_choice"] = kwargs.get("tool_choice", "auto")
            body["parallel_tool_calls"] = False
        strict_qwen_chat = (
            self.provider == "qwen_local"
            and get_env("QWEN_LOCAL_STRICT_CHAT_COMPLETIONS", "")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )
        response_format = kwargs.get("response_format")
        if response_format and not strict_qwen_chat:
            body["response_format"] = response_format
        if kwargs.get("capture_policy_tokens"):
            if self.provider not in {"qwen_local", "lmdeploy"}:
                raise ValueError(
                    "policy token capture is only supported for local "
                    "Qwen-compatible chat completions"
                )
            policy_topk = max(
                1,
                min(
                    100,
                    int(
                        kwargs.get(
                            "policy_topk",
                            get_env("EVISCENE_POLICY_TOPK", "20"),
                        )
                    ),
                ),
            )
            body.update(
                {
                    "return_token_ids": True,
                    "return_tokens_as_token_ids": True,
                    "logprobs": True,
                    "top_logprobs": policy_topk,
                }
            )
        generation_config = kwargs.get("generation_config")
        # Allow reasoning-only candidates only for schema-bound requests; StageRunner
        # validates them. Free-form reasoning must never become an executable action.
        reasoning_fallback_requested = bool(response_format) or (
            self.provider == "qwen_local"
            and not tools
            and isinstance(generation_config, dict)
            and bool(generation_config.get("enable_thinking"))
        )

        # For models with internal reasoning (gpt-5.5, o1), give enough space
        # for both reasoning and content output.
        if self.provider == "necodex":
            body["max_tokens"] = 32768

        # Merge extra_body (e.g., chat_template_kwargs for GPUStack)
        if self.extra_body:
            body.update(self.extra_body)

        # LMDeploy exposes Qwen's thinking switch through the chat template,
        # not as an OpenAI top-level generation field. Keep this local so other
        # OpenAI-compatible providers never receive an unsupported parameter.
        if (
            self.provider in {"qwen_local", "lmdeploy"}
            and isinstance(generation_config, dict)
            and "enable_thinking" in generation_config
        ):
            template_kwargs = body.get("chat_template_kwargs", {})
            template_kwargs = (
                dict(template_kwargs)
                if isinstance(template_kwargs, dict)
                else {}
            )
            template_kwargs["enable_thinking"] = bool(
                generation_config["enable_thinking"]
            )
            body["chat_template_kwargs"] = template_kwargs

        # vLLM exposes Qwen's reasoning wall and sampling controls as top-level
        # request fields, separately from chat-template switches.  Forward only
        # the explicit allowlist so internal stage metadata never leaks onto the
        # provider wire.
        if self.provider in {"qwen_local", "lmdeploy"} and isinstance(
            generation_config, dict
        ):
            qwen_generation_fields = (
                "thinking_token_budget",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "presence_penalty",
                "repetition_penalty",
                "seed",
            )
            # The official Transformers OpenAI-compatible server rejects
            # vLLM-specific Qwen controls as unknown top-level fields. Keep
            # the existing default for vLLM/LMDeploy, while allowing a
            # deliberately explicit strict OpenAI mode for such servers.
            if strict_qwen_chat:
                qwen_generation_fields = (
                    "temperature",
                    "top_p",
                    "presence_penalty",
                    "seed",
                )
            for key in qwen_generation_fields:
                if key in generation_config:
                    body[key] = generation_config[key]

        last_error: Optional[Exception] = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                client = self._get_shared_client()
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()

                # Handle empty response body
                if not response.content or not response.content.strip():
                    if attempt + 1 < attempts:
                        import asyncio
                        await asyncio.sleep(3 * (attempt + 1))
                        continue
                    raise RuntimeError("Chat Completions returned an empty response body.")

                data = response.json()

                choice = data["choices"][0]
                text = self._extract_chat_completion_text(
                    choice,
                    allow_reasoning_fallback=reasoning_fallback_requested,
                )
                if not text.strip():
                    # Retry empty HTTP-success responses within the transport budget.
                    # Schema-bound reasoning candidates are handled by StageRunner above.
                    if attempt + 1 < attempts:
                        last_error = RuntimeError(
                            "Chat Completions returned an empty model response."
                        )
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    raise RuntimeError(
                        self._empty_chat_response_detail(
                            data,
                            reasoning_fallback_requested=reasoning_fallback_requested,
                        )
                    )

                usage = data.get("usage", {})

                return LLMResponse(
                    text=text,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    raw=data,
                )
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout,
                    httpx.LocalProtocolError, httpx.ReadError) as e:
                last_error = e
                if attempt + 1 < attempts:
                    import asyncio
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                raise
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt + 1 < attempts:
                    last_error = e
                    import asyncio
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(self._http_status_error_detail(e)) from e
            except Exception as e:
                # Catch SSL errors and other transient network issues
                if "SSL" in str(type(e).__name__) or "ssl" in str(e).lower():
                    last_error = e
                    if attempt + 1 < attempts:
                        import asyncio
                        await asyncio.sleep(3 * (attempt + 1))
                        continue
                raise

        raise last_error or RuntimeError("All retries exhausted")

    @staticmethod
    def _empty_chat_response_detail(
        payload: Dict[str, Any],
        *,
        reasoning_fallback_requested: bool,
    ) -> str:
        """Describe a successful-but-unusable chat response without its body."""

        choices = payload.get("choices")
        choice_count = len(choices) if isinstance(choices, list) else 0
        choice = choices[0] if choice_count and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str):
            reasoning = message.get("reasoning")
        content_chars = len(content) if isinstance(content, str) else 0
        reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
        finish_reason = str(choice.get("finish_reason", "") or "")
        return (
            "Chat Completions returned an unusable response: "
            f"choices={choice_count}, finish_reason={finish_reason or 'unknown'}, "
            f"content_chars={content_chars}, reasoning_chars={reasoning_chars}, "
            f"reasoning_fallback_requested={reasoning_fallback_requested}"
        )

    @staticmethod
    def _http_status_error_detail(error: httpx.HTTPStatusError) -> str:
        """Preserve a bounded provider error body for durable diagnostics."""

        response = error.response
        try:
            detail = response.text.strip()
        except Exception:
            detail = ""
        if len(detail) > 4000:
            detail = detail[:4000] + "... [truncated]"
        suffix = f": {detail}" if detail else ""
        return (
            f"HTTP {response.status_code} {response.reason_phrase} for "
            f"{response.request.url}{suffix}"
        )

    @staticmethod
    def _openai_tool_schemas(
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in tools:
            if isinstance(item.get("function"), dict):
                normalized.append(item)
                continue
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "description": item.get("description", ""),
                        "parameters": item.get("parameters", {}),
                    },
                }
            )
        return normalized

    @staticmethod
    def _extract_chat_completion_text(
        choice: Dict[str, Any],
        *,
        allow_reasoning_fallback: bool = False,
    ) -> str:
        """Extract assistant text from OpenAI-compatible chat completion payloads.

        Different providers are slightly inconsistent here:
        - `message.content` may be a string
        - `message.content` may be a list of content parts
        - `message.content` may be omitted while provider-specific fields exist
        """
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            if len(tool_calls) != 1:
                return "<parallel_tool_calls>" + json.dumps(
                    tool_calls,
                    ensure_ascii=False,
                ) + "</parallel_tool_calls>"
            function = tool_calls[0].get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            serialized = json.dumps(
                {
                    "name": function.get("name", ""),
                    "arguments": arguments,
                },
                ensure_ascii=False,
            )
            return f"<tool_call>{serialized}</tool_call>"

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            if parts:
                return "\n".join(parts)

        # Accept either reasoning field for schema-bound requests only;
        # the stage parser must validate it before producing a canonical action.
        if allow_reasoning_fallback:
            for field in ("reasoning_content", "reasoning"):
                reasoning = message.get(field)
                if isinstance(reasoning, str) and reasoning.strip():
                    return reasoning

        return ""

    async def _get_responses_response(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        url = f"{self.base_url.rstrip('/')}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": kwargs.get("model", self.model_name),
            "input": OpenAICompatibleChatClient._to_responses_input(messages),
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            "text": {"format": {"type": "text"}},
        }
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            body["temperature"] = temperature

        last_error: Optional[Exception] = None
        for attempt in range(5):
            try:
                client = self._get_shared_client()
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
                text = OpenAICompatibleChatClient._extract_responses_text(data)
                usage = data.get("usage", {})
                return LLMResponse(
                    text=text,
                    prompt_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                    completion_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)),
                    raw=data,
                )
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout,
                    httpx.LocalProtocolError, httpx.ReadError) as e:
                last_error = e
                if attempt < 4:
                    import asyncio
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                raise
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < 4:
                    last_error = e
                    import asyncio
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                raise
        raise last_error or RuntimeError("All retries exhausted")

    async def _get_interactions_response(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        data = await self.create_interaction(
            input_payload=messages_to_input(messages),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            store=True,
        )
        _, status = validate_interaction_response(data)
        if status != "completed":
            raise RuntimeError(
                f"Gemini text completion requires status=completed, received status={status}."
            )
        text = extract_text(data)
        if not text.strip():
            raise RuntimeError("Gemini Interactions returned an empty model response.")
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        return LLMResponse(
            text=text,
            prompt_tokens=usage.get("total_input_tokens", usage.get("input_tokens", 0)),
            completion_tokens=usage.get("total_output_tokens", usage.get("output_tokens", 0)),
            raw=data,
        )

    async def create_interaction(
        self,
        *,
        input_payload: Any,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        previous_interaction_id: Optional[str] = None,
        response_format: Optional[Any] = None,
        store: bool = True,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        background: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if self.provider != "gemini":
            raise RuntimeError("create_interaction is only supported for Gemini.")
        if self.wire_api != "interactions":
            raise RuntimeError("Gemini requires wire_api='interactions'.")

        effective_generation_config: Dict[str, Any] = dict(generation_config or {})
        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if effective_max_tokens:
            effective_generation_config["max_output_tokens"] = effective_max_tokens
        effective_temperature = self.temperature if temperature is None else temperature
        if effective_temperature is not None:
            effective_generation_config["temperature"] = effective_temperature
        client = GeminiInteractionsClient(
            base_url=self._interactions_base_url(),
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_delay=float(
                get_env("GEMINI_RETRY_BASE_DELAY_SECONDS", "3")
            ),
            retry_jitter=float(
                get_env("GEMINI_RETRY_JITTER_SECONDS", "2")
            ),
            retry_max_delay=float(
                get_env("GEMINI_RETRY_MAX_DELAY_SECONDS", "60")
            ),
            request=self._post_gemini_request,
        )
        payload = await client.create(
            model=self.model_name,
            input=input_payload,
            system_instruction=system_instruction,
            tools=tools,
            previous_interaction_id=previous_interaction_id,
            response_format=response_format,
            generation_config=effective_generation_config or None,
            background=background,
            store=store,
        )
        retry_metadata = dict(getattr(client, "last_retry_metadata", {}) or {})
        if int(retry_metadata.get("retry_attempts", 0) or 0) > 0:
            payload = dict(payload)
            payload["__retry_metadata__"] = retry_metadata
        return payload
