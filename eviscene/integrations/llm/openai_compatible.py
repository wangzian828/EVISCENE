from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from eviscene.integrations.http_transport import provider_httpx_limits
from eviscene.integrations.llm.prefix_cache import validate_case_cache_salt


@dataclass
class OpenAICompatibleChatClient:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    wire_api: str = "chat_completions"
    timeout: float = 60.0
    max_retries: int = 2
    cache_salt: Optional[str] = None

    def __post_init__(self) -> None:
        self.wire_api = validate_wire_api(self.wire_api)
        if self.wire_api == "openai_compat":
            self.wire_api = "chat_completions"
        if self.wire_api == "interactions":
            raise ValueError(
                "OpenAICompatibleChatClient does not implement Gemini Interactions; "
                "use GeminiInteractionsClient."
            )
        if self.cache_salt is not None:
            self.cache_salt = validate_case_cache_salt(self.cache_salt)
        self._thread_local = threading.local()

    def _client_kwargs(self) -> Dict[str, Any]:
        is_local = self.base_url and ("127.0.0.1" in self.base_url or "localhost" in self.base_url)
        if is_local:
            return {
                "timeout": self.timeout,
                "trust_env": False,
                "limits": provider_httpx_limits(),
            }
        proxy = (
            os.environ.get("HTTPS_PROXY")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("http_proxy")
        )
        kwargs: Dict[str, Any] = {
            "timeout": self.timeout,
            "limits": provider_httpx_limits(),
        }
        if proxy:
            kwargs["proxy"] = proxy
            kwargs["trust_env"] = True
        return kwargs

    def close(self) -> None:
        """Close the client owned by the current synchronous worker thread."""

        client = getattr(self._thread_local, "client", None)
        self._thread_local.client = None
        if client is not None:
            client.close()

    def _get_client(self) -> httpx.Client:
        client = getattr(self._thread_local, "client", None)
        if client is None or client.is_closed:
            client = httpx.Client(**self._client_kwargs())
            self._thread_local.client = client
        return client

    def create_json_completion(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        response_schema: Optional[Dict[str, Any]] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        thinking_token_budget: Optional[int] = None,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.wire_api == "responses":
                    if thinking_token_budget is not None:
                        raise ValueError(
                            "thinking_token_budget is only supported by chat_completions"
                        )
                    return self._create_responses_json_completion(
                        model_name=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_schema=response_schema,
                    )
                if self.wire_api == "chat_completions":
                    return self._create_chat_json_completion(
                        model_name=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_schema=response_schema,
                        chat_template_kwargs=chat_template_kwargs,
                        thinking_token_budget=thinking_token_budget,
                    )
                raise RuntimeError(f"Unsupported wire API at runtime: {self.wire_api}")
            except httpx.HTTPStatusError as exc:
                last_error = exc
                raise
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("create_json_completion failed without a captured exception.")

    def _create_chat_json_completion(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        response_schema: Optional[Dict[str, Any]],
        chat_template_kwargs: Optional[Dict[str, Any]],
        thinking_token_budget: Optional[int],
    ) -> str:
        if not self.api_key:
            raise RuntimeError("LLM API key is not set.")
        base_url = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        payload: Dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "strict": True,
                        "schema": response_schema,
                    },
                }
                if response_schema is not None
                else {"type": "json_object"}
            ),
            "messages": messages,
        }
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(chat_template_kwargs)
        if thinking_token_budget is not None:
            payload["thinking_token_budget"] = int(thinking_token_budget)
        if self.cache_salt is not None:
            payload["cache_salt"] = self.cache_salt
        response = self._get_client().post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        content = self._extract_chat_json_text(data["choices"][0])
        if not content:
            raise RuntimeError("Chat Completions returned an empty JSON response.")
        return content

    @staticmethod
    def _extract_chat_json_text(choice: Dict[str, Any]) -> str:
        """Return only a JSON-object candidate from a constrained chat reply.

        Qwen Thinking servers may place a schema-constrained final object in
        ``reasoning_content`` (older runtimes) or ``reasoning`` (current vLLM)
        while leaving canonical ``content`` null.  This fallback is
        intentionally limited to this JSON-only client method and accepts an
        alternate field only when it is itself a complete object.
        """

        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

        for field in ("reasoning_content", "reasoning"):
            reasoning = message.get(field)
            if not isinstance(reasoning, str) or not reasoning.strip():
                continue
            candidate = reasoning.strip()
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return candidate
        return ""

    def _create_responses_json_completion(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        response_schema: Optional[Dict[str, Any]],
    ) -> str:
        if not self.api_key:
            raise RuntimeError("LLM API key is not set.")
        base_url = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        payload = {
            "model": model_name,
            "input": self._to_responses_input(messages),
            "max_output_tokens": max_tokens,
            "text": {"format": {"type": "json_object"}},
        }
        if temperature is not None:
            payload["temperature"] = temperature

        response = self._get_client().post(
            f"{base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        return self._extract_responses_text(response.json())

    @staticmethod
    def _to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, str):
                converted.append(
                    {
                        "role": role,
                        "content": [{"type": "input_text", "text": content}],
                    }
                )
                continue

            parts = []
            for item in content:
                item_type = item.get("type")
                if item_type == "text":
                    parts.append({"type": "input_text", "text": item.get("text", "")})
                elif item_type == "image_url":
                    image_url = item.get("image_url", {})
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url", "")
                    parts.append({"type": "input_image", "image_url": image_url})
            converted.append({"role": role, "content": parts})
        return converted

    @staticmethod
    def _extract_responses_text(payload: Dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        chunks: List[str] = []
        for output_item in payload.get("output", []) or []:
            for content_item in output_item.get("content", []) or []:
                text = content_item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        text = "\n".join(chunks).strip()
        if not text:
            raise RuntimeError("Responses API returned an empty response.")
        return text

def resolve_qwen_api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")


def resolve_gemini_api_key(api_key: Optional[str]) -> Optional[str]:
    if api_key is not None:
        raise ValueError(
            "Gemini credentials must come from GEMINI_API_KEY or GOOGLE_API_KEY."
        )
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def resolve_openai_api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.getenv("OPENAI_API_KEY")


def resolve_necodex_api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.getenv("NECODEX_API_KEY")


def resolve_model_api_key(provider: str, api_key: Optional[str]) -> Optional[str]:
    provider = provider.lower().strip()
    if provider == "gemini":
        return resolve_gemini_api_key(api_key)
    if provider == "qwen":
        return resolve_qwen_api_key(api_key)
    if provider == "openai":
        return resolve_openai_api_key(api_key)
    if provider == "necodex":
        return resolve_necodex_api_key(api_key)
    if provider == "lmdeploy":
        return api_key or os.getenv("LMDEPLOY_API_KEY", "none")
    if provider == "qwen_local":
        return api_key or os.getenv("QWEN_LOCAL_API_KEY", "none")
    return api_key


def resolve_model_base_url(
    provider: str,
    base_url: Optional[str],
    wire_api: Optional[str] = None,
) -> Optional[str]:
    provider = provider.lower().strip()
    if base_url:
        return base_url
    if provider == "gemini":
        resolve_model_wire_api(provider, wire_api)
        return os.getenv("GEMINI_INTERACTIONS_URL") or "https://generativelanguage.googleapis.com/v1beta/interactions"
    if provider == "qwen":
        return os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if provider == "openai":
        return os.getenv("OPENAI_BASE_URL")
    if provider == "necodex":
        configured = os.getenv("NECODEX_BASE_URL", "").strip()
        if not configured:
            raise ValueError(
                "NECODEX_BASE_URL must be configured when using the necodex provider"
            )
        return configured
    if provider == "lmdeploy":
        return os.getenv("LMDEPLOY_BASE_URL", "http://127.0.0.1:8899/v1")
    if provider == "qwen_local":
        return os.getenv("QWEN_LOCAL_BASE_URL", "http://127.0.0.1:8899/v1")
    return base_url


def resolve_model_wire_api(provider: str, wire_api: Optional[str]) -> str:
    provider = provider.lower().strip()
    if wire_api:
        value = validate_wire_api(wire_api)
        if value == "openai_compat":
            value = "chat_completions"
        if provider == "gemini" and value != "interactions":
            raise ValueError("Gemini requires wire_api='interactions'; protocol fallback is disabled.")
        return value
    if provider == "gemini":
        value = validate_wire_api(os.getenv("GEMINI_WIRE_API", "interactions"))
        if value != "interactions":
            raise ValueError("Gemini requires wire_api='interactions'; protocol fallback is disabled.")
        return "interactions"
    if provider == "necodex":
        return validate_wire_api(os.getenv("NECODEX_WIRE_API") or "responses")
    return "chat_completions"


def validate_wire_api(wire_api: str) -> str:
    value = str(wire_api or "").strip().lower()
    allowed = {"interactions", "responses", "chat_completions", "openai_compat"}
    if value not in allowed:
        raise ValueError(
            f"Unsupported wire_api {wire_api!r}; expected one of {', '.join(sorted(allowed))}."
        )
    return value
