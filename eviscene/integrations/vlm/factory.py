from __future__ import annotations

import os
from typing import Any, Optional

from eviscene.integrations.vlm.openai_vlm import OpenAIVisionClient
from eviscene.integrations.vlm.qwen_vl import QwenVLClient


def build_vlm_client(
    *,
    provider: str = "gemini",
    model_name: Optional[str] = None,
    wire_api: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
    max_retries: int = 2,
    cache_salt: Optional[str] = None,
) -> Any:
    provider = provider.lower().strip()
    if cache_salt is not None and provider not in {"qwen_local", "lmdeploy"}:
        raise ValueError("vision cache_salt is only supported for local serving")
    if provider == "gemini":
        if api_key is not None:
            raise ValueError(
                "Gemini credentials must come from GEMINI_API_KEY or GOOGLE_API_KEY."
            )
        timeout = max(
            timeout,
            float(os.getenv("GEMINI_VISION_TIMEOUT_SECONDS", "90")),
        )
        return OpenAIVisionClient(
            api_key=None,
            provider="gemini",
            base_url=base_url,
            wire_api=wire_api
            or os.getenv("GEMINI_VISION_WIRE_API")
            or os.getenv("VISION_LLM_WIRE_API")
            or "interactions",
            model_name=model_name or os.getenv("GEMINI_VISION_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.7-flash")),
            timeout=timeout,
            max_retries=max_retries,
        )
    if provider == "qwen":
        return QwenVLClient(
            api_key=api_key,
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name=model_name or "qwen3.6-plus",
            timeout=timeout,
            max_retries=max_retries,
        )
    if provider == "lmdeploy":
        resolved_model_name = model_name or os.getenv("LMDEPLOY_MODEL", "").strip()
        if not resolved_model_name:
            raise ValueError(
                "lmdeploy requires model_name or LMDEPLOY_MODEL; "
                "no machine-specific checkpoint is assumed"
            )
        return OpenAIVisionClient(
            api_key=api_key or os.getenv("LMDEPLOY_API_KEY", "none"),
            provider="lmdeploy",
            base_url=base_url or os.getenv("LMDEPLOY_BASE_URL", "http://127.0.0.1:8899/v1"),
            wire_api=wire_api,
            model_name=resolved_model_name,
            timeout=timeout,
            max_retries=max_retries,
            cache_salt=cache_salt,
        )
    if provider == "qwen_local":
        return OpenAIVisionClient(
            api_key=api_key or os.getenv("QWEN_LOCAL_API_KEY", "none"),
            provider="qwen_local",
            base_url=base_url
            or os.getenv("QWEN_LOCAL_BASE_URL", "http://127.0.0.1:8899/v1"),
            wire_api=wire_api or "chat_completions",
            model_name=model_name
            or os.getenv("QWEN_LOCAL_MODEL", "eviscene"),
            timeout=timeout,
            max_retries=max_retries,
            cache_salt=cache_salt,
        )
    if provider in {"openai", "necodex"}:
        return OpenAIVisionClient(
            api_key=api_key,
            provider=provider,
            base_url=base_url,
            wire_api=wire_api,
            model_name=model_name or ("gpt-5.5" if provider == "necodex" else "gpt-4o-mini"),
            timeout=timeout,
            max_retries=max_retries,
        )
    raise ValueError(f"Unsupported VLM provider: {provider}")
