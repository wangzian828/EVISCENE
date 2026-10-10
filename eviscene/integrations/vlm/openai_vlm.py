from __future__ import annotations

import atexit
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from eviscene.integrations.async_runtime import PersistentAsyncRuntime
from eviscene.integrations.gemini import (
    GeminiInteractionsClient,
    RUNTIME_METRICS_KEY,
    attach_runtime_metrics,
    interaction_runtime_metrics,
    missing_required_paths,
    normalize_json_schema,
    require_low_thinking,
)
from eviscene.integrations.llm.openai_compatible import (
    OpenAICompatibleChatClient,
    resolve_model_api_key,
    resolve_model_base_url,
    resolve_model_wire_api,
)
from eviscene.integrations.llm.prefix_cache import validate_case_cache_salt
from eviscene.integrations.vlm.qwen_vl import parse_json_object
from eviscene.tools.vision_utils import vision_tool_image_to_data_url


DEFAULT_JSON_OBJECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
}
DEFAULT_GEMINI_VISION_MIN_OUTPUT_TOKENS = 8192


@dataclass
class OpenAIVisionClient:
    api_key: Optional[str] = None
    provider: str = "openai"
    base_url: Optional[str] = None
    wire_api: Optional[str] = None
    model_name: str = "gpt-4o-mini"
    timeout: float = 60.0
    max_retries: int = 2
    cache_salt: Optional[str] = None

    def __post_init__(self) -> None:
        self.provider = self.provider.lower().strip()
        if self.cache_salt is not None:
            if self.provider not in {"qwen_local", "lmdeploy"}:
                raise ValueError("vision cache_salt is only supported for local serving")
            self.cache_salt = validate_case_cache_salt(self.cache_salt)
        if self.provider == "gemini" and self.api_key is not None:
            raise ValueError(
                "Gemini credentials must come from GEMINI_API_KEY or GOOGLE_API_KEY."
            )
        self.api_key = resolve_model_api_key(self.provider, self.api_key)
        self.wire_api = resolve_model_wire_api(self.provider, self.wire_api)
        self.base_url = resolve_model_base_url(self.provider, self.base_url, self.wire_api)
        self._async_runtime: Optional[PersistentAsyncRuntime] = (
            PersistentAsyncRuntime(thread_name="eviscene-gemini-vision")
            if self.provider == "gemini"
            else None
        )
        self._gemini_interactions_client: Optional[GeminiInteractionsClient] = None
        atexit.register(self.close)

    def close(self) -> None:
        """Close the reusable vision transport if it has been started."""

        if self._async_runtime is None:
            return
        if self._async_runtime.closed:
            return
        client = self._gemini_interactions_client
        self._async_runtime.close(
            (lambda: client.aclose()) if client is not None else None
        )

    def create_image_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_input: str,
        max_tokens: int,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        response_schema: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.create_images_json(
            system_prompt=system_prompt,
            user_text=user_text,
            image_inputs=[image_input],
            max_tokens=max_tokens,
            model_name=model_name,
            temperature=temperature,
            response_schema=response_schema,
        )

    def create_images_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_inputs: Sequence[str],
        max_tokens: int,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        response_schema: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            if self.provider == "lmdeploy":
                self.api_key = "none"
            else:
                if self.provider == "gemini":
                    env_name = "GEMINI_API_KEY"
                elif self.provider == "necodex":
                    env_name = "NECODEX_API_KEY"
                elif self.provider == "qwen":
                    env_name = "QWEN_API_KEY"
                else:
                    env_name = "OPENAI_API_KEY"
                raise RuntimeError(f"{env_name} is not set. Add it to the environment before using vision tools.")
        images = [str(item).strip() for item in image_inputs if str(item).strip()]
        if not images:
            raise ValueError("create_images_json requires at least one image input.")

        if self.provider == "gemini":
            try:
                minimum_tokens = max(
                    1,
                    int(
                        os.getenv(
                            "GEMINI_VISION_MIN_OUTPUT_TOKENS",
                            str(DEFAULT_GEMINI_VISION_MIN_OUTPUT_TOKENS),
                        )
                    ),
                )
            except ValueError:
                minimum_tokens = DEFAULT_GEMINI_VISION_MIN_OUTPUT_TOKENS
            return self._create_gemini_interactions_images_json(
                system_prompt=system_prompt,
                user_text=user_text,
                image_inputs=images,
                max_tokens=max(max_tokens, minimum_tokens),
                model_name=model_name or self.model_name,
                temperature=temperature,
                response_schema=response_schema,
            )

        schema = None
        effective_max_tokens = max_tokens
        if self.provider == "qwen_local":
            effective_max_tokens = max(
                max_tokens,
                max(
                    1,
                    int(
                        os.getenv(
                            "QWEN_LOCAL_VISION_MIN_OUTPUT_TOKENS",
                            "8192",
                        )
                    ),
                ),
            )
            if response_schema is not None:
                schema = normalize_json_schema(
                    response_schema,
                    require_all_properties=True,
                )

        chat = OpenAICompatibleChatClient(
            api_key=self.api_key,
            base_url=self.base_url,
            wire_api=self.wire_api or "chat_completions",
            timeout=self.timeout,
            max_retries=self.max_retries,
            cache_salt=self.cache_salt,
        )
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": vision_tool_image_to_data_url(
                                        image_input
                                    )
                                },
                            }
                            for image_input in images
                        ],
                    ],
                },
            ]
            content = chat.create_json_completion(
                model_name=model_name or self.model_name,
                max_tokens=effective_max_tokens,
                temperature=temperature,
                messages=messages,
                response_schema=schema,
                chat_template_kwargs=(
                    {"enable_thinking": False}
                    if self.provider == "qwen_local"
                    else None
                ),
            )
        finally:
            close = getattr(chat, "close", None)
            if callable(close):
                close()
        parsed = parse_json_object(content)
        if not parsed:
            raise RuntimeError("Vision model did not return a valid JSON object.")
        return parsed

    def _create_gemini_interactions_images_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_inputs: Sequence[str],
        max_tokens: int,
        model_name: str,
        temperature: float,
        response_schema: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if self.wire_api != "interactions":
            raise RuntimeError(
                "Gemini vision requires wire_api='interactions'; protocol fallback is disabled."
            )

        schema = normalize_json_schema(
            DEFAULT_JSON_OBJECT_SCHEMA if response_schema is None else response_schema,
            require_all_properties=response_schema is not None,
            strip_validation_constraints=self.provider == "gemini",
        )

        async def _request() -> Dict[str, Any]:
            client = self._gemini_interactions_client
            if client is None:
                client = GeminiInteractionsClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )
                self._gemini_interactions_client = client
            payload = await client.create(
                model=model_name,
                input=[
                    {"type": "text", "text": user_text},
                    *[
                        self._to_interactions_image(image_input)
                        for image_input in image_inputs
                    ],
                ],
                system_instruction=(
                    f"{system_prompt}\n\n"
                    "Return exactly one JSON object without markdown or commentary."
                ),
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                },
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                    "thinking_level": require_low_thinking(
                        os.getenv("GEMINI_VISION_THINKING_LEVEL", "low"),
                        env_name="GEMINI_VISION_THINKING_LEVEL",
                    ),
                },
                background=False,
                store=True,
            )
            runtime_metrics = interaction_runtime_metrics(payload)
            try:
                content = client.extract_text(payload)
                if not content.strip():
                    raise RuntimeError("Gemini Interactions vision response was empty.")
                parsed = parse_json_object(content)
                if not parsed:
                    raise RuntimeError(
                        "Gemini Interactions vision response was not a valid JSON object: "
                        + content[:500]
                    )
                missing = missing_required_paths(parsed, schema)
                if missing:
                    raise RuntimeError(
                        "Gemini Interactions vision response is missing required fields: "
                        + ", ".join(missing[:20])
                    )
            except Exception as exc:
                raise attach_runtime_metrics(exc, runtime_metrics)
            parsed[RUNTIME_METRICS_KEY] = runtime_metrics
            return parsed

        return self._run_async(_request())

    @staticmethod
    def _to_interactions_image(image_input: str) -> Dict[str, Any]:
        # Every VLM input, including a remote candidate image, goes through the
        # shared bounded serializer.  Passing remote URLs through as provider
        # URIs would bypass the 1024-pixel model-image contract.
        image_url = vision_tool_image_to_data_url(image_input)
        return OpenAIVisionClient._data_url_to_interactions_image(image_url)

    @staticmethod
    def _data_url_to_interactions_image(image_url: str) -> Dict[str, Any]:
        if not (image_url.startswith("data:") and ";base64," in image_url):
            return {"type": "image", "uri": image_url}
        header, data = image_url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] or "image/jpeg"
        return {"type": "image", "mime_type": mime_type, "data": data}

    def _run_async(self, coroutine: Any) -> Any:
        if self._async_runtime is None:
            raise RuntimeError("Reusable async vision runtime is unavailable.")
        return self._async_runtime.run(coroutine)
