from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from eviscene.integrations.gemini import missing_required_paths, normalize_json_schema
from eviscene.integrations.llm.openai_compatible import OpenAICompatibleChatClient, resolve_qwen_api_key
from eviscene.tools.vision_utils import vision_tool_image_to_data_url


@dataclass
class QwenVLClient:
    api_key: Optional[str] = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: str = "qwen3.6-plus"
    timeout: float = 60.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        self.api_key = resolve_qwen_api_key(self.api_key)

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
            raise RuntimeError("QWEN_API_KEY is not set. Add it to the environment before using Qwen VL tools.")

        images = [str(item).strip() for item in image_inputs if str(item).strip()]
        if not images:
            raise ValueError("create_images_json requires at least one image input.")
        chat = OpenAICompatibleChatClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        schema = (
            normalize_json_schema(
                response_schema,
                require_all_properties=True,
            )
            if response_schema is not None
            else None
        )
        schema_instruction = (
            "\n\nReturn exactly one JSON object matching this schema:\n"
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            if schema is not None
            else "\n\nReturn exactly one JSON object."
        )
        content = chat.create_json_completion(
            model_name=model_name or self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt + schema_instruction,
                },
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
            ],
        )
        parsed = parse_json_object(content)
        if schema is not None:
            error = _validate_schema_value(parsed, schema, path="$")
            if error:
                raise RuntimeError(
                    "Qwen structured vision response failed schema validation: "
                    + error
                )
            missing = missing_required_paths(parsed, schema)
            if missing:
                raise RuntimeError(
                    "Qwen structured vision response is missing required fields: "
                    + ", ".join(missing[:20])
                )
        return parsed


def parse_json_object(content: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        left = content.find("{")
        right = content.rfind("}")
        if left == -1 or right == -1 or left > right:
            raise ValueError("Model response does not contain a JSON object.")
        try:
            parsed = json.loads(content[left : right + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Model response contains malformed JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object.")
    return parsed


def _validate_schema_value(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str,
) -> str:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return f"{path} must be an object"
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                return f"{path} has unexpected fields: {', '.join(extras)}"
        for name, child in value.items():
            child_schema = properties.get(name)
            if isinstance(child_schema, Mapping):
                error = _validate_schema_value(
                    child,
                    child_schema,
                    path=f"{path}.{name}",
                )
                if error:
                    return error
    elif expected == "array":
        if not isinstance(value, list):
            return f"{path} must be an array"
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            return f"{path} must contain at least {minimum} items"
        if isinstance(maximum, int) and len(value) > maximum:
            return f"{path} must contain at most {maximum} items"
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                error = _validate_schema_value(
                    item,
                    item_schema,
                    path=f"{path}[{index}]",
                )
                if error:
                    return error
    elif expected == "string" and not isinstance(value, str):
        return f"{path} must be a string"
    elif expected == "integer" and (
        isinstance(value, bool) or not isinstance(value, int)
    ):
        return f"{path} must be an integer"
    elif expected == "number" and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        return f"{path} must be a number"
    elif expected == "boolean" and not isinstance(value, bool):
        return f"{path} must be a boolean"

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return f"{path} must be one of: {', '.join(map(str, enum))}"
    return ""
