"""Stage schemas for the unified ReAct runtime."""

from __future__ import annotations

from eviscene.environment import get_env
from copy import deepcopy
from typing import Any, Dict, List, Optional
from eviscene.integrations.gemini import normalize_json_schema
from .stage_types import StageStep

_MAX_GEMINI_DYNAMIC_ENUM_VALUES = 10


class StageSchemaMixin:
    """Internal execution component; StageRunner owns shared case state."""

    def _openai_response_format(self) -> Optional[Dict[str, Any]]:
        if self.output_schema is None:
            return None
        if str(getattr(self.llm, "provider", "")).lower() == "qwen_local" and get_env(
            "QWEN_LOCAL_STRICT_CHAT_COMPLETIONS", ""
        ).strip().lower() in {"1", "true", "yes", "on"}:
            # Strict local servers use a schema prompt rather than response_format.
            # Pydantic still validates the returned object.
            return None
        schema = self._normalized_output_schema()
        if str(getattr(self.llm, "provider", "")).lower() in {
            "qwen_local",
            "lmdeploy",
        }:
            schema = self._lmdeploy_response_schema(schema)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.output_schema.__name__,
                "schema": schema,
                "strict": True,
            },
        }

    @classmethod
    def _lmdeploy_response_schema(cls, schema: Any) -> Any:
        """Project strict application schemas onto LMDeploy's grammar subset.

        LMDeploy 0.13 warns that string length, pattern and format keywords are
        unsupported. Passing them in nested schemas can make guided decoding
        emit whitespace until the output budget is exhausted. Structural,
        enum, array and numeric constraints remain on the wire; Pydantic still
        enforces the complete original model after decoding.
        """

        if isinstance(schema, list):
            return [cls._lmdeploy_response_schema(item) for item in schema]
        if not isinstance(schema, dict):
            return schema
        unsupported = {"minLength", "maxLength", "pattern", "format"}
        return {
            key: cls._lmdeploy_response_schema(value)
            for key, value in schema.items()
            if key not in unsupported
        }

    def _build_native_tool_schemas(
        self,
        *,
        steps: Optional[List[StageStep]] = None,
    ) -> List[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = []
        for tool in self._available_tool_objects(steps=steps):
            parameters = deepcopy(tool.parameters or {})
            parameters.setdefault("type", "object")
            properties = parameters.setdefault("properties", {})
            required = list(parameters.get("required", []) or [])
            for property_name, allowed_values in self.tool_argument_constraints.get(
                tool.name, {}
            ).items():
                if property_name == "question_id" and property_name not in properties:
                    properties[property_name] = {
                        "type": "string",
                        "description": "The active runtime question identifier.",
                    }
                if property_name not in properties or not allowed_values:
                    continue
                property_schema = properties[property_name]
                property_type = property_schema.get("type")
                if (
                    property_type == "array"
                    or isinstance(property_type, list)
                    and "array" in property_type
                ):
                    property_schema.setdefault("items", {"type": "string"})
                    enum_target = property_schema["items"]
                else:
                    enum_target = property_schema
                normalized_allowed_values = list(dict.fromkeys(allowed_values))
                always_plain_dynamic_enum = (
                    tool.name == "visit" and property_name == "url"
                )
                if (
                    not always_plain_dynamic_enum
                    and len(normalized_allowed_values)
                    <= _MAX_GEMINI_DYNAMIC_ENUM_VALUES
                ):
                    enum_target["enum"] = normalized_allowed_values
                else:
                    # Dynamic URL enums can be rejected by Interactions.
                    # Expose a string and enforce executable routes in the runtime validator.
                    enum_target.pop("enum", None)
                    description = str(enum_target.get("description", "")).strip()
                    suffix = (
                        " Choose from the runtime-provided candidates; "
                        "the runtime validates membership."
                    )
                    if suffix not in description:
                        enum_target["description"] = (description + suffix).strip()
                # Constrained route selectors must be required, even if the base schema
                # makes them optional, so guided decoding cannot omit the executable route.
                if property_name not in required:
                    required.append(property_name)
            properties.pop("image_input", None)
            required = [name for name in required if name != "image_input"]
            parameters = self._normalize_native_schema(parameters)
            parameters["required"] = required
            parameters["additionalProperties"] = False
            schemas.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                }
            )
        return schemas

    @classmethod
    def _normalize_native_schema(cls, schema: Any) -> Any:
        if isinstance(schema, list):
            return [cls._normalize_native_schema(item) for item in schema]
        if not isinstance(schema, dict):
            return schema
        normalized = {
            key: cls._normalize_native_schema(value) for key, value in schema.items()
        }
        schema_type = normalized.get("type")
        if isinstance(schema_type, list):
            if "array" in schema_type:
                normalized["type"] = "array"
                normalized.setdefault("items", {"type": "string"})
            elif "string" in schema_type:
                normalized["type"] = "string"
            elif schema_type:
                normalized["type"] = schema_type[0]
        if normalized.get("type") == "array":
            normalized.setdefault("items", {"type": "string"})
        return normalized

    def _validate_native_tool_args(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        *,
        steps: Optional[List[StageStep]] = None,
    ) -> str:
        tool = self.tools.get(tool_name)
        if tool is None:
            return self._unknown_tool_message(tool_name)
        validation_args, _ = self._normalize_runtime_tool_args(
            tool_name,
            tool_args,
        )
        schema = next(
            (
                item["parameters"]
                for item in self._build_native_tool_schemas(steps=steps)
                if item["name"] == tool_name
            ),
            None,
        )
        if schema is None:
            return self._tool_unavailable_message(
                tool_name,
                self._available_tool_names(steps=steps),
            )
        properties = schema.get("properties", {}) or {}
        unknown = sorted(set(validation_args) - set(properties))
        if unknown:
            return f"Unknown argument(s) for {tool_name}: {', '.join(unknown)}"
        missing = [
            name
            for name in schema.get("required", [])
            if name not in validation_args
            and not (
                tool_name == "crop_and_inspect"
                and str(validation_args.get("visual_question_id", "")).strip()
                and name in {"bbox", "focus_question"}
            )
        ]
        if missing:
            return f"Missing required argument(s) for {tool_name}: {', '.join(missing)}"
        for name, value in validation_args.items():
            error = self._validate_schema_value(
                value,
                properties.get(name, {}),
                path=f"Argument '{name}' for {tool_name}",
            )
            if error:
                return error
        for name, allowed_values in self.tool_argument_constraints.get(
            tool_name, {}
        ).items():
            if name not in validation_args or not allowed_values:
                continue
            allowed = list(dict.fromkeys(allowed_values))
            value = validation_args[name]
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if item not in allowed:
                        return (
                            f"Argument '{name}' for {tool_name}[{index}] "
                            "must be one of: "
                            + ", ".join(str(candidate) for candidate in allowed)
                        )
            elif value not in allowed:
                return (
                    f"Argument '{name}' for {tool_name} must be one of: "
                    + ", ".join(str(candidate) for candidate in allowed)
                )
        return ""

    @classmethod
    def _validate_schema_value(
        cls,
        value: Any,
        spec: Dict[str, Any],
        *,
        path: str,
    ) -> str:
        expected = spec.get("type")
        if expected == "string":
            if not isinstance(value, str):
                return f"{path} must be a string."
            minimum = spec.get("minLength")
            maximum = spec.get("maxLength")
            normalized_length = len(value.strip())
            if isinstance(minimum, int) and normalized_length < minimum:
                return f"{path} must be a non-empty string."
            if isinstance(maximum, int) and len(value) > maximum:
                return f"{path} must contain at most {maximum} characters."
        if expected == "array":
            if not isinstance(value, list):
                return f"{path} must be an array."
            minimum = spec.get("minItems")
            maximum = spec.get("maxItems")
            if isinstance(minimum, int) and len(value) < minimum:
                return f"{path} must contain at least {minimum} items."
            if isinstance(maximum, int) and len(value) > maximum:
                return f"{path} must contain at most {maximum} items."
            item_spec = spec.get("items")
            if isinstance(item_spec, dict):
                for index, item in enumerate(value):
                    error = cls._validate_schema_value(
                        item,
                        item_spec,
                        path=f"{path}[{index}]",
                    )
                    if error:
                        return error
        if expected == "object":
            if not isinstance(value, dict):
                return f"{path} must be an object."
            properties = spec.get("properties", {}) or {}
            unknown = (
                sorted(set(value) - set(properties))
                if spec.get("additionalProperties") is False
                else []
            )
            if unknown:
                return f"{path} has unknown field(s): {', '.join(unknown)}."
            for name in spec.get("required", []) or []:
                if name not in value:
                    return f"{path} is missing required field '{name}'."
            for name, child in value.items():
                child_spec = properties.get(name)
                if isinstance(child_spec, dict):
                    error = cls._validate_schema_value(
                        child,
                        child_spec,
                        path=f"{path}.{name}",
                    )
                    if error:
                        return error
        if expected == "number" and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            return f"{path} must be a number."
        if expected == "integer" and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            return f"{path} must be an integer."
        if expected == "boolean" and not isinstance(value, bool):
            return f"{path} must be a boolean."
        allowed = spec.get("enum")
        if isinstance(allowed, list) and value not in allowed:
            return f"{path} must be one of: " + ", ".join(str(item) for item in allowed)
        return ""

    def _native_response_format(self) -> Optional[Dict[str, Any]]:
        if self.output_schema is None:
            return None
        return {
            "type": "text",
            "mime_type": "application/json",
            "schema": self._gemini_response_schema(self._normalized_output_schema()),
        }

    @classmethod
    def _gemini_response_schema(cls, schema: Any) -> Any:
        """Remove response-schema keywords rejected by Gemini Interactions.

        ``maxItems`` is accepted in tool schemas and some direct media requests,
        but Gemini rejects it in a structured response schema that continues a
        native function-call interaction. Pydantic still enforces the original
        list bounds after the response is returned.
        """

        if isinstance(schema, list):
            return [cls._gemini_response_schema(item) for item in schema]
        if not isinstance(schema, dict):
            return schema
        return {
            key: cls._gemini_response_schema(value)
            for key, value in schema.items()
            if key != "maxItems"
        }

    def _normalized_output_schema(self) -> Dict[str, Any]:
        if self.output_schema is None:
            return {}
        return normalize_json_schema(
            self.output_schema.model_json_schema(),
            require_all_properties=True,
        )
