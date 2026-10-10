"""JSON Schema helpers for direct Gemini Interactions REST requests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


_UNSUPPORTED_ANNOTATION_KEYS = {
    "$defs",
    "definitions",
    "default",
    "examples",
    "title",
}

_UNSUPPORTED_VALIDATION_KEYS = {
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "pattern",
    "format",
}


def normalize_json_schema(
    schema: Mapping[str, Any],
    *,
    require_all_properties: bool = True,
    strip_validation_constraints: bool = False,
) -> dict[str, Any]:
    """Inline local refs and emit the REST structured-output schema subset."""

    root = deepcopy(dict(schema))
    definitions = dict(root.get("$defs") or root.get("definitions") or {})

    def resolve(node: Any, stack: tuple[str, ...] = ()) -> Any:
        if isinstance(node, list):
            return [resolve(item, stack) for item in node]
        if not isinstance(node, dict):
            return node

        reference = node.get("$ref")
        if isinstance(reference, str):
            prefix = "#/$defs/" if reference.startswith("#/$defs/") else "#/definitions/"
            if reference.startswith(prefix):
                name = reference[len(prefix) :]
                if name in stack:
                    raise ValueError(f"Recursive JSON schema reference is unsupported: {reference}")
                target = definitions.get(name)
                if not isinstance(target, dict):
                    raise ValueError(f"Unknown JSON schema reference: {reference}")
                merged = deepcopy(target)
                merged.update({key: value for key, value in node.items() if key != "$ref"})
                return resolve(merged, (*stack, name))

        excluded_keys = set(_UNSUPPORTED_ANNOTATION_KEYS)
        if strip_validation_constraints:
            excluded_keys.update(_UNSUPPORTED_VALIDATION_KEYS)
        normalized = {
            key: resolve(value, stack)
            for key, value in node.items()
            if key not in excluded_keys and key != "$ref"
        }
        properties = normalized.get("properties")
        if isinstance(properties, dict):
            normalized["properties"] = {
                str(name): resolve(value, stack)
                for name, value in properties.items()
            }
            if require_all_properties:
                normalized["required"] = list(normalized["properties"].keys())
            normalized.setdefault("additionalProperties", False)
        return normalized

    result = resolve(root)
    if not isinstance(result, dict):
        raise TypeError("JSON schema root must be an object.")
    return result


def missing_required_paths(value: Any, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """Return required paths absent from a structured model response."""

    missing: list[str] = []
    schema_type = schema.get("type")
    if schema_type == "object" or isinstance(schema.get("properties"), Mapping):
        if not isinstance(value, Mapping):
            return [path]
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            child_path = f"{path}.{name}"
            if name not in value:
                missing.append(child_path)
                continue
            child_schema = properties.get(name)
            if isinstance(child_schema, Mapping):
                missing.extend(missing_required_paths(value[name], child_schema, child_path))
        return missing

    if schema_type == "array" and isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                missing.extend(missing_required_paths(item, item_schema, f"{path}[{index}]"))
    return missing
