"""Project configuration with compatibility for existing IFV environment files."""
from __future__ import annotations

import os
from collections.abc import Mapping


def get_env(
    name: str,
    default: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Read a setting; an explicit EVISCENE value wins, including an empty one."""
    values = os.environ if environ is None else environ
    if name in values:
        return values[name]
    if name.startswith("EVISCENE_"):
        return values.get("IFV_" + name[len("EVISCENE_"):], default)
    return values.get(name, default)


def normalize_environment(values: Mapping[str, str]) -> dict[str, str]:
    """Canonicalize legacy project keys without mutating the caller's mapping."""
    result = {key: value for key, value in values.items() if not key.startswith("IFV_")}
    for key, value in values.items():
        if key.startswith("IFV_"):
            result.setdefault("EVISCENE_" + key[len("IFV_"):], value)
    return result
