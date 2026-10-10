# -*- coding: utf-8 -*-
"""Tool availability metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class ToolHealth:
    available: bool
    error: str = ""


def summarize_health(health: Dict[str, ToolHealth]) -> Dict[str, Dict[str, str | bool]]:
    return {
        name: {"available": item.available, "error": item.error}
        for name, item in health.items()
    }


def require_tools(health: Dict[str, ToolHealth], required: Iterable[str]) -> None:
    failures = []
    for name in required:
        item = health.get(name)
        if item is None:
            failures.append(f"{name}: not registered")
        elif not item.available:
            failures.append(f"{name}: {item.error or 'unavailable'}")
    if failures:
        raise RuntimeError("Required tools are unavailable: " + "; ".join(failures))
