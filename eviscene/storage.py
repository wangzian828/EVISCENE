"""Shared storage paths for local development and server runs."""

from __future__ import annotations

from eviscene.environment import get_env

import os
import tempfile
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]


def data_root() -> Optional[Path]:
    raw = get_env("EVISCENE_DATA_ROOT", "").strip()
    return Path(raw).expanduser() if raw else None


def data_path(relative: PathLike, local_default: PathLike) -> Path:
    root = data_root()
    return root / Path(relative) if root is not None else Path(local_default)


def default_trace_dir() -> str:
    return str(data_path("runs/traces", "outputs/traces"))


def default_eval_root() -> Path:
    return data_path("runs/eval", "outputs/eval_runs")


def default_tool_cache_dir() -> str:
    return str(data_path("cache/tools", ".cache/tool_results"))


def runtime_lock_root() -> Path:
    """Return a cross-process lock root without assuming a server mount."""

    root = data_root()
    if root is not None:
        return root.expanduser().resolve() / "runs" / "_locks"
    user_scope = (
        str(os.getuid())
        if hasattr(os, "getuid")
        else get_env("USERNAME", get_env("USER", "user"))
    )
    return (
        Path(tempfile.gettempdir())
        # Keep the existing lock namespace shared with older workers.
        / f"image-factual-verifier-{user_scope}"
        / "locks"
    )
