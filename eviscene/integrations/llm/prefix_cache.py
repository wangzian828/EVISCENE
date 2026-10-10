"""Per-rollout prefix-cache isolation for local Qwen serving.

The active Qwen3.5/vLLM stack keeps automatic prefix caching disabled because
hybrid attention/Mamba cache hits have produced incorrect generations.  This
module only defines the request-side isolation contract needed by a future
validated backend; it does not enable server-side caching.
"""
from __future__ import annotations

from eviscene.environment import get_env

import re
import secrets
from typing import Optional


PREFIX_CACHE_MODE_ENV = "EVISCENE_PREFIX_CACHE_MODE"
REQUEST_ISOLATED = "request_isolated"
CASE_ISOLATED = "case_isolated"
# Wire protocol identifier: retained for existing serving gateways.
CASE_CACHE_SALT_PREFIX = "ifv-case-v1-"
_CASE_CACHE_SALT = re.compile(
    rf"{re.escape(CASE_CACHE_SALT_PREFIX)}[A-Za-z0-9_-]{{43}}"
)


def prefix_cache_mode() -> str:
    """Return the explicit cache-isolation mode.

    ``request_isolated`` preserves the existing safety behavior: callers do
    not request reuse and the gateway gives every request a unique salt.
    ``case_isolated`` is an opt-in canary mode in which one unguessable salt is
    reused only within one rollout.
    """

    raw = get_env(PREFIX_CACHE_MODE_ENV, REQUEST_ISOLATED).strip().lower()
    aliases = {
        "": REQUEST_ISOLATED,
        "off": REQUEST_ISOLATED,
        "disabled": REQUEST_ISOLATED,
        "request": REQUEST_ISOLATED,
        REQUEST_ISOLATED: REQUEST_ISOLATED,
        "case": CASE_ISOLATED,
        CASE_ISOLATED: CASE_ISOLATED,
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(
            f"{PREFIX_CACHE_MODE_ENV} must be request_isolated or case_isolated"
        ) from exc


def new_case_cache_salt() -> Optional[str]:
    """Mint one opaque 256-bit salt for a single rollout, when opted in."""

    if prefix_cache_mode() != CASE_ISOLATED:
        return None
    value = CASE_CACHE_SALT_PREFIX + secrets.token_urlsafe(32)
    return validate_case_cache_salt(value)


def new_rollout_cache_salts() -> tuple[Optional[str], Optional[str]]:
    """Mint independent Agent and page-extractor cache domains.

    The two prompt families have no useful common prefix.  Keeping them in
    separate private domains also prevents one family's hybrid-state cache
    entry from affecting the other family.
    """

    agent = new_case_cache_salt()
    if agent is None:
        return None, None
    extractor = new_case_cache_salt()
    while extractor == agent:  # Defensive even though a collision is negligible.
        extractor = new_case_cache_salt()
    return agent, extractor


def validate_case_cache_salt(value: object) -> str:
    """Reject predictable, malformed, or cross-protocol cache domains."""

    if not isinstance(value, str) or _CASE_CACHE_SALT.fullmatch(value) is None:
        raise ValueError(
            "case-isolated prefix caching requires an opaque 256-bit "
            f"{CASE_CACHE_SALT_PREFIX} cache_salt"
        )
    return value
