"""Shared HTTP transport policies for provider clients."""

from __future__ import annotations

from eviscene.environment import get_env


import httpx


DEFAULT_HTTPX_MAX_CONNECTIONS = 128
DEFAULT_HTTPX_MAX_KEEPALIVE_CONNECTIONS = 0
DEFAULT_HTTPX_KEEPALIVE_EXPIRY_SECONDS = 5.0


def provider_httpx_limits() -> httpx.Limits:
    """Build bounded limits that do not retain proxy keep-alive sockets.

    The egress proxy used by the rollout can half-close idle connections.  An
    unlimited keep-alive pool then retains those sockets until the owning
    client is closed, which is too late for long-running concurrent cases.
    Keep-alive reuse is therefore disabled by default; the values remain
    explicitly overridable for environments with a healthy proxy.
    """

    def _positive_int(name: str, default: int) -> int:
        raw = get_env(name, str(default)).strip()
        try:
            return max(1, int(raw))
        except ValueError:
            return default

    def _non_negative_int(name: str, default: int) -> int:
        raw = get_env(name, str(default)).strip()
        try:
            return max(0, int(raw))
        except ValueError:
            return default

    def _non_negative_float(name: str, default: float) -> float:
        raw = get_env(name, str(default)).strip()
        try:
            return max(0.0, float(raw))
        except ValueError:
            return default

    max_connections = _positive_int(
        "EVISCENE_HTTPX_MAX_CONNECTIONS",
        DEFAULT_HTTPX_MAX_CONNECTIONS,
    )
    max_keepalive_connections = min(
        max_connections,
        _non_negative_int(
            "EVISCENE_HTTPX_MAX_KEEPALIVE_CONNECTIONS",
            DEFAULT_HTTPX_MAX_KEEPALIVE_CONNECTIONS,
        ),
    )
    return httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        keepalive_expiry=_non_negative_float(
            "EVISCENE_HTTPX_KEEPALIVE_EXPIRY_SECONDS",
            DEFAULT_HTTPX_KEEPALIVE_EXPIRY_SECONDS,
        ),
    )
