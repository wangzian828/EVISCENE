"""Redact credentials and signed URL parameters before persistence."""
from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


REDACTED = "[REDACTED]"

_SENSITIVE_FIELD_NAMES = {
    "apikey",
    "authorization",
    "accesstoken",
    "refreshtoken",
    "clientsecret",
    "accesskeyid",
    "accesskeysecret",
    "secretaccesskey",
    "ossaccesskeyid",
    "signature",
    "xapikey",
    "xgoogapikey",
}

_SENSITIVE_QUERY_NAMES = {
    "key",
    "apikey",
    "token",
    "accesstoken",
    "credential",
    "googleaccessid",
    "ossaccesskeyid",
    "signature",
    "sig",
    "securitytoken",
}

_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def sanitize_for_persistence(value: Any) -> Any:
    """Return a JSON-compatible copy with credential material removed."""
    if isinstance(value, dict):
        return {
            key: REDACTED
            if _field_is_sensitive(str(key))
            else sanitize_for_persistence(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_for_persistence(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_persistence(item) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value)
    return value


def redact_url(url: str) -> str:
    """Redact authentication query parameters while retaining URL identity."""
    try:
        parsed = urlsplit(url)
    except ValueError:
        return url
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return url

    changed = False
    query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if _query_name_is_sensitive(key):
            query.append((key, REDACTED))
            changed = True
        else:
            query.append((key, value))
    if not changed:
        return url
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )


def _sanitize_string(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            pass
        else:
            if isinstance(parsed, (dict, list)):
                return json.dumps(
                    sanitize_for_persistence(parsed),
                    ensure_ascii=False,
                    indent=2,
                )

    def replace(match: re.Match[str]) -> str:
        candidate = match.group(0)
        suffix = ""
        while candidate and candidate[-1] in ".,);]":
            suffix = candidate[-1] + suffix
            candidate = candidate[:-1]
        return redact_url(candidate) + suffix

    return _URL_PATTERN.sub(replace, value)


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _field_is_sensitive(name: str) -> bool:
    normalized = _normalized_name(name)
    return normalized in _SENSITIVE_FIELD_NAMES or normalized.endswith("secret")


def _query_name_is_sensitive(name: str) -> bool:
    normalized = _normalized_name(name)
    return (
        normalized in _SENSITIVE_QUERY_NAMES
        or normalized.startswith("xamz")
        or normalized.startswith("xoss")
        or normalized.startswith("xgoog")
    )
