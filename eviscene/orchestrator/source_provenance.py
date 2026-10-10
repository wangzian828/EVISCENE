"""Deterministic source normalization and provenance classification."""
from __future__ import annotations

import hashlib
import ipaddress
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "source",
}

OFFICIAL_DOMAINS = {
    "apple.com",
    "esa.int",
    "europa.eu",
    "nasa.gov",
    "noaa.gov",
    "openai.com",
    "un.org",
}

NEWS_DOMAINS = {
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
}

UGC_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "medium.com",
    "reddit.com",
    "tiktok.com",
    "weibo.com",
    "x.com",
    "youtube.com",
}

OFFICIAL_PUBLIC_SUFFIXES = {
    "ac.uk",
    "edu.au",
    "edu.cn",
    "gov.au",
    "gov.cn",
    "gov.uk",
}

MULTIPART_SUFFIXES = {
    "ac.uk",
    "co.jp",
    "co.uk",
    "com.au",
    "com.br",
    "com.cn",
    "com.hk",
    "com.sg",
    "edu.au",
    "edu.cn",
    "gov.au",
    "gov.cn",
    "gov.uk",
    "net.au",
    "org.au",
    "org.cn",
    "org.uk",
}


@dataclass(frozen=True)
class SourceIdentity:
    canonical_url: str
    hostname: str
    registered_domain: str
    source_family: str
    source_class: str
    content_sha256: str
    risk_flags: tuple[str, ...]


def canonicalize_url(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "://" not in text:
        text = "https://" + text
    try:
        parsed = urlsplit(text)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except ValueError:
        return ""
    if not hostname:
        return ""
    scheme = parsed.scheme.lower() if parsed.scheme.lower() in {"http", "https"} else "https"
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = hostname if port is None or default_port else f"{hostname}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(
        sorted(
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in TRACKING_QUERY_KEYS
        ),
        doseq=True,
    )
    return urlunsplit((scheme, netloc, path, query, ""))


def registered_domain(hostname: str) -> str:
    host = str(hostname or "").lower().rstrip(".")
    if not host:
        return ""
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    labels = [label for label in host.split(".") if label]
    if len(labels) <= 2:
        return host
    suffix2 = ".".join(labels[-2:])
    return ".".join(labels[-3:]) if suffix2 in MULTIPART_SUFFIXES else suffix2


def domain_matches(hostname: str, domain: str) -> bool:
    host = str(hostname or "").lower().rstrip(".")
    target = str(domain or "").lower().rstrip(".")
    return bool(host and target and (host == target or host.endswith("." + target)))


def content_sha256(content: str | bytes) -> str:
    payload = content if isinstance(content, bytes) else str(content or "").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def classify_source(url: str, *, content: str = "", injection_flags: Optional[Iterable[str]] = None) -> SourceIdentity:
    canonical = canonicalize_url(url)
    hostname = (urlsplit(canonical).hostname or "") if canonical else ""
    domain = registered_domain(hostname)
    flags = {str(flag) for flag in (injection_flags or []) if str(flag)}

    if any(domain_matches(hostname, item) for item in UGC_DOMAINS):
        source_class = "ugc"
        flags.add("user_generated_content")
    elif (
        any(domain_matches(hostname, item) for item in OFFICIAL_DOMAINS)
        or any(
            domain_matches(hostname, item)
            for item in OFFICIAL_PUBLIC_SUFFIXES
        )
        or hostname.endswith((".gov", ".edu", ".int"))
    ):
        source_class = "official"
    elif any(domain_matches(hostname, item) for item in NEWS_DOMAINS):
        source_class = "news"
    else:
        source_class = "unknown"

    official_token = next((item for item in OFFICIAL_DOMAINS if item in hostname), "")
    if official_token and not domain_matches(hostname, official_token):
        flags.add("lookalike_official_domain")

    digest = content_sha256(content) if content else ""
    family = f"content:{digest}" if digest else f"domain:{domain or hostname or 'unknown'}"
    return SourceIdentity(
        canonical_url=canonical,
        hostname=hostname,
        registered_domain=domain,
        source_family=family,
        source_class=source_class,
        content_sha256=digest,
        risk_flags=tuple(sorted(flags)),
    )
