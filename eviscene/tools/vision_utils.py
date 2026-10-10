from __future__ import annotations

from eviscene.environment import get_env

import base64
import hashlib
import io
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from eviscene.integrations.http_sessions import new_provider_session

REMOTE_IMAGE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
}

DEFAULT_IMAGE_MAX_LONG_EDGE = 1024
DEFAULT_IMAGE_JPEG_QUALITY = 95


def _guess_mime_from_name(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")


def _remote_image_to_data_url(image_url: str) -> str:
    proxy = get_env("HTTPS_PROXY") or get_env("HTTP_PROXY") or get_env("https_proxy") or get_env("http_proxy")
    proxies = {"http": proxy, "https": proxy} if proxy else None
    session = new_provider_session()
    try:
        with session.get(
            image_url,
            headers=REMOTE_IMAGE_HEADERS,
            timeout=20,
            proxies=proxies,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
            mime = content_type or _guess_mime_from_name(urlparse(image_url).path)
            encoded = base64.b64encode(response.content).decode("utf-8")
            return f"data:{mime};base64,{encoded}"
    finally:
        session.close()


def image_to_data_url(image_input: str) -> str:
    if image_input.startswith("data:"):
        return image_input
    if image_input.startswith("http://") or image_input.startswith("https://"):
        return _remote_image_to_data_url(image_input)

    path = Path(image_input)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_input}")

    stat = path.stat()
    return _local_image_to_data_url(
        str(path.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
    )


@lru_cache(maxsize=64)
def _local_image_to_data_url(
    path_string: str,
    mtime_ns: int,
    size: int,
) -> str:
    """Serialize an unchanged local image only once per process.

    The file metadata is part of the cache key, so replacing an image at the
    same path cannot silently reuse an old payload. This cache is deliberately
    process-local: the canonical image hash and runtime cache remain the
    cross-process sources of truth.
    """

    path = Path(path_string)
    mime = _guess_mime_from_name(path.name)
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def controlled_image_to_data_url(
    image_input: str,
    *,
    max_long_edge: int | None = None,
    jpeg_quality: int | None = None,
) -> tuple[str, dict]:
    """Serialize a local image at a bounded resolution with auditable metadata."""

    edge = _resolve_image_edge(max_long_edge)
    quality = _resolve_image_quality(jpeg_quality)
    if image_input.startswith(("data:", "http://", "https://")):
        data_url = image_to_data_url(image_input)
        header, encoded = data_url.split(",", 1)
        content = base64.b64decode(encoded)
        return _controlled_image_bytes_to_data_url(
            content,
            source_kind="remote_or_inline",
            source_sha256=hashlib.sha256(content).hexdigest(),
            source_media_type=header[5:].split(";", 1)[0],
            edge=edge,
            quality=quality,
        )

    path = Path(image_input)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_input}")
    stat = path.stat()
    data_url, metadata = _controlled_local_image_to_data_url(
        str(path.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        edge,
        quality,
    )
    return data_url, dict(metadata)


def controlled_image_bytes_to_data_url(
    original_bytes: bytes,
    *,
    source_kind: str = "inline_bytes",
    source_media_type: str = "application/octet-stream",
    max_long_edge: int | None = None,
    jpeg_quality: int | None = None,
) -> tuple[str, dict]:
    """Serialize already-loaded image bytes using the model-image contract."""

    if not isinstance(original_bytes, (bytes, bytearray)) or not original_bytes:
        raise ValueError("image bytes must be non-empty")
    edge = _resolve_image_edge(max_long_edge)
    quality = _resolve_image_quality(jpeg_quality)
    source = bytes(original_bytes)
    return _controlled_image_bytes_to_data_url(
        source,
        source_kind=source_kind,
        source_sha256=hashlib.sha256(source).hexdigest(),
        source_media_type=source_media_type,
        edge=edge,
        quality=quality,
    )


def vision_tool_image_to_data_url(image_input: str) -> str:
    """Serialize an image for a VLM; the 1024-pixel bound cannot be bypassed."""

    data_url, _ = controlled_image_to_data_url(image_input)
    return data_url


def _resolve_image_edge(value: int | None) -> int:
    configured = int(
        value
        or get_env(
            "EVISCENE_IMAGE_MAX_LONG_EDGE",
            str(DEFAULT_IMAGE_MAX_LONG_EDGE),
        )
    )
    return max(256, min(DEFAULT_IMAGE_MAX_LONG_EDGE, configured))


def _resolve_image_quality(value: int | None) -> int:
    return max(
        55,
        min(
            95,
            int(
                value
                or get_env(
                    "EVISCENE_IMAGE_JPEG_QUALITY",
                    str(DEFAULT_IMAGE_JPEG_QUALITY),
                )
            ),
        ),
    )


def _controlled_image_bytes_to_data_url(
    original_bytes: bytes,
    *,
    source_kind: str,
    source_sha256: str,
    source_media_type: str,
    edge: int,
    quality: int,
) -> tuple[str, dict]:
    """Normalize local, remote, and inline image bytes to one model wire form."""

    from PIL import Image

    try:
        with Image.open(io.BytesIO(original_bytes)) as opened:
            sent_bytes, image_metadata = bounded_pil_image_to_jpeg_bytes(
                opened,
                max_long_edge=edge,
                jpeg_quality=quality,
            )
    except Exception as exc:
        raise ValueError(
            "image bytes could not be decoded as a raster image: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return (
        "data:image/jpeg;base64," + base64.b64encode(sent_bytes).decode("ascii"),
        {
            "source_kind": source_kind,
            "source_sha256": source_sha256,
            "sha256": hashlib.sha256(sent_bytes).hexdigest(),
            **image_metadata,
            "encoded_bytes": len(sent_bytes),
            "media_type": "image/jpeg",
            "max_long_edge": edge,
            "jpeg_quality": quality,
        },
    )


def bounded_pil_image_to_jpeg_bytes(
    image: object,
    *,
    max_long_edge: int | None = None,
    jpeg_quality: int | None = None,
) -> tuple[bytes, dict]:
    """Encode one PIL image using the shared model-image bounds."""

    from PIL import Image, ImageOps

    edge = _resolve_image_edge(max_long_edge)
    quality = _resolve_image_quality(jpeg_quality)
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    original_size = list(normalized.size)
    normalized.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    normalized.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue(), {
        "original_size": original_size,
        "sent_size": list(normalized.size),
        "max_long_edge": edge,
        "jpeg_quality": quality,
    }


@lru_cache(maxsize=32)
def _controlled_local_image_to_data_url(
    path_string: str,
    mtime_ns: int,
    size: int,
    edge: int,
    quality: int,
) -> tuple[str, dict]:
    path = Path(path_string)
    original_bytes = path.read_bytes()
    return _controlled_image_bytes_to_data_url(
        original_bytes,
        source_kind="local_file",
        source_sha256=hashlib.sha256(original_bytes).hexdigest(),
        source_media_type=_guess_mime_from_name(path.name),
        edge=edge,
        quality=quality,
    )
