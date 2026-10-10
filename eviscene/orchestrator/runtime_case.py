"""Image-only runtime case hashing and integrity checks."""

from __future__ import annotations

import hashlib

from eviscene.orchestrator.state import ImageOnlyRuntimeCase


def image_sha256(image_path: str) -> str:
    digest = hashlib.sha256()
    with open(image_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_case_image(case: ImageOnlyRuntimeCase, image_path: str) -> None:
    actual = image_sha256(image_path)
    if actual != case.image_sha256:
        raise ValueError(
            "ImageOnlyRuntimeCase image_sha256 mismatch: "
            f"expected {case.image_sha256}, got {actual}."
        )
