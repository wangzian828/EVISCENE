"""Create an image-only runtime package from local image/metadata pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

from eviscene.eval.release_adapter import (
    DATA_PIPELINE_DECISION_POLICY_VERSION,
    INPUT_MODE,
    RELEASE_SCHEMA_VERSION,
    RUNTIME_CASE_KEYS,
    RUNTIME_CONTRACT_VERSION,
)


def prepare_runtime_input(
    metadata: Path, image_root: Path, output: Path, *, release_id: str
) -> int:
    """Whitelist runtime fields; validate all sources before creating output."""
    if not release_id.strip():
        raise ValueError("release_id must not be empty")
    source_root = image_root.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError("output already exists; choose a new directory")
    sources = []
    seen = set()
    with metadata.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {number}: expected an object")
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
                raise ValueError(f"line {number}: empty or duplicate case_id")
            raw_path = row.get("image_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError(f"line {number}: missing image_path")
            relative = Path(raw_path)
            if relative.is_absolute() or relative.drive:
                raise ValueError(f"line {number}: image_path must be relative")
            source = (source_root / relative).resolve(strict=True)
            if not source.is_relative_to(source_root) or not source.is_file():
                raise ValueError(
                    f"line {number}: image_path escapes image_root or is not a file"
                )
            with Image.open(source) as image:
                image.verify()
            seen.add(case_id)
            sources.append((case_id, source))
    if not sources:
        raise ValueError("metadata contains no cases")
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime_input"
    images = runtime / "images"
    images.mkdir(parents=True)
    rows = []
    for index, (case_id, source) in enumerate(sources, 1):
        image_path = Path("images") / f"{index:06d}{source.suffix.lower()}"
        target = runtime / image_path
        shutil.copyfile(source, target)
        with target.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        rows.append(
            {
                "case_id": case_id,
                "image_path": image_path.as_posix(),
                "image_sha256": digest,
            }
        )
    (runtime / "cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    # Publish the manifest last: an interrupted copy is not a valid release.
    manifest = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
        "input_mode": INPUT_MODE,
        "decision_policy_version": DATA_PIPELINE_DECISION_POLICY_VERSION,
        "release_id": release_id,
        "release_stage": "custom_image_only",
        "runtime_contract": {
            "allowed_keys": sorted(RUNTIME_CASE_KEYS),
            "private_keys_absent": True,
        },
        "artifacts": {"agent_input": "runtime_input/cases.jsonl"},
        "source_access_policy": {"active": False},
        "case_count": len(rows),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", type=Path, required=True, help="JSONL with case_id and image_path"
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; never overwritten"
    )
    parser.add_argument(
        "--release-id",
        required=True,
        help="Your dataset version, not the frozen EviLens release ID",
    )
    args = parser.parse_args()
    count = prepare_runtime_input(
        args.metadata, args.image_root, args.output, release_id=args.release_id
    )
    print(f"Prepared {count} image-only cases")


if __name__ == "__main__":
    main()
