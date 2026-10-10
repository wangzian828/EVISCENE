"""First-class consumer for reviewed ``ifv-scoring-gold-v1`` packages.

This contract is intentionally separate from the historical v0.3 benchmark
release.  The Agent receives only the public three-field runtime rows; private
gold is exposed to post-rollout scoring through an explicit artifact path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from eviscene.eval.release_adapter import (
    INPUT_MODE,
    RUNTIME_CASE_KEYS,
    RUNTIME_CONTRACT_VERSION,
)


SCORING_PACKAGE_SCHEMA_VERSION = "ifv-existing-eval-package-v1"
SCORING_GOLD_SCHEMA_VERSION = "ifv-scoring-gold-v1"


@dataclass(frozen=True)
class ScoringReleaseArtifacts:
    agent_input: Path
    evaluation_gold: Path
    migration_audit: Path
    checksums: Path
    classification_protocol: None = None
    process_reference_protocol: None = None
    licenses: None = None
    source_access_policy: None = None


@dataclass(frozen=True)
class ScoringRuntimeRelease:
    root: Path
    manifest_path: Path
    manifest: Dict[str, Any]
    release_id: str
    release_stage: str
    schema_version: str
    runtime_contract_version: str
    input_mode: str
    decision_policy_version: str
    source_access_policy_active: bool
    artifacts: ScoringReleaseArtifacts


def _json_object(path: Path, *, name: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return payload


def _required_text(payload: Mapping[str, Any], key: str, *, name: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"{name} requires non-empty {key}")
    return value


def _release_path(root: Path, value: Any, *, name: str) -> Path:
    rendered = str(value or "").strip()
    if not rendered:
        raise ValueError(f"scoring manifest requires artifact path {name}")
    relative = Path(rendered)
    if relative.is_absolute():
        raise ValueError(f"scoring artifact path must be relative: {name}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"scoring artifact escapes release root: {name}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"missing scoring artifact {name}: {resolved}")
    return resolved


def load_scoring_release(benchmark_path: Path) -> ScoringRuntimeRelease:
    benchmark = benchmark_path.expanduser().resolve()
    if benchmark.name != "cases.jsonl" or benchmark.parent.name != "runtime_input":
        raise ValueError(
            "scoring benchmark must be runtime_input/cases.jsonl"
        )
    root = benchmark.parent.parent.resolve()
    manifest_path = root / "manifest.json"
    manifest = _json_object(manifest_path, name="scoring manifest")
    schema_version = _required_text(
        manifest, "schema_version", name="scoring manifest"
    )
    if schema_version != SCORING_PACKAGE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported scoring release contract: "
            f"schema_version={schema_version!r}, expected "
            f"{SCORING_PACKAGE_SCHEMA_VERSION!r}"
        )
    if _required_text(manifest, "input_mode", name="scoring manifest") != INPUT_MODE:
        raise ValueError("scoring release input_mode must be image_only")

    runtime_contract = manifest.get("runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        raise ValueError("scoring manifest requires runtime_contract object")
    allowed_keys = runtime_contract.get("allowed_keys")
    if (
        not isinstance(allowed_keys, list)
        or frozenset(str(item) for item in allowed_keys) != RUNTIME_CASE_KEYS
        or len(allowed_keys) != len(RUNTIME_CASE_KEYS)
    ):
        raise ValueError(
            "runtime_contract.allowed_keys must be exactly "
            "case_id, image_path, image_sha256"
        )
    if runtime_contract.get("private_keys_absent") is not True:
        raise ValueError("runtime_contract.private_keys_absent must be true")

    artifact_payload = manifest.get("artifacts")
    if not isinstance(artifact_payload, Mapping):
        raise ValueError("scoring manifest requires artifacts object")
    agent_input = _release_path(
        root, artifact_payload.get("runtime_input"), name="runtime_input"
    )
    if agent_input != benchmark:
        raise ValueError(
            "benchmark path does not match manifest artifacts.runtime_input"
        )

    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or int(counts.get("cases", 0) or 0) < 1:
        raise ValueError("scoring manifest requires a positive counts.cases")

    return ScoringRuntimeRelease(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        release_id=_required_text(manifest, "release_id", name="scoring manifest"),
        release_stage="scoring_release",
        schema_version=schema_version,
        runtime_contract_version=RUNTIME_CONTRACT_VERSION,
        input_mode=INPUT_MODE,
        decision_policy_version="unspecified-by-scoring-package",
        source_access_policy_active=False,
        artifacts=ScoringReleaseArtifacts(
            agent_input=agent_input,
            evaluation_gold=_release_path(
                root, artifact_payload.get("gold"), name="gold"
            ),
            migration_audit=_release_path(
                root,
                artifact_payload.get("migration_audit"),
                name="migration_audit",
            ),
            checksums=_release_path(root, "SHA256SUMS", name="checksums"),
        ),
    )


def resolve_scoring_image_path(
    sample: Mapping[str, Any],
    release: ScoringRuntimeRelease,
) -> Dict[str, Any]:
    """Resolve release-root-relative image paths without exposing private files."""

    resolved = dict(sample)
    image_path = Path(str(resolved.get("image_path") or ""))
    if image_path.is_absolute():
        raise ValueError("scoring release image_path must be relative")
    absolute = (release.root / image_path).resolve()
    runtime_root = (release.root / "runtime_input").resolve()
    try:
        absolute.relative_to(runtime_root)
    except ValueError as exc:
        raise ValueError("image_path escapes runtime_input") from exc
    if not absolute.is_file():
        raise FileNotFoundError(f"scoring release image is missing: {absolute}")
    resolved["image_path"] = str(absolute)
    return resolved


def adapt_scoring_gold_for_process(
    gold: Mapping[str, Any],
) -> Dict[str, Any]:
    """Expose only the label alias expected by the existing structural scorer."""

    schema_version = str(gold.get("schema_version") or "").strip()
    if schema_version != SCORING_GOLD_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported private gold schema {schema_version!r}; "
            f"expected {SCORING_GOLD_SCHEMA_VERSION!r}"
        )
    label = str(gold.get("label") or "").strip()
    if label not in {"supported", "refuted"}:
        raise ValueError("private scoring gold label must be supported or refuted")
    return {**dict(gold), "factual_status": label}
