"""Consumer boundary for immutable v0.3 image-only benchmark releases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from eviscene.orchestrator.state import ImageOnlyRuntimeCase


RELEASE_SCHEMA_VERSION = "ifv-image-only-benchmark-release-v0.3"
RUNTIME_CONTRACT_VERSION = "ifv-image-only-runtime-v1"
INPUT_MODE = "image_only"
DATA_PIPELINE_DECISION_POLICY_VERSION = "reinspect-v2"
RUNTIME_CASE_KEYS = frozenset({"case_id", "image_path", "image_sha256"})


@dataclass(frozen=True)
class ReleaseArtifacts:
    agent_input: Path
    evaluation_gold: Path
    classification_protocol: Path
    process_reference_protocol: Path
    licenses: Path
    source_access_policy: Path | None


@dataclass(frozen=True)
class RuntimeRelease:
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
    artifacts: ReleaseArtifacts


def _load_json_object(path: Path, *, name: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
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
        raise ValueError(f"release manifest requires artifact path {name}")
    relative = Path(rendered)
    if relative.is_absolute():
        raise ValueError(f"release artifact path must be relative: {name}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"release artifact escapes release root: {name}") from exc
    return resolved


def _require_artifact(root: Path, artifacts: Mapping[str, Any], key: str) -> Path:
    path = _release_path(root, artifacts.get(key), name=key)
    if not path.is_file():
        raise FileNotFoundError(f"missing release artifact {key}: {path}")
    return path


def load_runtime_release(benchmark_path: Path) -> RuntimeRelease:
    """Load the v0.3 descriptor for ``runtime_input/cases.jsonl``.

    The evaluator accepts no legacy row-only benchmark shape. The entrypoint must be
    the ``runtime_input/cases.jsonl`` declared by a valid v0.3 release manifest.
    """

    benchmark = benchmark_path.expanduser().resolve()
    if benchmark.parent.name != "runtime_input":
        raise ValueError(
            "benchmark must be a v0.3 release entrypoint at "
            "runtime_input/cases.jsonl"
        )

    root = benchmark.parent.parent.resolve()
    manifest_path = root / "manifest.json"
    manifest = _load_json_object(manifest_path, name="release manifest")

    schema_version = _required_text(
        manifest, "schema_version", name="release manifest"
    )
    runtime_contract_version = _required_text(
        manifest, "runtime_contract_version", name="release manifest"
    )
    input_mode = _required_text(manifest, "input_mode", name="release manifest")
    decision_policy_version = _required_text(
        manifest, "decision_policy_version", name="release manifest"
    )
    expected = {
        "schema_version": (schema_version, RELEASE_SCHEMA_VERSION),
        "runtime_contract_version": (
            runtime_contract_version,
            RUNTIME_CONTRACT_VERSION,
        ),
        "input_mode": (input_mode, INPUT_MODE),
        "decision_policy_version": (
            decision_policy_version,
            DATA_PIPELINE_DECISION_POLICY_VERSION,
        ),
    }
    mismatches = [
        f"{key}={actual!r}, expected {wanted!r}"
        for key, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError(
            "unsupported runtime release contract: " + "; ".join(mismatches)
        )

    runtime_contract = manifest.get("runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        raise ValueError("release manifest requires runtime_contract object")
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
        raise ValueError("release manifest requires artifacts object")
    agent_input = _require_artifact(root, artifact_payload, "agent_input")
    if agent_input != benchmark:
        raise ValueError(
            "benchmark path does not match manifest artifacts.agent_input"
        )

    policy_payload = manifest.get("source_access_policy")
    if not isinstance(policy_payload, Mapping):
        raise ValueError("release manifest requires source_access_policy object")
    policy_active = policy_payload.get("active")
    if not isinstance(policy_active, bool):
        raise ValueError("source_access_policy.active must be boolean")
    policy_path = None
    if policy_active:
        policy_path = _release_path(
            root,
            policy_payload.get("path"),
            name="source_access_policy.path",
        )
        if not policy_path.is_file():
            raise FileNotFoundError(
                f"missing active source-access policy: {policy_path}"
            )
    elif str(policy_payload.get("path") or "").strip():
        raise ValueError("inactive source_access_policy must not declare a path")

    return RuntimeRelease(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        release_id=_required_text(manifest, "release_id", name="release manifest"),
        release_stage=_required_text(
            manifest, "release_stage", name="release manifest"
        ),
        schema_version=schema_version,
        runtime_contract_version=runtime_contract_version,
        input_mode=input_mode,
        decision_policy_version=decision_policy_version,
        source_access_policy_active=policy_active,
        artifacts=ReleaseArtifacts(
            agent_input=agent_input,
            evaluation_gold=_require_artifact(
                root, artifact_payload, "evaluation_gold"
            ),
            classification_protocol=_require_artifact(
                root, artifact_payload, "classification_protocol"
            ),
            process_reference_protocol=_require_artifact(
                root, artifact_payload, "process_reference_protocol"
            ),
            licenses=_require_artifact(root, artifact_payload, "licenses"),
            source_access_policy=policy_path,
        ),
    )


def resolve_runtime_image_path(
    sample: Mapping[str, Any],
    benchmark_path: Path,
) -> Dict[str, Any]:
    """Resolve a runtime-input-relative image path without mutating the public row."""

    resolved = dict(sample)
    image_path = Path(str(resolved.get("image_path") or ""))
    if image_path.is_absolute():
        raise ValueError("image-only release image_path must be relative")
    runtime_root = benchmark_path.expanduser().resolve().parent
    absolute = (runtime_root / image_path).resolve()
    try:
        absolute.relative_to(runtime_root)
    except ValueError as exc:
        raise ValueError("image_path escapes runtime_input") from exc
    resolved["image_path"] = str(absolute)
    return resolved


def image_only_case_from_runtime_row(
    sample: Mapping[str, Any],
) -> ImageOnlyRuntimeCase:
    """Parse one exact public v0.3 row into the runtime-owned input model."""

    actual_keys = frozenset(sample)
    missing = sorted(RUNTIME_CASE_KEYS - actual_keys)
    unexpected = sorted(actual_keys - RUNTIME_CASE_KEYS)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ValueError(
            "image-only runtime row must match the exact public field set ("
            + "; ".join(details)
            + ")"
        )
    return ImageOnlyRuntimeCase.model_validate(dict(sample))
