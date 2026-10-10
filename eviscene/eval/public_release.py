from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from eviscene.eval.release_adapter import (
    _release_path,
    DATA_PIPELINE_DECISION_POLICY_VERSION,
    INPUT_MODE,
    RELEASE_SCHEMA_VERSION,
    RUNTIME_CASE_KEYS,
    RUNTIME_CONTRACT_VERSION,
)
from eviscene.eval.run_artifacts import load_json_object
from eviscene.eval.scoring_release_adapter import SCORING_PACKAGE_SCHEMA_VERSION


@dataclass(frozen=True)
class PublicCaseRelease:
    root: Path
    manifest_path: Path
    manifest: Dict[str, Any]
    release_id: str
    release_stage: str
    schema_version: str
    runtime_contract_version: str
    input_mode: str
    decision_policy_version: str
    source_access_policy_path: Path | None
    image_path_mode: str


def required_text(payload: Mapping[str, Any], key: str, *, name: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"{name} requires non-empty {key}")
    return value


def release_path(root: Path, value: Any, *, name: str) -> Path:
    return _release_path(root, value, name=name)


def _require_public_runtime_contract(manifest: Mapping[str, Any]) -> None:
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


def _source_policy_path(root: Path, manifest: Mapping[str, Any]) -> Path | None:
    policy_payload = manifest.get("source_access_policy")
    if policy_payload is None:
        return None
    if not isinstance(policy_payload, Mapping):
        raise ValueError("source_access_policy must be an object when declared")
    active = policy_payload.get("active")
    if active is None:
        return None
    if not isinstance(active, bool):
        raise ValueError("source_access_policy.active must be boolean")
    if not active:
        return None
    policy_path = release_path(
        root,
        policy_payload.get("path"),
        name="source_access_policy.path",
    )
    if not policy_path.is_file():
        raise FileNotFoundError(f"missing active source-access policy: {policy_path}")
    return policy_path


def load_public_release(benchmark_path: Path) -> PublicCaseRelease:
    benchmark = benchmark_path.expanduser().resolve()
    if benchmark.name != "cases.jsonl" or benchmark.parent.name != "runtime_input":
        raise ValueError("benchmark must be runtime_input/cases.jsonl")
    root = benchmark.parent.parent.resolve()
    manifest_path = root / "manifest.json"
    manifest = load_json_object(manifest_path, name="release manifest")
    schema_version = required_text(
        manifest, "schema_version", name="release manifest"
    )
    if required_text(manifest, "input_mode", name="release manifest") != INPUT_MODE:
        raise ValueError("release input_mode must be image_only")
    _require_public_runtime_contract(manifest)

    artifact_payload = manifest.get("artifacts")
    if not isinstance(artifact_payload, Mapping):
        raise ValueError("release manifest requires artifacts object")

    if schema_version == RELEASE_SCHEMA_VERSION:
        runtime_contract_version = required_text(
            manifest, "runtime_contract_version", name="release manifest"
        )
        if runtime_contract_version != RUNTIME_CONTRACT_VERSION:
            raise ValueError(
                "unsupported runtime_contract_version="
                f"{runtime_contract_version!r}"
            )
        decision_policy_version = required_text(
            manifest, "decision_policy_version", name="release manifest"
        )
        if decision_policy_version != DATA_PIPELINE_DECISION_POLICY_VERSION:
            raise ValueError(
                "unsupported decision_policy_version="
                f"{decision_policy_version!r}"
            )
        agent_input = release_path(
            root,
            artifact_payload.get("agent_input"),
            name="artifacts.agent_input",
        )
        if agent_input != benchmark:
            raise ValueError(
                "benchmark path does not match manifest artifacts.agent_input"
            )
        return PublicCaseRelease(
            root=root,
            manifest_path=manifest_path,
            manifest=dict(manifest),
            release_id=required_text(
                manifest, "release_id", name="release manifest"
            ),
            release_stage=required_text(
                manifest, "release_stage", name="release manifest"
            ),
            schema_version=schema_version,
            runtime_contract_version=runtime_contract_version,
            input_mode=INPUT_MODE,
            decision_policy_version=decision_policy_version,
            source_access_policy_path=_source_policy_path(root, manifest),
            image_path_mode="runtime_input_relative",
        )

    if schema_version == SCORING_PACKAGE_SCHEMA_VERSION:
        agent_input = release_path(
            root,
            artifact_payload.get("runtime_input"),
            name="artifacts.runtime_input",
        )
        if agent_input != benchmark:
            raise ValueError(
                "benchmark path does not match manifest artifacts.runtime_input"
            )
        return PublicCaseRelease(
            root=root,
            manifest_path=manifest_path,
            manifest=dict(manifest),
            release_id=required_text(
                manifest, "release_id", name="scoring manifest"
            ),
            release_stage="scoring_release",
            schema_version=schema_version,
            runtime_contract_version=RUNTIME_CONTRACT_VERSION,
            input_mode=INPUT_MODE,
            decision_policy_version="unspecified-by-scoring-package",
            source_access_policy_path=None,
            image_path_mode="release_root_relative",
        )

    raise ValueError(f"unsupported release schema_version={schema_version!r}")


def resolve_image_path(
    sample: Mapping[str, Any],
    *,
    release: PublicCaseRelease,
    benchmark_path: Path,
) -> Dict[str, Any]:
    resolved = dict(sample)
    image_path = Path(str(resolved.get("image_path") or ""))
    if image_path.is_absolute():
        raise ValueError("image-only release image_path must be relative")
    if release.image_path_mode == "runtime_input_relative":
        runtime_root = benchmark_path.expanduser().resolve().parent
        absolute = (runtime_root / image_path).resolve()
    else:
        absolute = (release.root / image_path).resolve()
        runtime_root = (release.root / "runtime_input").resolve()
    try:
        absolute.relative_to(runtime_root)
    except ValueError as exc:
        raise ValueError("image_path escapes runtime_input") from exc
    if not absolute.is_file():
        raise FileNotFoundError(f"runtime image is missing: {absolute}")
    resolved["image_path"] = str(absolute)
    return resolved
