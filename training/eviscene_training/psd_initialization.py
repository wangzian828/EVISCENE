"""Bind the actual Swift student initialization to the targets' frozen teacher."""
from __future__ import annotations

from pathlib import Path

from .artifact_receipts import artifact_identity
from .io import iter_jsonl, load_json, sha256_file
from .psd import _validated_model_roles
from .psd_topk import _validate_attestation


def frozen_base_binding(manifest, base):
    """Preserve the original dense base hashes across adapter-only rounds."""
    base = Path(base).resolve()
    checkpoint = Path(manifest["checkpoint"]["path"]).resolve()
    if base == checkpoint:
        binding = {"path": str(base), "artifacts": [item for item in manifest.get("artifacts", [])
                    if item.get("scope") == "model"]}
    else:
        binding = manifest.get("base_model_binding", {})
    if (not binding.get("path") or Path(binding["path"]).resolve() != base
            or not binding.get("artifacts")):
        raise ValueError("PSD adapter manifest lacks the immutable base model binding")
    has_weights = False
    for item in binding["artifacts"]:
        path = (base / item["path"]).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise ValueError("PSD frozen base model artifact is missing")
        has_weights = has_weights or path.suffix in {".safetensors", ".bin"}
    if not has_weights:
        raise ValueError("PSD frozen base binding contains no weight artifacts")
    return binding


def verify_initialization(*, datum_manifest_path, serving_profile_path,
                          checkpoint_manifest_path, model_path, adapter_path=None):
    manifest = load_json(datum_manifest_path)
    source = manifest["source"]
    targets_path = Path(source["targets"])
    if source.get("targets_identity") is not None:
        if artifact_identity(targets_path) != source["targets_identity"]:
            raise ValueError("PSD source targets changed after materialization")
    elif sha256_file(targets_path) != source["targets_sha256"]:
        raise ValueError("PSD source targets changed after materialization")
    profile, checkpoint, checkpoint_sha = _validate_attestation(
        serving_profile_path=serving_profile_path, checkpoint_manifest_path=checkpoint_manifest_path)
    base = profile.get("engine_model_path") or checkpoint
    expected_adapter = checkpoint if base != checkpoint else None
    if Path(model_path).resolve() != Path(base).resolve():
        raise ValueError("Swift student base model differs from frozen teacher")
    actual_adapter = str(Path(adapter_path).resolve()) if adapter_path else None
    if actual_adapter != (str(Path(expected_adapter).resolve()) if expected_adapter else None):
        raise ValueError("Swift student must load the exact round-start adapter")
    target_count = 0
    for row in iter_jsonl(targets_path):
        target_count += 1
        roles = _validated_model_roles(row.get("model_roles"))
        teacher, student = roles["frozen_self_teacher"], roles["trainable_student"]
        if (teacher["round_start_checkpoint"] != checkpoint or student["initial_checkpoint"] != checkpoint
                or teacher["checkpoint_manifest_sha256"] != checkpoint_sha
                or student["checkpoint_manifest_sha256"] != checkpoint_sha
                or teacher["model"] != profile["profile_id"]):
            raise ValueError("PSD target policy differs from actual student initialization")
    if not target_count:
        raise ValueError("PSD source targets are empty")
    checkpoint_manifest = load_json(checkpoint_manifest_path)
    if expected_adapter:
        frozen_base_binding(checkpoint_manifest, base)
    model_files = [item for item in checkpoint_manifest.get("artifacts", []) if item.get("scope") == "model"]
    if not model_files:
        raise ValueError("round-start manifest lacks model artifact hashes")
    root = Path(checkpoint).resolve()
    for item in model_files:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("round-start model artifact is missing")
    return {"schema_version": "ifv-psd-initialization-gate-v1", "passed": True,
            "model": str(Path(base).resolve()), "adapter": actual_adapter,
            "checkpoint_manifest_sha256": checkpoint_sha,
            "targets_binding": source.get("targets_identity") or source.get("targets_sha256"),
            "targets": target_count, "new_round_optimizer": "fresh unless explicit --resume_from_checkpoint"}
