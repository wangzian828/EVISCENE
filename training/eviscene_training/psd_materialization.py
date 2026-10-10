"""Idempotent local PSD package stages; no provider calls inside a stage.

Commit all generated file hashes after a successful builder. On interruption,
retain a bounded diagnostic copy and rebuild only the unfinished local stage.
Media paths remain stable because the final directory is never renamed.
"""
from __future__ import annotations

from pathlib import Path
import uuid

from .artifact_receipts import freeze_artifact, verify_artifact
from .io import load_json, sha256_file
from .psd_repair_storage import load_bound, save_bound
from .psd_repair_search import search_lock


def completed_package(*, output_dir, input_files, build, parameters=None):
    output_dir = Path(output_dir).absolute()
    if output_dir.is_symlink():
        raise ValueError("PSD package directory cannot be a symlink")
    output_dir = output_dir.resolve()
    control = output_dir.parent / ("." + output_dir.name + "-stage")
    with search_lock(control):
        marker = control / "state.json"
        # Keep historical packages readable.  New packages never hash payloads;
        # they use O(1) stat-bound immutable receipts.
        marker_header = load_json(marker) if marker.exists() else {}
        marker_identity = marker_header.get("identity", {})
        if marker_identity.get("schema_version") == "ifv-psd-materialization-stage-v2":
            identity = marker_identity
            if identity.get("output") != str(output_dir) or identity.get("parameters") != (parameters or {}):
                raise ValueError("PSD package stage identity changed")
            expected_inputs = {str(Path(path).resolve()) for path in input_files}
            if set(identity.get("inputs", {})) != expected_inputs:
                raise ValueError("PSD package stage inputs changed")
            for record in identity["inputs"].values():
                receipt = Path(record["receipt"])
                try:
                    verified = verify_artifact(receipt)
                except ValueError as error:
                    raise ValueError("PSD package input binding changed") from error
                if verified != record["artifact"]:
                    raise ValueError("PSD package input receipt changed")
        elif marker.exists():
            identity = {"schema_version": "ifv-psd-materialization-stage-v1",
                "output": str(output_dir), "parameters": parameters or {},
                "inputs": {str(Path(path).resolve()): sha256_file(Path(path)) for path in input_files}}
        else:
            inputs = {}
            for index, path in enumerate(input_files):
                resolved = Path(path).resolve()
                receipt = control / "input-receipts" / f"{index:08d}.json"
                inputs[str(resolved)] = {"receipt": str(receipt),
                    "artifact": freeze_artifact(resolved, receipt)}
            identity = {"schema_version": "ifv-psd-materialization-stage-v2",
                "output": str(output_dir), "parameters": parameters or {}, "inputs": inputs}
        if marker.exists():
            saved = load_bound(marker, identity=identity)
            if saved["status"] == "complete":
                if identity["schema_version"].endswith("v2"):
                    current_names = {str(path.relative_to(output_dir))
                        for path in output_dir.rglob("*") if path.is_file()}
                    if current_names != set(saved["files"]):
                        raise ValueError("completed PSD package files changed")
                    for record in saved["files"].values():
                        try:
                            verified = verify_artifact(Path(record["receipt"]))
                        except ValueError as error:
                            raise ValueError("completed PSD package files changed") from error
                        if verified != record["artifact"]:
                            raise ValueError("completed PSD package file changed")
                else:
                    current = {str(path.relative_to(output_dir)): sha256_file(path)
                               for path in output_dir.rglob("*") if path.is_file()}
                    if current != saved["files"]:
                        raise ValueError("completed PSD package files changed")
                return saved["result"]
        else:
            if output_dir.exists() and any(output_dir.iterdir()):
                raise ValueError("refusing to adopt or overwrite an unbound PSD package")
            save_bound(marker, identity=identity, payload={"status": "building"})
        if output_dir.exists() and any(output_dir.iterdir()):
            archives = list(control.glob("interrupted-*"))
            if len(archives) >= 2:
                raise ValueError("PSD stage has two interrupted copies; review storage before retry")
            destination = control / ("interrupted-" + uuid.uuid4().hex)
            # Both resolved endpoints are children of this exact stage parent.
            if output_dir.parent != control.parent or destination.parent != control:
                raise ValueError("invalid PSD stage recovery paths")
            output_dir.rename(destination)
        result = build(output_dir)
        if identity["schema_version"].endswith("v2"):
            for record in identity["inputs"].values():
                try:
                    verified = verify_artifact(Path(record["receipt"]))
                except ValueError as error:
                    raise ValueError("PSD package input changed while building") from error
                if verified != record["artifact"]:
                    raise ValueError("PSD package input changed while building")
            files = {}
            output_paths = sorted(path for path in output_dir.rglob("*") if path.is_file())
            for index, path in enumerate(output_paths):
                if not path.is_file():
                    continue
                relative = str(path.relative_to(output_dir))
                receipt = control / "output-receipts" / f"{index:08d}.json"
                files[relative] = {"receipt": str(receipt),
                    "artifact": freeze_artifact(path, receipt)}
        else:
            if identity["inputs"] != {path: sha256_file(Path(path)) for path in identity["inputs"]}:
                raise ValueError("PSD package inputs changed while building")
            files = {str(path.relative_to(output_dir)): sha256_file(path)
                     for path in output_dir.rglob("*") if path.is_file()}
        if "manifest.json" not in files:
            raise ValueError("PSD builder did not emit a manifest")
        save_bound(marker, identity=identity, payload={"status": "complete", "result": result, "files": files})
        return result


def materialize_bank(*, output_dir, repair_candidates, repair_attempts,
                     preservation_candidates, serving_profile=None,
                     checkpoint_manifest=None, teacher_device="cpu", score_missing_topk=False):
    from .psd_repairs import assemble_psd_repair_package
    from .psd import build_psd_target_package, materialize_psd_topk_cache
    from .psd_datums import build_sparse_topk_package
    from .psd_preflight import verify_psd_training_input
    root = Path(output_dir)
    assembled, targets, datums = root / "assembled", root / "targets", root / "datums"
    result = {}
    result["assembly"] = completed_package(output_dir=assembled,
        input_files=[repair_candidates, repair_attempts, preservation_candidates],
        build=lambda destination: assemble_psd_repair_package(repair_candidates_path=repair_candidates,
            repair_attempts_path=repair_attempts, preservation_candidates_path=preservation_candidates,
            output_dir=destination))
    if result["assembly"]["status"] != "ready_for_target_build":
        return {**result, "status": "blocked_missing_verified_source_kind"}
    result["targets"] = completed_package(output_dir=targets,
        input_files=[assembled / "repairs.jsonl", assembled / "preservation.jsonl"],
        build=lambda destination: build_psd_target_package(repairs_path=assembled / "repairs.jsonl",
            preservation_path=assembled / "preservation.jsonl", output_dir=destination))
    targets_path = targets / "targets.jsonl"
    if result["targets"]["status"] == "ready_for_topk_cache":
        if not score_missing_topk:
            return {**result, "status": "requires_frozen_teacher_topk"}
        if serving_profile is None or checkpoint_manifest is None:
            raise ValueError("missing frozen teacher attestation for top-k scoring")
        from .psd_topk import collect_psd_topk_cache
        cache = root / "topk-cache"
        result["topk"] = collect_psd_topk_cache(targets_path=targets_path,
            serving_profile_path=serving_profile, checkpoint_manifest_path=checkpoint_manifest,
            output_dir=cache, topk=20, backend="transformers", device=teacher_device)
        if result["topk"]["status"] != "ready_for_materialization":
            return {**result, "status": "paused_topk_collection"}
        resolved = root / "resolved-targets"
        result["resolved_targets"] = completed_package(output_dir=resolved,
            input_files=[targets_path, cache / "teacher_topk_cache.jsonl"],
            build=lambda destination: materialize_psd_topk_cache(targets_path=targets_path,
                cache_path=cache / "teacher_topk_cache.jsonl", output_dir=destination, topk=20))
        if result["resolved_targets"]["status"] != "ready_for_training":
            return {**result, "status": "blocked_invalid_teacher_cache"}
        targets_path = resolved / "targets.jsonl"
    elif result["targets"]["status"] != "ready_for_training":
        return {**result, "status": "blocked_invalid_targets"}
    result["datums"] = completed_package(output_dir=datums, input_files=[targets_path],
        build=lambda destination: build_sparse_topk_package(targets_path=targets_path,
            output_dir=destination, topk=20, max_sequence_length=131072, require_both_kinds=True))
    if result["datums"]["status"] != "ready_for_trainer":
        return {**result, "status": "blocked_invalid_datums"}
    result["verification"] = verify_psd_training_input(datums_path=datums / "datums.jsonl",
        manifest_path=datums / "manifest.json", expected_topk=20, max_context=131072)
    return {**result, "status": "ready_for_trainer" if result["verification"]["passed"] else "blocked_datum_gate",
            "datums_path": str((datums / "datums.jsonl").resolve())}
