"""Bind same-round optimizer resumption to the original targets and recipe."""
from pathlib import Path
from .io import load_json, sha256_file, write_json
from .checkpoints import _optimizer_state_files, _scheduler_state_files, _rng_state_files


def bind_resume(*, output_root, datum_manifest_path, initialization_gate_path,
                profile_gate_path, checkpoint=None, seed=0, max_grad_norm=1.0):
    initialization = load_json(initialization_gate_path)
    profile = load_json(profile_gate_path)
    if initialization.get("passed") is not True or profile.get("passed") is not True:
        raise ValueError("PSD resume binding requires successful launch gates")
    identity = {"schema_version": "ifv-psd-resume-binding-v1",
        "datum_manifest_sha256": sha256_file(datum_manifest_path),
        "initialization": initialization, "training_profile": profile,
        "seed": seed, "max_grad_norm": max_grad_norm}
    if checkpoint is not None:
        checkpoint = Path(checkpoint).resolve()
        marker = next((parent / "psd-resume-binding.json" for parent in list(checkpoint.parents)[:10]
                       if (parent / "psd-resume-binding.json").is_file()), None)
        if marker is None or load_json(marker) != identity:
            raise ValueError("resume checkpoint belongs to another PSD target bank/recipe/round")
        state = load_json(checkpoint / "trainer_state.json")
        if (type(state.get("global_step")) is not int or state["global_step"] <= 0
            or not _optimizer_state_files(checkpoint) or not _scheduler_state_files(checkpoint)
            or not _rng_state_files(checkpoint)):
            raise ValueError("PSD resume checkpoint lacks complete optimizer/scheduler/RNG state")
    write_json(Path(output_root) / "psd-resume-binding.json", identity)
    return {"passed": True, "resume_checkpoint": str(checkpoint) if checkpoint else None}
