"""Training-only grouped collector; native Agent semantics with explicit T=0.7.

Run in a separate process. Never patch the main evaluation server/process.
The standard run_cases CLI follows --train-cases <private split manifest>.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from eviscene.workflow import VerificationWorkflow
from eviscene_training.io import load_jsonl, load_json, write_json, sha256_file


class PSDWorkflow(VerificationWorkflow):
    def _get_orchestrator(self, *, validate_startup=True):
        policy = super()._get_orchestrator(validate_startup=validate_startup)
        if not getattr(policy, "_psd_sampling_bound", False):
            original = policy._stage_generation_config
            def generation(stage):
                config = original(stage)
                if stage.upper() in {"UNIFIED_REACT", "UNIFIED_JUDGMENT"}:
                    config = {**config, "temperature": 0.7}
                return config
            policy._stage_generation_config = generation
            from eviscene_training.psd_infrastructure_retry import guard_policy_backend
            guard_policy_backend(policy.llm)
            policy._psd_sampling_bound = True
        return policy

    def _new_batch_child(self, config):
        return PSDWorkflow(config)

    async def run_single(self, image_path, image_id="", *, runtime_case=None):
        from eviscene_training.psd_infrastructure_retry import (
            retry_episode, PolicyInfrastructureFailure, InfrastructureRetriesExhausted)
        from eviscene.orchestrator.runtime_events import atomic_write_json
        from eviscene.orchestrator.runtime_case import image_sha256, verify_case_image
        from eviscene.redaction import sanitize_for_persistence
        episode_id = image_id or (runtime_case.case_id if runtime_case else Path(image_path).name)
        trace_dir = Path(self.config.output_dir).resolve()
        slot = trace_dir.parent / "psd-infrastructure-attempts" / hashlib.sha256(episode_id.encode()).hexdigest()
        image_hash = image_sha256(image_path)
        if runtime_case is not None:
            verify_case_image(runtime_case, image_path)
        identity = {"episode_id": episode_id, "case_id": runtime_case.case_id if runtime_case else episode_id,
                    "image_sha256": image_hash, "sampling_seed": self.config.sampling_seed,
                    "model": self.config.model_name, "provider": self.config.provider,
                    "base_url": self.config.llm_base_url, "temperature": .7,
                    "timeout": self.config.timeout}
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in episode_id) + ".json"

        async def generate(directory):
            if image_sha256(image_path) != image_hash:
                raise ValueError("PSD input image changed between infrastructure attempts")
            config = copy.copy(self.config)
            config.output_dir = str(directory / "traces")
            config.resume_from = None  # No failed model history or recovered hint context.
            child = PSDWorkflow(config)
            try:
                try:
                    result = await VerificationWorkflow.run_single(child, image_path, episode_id,
                                                                  runtime_case=runtime_case)
                except PolicyInfrastructureFailure:
                    raise
                except Exception as error:
                    # Preserve the native failed trace, then let the answer-blind
                    # completion validator decide whether the slot needs recovery.
                    result = getattr(error, "_eviscene_result", None)
                    if not isinstance(result, dict):
                        raise
                saved_trace = Path(config.output_dir) / safe_name
                return load_json(saved_trace) if saved_trace.exists() else sanitize_for_persistence(result)
            finally:
                await child.aclose()

        try:
            from eviscene_training.psd_source_completion import source_completion_failure
            result = await retry_episode(root=slot, identity=identity, generate=generate,
                                         validate_result=source_completion_failure)
        except InfrastructureRetriesExhausted as error:
            # No canonical trace for a poisoned/unresolved slot. The native result
            # table still records the episode, keeping the denominator explicit.
            return {"image_id": episode_id, "image_path": str(image_path), "verdict": "error",
                    "termination": "error", "error": str(error),
                    "psd_infrastructure_pending": True}
        atomic_write_json(trace_dir / safe_name, result)
        return result


def main():
    from eviscene_training.psd_capture_semantics import install_capture_semantics
    install_capture_semantics()
    from eviscene_training.psd_collection import require_token_capture_environment
    require_token_capture_environment(os.environ)
    from eviscene.eval import run_cases
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-cases", type=Path, required=True)
    private_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = run_cases._parse_args()
    if args.rollouts_per_case != 8:
        raise ValueError("published PSD collection requires eight rollouts per task")
    if not args.benchmark or not args.output_dir:
        raise ValueError("PSD collector requires explicit public benchmark and new output directory")
    rows = load_jsonl(private_args.train_cases)
    if not rows or any(r.get("split") != "train" for r in rows):
        raise ValueError("PSD collection requires an explicitly training-only split")
    public = load_jsonl(Path(args.benchmark))
    if not {r["case_id"] for r in public} <= {r["case_id"] for r in rows}:
        raise ValueError("PSD public benchmark contains non-training cases")
    # Bind the PSD workflow only in this collector process and its batch children.
    run_cases.VerificationWorkflow = PSDWorkflow
    summary = asyncio.run(run_cases._run_cases(args))
    manifest_path = Path(args.output_dir) / "run_manifest.json"
    manifest = load_json(manifest_path)
    manifest["agent"]["psd_sampling"] = {"temperature": 0.7, "rollouts_per_case": 8,
        "capture_policy_tokens": True, "policy_topk": 20,
        "collector_sha256": sha256_file(Path(__file__)), "stages": ["UNIFIED_REACT", "UNIFIED_JUDGMENT"]}
    from eviscene_training.psd_source_completion import VERSION as COMPLETION_POLICY
    manifest['agent']['psd_sampling']['source_completion_policy'] = COMPLETION_POLICY
    from eviscene_training.psd_infrastructure_retry import VERSION, MAX_ATTEMPTS
    slots = list((Path(args.output_dir) / "psd-infrastructure-attempts").glob("*/retry-state.json"))
    unresolved = [str(p.parent) for p in slots if not (p.parent / "result.json").is_file()]
    extended = [{'episode_id': load_json(p)['identity']['inputs']['episode_id'],
                 'max_attempts': load_json(p)['identity']['max_attempts'],
                 'allowance': str(p.parent/'recovery-allowance.json')}
                for p in slots if load_json(p)['identity']['max_attempts'] != MAX_ATTEMPTS]
    manifest["agent"]["psd_sampling"]["infrastructure_retry"] = {
        "version": VERSION, "default_max_attempts": MAX_ATTEMPTS,
        "max_attempts": max([MAX_ATTEMPTS, *(r['max_attempts'] for r in extended)]),
        "budget_extensions": extended, "slots": len(slots),
        "unresolved": unresolved, "selection_by_answer": False}
    write_json(manifest_path, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if unresolved:
        raise RuntimeError(f"PSD collection has {len(unresolved)} unresolved infrastructure slots; not ready for targets")


if __name__ == "__main__":
    main()
