"""Fail CI on accidental secrets or non-code payloads in the Git index."""

from __future__ import annotations

import re
import io
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

RELEASE_FILES = frozenset({
    ".github/workflows/ci.yml",
    ".gitignore",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "NOTICE",
    "README.md",
    "README_zh-CN.md",
    "SECURITY.md",
    "assets/architecture.webp",
    "assets/badge-arxiv.svg",
    "assets/badge-hf-paper.svg",
    "assets/badge-models-data.svg",
    "assets/badge-website.svg",
    "docs/data.md",
    "eviscene/__init__.py",
    "eviscene/__main__.py",
    "eviscene/environment.py",
    "eviscene/eval/__init__.py",
    "eviscene/eval/agent_private_gold.py",
    "eviscene/eval/archive_adapter.py",
    "eviscene/eval/archive_baseline.py",
    "eviscene/eval/case_selection.py",
    "eviscene/eval/evaluator_private_gold.py",
    "eviscene/eval/gemini_run_guard.py",
    "eviscene/eval/postprocess_run.py",
    "eviscene/eval/private_gold_judge_contract.py",
    "eviscene/eval/private_gold_metrics.py",
    "eviscene/eval/public_release.py",
    "eviscene/eval/release_adapter.py",
    "eviscene/eval/result_records.py",
    "eviscene/eval/rollout.py",
    "eviscene/eval/run_artifacts.py",
    "eviscene/eval/run_cases.py",
    "eviscene/eval/run_eval.py",
    "eviscene/eval/score_reference_chain.py",
    "eviscene/eval/score_semantic_reward.py",
    "eviscene/eval/score_sft_eligibility.py",
    "eviscene/eval/scoring_release_adapter.py",
    "eviscene/integrations/__init__.py",
    "eviscene/integrations/async_runtime.py",
    "eviscene/integrations/browse/__init__.py",
    "eviscene/integrations/browse/jina_reader.py",
    "eviscene/integrations/browse/passage_selection.py",
    "eviscene/integrations/clock/__init__.py",
    "eviscene/integrations/clock/system_clock.py",
    "eviscene/integrations/gemini/__init__.py",
    "eviscene/integrations/gemini/interactions.py",
    "eviscene/integrations/gemini/runtime_metrics.py",
    "eviscene/integrations/gemini/schema.py",
    "eviscene/integrations/http_sessions.py",
    "eviscene/integrations/http_transport.py",
    "eviscene/integrations/llm/__init__.py",
    "eviscene/integrations/llm/openai_compatible.py",
    "eviscene/integrations/llm/prefix_cache.py",
    "eviscene/integrations/ocr/__init__.py",
    "eviscene/integrations/ocr/baidu.py",
    "eviscene/integrations/search/__init__.py",
    "eviscene/integrations/search/jina_reranker.py",
    "eviscene/integrations/search/serper.py",
    "eviscene/integrations/search/visual_search.py",
    "eviscene/integrations/vlm/__init__.py",
    "eviscene/integrations/vlm/factory.py",
    "eviscene/integrations/vlm/openai_vlm.py",
    "eviscene/integrations/vlm/qwen_vl.py",
    "eviscene/orchestrator/__init__.py",
    "eviscene/orchestrator/evidence_policy.py",
    "eviscene/orchestrator/evidence_semantics.py",
    "eviscene/orchestrator/investigation_models.py",
    "eviscene/orchestrator/llm_backend.py",
    "eviscene/orchestrator/pipeline.py",
    "eviscene/orchestrator/react_runtime.py",
    "eviscene/orchestrator/route_policy.py",
    "eviscene/orchestrator/runtime_case.py",
    "eviscene/orchestrator/runtime_events.py",
    "eviscene/orchestrator/source_access.py",
    "eviscene/orchestrator/source_provenance.py",
    "eviscene/orchestrator/stage_native.py",
    "eviscene/orchestrator/stage_runner.py",
    "eviscene/orchestrator/stage_schemas.py",
    "eviscene/orchestrator/stage_tools.py",
    "eviscene/orchestrator/stage_types.py",
    "eviscene/orchestrator/state.py",
    "eviscene/orchestrator/tool_ablation.py",
    "eviscene/orchestrator/tool_cache.py",
    "eviscene/orchestrator/tool_execution.py",
    "eviscene/orchestrator/tool_health.py",
    "eviscene/orchestrator/tool_registry.py",
    "eviscene/orchestrator/tool_result.py",
    "eviscene/orchestrator/unified_prompts.py",
    "eviscene/provider_profiles.py",
    "eviscene/redaction.py",
    "eviscene/render_trace_html.py",
    "eviscene/storage.py",
    "eviscene/tools/__init__.py",
    "eviscene/tools/base.py",
    "eviscene/tools/check_consistency.py",
    "eviscene/tools/compare_reference.py",
    "eviscene/tools/context_memory.py",
    "eviscene/tools/count_objects.py",
    "eviscene/tools/crop_and_inspect.py",
    "eviscene/tools/crop_and_search.py",
    "eviscene/tools/current_time.py",
    "eviscene/tools/focused_visual_inspection.py",
    "eviscene/tools/ocr_with_position.py",
    "eviscene/tools/perceive_scene.py",
    "eviscene/tools/reverse_image_search.py",
    "eviscene/tools/text_image_search.py",
    "eviscene/tools/text_search.py",
    "eviscene/tools/vision_utils.py",
    "eviscene/tools/visit.py",
    "eviscene/tools/visual_anomaly.py",
    "eviscene/trace_viewer.py",
    "eviscene/trajectory/__init__.py",
    "eviscene/trajectory/exporter.py",
    "eviscene/trajectory/media_projection.py",
    "eviscene/trajectory/perception_exporter.py",
    "eviscene/trajectory/reference_chain.py",
    "eviscene/trajectory/schema.py",
    "eviscene/trajectory/scoring.py",
    "eviscene/trajectory/semantic_reward.py",
    "eviscene/trajectory/sft_eligibility.py",
    "eviscene/workflow.py",
    "pyproject.toml",
    "scripts/__init__.py",
    "scripts/audit_agent_private_gold.py",
    "scripts/audit_direct_qa_baseline.py",
    "scripts/audit_real_trace.py",
    "scripts/check_release.py",
    "scripts/collect_psd_rollouts.py",
    "scripts/prepare_psd_paired_eval.py",
    "scripts/prepare_runtime_input.py",
    "scripts/run_agent_test_rollout.py",
    "scripts/run_direct_qa_baseline.py",
    "scripts/run_psd_judge.py",
    "scripts/run_psd_repair_driver.py",
    "scripts/server/retry_stream_agent_judge_failures.py",
    "scripts/server/stream_agent_judges.py",
    "scripts/summarize_agent_test_evaluation.py",
    "scripts/trajectory/__init__.py",
    "scripts/trajectory/build_sft_case_split.py",
    "scripts/trajectory/build_sft_training_package.py",
    "scripts/trajectory/export_dataset.py",
    "scripts/trajectory/run_teacher_rollout_autopilot.py",
    "scripts/trajectory/stage_accepted_teacher_release.py",
    "training/LICENSE",
    "training/MANIFEST.in",
    "training/NOTICE",
    "training/README.md",
    "training/eviscene_training/__init__.py",
    "training/eviscene_training/__main__.py",
    "training/eviscene_training/_repo_import.py",
    "training/eviscene_training/artifact_receipts.py",
    "training/eviscene_training/audit.py",
    "training/eviscene_training/checkpoint_io.py",
    "training/eviscene_training/checkpoint_selection.py",
    "training/eviscene_training/checkpoints.py",
    "training/eviscene_training/cli.py",
    "training/eviscene_training/contracts.py",
    "training/eviscene_training/encode_cache.py",
    "training/eviscene_training/io.py",
    "training/eviscene_training/long_context.py",
    "training/eviscene_training/manifests.py",
    "training/eviscene_training/perception.py",
    "training/eviscene_training/policy.py",
    "training/eviscene_training/policy_contract.py",
    "training/eviscene_training/profile.py",
    "training/eviscene_training/psd.py",
    "training/eviscene_training/psd_candidate_binding.py",
    "training/eviscene_training/psd_candidates.py",
    "training/eviscene_training/psd_capture_semantics.py",
    "training/eviscene_training/psd_case_pool.py",
    "training/eviscene_training/psd_collection.py",
    "training/eviscene_training/psd_collection_recovery.py",
    "training/eviscene_training/psd_datums.py",
    "training/eviscene_training/psd_deterministic_sampler.py",
    "training/eviscene_training/psd_fast_length.py",
    "training/eviscene_training/psd_fsdp_resume.py",
    "training/eviscene_training/psd_gemini_judge.py",
    "training/eviscene_training/psd_hf_teacher.py",
    "training/eviscene_training/psd_infrastructure_retry.py",
    "training/eviscene_training/psd_initialization.py",
    "training/eviscene_training/psd_judge_media.py",
    "training/eviscene_training/psd_materialization.py",
    "training/eviscene_training/psd_media.py",
    "training/eviscene_training/psd_modality.py",
    "training/eviscene_training/psd_model_route.py",
    "training/eviscene_training/psd_ms_swift.py",
    "training/eviscene_training/psd_pipeline_plan.py",
    "training/eviscene_training/psd_preflight.py",
    "training/eviscene_training/psd_repair.py",
    "training/eviscene_training/psd_repair_finalize.py",
    "training/eviscene_training/psd_repair_runtime.py",
    "training/eviscene_training/psd_repair_search.py",
    "training/eviscene_training/psd_repair_storage.py",
    "training/eviscene_training/psd_repair_verifier.py",
    "training/eviscene_training/psd_repairs.py",
    "training/eviscene_training/psd_resume.py",
    "training/eviscene_training/psd_round.py",
    "training/eviscene_training/psd_slate.py",
    "training/eviscene_training/psd_slate_feedback.py",
    "training/eviscene_training/psd_slate_search.py",
    "training/eviscene_training/psd_source_completion.py",
    "training/eviscene_training/psd_source_review.py",
    "training/eviscene_training/psd_storage_admission.py",
    "training/eviscene_training/psd_target_balance.py",
    "training/eviscene_training/psd_topk.py",
    "training/eviscene_training/resource_monitor.py",
    "training/eviscene_training/rewards.py",
    "training/eviscene_training/sft_loss_scale.py",
    "training/eviscene_training/watchdog.py",
    "training/pyproject.toml",
})

ARTIFACT_SUFFIXES = {
    ".jsonl",
    ".jsonl.gz",
    ".csv",
    ".tsv",
    ".parquet",
    ".arrow",
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".zip",
    ".tar",
    ".tar.gz",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".log",
    ".whl",
    ".pyc",
    ".pyo",
}
PATTERNS = {
    "access token": re.compile(
        r"(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}"
        r"|github_pat_[A-Za-z0-9_]{40,}|AIza[\w-]{30,}"
        r"|sk-(?:proj-)?[A-Za-z0-9_-]{30,}|AKIA[0-9A-Z]{16})"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "local user path": re.compile(
        r"[A-Za-z]:[\\/]+Users[\\/]+(?!<|YOUR_)[\w.-]+[\\/]", re.I
    ),
}
ALLOWED_ASSETS = {"assets/architecture.webp"}
FORBIDDEN_PARTS = {
    "tests",
    "operations",
    "private",
    "evaluation",
    "runs",
    "checkpoints",
    "datasets",
    "weights",
    "models",
    "data",
    "outputs",
    "cache",
    ".cache",
    "logs",
    "traces",
    "artifacts",
    "exports",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".git",
}


def _allowed_path(path: str, *, distribution: bool) -> bool:
    if path in RELEASE_FILES:
        return True
    if not distribution:
        return False
    if path.startswith("eviscene_training/"):
        return "training/" + path in RELEASE_FILES
    if path in {"PKG-INFO", "setup.cfg"}:
        return True
    return bool(re.fullmatch(
        r"(?:eviscene|eviscene_training)(?:-[0-9][\w.]*)?\.dist-info/"
        r"(?:METADATA|WHEEL|RECORD|entry_points\.txt|top_level\.txt|licenses/(?:LICENSE|NOTICE))"
        r"|(?:eviscene|eviscene_training)\.egg-info/"
        r"(?:PKG-INFO|SOURCES\.txt|dependency_links\.txt|entry_points\.txt|requires\.txt|top_level\.txt)",
        path,
    ))


def inspect_file(path: str, content: bytes, *, distribution: bool = False) -> list[str]:
    reasons = []
    normalized = path.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or ".." in relative.parts or ":" in normalized:
        reasons.append("invalid release path")
    if not _allowed_path(normalized, distribution=distribution):
        reasons.append("not in release file allowlist")
    parts = tuple(part.lower() for part in relative.parts)
    if (
        any(part in FORBIDDEN_PARTS for part in parts)
        or relative.name.lower().startswith(".env")
        or path.lower().endswith((".pem", ".key"))
    ):
        reasons.append("private/local file")
    if (
        any(path.lower().endswith(suffix) for suffix in ARTIFACT_SUFFIXES)
        and path not in ALLOWED_ASSETS
    ):
        reasons.append("non-code payload")
    if path not in ALLOWED_ASSETS:
        text = content.decode("utf-8", errors="replace")
        reasons.extend(
            name for name, pattern in PATTERNS.items() if pattern.search(text)
        )
    return reasons


def inspect_distributions(root: Path) -> None:
    failures = []
    archives = sorted(root.glob("dist/*.whl")) + sorted(root.glob("dist/*.tar.gz"))
    archives += sorted(root.glob("training/dist/*.whl"))
    archives += sorted(root.glob("training/dist/*.tar.gz"))
    if not archives:
        raise SystemExit("No built distributions to inspect")
    for archive in archives:
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as package:
                entries = [(name, package.read(name)) for name in package.namelist()
                           if not name.endswith("/")]
        else:
            with tarfile.open(archive, "r:gz") as package:
                entries = [(str(PurePosixPath(member.name).relative_to(
                    PurePosixPath(member.name).parts[0])), package.extractfile(member).read())
                    for member in package.getmembers() if member.isfile()]
        for name, content in entries:
            failures.extend(f"{archive.name}: {name}: {reason}"
                            for reason in inspect_file(name, content, distribution=True))
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Distribution hygiene passed ({len(archives)} archives)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distributions", action="store_true",
                        help="Inspect built wheels and source archives instead of the Git index")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.distributions:
        inspect_distributions(root)
        return
    names = list(
        filter(
            None,
            subprocess.check_output(["git", "ls-files", "-z", "--cached"], cwd=root)
            .decode("utf-8")
            .split("\0"),
        )
    )
    # Read staged blobs in one process; never inspect ignored local replacements.
    queries = "".join(":" + name + "\n" for name in names).encode("utf-8")
    batch = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=root,
        input=queries,
        stdout=subprocess.PIPE,
        check=True,
    )
    stream = io.BytesIO(batch.stdout)
    failures = []
    for name in names:
        header = stream.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise SystemExit(f"Cannot inspect staged blob: {name}")
        content = stream.read(int(header[2]))
        if stream.read(1) != b"\n":
            raise SystemExit("Malformed Git blob stream")
        failures.extend(f"{name}: {reason}" for reason in inspect_file(name, content))
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Release hygiene passed ({len(names)} tracked files)")


if __name__ == "__main__":
    main()
