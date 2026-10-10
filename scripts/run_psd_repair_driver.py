"""Run one auditable EviScene PSD repair attempt on the server."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TRAINING_ROOT = REPO_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from eviscene.orchestrator.pipeline import Orchestrator
from eviscene.orchestrator.llm_backend import APIBackend
from eviscene.orchestrator.runtime_events import CaseRuntimeStore
from eviscene.orchestrator.unified_prompts import UNIFIED_JUDGMENT_SYSTEM_PROMPT, UNIFIED_REACT_SYSTEM_PROMPT
from eviscene_training.io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from eviscene_training.psd_repair import (
    PSDModelRoles,
    HintProposal,
    build_psd_attempt_record,
    locate_failure_site,
)
from eviscene_training.psd_repair_runtime import QwenContinuationAdapter
from eviscene_training.psd_repair_verifier import (
    verify_continuation_pair,
    verify_source_rollout_failure,
)


PSD_VERIFICATION_BUNDLE_SCHEMA_VERSION = "ifv-psd-repair-verification-bundle-v1"


def _policy_runtime_kwargs(args, policy_base_url, source_policy):
    # Explicitly bind BOTH policy and visual tools to the attested service.
    # Orchestrator does not inherit llm_base_url into vlm_base_url; leaving it
    # unset silently falls back to the unrelated 8899 endpoint.
    return dict(provider=args.policy_provider, model_name=args.policy_model,
                llm_base_url=policy_base_url, llm_wire_api=args.policy_wire_api,
                vlm_provider=args.policy_provider, vlm_model=args.policy_model,
                vlm_base_url=policy_base_url, vlm_wire_api=args.policy_wire_api,
                image_access_mode="direct_multimodal", validate_startup=True,
                source_access_policy=source_policy)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _capture_from_steps(
    steps: list[Any],
    *,
    failure_stage: str,
) -> dict[str, Any]:
    expected_stage = (
        "psd_teacher_judgment"
        if failure_stage == "unified_judgment"
        else "psd_teacher_repair"
    )
    expected_action = (
        "output" if failure_stage == "unified_judgment" else "tool_call"
    )
    for step in steps:
        if (
            _text(getattr(step, "stage_name", "")) != expected_stage
            or _text(getattr(step, "action_type", "")) != expected_action
        ):
            continue
        capture = _mapping(getattr(step, "metadata", {}).get("policy_token_capture"))
        if _text(capture.get("status")) == "complete":
            return dict(capture)
    return {}


def _capture_from_source_step(
    trace: Mapping[str, Any],
    *,
    source_step_index: int | None,
) -> dict[str, Any]:
    rows = _mapping(trace.get("state")).get("all_steps")
    if (
        not isinstance(rows, list)
        or not isinstance(source_step_index, int)
        or isinstance(source_step_index, bool)
        or not (0 <= source_step_index < len(rows))
    ):
        return {}
    step = rows[source_step_index]
    if not isinstance(step, Mapping):
        return {}
    return dict(
        _mapping(_mapping(step.get("metadata")).get("policy_token_capture"))
    )


def _step_ids(capture: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    prompt = capture.get("prompt_token_ids")
    completion = capture.get("completion_token_ids")
    if not isinstance(prompt, list) or not isinstance(completion, list):
        return [], []
    return [int(item) for item in prompt], [int(item) for item in completion]


def _load_bound_artifact(value: Any, *, root: Path) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    path_value = _text(value)
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    loaded = load_json(path)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"verification artifact must be an object: {path}")
    return loaded


def _load_verification_bundle(path: Path | None) -> dict[int, Mapping[str, Any]]:
    if path is None:
        return {}
    value = load_json(path)
    if value.get("schema_version") != PSD_VERIFICATION_BUNDLE_SCHEMA_VERSION:
        raise ValueError("PSD repair verification bundle schema is invalid")
    rows = value.get("attempts")
    if not isinstance(rows, list):
        raise ValueError("PSD repair verification bundle requires attempts")
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("PSD repair verification attempt must be an object")
        index = row.get("hint_index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("PSD repair verification hint_index is invalid")
        if index in result:
            raise ValueError(f"duplicate PSD repair verification hint_index: {index}")
        result[index] = row
    return result


def _load_policy_serving_attestation(
    args: argparse.Namespace,
) -> tuple[Mapping[str, Any], str, str]:
    profile = load_json(args.policy_serving_profile)
    if profile.get("schema_version") != "ifv-qwen-serving-profile-v1":
        raise ValueError("policy serving profile schema is invalid")
    if _text(profile.get("profile_id")) != _text(args.policy_model):
        raise ValueError("policy model does not match serving profile ID")
    if _text(profile.get("wire_api")) != _text(args.policy_wire_api):
        raise ValueError("policy wire API does not match serving profile")
    if _text(profile.get("model_path")) != _text(args.round_start_checkpoint):
        raise ValueError("round-start checkpoint does not match served model path")
    profile_base_url = _text(profile.get("base_url")).rstrip("/")
    requested_base_url = _text(args.policy_base_url).rstrip("/")
    if requested_base_url and requested_base_url != profile_base_url:
        raise ValueError("policy base URL does not match serving profile")
    checkpoint_manifest_sha256 = sha256_file(
        args.round_start_checkpoint_manifest
    )
    if _text(profile.get("checkpoint_manifest_sha256")).casefold() != (
        checkpoint_manifest_sha256.casefold()
    ):
        raise ValueError(
            "checkpoint manifest does not match the serving profile attestation"
        )
    return profile, profile_base_url, checkpoint_manifest_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True,
                        help="One repair seed JSON from the training candidate bank")
    parser.add_argument("--candidate-offset", type=int)
    parser.add_argument("--candidate-length", type=int)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--semantic-verification", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--public-context", type=Path, required=True)
    parser.add_argument("--private-context", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-provider", default="qwen_local")
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--policy-base-url")
    parser.add_argument("--policy-wire-api", default="chat_completions")
    parser.add_argument("--policy-serving-profile", type=Path, required=True)
    parser.add_argument("--hint-constructor-provider", required=True)
    parser.add_argument("--hint-constructor-model", required=True)
    parser.add_argument("--hint-constructor-base-url")
    parser.add_argument("--hint-constructor-wire-api")
    parser.add_argument(
        "--hint-constructor-thinking-level",
        choices=("minimal", "low", "medium", "high"),
        default="low",
    )
    parser.add_argument("--round-start-checkpoint", required=True)
    parser.add_argument(
        "--round-start-checkpoint-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--hint-count", type=int, default=4,
                        help="Proposal count in single mode; feedback mode always proposes one at a time")
    parser.add_argument("--hint-level", type=int, default=1)
    parser.add_argument("--max-suffix-actions", type=int,
                        help="Optional diagnostic cap; by default retain the Agent's remaining action budget")
    parser.add_argument("--run-student-diagnostic", action="store_true")
    parser.add_argument("--verification-bundle", type=Path)
    parser.add_argument("--train-cases", type=Path, required=True)
    parser.add_argument("--source-access-policy", type=Path, required=True,
                        help="The exact source-access policy used for the original rollout")
    parser.add_argument("--judge-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--skip-auto-judge", action="store_true",
                        help="Persist pending attempts for offline task verification")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generation-retries", type=int, default=1,
                        help="Bounded retries of failed/incomplete generations; completed calls are cached")
    parser.add_argument("--search-mode", choices=("slate", "feedback", "single"), default="feedback")
    parser.add_argument("--policy-temperature", type=float, default=0.7)
    parser.add_argument("--repair-attempts", type=int, default=6)
    parser.add_argument("--proposal-rounds", type=int, default=12)
    parser.add_argument("--search-seconds", type=int, default=1800,
                        help="Soft budget checked between rounds, never cancels an in-flight provider call")
    parser.add_argument("--search-media", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--precomputed-slate-cache", type=Path,
                        help="Immutable first-round Gemini cache; a miss is an error")
    return parser


def _candidate_seed(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read one indexed candidate without copying or hashing the whole bank."""
    offset, length = args.candidate_offset, args.candidate_length
    if offset is None and length is None:
        return load_json(args.candidate), None
    if type(offset) is not int or type(length) is not int or offset < 0 or length <= 0:
        raise ValueError("candidate offset and length must be positive integers")
    with args.candidate.open("rb") as source:
        source.seek(offset)
        raw = source.read(length)
    if len(raw) != length or not raw.endswith(b"\n"):
        raise ValueError("indexed PSD candidate row is incomplete")
    seed = json.loads(raw)
    if not isinstance(seed, dict) or not seed.get("candidate_id"):
        raise ValueError("indexed PSD candidate is invalid")
    return seed, {"path": str(args.candidate.resolve()), "offset": offset,
                  "length": length, "sha256": hashlib.sha256(raw).hexdigest()}


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.search_mode == "slate" and (args.skip_auto_judge or args.verification_bundle):
        raise ValueError("slate search requires full-episode live/cached verification")
    if args.search_mode == "slate":
        from eviscene_training.psd_repair_search import search_lock
        root = args.output_dir.resolve()
        # Acquire before localization/input writes/provider calls. The lock
        # lives next to the output because a new driver requires a new root.
        # Process-owned advisory locks release on crash without PID guessing.
        with search_lock(root.parent / ('.' + root.name + '-writer')):
            return await _run_single(args)
    if args.search_mode == "single":
        return await _run_single(args)
    if args.skip_auto_judge or args.verification_bundle:
        raise ValueError("feedback search requires live/cached per-attempt verification; use single mode for offline bundles")
    from eviscene_training.psd_repair_search import run_search, reusable_localization
    # Bind every original input, not just filesystem paths. The single-attempt
    # driver repeats the source/train/checkpoint gates before any policy call.
    identity = {}
    for key, value in vars(args).items():
        if key == "resume":
            continue
        identity[key] = ({"path": str(value.resolve()), "sha256": sha256_file(value)}
                         if isinstance(value, Path) and key != "output_dir" else
                         str(value.resolve()) if isinstance(value, Path) else value)
    original_context = load_json(args.public_context)

    async def execute_round(directory, feedback, history):
        child = copy.copy(args)
        child.search_mode, child.hint_count = "single", 1
        child.output_dir = directory
        child.resume = (directory / "run-inputs.json").exists()
        if child.resume:
            # Resume an existing round with its exact original localization input
            # (including older rounds which localized in their own cache).
            original = load_json(directory / "run-inputs.json")["identity"].get("semantic_verification")
            child.semantic_verification = Path(original["path"]) if original else None
        else:
            child.semantic_verification = reusable_localization(history, feedback) or args.semantic_verification
        context = {**original_context, "repair_search": feedback}
        inputs = directory.parent.parent / "round-inputs" / directory.name
        context_path = inputs / "public.json"
        media_path = inputs / "media.json"
        media = {"episodes": [{"path": name, "sha256": digest}
                 for row in history for name, digest in row["files"].items()
                 if name.endswith("-teacher.json")]}
        for path, payload in ((context_path, context), (media_path, media)):
            if path.exists():
                if load_json(path) != payload:
                    raise ValueError("search child input changed on resume")
            else:
                write_json(path, payload)
        child.public_context, child.search_media = context_path, media_path
        await _run_single(child)

    return await run_search(root=args.output_dir, identity=identity, resume=args.resume,
        execute_round=execute_round, max_attempts=args.repair_attempts,
        max_proposals=args.proposal_rounds, max_seconds=args.search_seconds)


async def _run_single(args: argparse.Namespace) -> dict[str, Any]:
    policy_profile, policy_base_url, checkpoint_manifest_sha256 = (
        _load_policy_serving_attestation(args)
    )
    trace = load_json(args.trace)
    audit = load_json(args.audit)
    semantic_verification = (
        load_json(args.semantic_verification)
        if args.semantic_verification
        else None
    )
    gold = load_json(args.gold)
    seed, candidate_slice = _candidate_seed(args)
    source_trace_sha256 = (
        str(seed.get("source", {}).get("source_trace_sha256") or "")
        if candidate_slice else sha256_file(args.trace)
    )
    if len(source_trace_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_trace_sha256
    ):
        raise ValueError("indexed PSD source trace lacks its frozen digest")
    if seed.get("source", {}).get("source_access_policy_sha256") != sha256_file(args.source_access_policy):
        raise ValueError("PSD repair source-access policy differs from original rollout")
    public_context = load_json(args.public_context)
    # Available only to the local hint audit, never serialized into its prompt.
    private_context = load_json(args.private_context) if args.private_context else gold
    from eviscene_training.psd_gemini_judge import require_training_case
    require_training_case(trace, args.train_cases)
    from eviscene_training.psd_source_review import source_review_reference
    source_task_review = source_review_reference(seed.get("source", {}))
    source_audit = audit if audit.get("source_trace_canonical_sha256") else None
    # This strict recursive/private-field audit is CPU-heavy for full Agent
    # traces.  Running it inline serialized the nominally concurrent case pool
    # before any task reached provider I/O.  It remains the identical gate;
    # only its execution moves to the bounded asyncio worker pool.
    source_verification = await asyncio.to_thread(
        verify_source_rollout_failure, trace, gold=gold,
        source_task_review=source_task_review, source_audit=source_audit)
    if source_verification["passed"] is not True:
        raise RuntimeError(
            "source no-hint rollout was not explicitly verified as failed: "
            + ",".join(source_verification["reasons"])
        )
    output_dir = args.output_dir
    from eviscene_training.psd_repair_storage import save_bound, load_bound, cached_continuation
    from eviscene_training.psd_repair import _sha
    if args.generation_retries < 0 or args.generation_retries > 3:
        raise ValueError("generation retries must be between 0 and 3")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in {"resume", "skip_auto_judge", "generation_retries"}}
    config["continuation_policy_version"] = "native-contract-and-observed-slate-positions-v6"
    if args.search_mode == 'slate':
        from eviscene_training.psd_slate import SLATE_REVIEW_POLICY
        config['slate_review_policy'] = SLATE_REVIEW_POLICY
    for key in ("trace", "candidate", "audit", "gold", "public_context", "private_context",
                "image", "train_cases", "policy_serving_profile", "round_start_checkpoint_manifest",
                "semantic_verification", "verification_bundle", "source_access_policy", "search_media"):
        value = getattr(args, key)
        if value is not None:
            config[key] = (candidate_slice if key == "candidate" and candidate_slice
                           else {"path": str(value.resolve()), "sha256": (
                               source_trace_sha256 if key == "trace" and candidate_slice
                               else sha256_file(value))})
    config_path = output_dir / "run-inputs.json"
    if args.resume:
        load_bound(config_path, identity=config)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        save_bound(config_path, identity=config, payload={"status": "initialized"})
    # Slate search replays the full episode and chooses observed interventions.
    # Its already-verified failed source must not face a second LLM veto gate.
    if args.search_mode == "slate":
        from eviscene_training.psd_slate_feedback import slate_replay_anchor
        site = slate_replay_anchor(trace)
    else:
        site = locate_failure_site(trace, audit, semantic_verification)
    search_feedback = public_context.get("repair_search", {})
    if args.search_mode != "slate" and search_feedback.get("needs_relocalization"):
        site = None
    if site is None and not args.skip_auto_judge:
        from eviscene_training.psd_gemini_judge import localize_failure
        from eviscene.integrations.gemini import GeminiInteractionsClient
        async with GeminiInteractionsClient(timeout=240, max_retries=2) as client:
            semantic_verification = await localize_failure(client, trace, gold=gold,
                image_path=args.image, model=args.judge_model, cache_dir=output_dir / "judge-cache",
                feedback=search_feedback.get("anchor_feedback") if search_feedback.get("needs_relocalization") else None)
        write_json(output_dir / "semantic-localization.json", semantic_verification)
        site = locate_failure_site(trace, audit, semantic_verification)
    if site is None:
        write_jsonl(output_dir / "repair_candidates.jsonl", [])
        write_jsonl(output_dir / "repair_attempts.jsonl", [])
        result = {"status": "no_recoverable_site", "candidate_count": 0,
                  "accepted_count": 0, "pending_hinted_episode_count": 0}
        write_json(output_dir / "manifest.json", result)
        return result
    if semantic_verification is not None:
        write_json(output_dir / "semantic-localization.json", semantic_verification)
    verification_rows = _load_verification_bundle(args.verification_bundle)
    from eviscene_training.psd_candidate_binding import bind_localized_candidate
    candidate, site = bind_localized_candidate(
        seed, trace, site, source_trace_sha256=source_trace_sha256)
    if args.search_mode != "slate":
        write_jsonl(output_dir / "repair_candidates.jsonl", [candidate])
    runtime_store = CaseRuntimeStore(
        output_dir,
        case_id=_text(_mapping(trace.get("state")).get("runtime_case", {}).get("case_id"))
        or _text(trace.get("image_id")),
        attempt_id=f"psd-proposer-{uuid.uuid4().hex[:12]}",
    )
    runtime_root = runtime_store.root
    # An interrupted run remains inspectable/finalizable after each attempt.
    manifest_path = output_dir / "manifest.json"
    reopening = (manifest_path.exists()
        and load_json(manifest_path).get("status") == "no_recoverable_site"
        and args.search_mode == "slate")
    if reopening:
        historical = output_dir / "pre-checker-feedback-manifest.json"
        if not historical.exists():
            write_json(historical, load_json(manifest_path))
    if not manifest_path.exists() or reopening:
        write_json(output_dir / "manifest.json", {
        "schema_version": "ifv-psd-repair-driver-result-v1", "status": "generating",
        "trace": str(args.trace), "runtime_archive": str(runtime_root),
        "source_task_review": seed.get("source", {}).get("source_task_review"),
        "source_audit": {"path": str(args.audit.resolve()), "sha256": sha256_file(args.audit)} if source_audit else None,
        "train_cases_sha256": sha256_file(args.train_cases),
        "source_access_policy": {"path": str(args.source_access_policy.resolve()),
                                 "sha256": sha256_file(args.source_access_policy)},
        "candidate_count": 0, "accepted_count": 0,
        })
    from eviscene.orchestrator.source_access import SourceAccessPolicy
    source_policy = SourceAccessPolicy.load(args.source_access_policy)
    if not source_policy.active:
        raise ValueError("PSD training repair requires an active source-exclusion policy")
    policy_orchestrator = Orchestrator(**_policy_runtime_kwargs(args, policy_base_url, source_policy))
    hint_constructor_llm = APIBackend(
        provider=args.hint_constructor_provider,
        model_name=args.hint_constructor_model,
        base_url=args.hint_constructor_base_url,
        wire_api=args.hint_constructor_wire_api,
        temperature=0.0,
        max_tokens=2048,
    )
    if args.hint_constructor_provider.strip().lower() == "gemini":
        if hint_constructor_llm.wire_api != "interactions":
            raise ValueError(
                "Gemini PSD hint constructor requires wire_api=interactions"
            )
        if not hint_constructor_llm.api_key:
            raise RuntimeError(
                "Gemini PSD hint constructor requires GEMINI_API_KEY or "
                "GOOGLE_API_KEY"
            )
    model_roles = PSDModelRoles(
        hint_constructor_provider=args.hint_constructor_provider,
        hint_constructor_model=args.hint_constructor_model,
        frozen_self_teacher_provider=args.policy_provider,
        frozen_self_teacher_model=args.policy_model,
        round_start_checkpoint=args.round_start_checkpoint,
        round_start_checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        trainable_student_provider=args.policy_provider,
        trainable_student_model=args.policy_model,
        trainable_student_initial_checkpoint=args.round_start_checkpoint,
    )
    adapter = QwenContinuationAdapter(
        policy_llm=policy_orchestrator.llm,
        hint_constructor_llm=hint_constructor_llm,
        model_roles=model_roles,
        tools=list(policy_orchestrator.all_tools.values()),
        image_path=str(args.image),
        runtime_store=runtime_store,
        tool_cache=policy_orchestrator.tool_cache,
        cacheable_tools=list(policy_orchestrator.cacheable_tools),
        source_access_policy=policy_orchestrator.source_access_policy,
        tool_call_limits=policy_orchestrator.verification_tool_limits,
        source_runtime_store_path=site.runtime_store_path,
        hint_constructor_thinking_level=args.hint_constructor_thinking_level,
        max_output_tokens=policy_orchestrator._stage_output_tokens("UNIFIED_REACT", 8192),
        judgment_max_output_tokens=policy_orchestrator._stage_output_tokens("UNIFIED_JUDGMENT", 8192),
        generation_config={**policy_orchestrator._stage_generation_config(
            "UNIFIED_REACT"), "temperature": args.policy_temperature},
        judgment_generation_config={**policy_orchestrator._stage_generation_config(
            "UNIFIED_JUDGMENT"), "temperature": args.policy_temperature},
        judgment_system_prompt=policy_orchestrator._sp(
            UNIFIED_JUDGMENT_SYSTEM_PROMPT
        ),
        react_system_prompt=policy_orchestrator._sp(UNIFIED_REACT_SYSTEM_PROMPT),
        request_timeout_seconds=policy_orchestrator.stage_request_timeout_seconds,
        tool_timeout_seconds=policy_orchestrator.tool_action_timeout_seconds,
    )
    try:
        async def guard_live_policy():
            import httpx
            from eviscene_training.psd_repair_search import validate_live_model
            key = policy_orchestrator.llm.api_key
            headers = {"Authorization": "Bearer " + key} if key else {}
            async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
                response = await client.get(policy_base_url.rstrip("/") + "/models", headers=headers)
                response.raise_for_status()
                attestation = validate_live_model(policy_profile, response.json())
            write_json(output_dir / "live-serving-check.json", attestation)
        adapter.before_policy_call = guard_live_policy
        if args.search_mode == "slate":
            from eviscene_training.psd_slate_search import run_slate_search
            return await run_slate_search(args=args, adapter=adapter, site=site, candidate=candidate,
                source_failure=source_verification,
                trace=trace, gold=gold, private_context=private_context, source_task_review=source_task_review,
                source_audit=source_audit, source_policy=source_policy, roles=model_roles,
                profile=policy_profile, config=config)
        if args.search_media:
            from eviscene_training.psd_gemini_judge import review_images
            adapter.search_review_images = []
            seen_images = set()
            for media in load_json(args.search_media)["episodes"]:
                path = Path(media["path"])
                if sha256_file(path) != media["sha256"]:
                    raise ValueError("search feedback episode changed")
                images, _ = review_images({"source": trace, "repaired": load_json(path)}, image_path=args.image)
                for position in range(0, len(images), 2):
                    label = images[position]["text"]
                    if label not in seen_images:
                        seen_images.add(label)
                        adapter.search_review_images.extend(images[position:position + 2])
        adapter.proposer_cache_path = output_dir / "proposer-response.json"
        proposals_path = output_dir / "hint-proposals.json"
        if proposals_path.exists():
            proposals = [HintProposal(**row) for row in load_bound(proposals_path, identity=config)]
        else:
            proposals = await adapter.propose_hints(
                failure_site=site, public_trace_context=public_context,
                private_context=private_context, hint_count=args.hint_count, hint_level=args.hint_level)
            from eviscene_training.psd_repair_search import revision_rejection
            filtered = []
            audit_path = output_dir / "proposer-hint-audits.json"
            audits = load_json(audit_path).get("proposals", []) if audit_path.exists() else []
            for hint in proposals:
                reason = revision_rejection(hint.text, search_feedback)
                if reason:
                    audits.append({"passed": False, "reason": reason})
                else:
                    filtered.append(hint)
            proposals = filtered
            write_json(audit_path, {"proposals": audits})
            save_bound(proposals_path, identity=config, payload=[
                {"text": hint.text, "level": hint.level, "proposal_id": hint.proposal_id,
                 "provider": hint.provider, "model": hint.model, "audit": dict(hint.audit)} for hint in proposals])
        records = load_jsonl(output_dir / "repair_attempts.jsonl")
        for index, hint in enumerate(proposals):
            if index < len(records):
                record = records[index]
                if record["source_trace_sha256"] != source_trace_sha256 or record["hint_record"]["proposal_id"] != hint.proposal_id:
                    raise ValueError("saved PSD attempt changed on resume")
                episode = record["continuation"]
                if sha256_file(output_dir / episode["hinted_teacher_episode_trace"]) != episode["hinted_teacher_episode_trace_sha256"]:
                    raise ValueError("saved teacher episode changed on resume")
                continue

            async def generate():
                for retry in range(args.generation_retries + 1):
                    # A new context-ledger namespace prevents request-ID reuse
                    # after process restarts; all failed tries remain archived.
                    adapter.runtime_store = CaseRuntimeStore(output_dir, case_id=candidate["case_id"],
                        attempt_id=f"hint-{index:02d}-{uuid.uuid4().hex[:12]}", resume_from=output_dir)
                    try:
                        return await adapter.run_hinted_episode(failure_site=site, hint=hint,
                            base_trace=trace, max_suffix_actions=args.max_suffix_actions,
                            run_student_diagnostic=args.run_student_diagnostic)
                    except Exception as exc:
                        write_json(output_dir / "generation-last-error.json", {
                            "hint_index": index, "retry": retry, "error_type": type(exc).__name__,
                            "archive": str(adapter.runtime_store.root)})
                        if retry == args.generation_retries:
                            raise

            continuation = await cached_continuation(output_dir / "continuations" / f"hint-{index:02d}.json",
                identity={"run": _sha(config), "hint": hint.audit["hint_sha256"]}, generate=generate)
            runtime_root = Path(continuation.teacher_episode_trace["state"]["runtime_store"]["runtime_path"])
            bound = verification_rows.get(index, {})
            expected_hint_sha = _text(hint.audit.get("hint_sha256"))
            if not expected_hint_sha:
                raise RuntimeError("audited hint is missing its SHA-256 binding")
            if bound and _text(bound.get("hint_sha256")) != expected_hint_sha:
                raise ValueError(
                    f"verification bundle hint hash mismatch at index {index}"
                )
            bundle_root = (
                args.verification_bundle.parent
                if args.verification_bundle is not None
                else Path.cwd()
            )
            local_verification = _load_bound_artifact(
                bound.get("local_verification"),
                root=bundle_root,
            )
            hinted_teacher_episode_trace = _load_bound_artifact(
                bound.get("hinted_teacher_episode_trace"),
                root=bundle_root,
            )
            generated_teacher_trace = continuation.teacher_episode_trace
            if generated_teacher_trace is None:
                raise RuntimeError("repair driver did not produce a teacher episode")
            if (
                hinted_teacher_episode_trace is not None
                and json.dumps(
                    hinted_teacher_episode_trace,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                != json.dumps(
                    generated_teacher_trace,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ):
                raise ValueError(
                    "verification bundle teacher trace differs from generated episode"
                )
            hinted_teacher_episode_trace = generated_teacher_trace
            unhinted_student_episode_trace = _load_bound_artifact(
                bound.get("unhinted_student_episode_trace"),
                root=bundle_root,
            )
            verification, artifacts = verify_continuation_pair(
                base_trace=trace,
                source_trace_sha256=source_trace_sha256,
                teacher_steps=continuation.teacher_steps,
                student_steps=continuation.student_steps,
                hinted_teacher_episode_trace=hinted_teacher_episode_trace,
                unhinted_student_episode_trace=unhinted_student_episode_trace,
                gold=gold,
                local_verification=local_verification,
                repair_step_id=site.step_id,
                hint_sha256=expected_hint_sha,
                source_access_policy=source_policy,
                source_task_review=source_task_review,
                source_audit=source_audit,
            )
            teacher_capture = _capture_from_steps(
                continuation.teacher_steps,
                failure_stage=site.stage,
            )
            teacher_prompt_ids, completion_ids = _step_ids(teacher_capture)
            if not teacher_prompt_ids or not completion_ids:
                raise RuntimeError(
                    "hinted teacher action lacks prompt/completion token IDs"
                )
            source_capture = _capture_from_source_step(
                trace,
                source_step_index=site.source_step_index,
            )
            student_prompt_ids, _student_completion_ids = _step_ids(source_capture)
            if not student_prompt_ids:
                raise RuntimeError(
                    "source failure step lacks the no-hint student prompt token IDs"
                )
            record = build_psd_attempt_record(
                candidate_id=candidate["candidate_id"],
                case_id=candidate["case_id"],
                episode_id=candidate["episode_id"],
                failure_site=site,
                hint=hint,
                model_roles=model_roles,
                verification=verification,
                local_verification=local_verification,
                student_prompt_ids=student_prompt_ids,
                teacher_prompt_ids=teacher_prompt_ids,
                completion_ids=completion_ids,
                teacher_token_capture=teacher_capture,
                source_trace_sha256=source_trace_sha256,
            )
            record["continuation"] = {
                "teacher_complete": continuation.teacher_complete,
                "student_complete": continuation.student_complete,
                "stop_reason": continuation.stop_reason,
                "teacher_step_count": len(continuation.teacher_steps),
                "student_step_count": len(continuation.student_steps),
                "hinted_teacher_episode_verified": artifacts[
                    "hinted_teacher_episode_verified"
                ],
                "unhinted_student_diagnostic": artifacts[
                    "unhinted_student_diagnostic"
                ],
                "unhinted_student_affects_acceptance": False,
            }
            if 248056 in student_prompt_ids or 248056 in teacher_prompt_ids:
                from eviscene_training.psd_media import media_from_archive
                processor_path = str(policy_profile.get("engine_model_path") or args.round_start_checkpoint)
                source_media = media_from_archive(
                    {"runtime_store_path": site.runtime_store_path,
                     "context_request_id": site.context_request_id},
                    processor_path=processor_path, output_dir=output_dir / "media",
                    prompt_ids=student_prompt_ids,
                )
                matching_steps = [step for step in continuation.teacher_steps
                                  if step.metadata.get("policy_token_capture", {}).get("prompt_token_ids") == teacher_prompt_ids]
                if len(matching_steps) != 1:
                    raise ValueError("cannot bind exact teacher request to PSD media")
                teacher_media = media_from_archive(
                    {"runtime_store_path": str(runtime_root),
                     "context_request_id": matching_steps[0].metadata["context_request_id"]},
                    processor_path=processor_path, output_dir=output_dir / "media",
                    prompt_ids=teacher_prompt_ids,
                )
                if source_media != teacher_media:
                    raise ValueError("teacher and student visual conditioning differs")
                record["psd_media"] = source_media
            episode_dir = output_dir / "episodes"
            episode_dir.mkdir(parents=True, exist_ok=True)
            teacher_episode_name = f"hint-{index:02d}-teacher.json"
            write_json(
                episode_dir / teacher_episode_name,
                hinted_teacher_episode_trace,
            )
            record["continuation"]["hinted_teacher_episode_trace"] = (
                f"episodes/{teacher_episode_name}"
            )
            record["continuation"]["hinted_teacher_episode_trace_sha256"] = (
                sha256_file(episode_dir / teacher_episode_name)
            )
            records.append(record)
            # Preserve every completed provider/tool attempt immediately.  A
            # resumed offline finalization never repeats successful calls.
            write_jsonl(output_dir / "repair_attempts.jsonl", records)
            manifest = load_json(output_dir / "manifest.json")
            manifest.update({"candidate_count": len(records),
                "accepted_count": sum(row.get("accepted") is True for row in records),
                "pending_hinted_episode_count": sum(not row.get("local_verification") for row in records)})
            write_json(output_dir / "manifest.json", manifest)
        write_jsonl(output_dir / "repair_attempts.jsonl", records)
        result = {
            "schema_version": "ifv-psd-repair-driver-result-v1",
            "status": "generated" if records else "no_admissible_hints",
            "train_cases_sha256": sha256_file(args.train_cases),
            "source_access_policy": {"path": str(args.source_access_policy.resolve()),
                                     "sha256": sha256_file(args.source_access_policy)},
            "trace": str(args.trace),
            "source_task_review": seed.get("source", {}).get("source_task_review"),
            "source_audit": {"path": str(args.audit.resolve()), "sha256": sha256_file(args.audit)} if source_audit else None,
            "runtime_archive": str(runtime_root),
            "policy_serving_attestation": {
                "profile": str(args.policy_serving_profile),
                "profile_sha256": sha256_file(args.policy_serving_profile),
                "profile_id": _text(policy_profile.get("profile_id")),
                "model_path": _text(policy_profile.get("model_path")),
                "checkpoint_manifest": str(args.round_start_checkpoint_manifest),
                "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            },
            "hint_constructor": {
                "provider": args.hint_constructor_provider,
                "model": args.hint_constructor_model,
                "wire_api": hint_constructor_llm.wire_api,
                "thinking_level": args.hint_constructor_thinking_level,
                "supplies_training_distribution": False,
            },
            "candidate_count": len(records),
            "accepted_count": sum(1 for row in records if row.get("accepted") is True),
            "pending_hinted_episode_count": sum(
                1
                for row in records
                if not row.get("local_verification")
            ),
            "artifacts": {"repair_attempts": "repair_attempts.jsonl"},
        }
        write_json(output_dir / "manifest.json", result)
        if not args.skip_auto_judge and records:
            from eviscene_training.psd_gemini_judge import judge_run
            from eviscene.integrations.gemini import GeminiInteractionsClient
            async with GeminiInteractionsClient(timeout=240, max_retries=2) as client:
                result["task_judge"] = await judge_run(run_dir=output_dir,
                    source_trace_path=args.trace, gold_path=args.gold, image_path=args.image,
                    train_cases_path=args.train_cases, model=args.judge_model, client=client)
            result.update(load_json(output_dir / "manifest.json"))
        return result
    finally:
        await hint_constructor_llm.aclose()
        await policy_orchestrator.aclose()


def main() -> None:
    args = _parser().parse_args()
    print(json.dumps(asyncio.run(_run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
