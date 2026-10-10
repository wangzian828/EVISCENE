"""Sequential, cached full-episode slate search on one frozen policy snapshot.

Case-level concurrency belongs to the caller. Attempts within a case depend on
actual verifier feedback; speculative hints or unverified targets never train.
"""
from __future__ import annotations

from eviscene.environment import get_env

from functools import partial
from pathlib import Path
import time
import uuid

from .io import load_json, load_jsonl, write_json, write_jsonl, sha256_file
from .psd_repair import _sha
from .psd_repair_storage import bound_artifact_paths, load_bound, save_bound
from .psd_slate import (capture_target, decision_map, propose_slate, review_slate,
                        assemble_slate_attempts, SlateProposalRejected,
                        SlateReviewRejected)
from .psd_slate_feedback import (POLICY, checker_feedback, diagnostic_position,
                                load_slate_state)


SUCCESS_STOP_POLICY = "first_verified_full_episode_pass"


def _case_ledger(root: Path, stem: str) -> Path:
    """Keep historical plain ledgers readable; new ledgers are gzip-only."""
    plain, compressed = root / f"{stem}.jsonl", root / f"{stem}.jsonl.gz"
    if plain.exists() and compressed.exists():
        raise ValueError(f"duplicate PSD case ledger formats: {stem}")
    return plain if plain.exists() else compressed


async def propose_with_budget(*, state, round_index, budget, persist, judge, kwargs):
    """Reserve accepted proposals before generation; only invalid hints advance.

    Transport/checker errors propagate instead of being labeled model failures.
    Completed rejected proposal responses remain in the normal SHA-bound cache.
    """
    from .psd_repair import HintProposal
    pending = state.get('pending_proposal')
    if pending is not None:
        if pending['round_index'] != round_index:
            raise ValueError('PSD pending proposal belongs to another rerun')
        return {int(k):HintProposal(**v) for k,v in pending['hints'].items()}, pending['provenance']
    attempts = state.setdefault('proposals', [])
    while len(attempts) < budget:
        rejected = sum(r['round_index']==round_index and r['status']=='rejected' for r in attempts)
        options = dict(kwargs)
        if rejected:
            options['proposal_feedback']={'rejected_proposals':rejected,'reason':'invalid_or_nonprocedural_slate'}
        try:
            hints, provenance = await propose_slate(judge, **options)
        except SlateProposalRejected as error:
            attempts.append({'round_index':round_index,'status':'rejected','provenance':error.provenance})
            persist()
            continue
        attempts.append({'round_index':round_index,'status':'accepted' if hints else 'no_hint',
                         'provenance':provenance})
        state['pending_proposal']={'round_index':round_index,'hints':{str(k):vars(v) for k,v in hints.items()},
                                   'provenance':provenance}
        persist()
        return hints, provenance
    return None, None


async def run_slate_search(*, args, adapter, site, candidate, trace, gold, private_context,
                           source_task_review, source_audit, source_policy, roles, profile, config,
                           source_failure=None):
    from eviscene.integrations.gemini import GeminiInteractionsClient
    from eviscene.orchestrator.runtime_events import CaseRuntimeStore
    from .psd_gemini_judge import review_images, trace_steps
    import httpx
    if args.max_suffix_actions is not None or args.run_student_diagnostic:
        raise ValueError("published slate mode retains native action budget and needs no student resampling")
    if not 1 <= args.repair_attempts <= 6:
        raise ValueError("published slate mode allows at most six complete repair reruns")
    proposal_budget = getattr(args, 'proposal_rounds', 12)
    if type(proposal_budget) is not int or not args.repair_attempts <= proposal_budget <= 64:
        raise ValueError('PSD proposal budget must cover the rerun budget and be at most 64')
    root = args.output_dir
    marker = root / "slate-state.json"
    identity = {"version": "slate-search-v3-observed-positions", "inputs": _sha(config), 'proposal_budget':proposal_budget}
    state = load_slate_state(marker, identity=identity) if marker.exists() else {
        "rounds": [], "elapsed_seconds": 0.0, "status": "repairing"}
    # An infrastructure-only round has no semantic failure record.  It must
    # stay resumable instead of being treated as a case terminal.  Keep the
    # accepted Gemini proposal and use a fresh Qwen episode directory on the
    # next owner pass; the original ledger remains immutable evidence.
    if state["status"] in {"infrastructure_budget_exhausted", "unresolved_infrastructure"}:
        state["status"] = "repairing"
        state.pop("output_files", None)
        save_bound(marker, identity=identity, payload=state)
    for row in state["rounds"]:
        for raw, digest in row["files"].items():
            path = Path(raw).resolve()
            path.relative_to(root.resolve())
            if sha256_file(path) != digest:
                raise ValueError("PSD slate completed round changed")
    if state["status"] == "no_further_grounded_hint":
        # Preserve the old terminal artifact, then continue the SAME budget
        # and accepted empty proposal. No completed rerun is discarded.
        historical = root / "pre-checker-feedback-slate-state.json"
        if not historical.exists():
            write_json(historical, load_json(marker))
        state["status"] = "repairing"
        state.pop("output_files", None)
    # A task-level wall-clock cutoff is not an admission rule.  Reopen a
    # persisted time/strict-audit stop while its six complete-rerun budget is
    # still available.  Attempt/proposal caps remain hard and are never reset.
    if (state["status"] in {"time_budget_exhausted", "paused_strict_audit_failed"}
            and len(state["rounds"]) < args.repair_attempts):
        state["status"] = "repairing"
        state.pop("pending_proposal", None)
        state.pop("output_files", None)
    if state["status"] != "repairing":
        audit_slate_search(root)
        return load_json(root / "manifest.json")
    initial_state, _ = adapter._initial_runtime_state(base_trace=trace, failure_site=site)
    precomputed_cache = getattr(args, "precomputed_slate_cache", None)
    if precomputed_cache is not None and not precomputed_cache.is_dir():
        raise FileNotFoundError("frozen first-round PSD Gemini cache is missing")
    # Precomputation used the source review alone. Preserve that exact public
    # first-round packet so the already-paid response is consumed verbatim.
    source_feedback = checker_feedback(source_task_review, trace,
        source_failure=None if precomputed_cache is not None else source_failure,
        withhold_invalid_citations=True)
    initial = (diagnostic_position(source_feedback, trace) if source_task_review else
               24 if site.stage == "unified_judgment" else initial_state.action_count)
    state["active_feedback_policy"] = POLICY
    from .psd_repair import FailureSite, project_policy_steps
    first = next(row for row in project_policy_steps(trace) if row["action_type"] in {"tool_call", "output"})
    replay_site = FailureSite(step_index=0, **{k: first[k] for k in (
        "step_id", "stage", "example_type", "policy_input", "policy_action", "source_step_index",
        "context_request_id", "runtime_store_path")})
    started = time.monotonic()
    elapsed_before = state["elapsed_seconds"]
    attempts_path = _case_ledger(root, "repair_attempts")
    candidates_path = _case_ledger(root, "repair_candidates")
    records = load_jsonl(attempts_path) if attempts_path.exists() else []
    candidates = load_jsonl(candidates_path) if candidates_path.exists() else []
    source_hash = (candidate.get("source", {}).get("source_trace_sha256")
                   if getattr(args, "candidate_offset", None) is not None
                   else sha256_file(args.trace))
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("indexed PSD source trace digest is missing")
    token_url = str(profile["base_url"]).rstrip("/").removesuffix("/v1") + "/tokenize"
    key = adapter.policy_llm.api_key
    headers = {"Authorization": "Bearer " + key} if key else {}
    async with httpx.AsyncClient(timeout=240, trust_env=False, headers=headers) as token_client, \
            GeminiInteractionsClient(timeout=240, max_retries=2) as judge:
        async def tokenize(body):
            response = await token_client.post(token_url, json=body)
            response.raise_for_status()
            return response.json()["tokens"]
        capture = partial(capture_target, tokenize=tokenize, model=args.policy_model)
        while len(state["rounds"]) < args.repair_attempts:
            number = len(state["rounds"])
            previous, passing, failed, observed = {}, [], initial, trace
            reviewed_rounds = [row for row in state["rounds"] if "review" in row]
            if reviewed_rounds:
                last = reviewed_rounds[-1]
                if "result_path" in last:
                    observed = load_bound(Path(last["result_path"]),
                        identity=last["result_identity"])["teacher_episode_trace"]
                else:
                    # Compatibility with already committed historical rounds.
                    observed = load_json(Path(last["episode_path"]))
                previous = {int(k): v for k, v in last["hints"].items()}
                passing = last["review"]["decision"]["passing_positions"]
                failed = last["review"]["decision"]["failed_position"]
                if failed < 0:
                    # A strict assembly rejection can follow a semantic pass,
                    # whose review has no failed position.  Reuse the latest
                    # public failing position as the grounded anchor; never
                    # expose the private strict-audit reason to the proposer.
                    for prior in reversed(reviewed_rounds[:-1]):
                        candidate_failed = prior["review"]["decision"].get("failed_position", -1)
                        if candidate_failed >= 0:
                            failed = candidate_failed
                            break
                    if failed < 0:
                        failed = initial
            images, _ = review_images({"source": trace, **({"repaired": observed} if number else {})}, image_path=args.image)
            if reviewed_rounds:
                feedback_review = last["review"]
                # A strict assembly failure can follow a semantic pass.  The
                # private strict reason must stay private, so ground the next
                # proposer on the latest public failing review instead.
                if feedback_review["decision"].get("status") != "fail":
                    for prior in reversed(reviewed_rounds[:-1]):
                        if prior["review"]["decision"].get("status") == "fail":
                            feedback_review = prior["review"]
                            break
                if feedback_review["decision"].get("status") == "fail":
                    feedback = checker_feedback(feedback_review, observed,
                        repaired=True, hints=previous, withhold_invalid_citations=True)
                else:
                    # A strict rejection has no public explanation: reuse verified public feedback,
                    # not the private rejection reason or the unusable review cache.
                    feedback = source_feedback
                    previous, passing, failed, observed = {}, [], initial, trace
            else:
                feedback = source_feedback
            public = {"source_steps": trace_steps(trace), "observed_steps": trace_steps(observed),
                "decision_map": decision_map(observed), "checker_feedback": feedback}
            hints, provenance = await propose_with_budget(state=state,round_index=number,
                budget=proposal_budget,persist=lambda:save_bound(marker,identity=identity,payload=state),judge=judge,
                kwargs=dict(public_context=public, previous=previous, passing_positions=passing,
                    failed_position=failed, model=args.hint_constructor_model,
                    cache_dir=precomputed_cache if number == 0 and precomputed_cache is not None
                    else root / "judge-cache",
                    cache_only=number == 0 and precomputed_cache is not None,
                    private_context=private_context,images=images))
            if hints is None:
                state['status']='proposal_budget_exhausted'
                break
            # ``number`` is the semantic rerun count.  An infrastructure-only
            # retry must not consume that count, so its filesystem slot has a
            # separate monotone cursor and can never reopen an old ledger.
            slot = int(state.get("next_round_slot", len(state["rounds"])))
            directory = root / "slate-rounds" / f"{slot:02d}"
            state["next_round_slot"] = slot + 1
            # Commit the slot before creating any provider/episode artifact so
            # an owner crash cannot reopen the same filesystem ledger.
            save_bound(marker, identity=identity, payload=state)
            directory.mkdir(parents=True, exist_ok=True)
            retry_inputs = {"inputs": _sha(config),
                "hints": {str(k): h.text for k, h in hints.items()}}
            async def generate():
                import copy
                from .psd_infrastructure_retry import retry_episode, guard_policy_backend
                from .psd_repair_storage import continuation_payload, continuation_from_payload
                guard_policy_backend(adapter.policy_llm, max_request_retries=int(
                    get_env("EVISCENE_PSD_POLICY_REQUEST_RETRIES", "0")))
                async def generate_attempt(attempt_directory):
                    fresh = copy.copy(adapter)
                    fresh.runtime_store = CaseRuntimeStore(attempt_directory, case_id=candidate["case_id"],
                        attempt_id=f"slate-{number:02d}-{uuid.uuid4().hex[:12]}", resume_from=None)
                    fresh.tool_cache = None
                    # Restart at the first decision with the SAME hint slate.
                    # Infra failures do not consume another hint/review round.
                    result = await fresh.run_hinted_episode(failure_site=replay_site, hint=None,
                        base_trace=trace, hints_by_action=hints, capture_local_target=capture)
                    return continuation_payload(result)
                payload = await retry_episode(root=directory / "infrastructure-attempts",
                    identity=retry_inputs,
                    generate=generate_attempt)
                return continuation_from_payload(payload)
            result_path = directory / "infrastructure-attempts" / "result.json"
            try:
                # retry_episode is the one authoritative continuation cache.
                # Do not write the same 80+ MB payload again as continuation.json.
                continuation = await generate()
            except Exception as error:
                from .psd_infrastructure_retry import InfrastructureRetriesExhausted
                if not isinstance(error, InfrastructureRetriesExhausted):
                    raise
                # Preserve exhausted round evidence but keep the case resumable;
                # an incomplete round must not produce a terminal case receipt.
                state["status"] = "unresolved_infrastructure"
                save_bound(marker, identity=identity, payload=state)
                break
            # Persist the retry layer's complete binding (including its fixed
            # version and budget), rather than reconstructing it on resume.
            result_identity = load_json(result_path)["identity"]
            episode = continuation.teacher_episode_trace
            try:
                review = await review_slate(judge, source=trace, episode=episode, gold=gold,
                    image_path=args.image, model=args.judge_model, cache_dir=root / "judge-cache",
                    targets=continuation.local_targets)
            except SlateReviewRejected as error:
                # A completed rerun consumes budget even if its review is inadmissible.
                # Keep evidence and resume from verified public feedback.
                round_state = {"hints": {str(k): h.text for k, h in hints.items()},
                    "review_rejection": {"status": "rejected_contract", "reason": error.reason},
                    "proposal": provenance, "result_path": str(result_path),
                    "result_identity": result_identity,
                    "feedback_policy": POLICY, "checker_feedback_sha256": _sha(feedback),
                    "plain_retry": not hints,
                    "files": {str(p): sha256_file(p) for p in bound_artifact_paths(result_path)}}
                state["rounds"].append(round_state)
                state["elapsed_seconds"] = elapsed_before + time.monotonic() - started
                state.pop("pending_proposal", None)
                save_bound(marker, identity=identity, payload=state)
                continue
            write_json(directory / "review.json", review)
            round_state = {"hints": {str(k): h.text for k, h in hints.items()}, "review": review,
                "proposal": provenance, "result_path": str(result_path),
                "result_identity": result_identity,
                "feedback_policy": POLICY, "checker_feedback_sha256": _sha(feedback),
                "plain_retry": not hints,
                "files": {str(p): sha256_file(p) for p in (
                    [directory / "review.json"] + bound_artifact_paths(result_path))}}
            state["rounds"].append(round_state)
            state["elapsed_seconds"] = elapsed_before + time.monotonic() - started
            # Generation and judge are already independently cached. Advance the
            # search cursor only after materialization, so a crash here resumes
            # this SAME round without re-sampling or losing a passing result.
            if review["decision"]["status"] == "pass":
                from .psd_media import media_from_archive, validate_media
                for target in continuation.local_targets:
                    if 248056 in target["teacher_prompt_ids"]:
                        media = media_from_archive(target, processor_path=str(profile.get("engine_model_path")
                            or args.round_start_checkpoint), output_dir=root / "media",
                            prompt_ids=target["teacher_prompt_ids"])
                        validate_media(media, target["student_prompt_ids"])
                        target["psd_media"] = media
                new_candidates, new_records = assemble_slate_attempts(seed=candidate, source=trace,
                    source_hash=source_hash, episode=episode, targets=continuation.local_targets,
                    review=review, gold=gold, source_task_review=source_task_review, source_audit=source_audit,
                    source_policy=source_policy, roles=roles)
                candidates.extend(new_candidates)
                records.extend(new_records)
                if not continuation.local_targets:
                    state["status"] = "passed_without_intervention"
                    break
                if new_records and all(r["accepted"] for r in new_records):
                    state["status"] = "converged"
                    break
                # Keep the strict gate.  Continue with the next grounded
                # proposal/re-run while the six-rerun budget remains.
                state["status"] = "repairing"
                state.pop("pending_proposal", None)
                save_bound(marker, identity=identity, payload=state)
                continue
            if review["decision"]["status"] == "unresolved":
                state["status"] = "paused_unresolved_task_review"
                break
            state.pop('pending_proposal', None)
            save_bound(marker, identity=identity, payload=state)
        else:
            state["status"] = "attempt_budget_exhausted"
    write_jsonl(candidates_path, candidates)
    write_jsonl(attempts_path, records)
    state["elapsed_seconds"] = elapsed_before + time.monotonic() - started
    result = {"schema_version": "ifv-psd-slate-search-v1", "status": state["status"],
        "complete_reruns": len(state["rounds"]), "candidate_count": len(records),
        'proposal_count':len(state.get('proposals',[])),
        'rejected_proposal_count':sum(r['status']=='rejected' for r in state.get('proposals',[])),
        "accepted_count": sum(r["accepted"] for r in records), "elapsed_seconds": state["elapsed_seconds"],
        "training_started": False, "per_target_weight": 1.0, "feedback_policy": POLICY,
        # Attempts within one case are deliberately serial. The first
        # full-episode verifier pass is terminal; later alternatives must not
        # be sampled merely to rank multiple successful trajectories.
        "success_stop_policy": SUCCESS_STOP_POLICY}
    write_json(root / "manifest.json", result)
    # The bound state is the terminal commit marker. Write every output first:
    # a crash must not leave a completed state pointing at a missing manifest.
    state["output_files"] = {str(path): sha256_file(path)
        for path in (candidates_path, attempts_path, root / "manifest.json")}
    save_bound(marker, identity=identity, payload=state)
    return result


def audit_slate_search(root):
    marker = root / "slate-state.json"
    saved = load_json(marker)
    state = load_slate_state(marker, identity=saved["identity"])
    for row in [*state["rounds"], {"files": state.get("output_files", {})}]:
        for raw, digest in row["files"].items():
            path = Path(raw).resolve()
            path.relative_to(root.resolve())
            if sha256_file(path) != digest:
                raise ValueError("PSD slate artifact changed after completion")
    result = load_json(root / "manifest.json")
    policy = result.get("success_stop_policy")
    if (result["status"] != state["status"]
            or result["complete_reruns"] != len(state["rounds"])
            # Historical terminal packages predate this explicit receipt.
            # New writers always emit it; keep old immutable packages readable.
            or (policy is not None and policy != SUCCESS_STOP_POLICY)):
        raise ValueError("PSD slate manifest differs from checkpoint")
    return {"passed": not state["status"].startswith("paused_") and state["status"] != "repairing"}
