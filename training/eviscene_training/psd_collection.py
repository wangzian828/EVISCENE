"""Explicit grouped sampling and upstream failed-episode selection for PSD."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from eviscene.eval.rollout import rollout_specs
from eviscene.environment import get_env


def repairable_terminal_model_failure(trace):
    """Recognize a recorded unusable model action, not a transport failure.

    It is only a repair SEED: preservation still needs full task success. The
    native error receipt must agree, and an intact finite captured prefix must
    exist. This does not claim to have diagnosed the parser/model root cause.
    """
    import math
    import re
    from pathlib import Path
    from .io import load_json
    error = str(trace.get('error') or '')
    if (trace.get('termination') != 'error' or not re.fullmatch(
            r'RuntimeError: Chat Completions returned an unusable response: '
            r'choices=1, finish_reason=(?:tool_calls|stop|length), content_chars=0, '
            r'reasoning_chars=\d+, reasoning_fallback_requested=False', error)):
        return None
    state = trace.get('state', {})
    runtime_name = state.get('runtime_store', {}).get('runtime_path')
    if not runtime_name:
        return None
    receipts = [load_json(path) for path in (Path(runtime_name)/'context').glob('*.json')]
    if not any(r.get('error') == error and r.get('status') == 'error'
            and str(r.get('stage', '')).lower() in {'unified_react', 'unified_judgment'} for r in receipts):
        return None
    captures = 0
    for step in state.get('all_steps', []):
        metadata = step.get('metadata', {})
        if metadata.get('deterministic_segment_boundary'):
            continue
        cap = metadata.get('policy_token_capture')
        if cap is None:
            continue
        values = cap.get('completion_logprobs', [])
        if (cap.get('status') != 'complete' or not values
                or len(values) != len(cap.get('completion_token_ids', []))
                or any(not math.isfinite(value) for value in values)):
            return None
        captures += 1
    return 'unusable_policy_output_with_captured_prefix' if captures else None


def require_completed_collection(run_dir, manifest):
    """Finished sampling is not the same as every policy outcome succeeding.

    Legacy successful banks keep their existing contract. A bank ending with
    errors is admissible only with full published slot coverage and a durable
    completed retry ledger/result for every slot. Per-trace candidate quality
    gates remain independent; this does NOT admit failed traces to preservation.
    """
    from pathlib import Path
    import hashlib
    from .io import load_json, load_jsonl
    from .psd_repair_storage import load_bound
    status = manifest.get('status')
    if status == 'completed':
        recovery = manifest.get('agent', {}).get('psd_sampling', {}).get('infrastructure_retry')
        if recovery is not None and recovery.get('unresolved'):
            raise ValueError('Completed manifest still has unresolved infrastructure slots')
        return {'passed': True, 'native_status': status}
    if status != 'completed_with_errors':
        raise ValueError('PSD source collection is not completed')
    run_dir = Path(run_dir).resolve()
    rows = load_jsonl(run_dir/'run_results.jsonl')
    recovery = manifest.get('agent', {}).get('psd_sampling', {}).get('infrastructure_retry')
    if not recovery or not rows:
        raise ValueError('Error-bearing collection lacks durable infrastructure attestation')
    ids = list(dict.fromkeys(row['case_id'] for row in rows))
    verify_collection(rows, case_ids=ids, manifest=manifest, expected_rollouts=8)
    benchmark = manifest.get('benchmark', {})
    if benchmark.get('sample_count') != len(ids) or benchmark.get('episode_count') != len(rows):
        raise ValueError('Error-bearing collection has incomplete declared coverage')
    for row in rows:
        slot = run_dir/'psd-infrastructure-attempts'/hashlib.sha256(row['episode_id'].encode()).hexdigest()
        saved = load_json(slot/'retry-state.json')
        identity = saved['identity']
        state = load_bound(slot/'retry-state.json', identity=identity)
        inputs = identity.get('inputs', {})
        if (inputs.get('episode_id') != row['episode_id'] or inputs.get('case_id') != row['case_id']
                or inputs.get('sampling_seed') != row['sampling_seed']
                or inputs.get('model') != manifest['agent']['model']
                or not state.get('attempts') or state['attempts'][-1]['status'] != 'completed'):
            raise ValueError('Error-bearing collection contains unbound or unresolved slots')
        result = load_bound(slot/'result.json', identity=identity)
        trace_path = (run_dir/str(row.get('trace_path') or '')).resolve()
        trace_path.relative_to(run_dir)
        if not trace_path.is_file() or load_json(trace_path) != result:
            raise ValueError('Error-bearing collection has missing or inconsistent canonical trace')
        # Numerical transport failures require infrastructure recovery.
        prefix = 'RuntimeError: HTTP 400 Bad Request for ' + str(inputs.get('base_url', '')).rstrip('/') + '/chat/completions: '
        error = str(result.get('error') or '')
        if error.startswith(prefix):
            import json
            from .psd_infrastructure_retry import is_nonfinite_serialization_response
            try:
                payload = json.loads(error[len(prefix):])
            except ValueError:
                payload = None
            if is_nonfinite_serialization_response(400, payload):
                raise ValueError('Misclassified numerical failure requires infrastructure recovery')
    return {'passed': True, 'native_status': status, 'episodes': len(rows),
            'model_failures_retained': sum(row.get('status') == 'error' for row in rows)}


def require_token_capture_environment(environ):
    if str(get_env("EVISCENE_CAPTURE_POLICY_TOKENS", "", environ=environ)).strip().lower() not in {"1", "true", "yes", "on"}:
        raise ValueError("PSD source collection requires native policy token/logprob capture before dispatch")
    if str(get_env("EVISCENE_POLICY_TOPK", "20", environ=environ)).strip() != "20":
        raise ValueError("PSD source collection requires top-20 policy capture")


def verify_collection(rows, *, case_ids, manifest, expected_rollouts):
    """Count slots, not just case IDs; do not accept 400 rows as 400 x 8."""
    if type(expected_rollouts) is not int or expected_rollouts < 1:
        raise ValueError("PSD rollouts per case must be a positive integer")
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("PSD collection case IDs must be nonempty and unique")
    agent = manifest.get("agent", {})
    recovery = agent.get("psd_sampling", {}).get("infrastructure_retry")
    if recovery is not None and (recovery.get("unresolved") or recovery.get("slots") != len(rows)
                                 or recovery.get("selection_by_answer") is not False):
        raise ValueError("PSD collection has unresolved or unbound infrastructure sampling slots")
    if agent.get("rollouts_per_case") != expected_rollouts:
        raise ValueError("PSD collection sampling budget differs from requested group size")
    if expected_rollouts == 8 and (agent.get("psd_sampling", {}).get("temperature") != .7
            or not agent.get("psd_sampling", {}).get("collector_sha256")):
        raise ValueError("PSD collection lacks explicit published sampling provenance")
    if type(agent.get("base_sampling_seed")) is not int or not manifest.get("git_commit") or not agent.get("model"):
        raise ValueError("PSD collection lacks sampling/policy identity")
    expected = rollout_specs([{} for _ in case_ids], [SimpleNamespace(case_id=c) for c in case_ids],
        rollouts_per_case=expected_rollouts, base_sampling_seed=agent["base_sampling_seed"],
        policy_revision=manifest["git_commit"], model=agent["model"],
        episode_namespace=agent.get("episode_namespace"))
    # The native run_result_record serializes slot identity but not group_size.
    # Group size remains bound by the manifest, hashed prompt_group_id, and the
    # exact complete slot set below; do not rewrite historical result rows.
    keys = ("case_id", "episode_id", "prompt_group_id", "rollout_index", "sampling_seed")
    expected_by_id = {r["episode_id"]: r for r in expected}
    if len(expected_by_id) != len(expected):
        raise ValueError("PSD generated episode ID collision")
    seen = set()
    for row in rows:
        episode = row.get("episode_id")
        if episode in seen or episode not in expected_by_id:
            raise ValueError("PSD collection contains duplicate or unexpected episodes")
        if any(row.get(k) != expected_by_id[episode][k] for k in keys):
            raise ValueError("PSD collection slot seed/index/policy binding mismatch")
        if "group_size" in row and row["group_size"] != expected_rollouts:
            raise ValueError("PSD collection explicit group size mismatch")
        seen.add(episode)
    if seen != set(expected_by_id):
        raise ValueError("PSD collection is incomplete; missing sampling slots")
    return {"passed": True, "cases": len(case_ids), "episodes": len(rows),
        "rollouts_per_case": expected_rollouts, "selection_by_success": False,
        "group_size_attestation": "manifest_and_hashed_group_and_complete_slot_set",
        "rows_without_redundant_group_size": sum("group_size" not in row for row in rows)}


def select_task_repair_sources(candidates, *, load_trace):
    """Upstream load_collection_rollout defaults to the longest failed episode.

    Keep every input in the source bank, but spend one task's repair search
    budget on one source episode. Preservation collection is not filtered here.
    """
    groups = defaultdict(list)
    for row in candidates:
        if row.get("class") != "repair_seed" or not row.get("case_id") or not row.get("episode_id"):
            raise ValueError("invalid PSD repair seed")
        groups[row["case_id"]].append(row)
    selected, records = [], []
    for case, group in sorted(groups.items()):
        if len({r["episode_id"] for r in group}) != len(group):
            raise ValueError("duplicate repair source episode")
        ranked = []
        for row in group:
            trace = load_trace(row)
            steps = trace.get("state", {}).get("all_steps")
            if not isinstance(steps, list) or not steps:
                raise ValueError("repair source lacks recorded steps")
            ranked.append((len(steps), row))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["episode_id"]))
        count, chosen = ranked[0]
        selected.append(chosen)
        records.append({"case_id": case, "selected_episode_id": chosen["episode_id"],
            "source_steps": count, "failed_episodes": len(group),
            "unselected_episode_ids": [r["episode_id"] for _, r in ranked[1:]]})
    return selected, {"policy": "longest_failed_episode_per_task; episode_id_tie_break",
        "original_candidates": len(candidates), "selected_tasks": len(selected), "tasks": records}


def _measure_repair_source(job):
    path = Path(job["trace_path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != job["trace_sha256"]:
        raise ValueError("PSD source changed before task selection")
    trace = json.loads(raw)
    steps = trace.get("state", {}).get("all_steps") if isinstance(trace, dict) else None
    if not isinstance(steps, list) or not steps:
        raise ValueError("repair source lacks recorded steps")
    return {**job, "source_steps": len(steps)}


def _atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_task_repair_selection_parallel(*, candidates_path, run_dir, output_dir,
                                         workers=16):
    """Select one longest failed trace per task without loading the source bank.

    Candidate rows can contain hundreds of thousands of token IDs.  Keep only
    small identities while measuring traces, then copy the chosen original rows
    byte-for-byte in deterministic task order.
    """
    from .io import require_new_or_empty, sha256_file, write_json

    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("PSD task-selection workers must be in [1, 32]")
    candidates_path = Path(candidates_path).resolve()
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir)
    require_new_or_empty(output_dir)
    jobs, seen_ids = [], set()
    with candidates_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = row.get("candidate_id")
            case_id, episode_id = row.get("case_id"), row.get("episode_id")
            binding = row.get("source") if isinstance(row.get("source"), dict) else {}
            if (row.get("class") != "repair_seed" or not candidate_id or not case_id
                    or not episode_id or candidate_id in seen_ids):
                raise ValueError(f"invalid PSD repair seed at line {line_number}")
            seen_ids.add(candidate_id)
            trace = (run_dir / str(binding.get("source_trace_path") or "")).resolve()
            try:
                trace.relative_to(run_dir)
            except ValueError as exc:
                raise ValueError("repair source escaped rollout directory") from exc
            digest = binding.get("source_trace_sha256")
            if not trace.is_file() or not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("repair source binding is incomplete")
            jobs.append({"candidate_id": candidate_id, "case_id": case_id,
                "episode_id": episode_id, "trace_path": str(trace),
                "trace_sha256": digest})

    progress = output_dir / "selection-progress.json"
    _atomic_json(progress, {"status": "measuring", "completed": 0,
        "total": len(jobs), "workers": workers})
    measured = []
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
    try:
        for completed, result in enumerate(
                executor.map(_measure_repair_source, jobs, chunksize=1), 1):
            measured.append(result)
            if completed % 10 == 0 or completed == len(jobs):
                _atomic_json(progress, {"status": "measuring", "completed": completed,
                    "total": len(jobs), "workers": workers})
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    groups = defaultdict(list)
    for row in measured:
        groups[row["case_id"]].append(row)
    selected_ids, records = [], []
    for case_id, group in sorted(groups.items()):
        if len({row["episode_id"] for row in group}) != len(group):
            raise ValueError("duplicate repair source episode")
        ranked = sorted(group, key=lambda row: (-row["source_steps"], row["episode_id"]))
        chosen = ranked[0]
        selected_ids.append(chosen["candidate_id"])
        records.append({"case_id": case_id,
            "selected_episode_id": chosen["episode_id"],
            "source_steps": chosen["source_steps"],
            "failed_episodes": len(group),
            "unselected_episode_ids": [row["episode_id"] for row in ranked[1:]]})
    selection = {"policy": "longest_failed_episode_per_task; episode_id_tie_break",
        "original_candidates": len(jobs), "selected_tasks": len(selected_ids),
        "tasks": records}

    selected = set(selected_ids)
    shards = output_dir / ".rows"
    found = set()
    with candidates_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = row.get("candidate_id")
            if candidate_id not in selected:
                continue
            shard = shards / f"{candidate_id.replace(':', '-')}.jsonl"
            shard.parent.mkdir(parents=True, exist_ok=True)
            shard.write_text(line if line.endswith("\n") else line + "\n", encoding="utf-8")
            found.add(candidate_id)
    if found != selected:
        raise ValueError("selected PSD repair source disappeared during streaming copy")
    selected_path = output_dir / "selected_candidates.jsonl"
    with selected_path.open("wb") as destination:
        for candidate_id in selected_ids:
            shard = shards / f"{candidate_id.replace(':', '-')}.jsonl"
            with shard.open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    shutil.rmtree(shards)
    write_json(output_dir / "selection.json", selection)
    manifest = {"schema_version": "ifv-psd-task-source-selection-v1",
        "source_candidates": str(candidates_path),
        "source_candidates_sha256": sha256_file(candidates_path),
        "selected_candidates": "selected_candidates.jsonl",
        "selected_candidates_sha256": sha256_file(selected_path),
        "selection": "selection.json", "workers": workers,
        "original_candidates": len(jobs), "selected_tasks": len(selected_ids)}
    write_json(output_dir / "manifest.json", manifest)
    _atomic_json(progress, {"status": "completed", "completed": len(jobs),
        "total": len(jobs), "workers": workers,
        "selected_tasks": len(selected_ids)})
    return manifest


def build_task_repair_selection_from_rewards(*, candidates_path, rewards_path,
                                              rollout_gate_path, output_dir):
    """Select one longest failed episode per task without reopening source traces.

    The reviewed-only postprocess already recorded the exact policy step IDs and
    the rollout gate binds that reward file.  Stream the large candidate file
    once, retaining only the current winner for each case on disk.
    """
    from .io import load_json, load_jsonl, require_new_or_empty, sha256_file, write_json

    candidates_path = Path(candidates_path).resolve()
    rewards_path = Path(rewards_path).resolve()
    rollout_gate_path = Path(rollout_gate_path).resolve()
    output_dir = Path(output_dir)
    require_new_or_empty(output_dir)
    gate = load_json(rollout_gate_path)
    expected_rewards = str(gate.get("run", {}).get("post_rollout_rewards_sha256") or "")
    actual_rewards = sha256_file(rewards_path)
    if gate.get("passed") is not True or expected_rewards != actual_rewards:
        raise ValueError("reward-only task selection requires its passed rollout gate")

    step_counts = {}
    for row in load_jsonl(rewards_path):
        episode_id = str(row.get("episode_id") or "")
        step_ids = row.get("step_ids")
        if not episode_id or not isinstance(step_ids, list) or episode_id in step_counts:
            raise ValueError("postprocess reward step identities are invalid")
        step_counts[episode_id] = len(step_ids)

    rows_dir = output_dir / ".rows"
    rows_dir.mkdir(parents=True)
    chosen = {}
    seen_ids = set()
    original_candidates = 0
    with candidates_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = row.get("candidate_id")
            case_id, episode_id = row.get("case_id"), row.get("episode_id")
            if (row.get("class") != "repair_seed" or not candidate_id or not case_id
                    or not episode_id or candidate_id in seen_ids):
                raise ValueError(f"invalid PSD repair seed at line {line_number}")
            seen_ids.add(candidate_id)
            if episode_id not in step_counts:
                raise ValueError("repair candidate lacks bound postprocess reward")
            original_candidates += 1
            rank = (-step_counts[episode_id], episode_id)
            previous = chosen.get(case_id)
            if previous is not None and previous["rank"] <= rank:
                continue
            shard = rows_dir / (hashlib.sha256(case_id.encode()).hexdigest() + ".jsonl")
            shard.write_text(line if line.endswith("\n") else line + "\n", encoding="utf-8")
            chosen[case_id] = {"rank": rank, "candidate_id": candidate_id,
                "episode_id": episode_id, "source_steps": step_counts[episode_id],
                "path": shard}

    selected_path = output_dir / "selected_candidates.jsonl"
    records = []
    with selected_path.open("wb") as destination:
        for case_id in sorted(chosen):
            item = chosen[case_id]
            with item["path"].open("rb") as selected:
                shutil.copyfileobj(selected, destination, length=1024 * 1024)
            records.append({"case_id": case_id,
                "selected_episode_id": item["episode_id"],
                "source_steps": item["source_steps"]})
    shutil.rmtree(rows_dir)
    selection = {"policy": "longest_failed_episode_per_task_from_gate_bound_step_ids",
        "original_candidates": original_candidates, "selected_tasks": len(chosen),
        "tasks": records}
    write_json(output_dir / "selection.json", selection)
    stat = candidates_path.stat()
    manifest = {"schema_version": "ifv-psd-task-source-selection-v2",
        "source_candidates": {"path": str(candidates_path), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns},
        "rewards": {"path": str(rewards_path), "sha256": actual_rewards},
        "rollout_gate": {"path": str(rollout_gate_path),
            "sha256": sha256_file(rollout_gate_path)},
        "selected_candidates": "selected_candidates.jsonl",
        "selected_candidates_sha256": sha256_file(selected_path),
        "selection": "selection.json", "trace_files_reopened": 0,
        "original_candidates": original_candidates, "selected_tasks": len(chosen)}
    write_json(output_dir / "manifest.json", manifest)
    return manifest
