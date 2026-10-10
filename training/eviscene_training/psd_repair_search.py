"""Bounded verifier-feedback search; private judge prose never reaches the actor.

EviScene has one task image and one selected source decision, not BFCL user-turn
slates. A locally verified hint is preserved verbatim while additional advice
may address observed downstream failures at that SAME anchor. No downstream
patches or hint-bearing prefixes become independent on-policy training seeds.
"""
from __future__ import annotations

import contextlib
import copy
import os
import time
from pathlib import Path

from .io import load_json, load_jsonl, sha256_file, write_jsonl
from .psd_repair import _sha
from .psd_repair_storage import load_bound, save_bound
from .psd_gemini_judge import CHECKS, _atomic_json, trace_steps

VERSION = "ifv-psd-feedback-search-v2"
MAX_SEARCH_ARTIFACT_BYTES = 2 * 1024 ** 3


@contextlib.contextmanager
def search_lock(root):
    """Process-owned advisory lock: crashes release it; no stale PID guessing."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "search.lock").open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def public_attempt_feedback(record, episode):
    """Allowlist checker bits + raw policy observations; never copy explanations.

    Judge explanations may disclose private reference facts. Even evidence
    strings are omitted here: the full observed steps are already available.
    """
    review = record.get("local_verification", {}).get("review", {})
    return {
        "hint": record["hint_record"]["text"],
        "selected_source_step": record["repair_site"]["source_step_index"],
        "checker": {name: review.get(name) for name in (*CHECKS, "anchor_matches", "episode_complete")
                    if type(review.get(name)) is bool},
        "earliest_error_step": review.get("earliest_error_step", -1),
        "task_passed": record.get("hinted_episode_pass") is True,
        "strict_trace_passed": record.get("hinted_strict_trace_audit_pass") is True,
        "accepted": record.get("accepted") is True,
        "repaired_steps": trace_steps(episode),
    }


def revision_context(history):
    attempts = [attempt for row in history for attempt in row["feedback"]["attempts"]]
    locked = ""
    for attempt in attempts:
        checks = attempt["checker"]
        if all(checks.get(k) is True for k in (
                "source_error_confirmed", "repaired_at_selected_step", "procedural_hint", "anchor_matches")):
            # Preserve already working advice, including earlier locked clauses.
            if not locked or attempt["hint"] == locked or attempt["hint"].startswith(locked + "\n"):
                locked = attempt["hint"]
    last = attempts[-1] if attempts else {}
    previous_rounds = []
    attempt_index = 0
    for row in history:
        feedback = copy.deepcopy(row["feedback"])
        compacted = []
        for attempt in feedback["attempts"]:
            attempt_index += 1
            if attempt_index < len(attempts):
                # Keep every public checker decision and hint, but do not
                # duplicate all prior full trajectories in every later prompt.
                # The latest observed rerun remains complete for diagnosis.
                attempt.pop("repaired_steps", None)
            compacted.append(attempt)
        feedback["attempts"] = compacted
        previous_rounds.append(feedback)
    return {
        "previous_rounds": previous_rounds,
        "locked_hint": locked,
        "excluded_hints": [attempt["hint"] for attempt in attempts],
        "needs_relocalization": bool(last and not locked and
            last.get("checker", {}).get("anchor_matches") is False),
        "anchor_feedback": {k: last[k] for k in ("selected_source_step", "earliest_error_step", "checker") if k in last},
        "feedback_compaction": {
            "full_repaired_steps_kept": int(bool(attempts)),
            "prior_repaired_steps_omitted": max(0, len(attempts) - 1),
        },
    }


def revision_rejection(hint, feedback):
    """Enforce the search decisions locally, not just through prompt wording."""
    if hint.strip() in {text.strip() for text in feedback.get("excluded_hints", [])}:
        return "repeated_completed_hint"
    locked = feedback.get("locked_hint", "")
    if locked and not hint.startswith(locked + "\n"):
        return "changed_verified_hint"
    return ""


def reusable_localization(history, feedback):
    """Keep the same source anchor unless the actual checker disputes it."""
    if feedback.get("needs_relocalization"):
        return None
    for row in reversed(history):
        path = Path(row["directory"]) / "semantic-localization.json"
        if path.is_file():
            if str(path.resolve()) in row["files"] and sha256_file(path) != row["files"][str(path.resolve())]:
                raise ValueError("saved localization changed")
            return path
    return None


def validate_live_model(profile, models):
    """An unchanged alias is not proof that it still serves the round's weights."""
    rows = [row for row in models.get("data", []) if row.get("id") == profile["profile_id"]]
    if len(rows) != 1:
        raise ValueError("live PSD model alias missing or ambiguous")
    row = rows[0]
    expected = profile["model_path"]
    if not row.get("root") or Path(row["root"]).resolve() != Path(expected).resolve():
        raise ValueError("live PSD service no longer serves the frozen checkpoint")
    context_card = row
    base = profile.get("engine_model_path") or expected
    if Path(base).resolve() != Path(expected).resolve():
        # vLLM adapter cards use root=adapter and parent=base alias. Their
        # context limit is inherited from the base card, not repeated here.
        parents = [card for card in models.get("data", []) if card.get("id") == row.get("parent")]
        if len(parents) != 1 or not parents[0].get("root") or Path(parents[0]["root"]).resolve() != Path(base).resolve():
            raise ValueError("live PSD adapter parent differs from frozen base")
        context_card = parents[0]
    if context_card.get("max_model_len") != profile["context_length"]:
        raise ValueError("live PSD context cap differs from frozen profile")
    return {"model_id": row["id"], "root": row["root"], "parent": row.get("parent"),
            "max_model_len": context_card["max_model_len"]}


def inspect_round(directory):
    """Snapshot only a finalized round. Pending votes must not drive search."""
    manifest = load_json(directory / "manifest.json")
    records = load_jsonl(directory / "repair_attempts.jsonl")
    candidates = load_jsonl(directory / "repair_candidates.jsonl")
    files = {str(path.resolve()): sha256_file(path) for path in (
        directory / "manifest.json", directory / "repair_attempts.jsonl", directory / "repair_candidates.jsonl")}
    attempts = []
    for record in records:
        episode = directory / record["continuation"]["hinted_teacher_episode_trace"]
        if sha256_file(episode) != record["continuation"]["hinted_teacher_episode_trace_sha256"]:
            raise ValueError("search teacher episode hash changed")
        files[str(episode.resolve())] = sha256_file(episode)
        if not record.get("local_verification"):
            return None
        attempts.append(public_attempt_feedback(record, load_json(episode)))
        record["continuation"]["hinted_teacher_episode_trace"] = str(episode.resolve())
    if manifest.get("pending_hinted_episode_count", 0):
        return None
    audits_path = directory / "proposer-hint-audits.json"
    audits = load_json(audits_path).get("proposals", []) if audits_path.exists() else []
    # Keep only procedural audit codes, not raw rejected hints that could leak an answer.
    rejected = [row.get("reason", "hint_audit_failed") for row in audits if not row.get("passed")]
    for path in (audits_path, directory / "hint-proposals.json", directory / "proposer-response.json",
                 directory / "live-serving-check.json", directory / "semantic-localization.json"):
        if path.exists():
            files[str(path.resolve())] = sha256_file(path)
    return {"directory": str(directory.resolve()), "files": files,
            "records": records, "candidates": candidates,
            "feedback": {"attempts": attempts, "rejected_proposal_reasons": rejected,
                         "status": manifest.get("status")},
            "no_further_hint": not records and not audits}


def verify_snapshot(row):
    for name, digest in row["files"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError("completed search round changed: " + name)


async def run_search(*, root, identity, resume, execute_round, max_attempts=6,
                     max_proposals=12, max_seconds=3600, clock=time.monotonic):
    """Finite state machine. Callback generates <=1 hint, then runs and judges it.

    Time limit is checked between complete rounds, never kills a paid in-flight
    call. Execution exceptions/pending judges pause, not reject or consume fresh
    alternatives. Resumption re-enters that same child's provider caches.
    """
    if type(max_attempts) is not int or not 1 <= max_attempts <= 32:
        raise ValueError("repair attempt budget must be 1..32")
    if type(max_proposals) is not int or not max_attempts <= max_proposals <= 64:
        raise ValueError("proposal budget must be between attempt budget and 64")
    if not 1 <= max_seconds <= 86400:
        raise ValueError("search wall budget must be 1..86400 seconds")
    identity = {"version": VERSION, "inputs": identity, "max_attempts": max_attempts,
                "max_proposals": max_proposals, "max_seconds": max_seconds}
    marker = root / "search-state.json"
    with search_lock(root):
        if resume:
            state = load_bound(marker, identity=identity)
        else:
            if marker.exists():
                raise ValueError("search already exists; use --resume")
            state = {"status": "searching", "elapsed_seconds": 0.0, "rounds": []}
            save_bound(marker, identity=identity, payload=state)
        for row in state["rounds"]:
            verify_snapshot(row)
        started, previous_elapsed = clock(), state["elapsed_seconds"]

        def persist(status):
            state.update(status=status, elapsed_seconds=previous_elapsed + clock() - started)
            save_bound(marker, identity=identity, payload=state)
            records = [r for item in state["rounds"] for r in item["records"]]
            candidates = {r["candidate_id"]: r for item in state["rounds"] for r in item["candidates"]}
            write_jsonl(root / "repair_candidates.jsonl", list(candidates.values()))
            write_jsonl(root / "repair_attempts.jsonl", records)
            result = {"schema_version": VERSION, "status": status,
                "converged": any(r.get("accepted") is True for r in records),
                "candidate_count": len(records), "accepted_count": sum(r.get("accepted") is True for r in records),
                "proposal_rounds": len(state["rounds"]), "attempt_budget": max_attempts,
                "proposal_budget": max_proposals, "elapsed_seconds": state["elapsed_seconds"],
                "stop_reason": status, "rounds": [r["directory"] for r in state["rounds"]],
                "execution_error": state.get("execution_error"),
                "training_distribution_provider": "frozen_self_teacher_only"}
            _atomic_json(root / "manifest.json", result)
            return result

        terminal = {"converged", "attempt_budget_exhausted", "proposal_budget_exhausted",
                    "time_budget_exhausted", "no_further_grounded_hint"}
        if state["status"] in terminal:
            return persist(state["status"])
        state.pop("execution_error", None)
        while True:
            history = state["rounds"]
            if any(r.get("accepted") is True for row in history for r in row["records"]):
                return persist("converged")
            if sum(len(row["records"]) for row in history) >= max_attempts:
                return persist("attempt_budget_exhausted")
            if len(history) >= max_proposals:
                return persist("proposal_budget_exhausted")
            if previous_elapsed + clock() - started >= max_seconds:
                return persist("time_budget_exhausted")
            if history and history[-1]["no_further_hint"]:
                return persist("no_further_grounded_hint")
            # Shared-filesystem free space is NOT the user's quota. Bound this
            # diagnostic's own growth conservatively and preserve, never delete,
            # evidence on exhaustion. One in-flight round may exceed the cap.
            used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
            if used >= MAX_SEARCH_ARTIFACT_BYTES:
                return persist("paused_storage_budget")
            index = len(history)
            directory = root / "rounds" / f"round-{index:02d}"
            feedback = revision_context(history)
            request_path = root / "requests" / f"round-{index:02d}.json"
            binding = {"search": _sha(identity), "round": index}
            if request_path.exists():
                if load_bound(request_path, identity=binding) != feedback:
                    raise ValueError("search feedback changed on resume")
            else:
                save_bound(request_path, identity=binding, payload=feedback)
            persist("searching")
            try:
                await execute_round(directory, feedback, history)
                row = inspect_round(directory)
            except Exception as exc:
                state["execution_error"] = {"round": index, "type": type(exc).__name__}
                return persist("paused_execution_error")
            if row is None:
                return persist("paused_pending_verifier")
            if len(row["records"]) > 1:
                raise ValueError("feedback search must judge after each individual continuation")
            history.append(row)
            persist("searching")
