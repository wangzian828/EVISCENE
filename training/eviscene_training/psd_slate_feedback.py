"""Checker-grounded slate search, without a second localizer admission gate.

The private checker's free-form explanation can contain reference-only facts.
Project only its status and literal observations, never that explanation/gold.
The full original review remains private and immutable for provenance.
"""
from __future__ import annotations

import copy

from .psd_repair import FailureSite, _sha, project_policy_steps

POLICY = "checker-grounded-no-localizer-veto-v1"
CRITERIA = (
    "A correct verdict alone is insufficient. Material claims and evidence "
    "attributions must be supported by actual observations. A failed tool or "
    "empty search is not proof of a factual conclusion. Cited passages locate "
    "the checker's concern; inspect their surrounding observations rather than "
    "treating a passage as a reference answer. Propose procedural corrections, "
    "not a verdict or new facts."
)


def checker_feedback(review, trace, *, repaired=False, hints=None, source_failure=None,
                     withhold_invalid_citations=False):
    """Copy only independently validated public quotes, including hint audits."""
    from .psd_gemini_judge import trace_steps, _quote_in_step
    from .psd_slate import decision_map
    decision = (review or {}).get("decision", {})
    # Source admission combines semantic, deterministic verdict and structural
    # checks. A semantic pass must not veto an independently verified failure.
    independently_failed = (not repaired and isinstance(source_failure, dict)
        and source_failure.get("passed") is True
        and (source_failure.get("result_correct") is False
             or source_failure.get("structural_failure") is True))
    if decision and decision.get("status") != "fail" and not independently_failed:
        raise ValueError("slate repair feedback requires a failed checker decision")
    steps = {row["index"]: row for row in trace_steps(trace)}
    positions = decision_map(trace)
    # The verifier may quote the actual injected hint, a public intervention,
    # at the corresponding decision. Reproduce exactly its review annotation.
    if repaired:
        for position, hint in (hints or {}).items():
            if int(position) in positions:
                steps[positions[int(position)]]["injected_procedural_hint"] = hint
    citations = []
    withheld = 0
    for row in decision.get("evidence", []):
        index, quote = row.get("step_index"), row.get("quote")
        if (row.get("trace") != ("repaired" if repaired else "source")
                or type(index) is not int or index not in steps
                or not isinstance(quote, str) or not quote.strip()
                or not _quote_in_step(quote, steps[index], decode_json_strings=(
                    review.get("evidence_encoding") == "ifv-psd-literal-json-strings-v1"))):
            if withhold_invalid_citations:
                withheld += 1
                continue
            raise ValueError("checker feedback must quote an actual public observation")
        citations.append({"step_index": index, "quote": quote})
    return {"policy": POLICY, "status": "fail", "criteria": CRITERIA,
            "cited_observations": citations,
            "invalid_citations_withheld": withheld,
            "semantic_reviewer_status": decision.get("status", "not_available"),
            "independent_task_check_failed": independently_failed,
            "reference_explanation_withheld": True}


def diagnostic_position(feedback, trace):
    """A cited decision is a diagnostic anchor, not an exclusive edit boundary."""
    from .psd_slate import decision_map
    positions = decision_map(trace)
    if not positions:
        raise ValueError("failed source has no replayable native decisions")
    cited = {row["step_index"] for row in feedback["cited_observations"]}
    direct = [p for p, i in positions.items() if i in cited]
    return min(direct) if direct else max(positions)


def slate_replay_anchor(trace):
    """Mechanical replay origin; never claim that the first action was wrong."""
    rows = project_policy_steps(trace)
    index, row = next((i, row) for i, row in enumerate(rows)
                      if row["action_type"] in {"tool_call", "output"})
    return FailureSite(step_index=index, **{key: row[key] for key in (
        "step_id", "stage", "example_type", "policy_input", "policy_action",
        "source_step_index", "context_request_id", "runtime_store_path")},
        localization_kind="mechanical_full_episode_replay_not_error_localization")


def load_slate_state(path, *, identity):
    """Read the historical numeric-hint-key hash only when it verifies exactly.

    Older writers hashed integer keys before JSON changed them to strings.
    This is NOT permission to ignore a failed checksum or modify artifacts.
    """
    from .io import load_json
    from .psd_repair_storage import load_bound
    try:
        return load_bound(path, identity=identity)
    except ValueError:
        saved = load_json(path)
        if (saved.get("identity") != identity
                or identity.get("version") != "slate-search-v3-observed-positions"):
            raise
        legacy = copy.deepcopy(saved.get("payload", {}))
        try:
            for row in legacy["rounds"]:
                hints = row["hints"]
                if any(not isinstance(k, str) or str(int(k)) != k for k in hints):
                    raise ValueError("noncanonical legacy hint position")
                row["hints"] = {int(k): v for k, v in hints.items()}
        except (KeyError, TypeError, ValueError):
            raise ValueError("PSD slate checkpoint binding changed") from None
        if _sha(legacy) != saved.get("payload_sha256"):
            raise ValueError("PSD slate checkpoint binding changed") from None
        return saved["payload"]
