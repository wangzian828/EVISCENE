"""Bind driver localization to the candidate IDs consumed by assembly."""
from dataclasses import replace

from .psd_candidates import _candidate_id, _policy_steps, _trace_case_id


def bind_localized_candidate(candidate, trace, site, *, source_trace_sha256):
    source = candidate.get("source", {})
    if source.get("source_trace_sha256") != source_trace_sha256:
        raise ValueError("repair candidate source trace hash mismatch")
    if candidate.get("case_id") != _trace_case_id(trace):
        raise ValueError("repair candidate case mismatch")
    if candidate.get("class") != "repair_seed":
        raise ValueError("repair candidate must be a repair_seed")
    identity = dict(kind="repair", case_id=candidate["case_id"], episode_id=candidate["episode_id"],
                    source_trace_sha256=source_trace_sha256)
    if candidate.get("candidate_id") != _candidate_id(**identity, step_id=candidate["repair_site"]["step_id"]):
        raise ValueError("repair candidate ID is not bound to its source")
    steps = _policy_steps(trace, episode_id=candidate["episode_id"])
    selected = [step for step in steps if step["source_step_index"] == site.source_step_index]
    if len(selected) != 1 or selected[0]["stage"] != site.stage:
        raise ValueError("localized site is not a source policy step")
    selected = selected[0]
    if selected["rollout_token_capture"].get("status") != "complete":
        raise ValueError("localized candidate lacks exact token capture")
    localized = dict(candidate, repair_site=selected,
                     candidate_id=_candidate_id(**identity, step_id=selected["step_id"]),
                     parent_candidate_id=candidate["candidate_id"], candidate_status="localized",
                     failure_localization={**candidate.get("failure_localization", {}),
                         "repair_anchor_source_step_index": selected["source_step_index"],
                         "repair_anchor_step_index": steps.index(selected),
                         "repair_anchor_step_id": selected["step_id"]})
    return localized, replace(site, step_id=selected["step_id"], example_type=selected["example_type"])
