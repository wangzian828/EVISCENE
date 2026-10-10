"""Answer-blind completeness gate for source resampling.

An incorrect but complete report is a source outcome, not a retry condition.
Historical empty/failed tools remain observations when the episode completes.
This gate does not call a judge, inspect gold, or modify the frozen Agent.
"""
import math
from collections.abc import Mapping

VERSION = 'ifv-psd-complete-source-v1'


def source_completion_failure(trace):
    if not isinstance(trace, Mapping):
        return 'source_result_not_object'
    if trace.get('error') or trace.get('termination') != 'success':
        return 'source_episode_not_completed'
    state = trace.get('state')
    if not isinstance(state, Mapping):
        return 'source_state_missing'
    steps = state.get('all_steps')
    if not isinstance(steps, list) or not steps or any(not isinstance(s, Mapping) for s in steps):
        return 'source_steps_missing_or_malformed'
    judgment = trace.get('judgment') or state.get('judgment')
    from eviscene.orchestrator.investigation_models import DiscrepancyJudgment
    from pydantic import ValidationError
    try:
        parsed = DiscrepancyJudgment.model_validate(judgment)
    except ValidationError:
        return 'source_report_contract_invalid'
    if parsed.fact_check_report is None:
        return 'source_final_report_missing'
    if parsed.verdict != trace.get('verdict'):
        return 'source_verdict_record_mismatch'
    final = [s for s in steps if s.get('stage') == 'unified_judgment' and s.get('action_type') == 'output']
    if not final or not final[-1].get('output'):
        return 'source_final_action_missing'
    captures = 0
    for step in steps:
        metadata = step.get('metadata') or {}
        if not isinstance(metadata, Mapping):
            return 'source_metadata_invalid'
        if (metadata.get('deterministic_segment_boundary')
                or step.get('stage') not in ('unified_react', 'unified_judgment')
                or step.get('action_type') not in ('tool_call', 'output')):
            continue
        cap = metadata.get('policy_token_capture') or {}
        if not isinstance(cap, Mapping):
            return 'source_token_capture_invalid'
        ids, probabilities = cap.get('completion_token_ids'), cap.get('completion_logprobs')
        if (cap.get('status') != 'complete' or not cap.get('prompt_token_ids')
                or not isinstance(ids, list) or not ids or not isinstance(probabilities, list)
                or len(ids) != len(probabilities)):
            return 'source_token_capture_incomplete'
        if any(not isinstance(p, (int, float)) or not math.isfinite(p) for p in probabilities):
            return 'source_token_capture_nonfinite'
        captures += 1
    return None if captures else 'source_no_policy_capture'
