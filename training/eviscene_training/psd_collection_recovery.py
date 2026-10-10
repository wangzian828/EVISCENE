"""Import persisted sampling slots after a verified retry-classification defect.

Never alter the original run. Successful and ordinary failed policy outcomes
are reused exactly. Only the exact provider numerical error, independently
present in the native request receipt, is reclassified for remaining retries.
Already spent attempts remain charged. Completed evidence-bound five-attempt
slots may be carried forward, but this never creates another budget extension.
"""
import copy
import hashlib
import json
import os
from pathlib import Path

from .io import load_json, sha256_file
from .psd_infrastructure_retry import VERSION, MAX_ATTEMPTS, is_nonfinite_serialization_response, recovery_attempt_budget
from .psd_repair_storage import load_bound, save_bound


def numerical_failure_evidence(trace, *, inputs, source_slot):
    error = str(trace.get('error') or '')
    prefix = 'RuntimeError: HTTP 400 Bad Request for ' + inputs['base_url'].rstrip('/') + '/chat/completions: '
    if not error.startswith(prefix):
        return None
    try:
        payload = json.loads(error[len(prefix):])
    except ValueError:
        return None
    if not is_nonfinite_serialization_response(400, payload):
        return None
    runtime = Path(trace.get('state', {}).get('runtime_store', {}).get('runtime_path', '')).resolve()
    runtime.relative_to(Path(source_slot).resolve())
    receipts = []
    for path in (runtime/'context').glob('*.json'):
        receipt = load_json(path)
        if (receipt.get('status') == 'error' and receipt.get('error') == error
                and receipt.get('model') == inputs['model']
                and str(receipt.get('stage', '')).lower() in {'unified_react', 'unified_judgment'}):
            receipts.append({'path': str(path), 'sha256': sha256_file(path)})
    if not receipts:
        raise ValueError('Numerical failure lacks an independent native request receipt')
    return {'reason': 'model_http_400_nonfinite_serialization', 'request_receipts': receipts}


def import_prior_slots(*, source, destination, episode_ids, seeds, numerical_recovery=None,
                       require_complete=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if len(episode_ids) != len(seeds) or len(set(episode_ids)) != len(episode_ids):
        raise ValueError('Invalid destination slot identities')
    expected = dict(zip(episode_ids, seeds))
    output = destination/'psd-infrastructure-attempts'
    if output.exists():
        raise ValueError('Recovery import requires an empty destination ledger')
    work = []
    # Prevalidate every old record before creating the new ledger bank.
    for marker in sorted((source/'psd-infrastructure-attempts').glob('*/retry-state.json')):
        saved = load_json(marker)
        identity = saved['identity']
        if type(identity.get('max_attempts')) is not int or identity['max_attempts'] not in (MAX_ATTEMPTS, 5):
            raise ValueError('Retry budget differs; cannot reset it during recovery')
        state = load_bound(marker, identity=identity)
        inputs = identity['inputs']
        episode = inputs['episode_id']
        if episode not in expected or inputs['sampling_seed'] != expected[episode]:
            raise ValueError('Recovered episode/seed differs from formal sampling plan')
        if marker.parent.name != hashlib.sha256(episode.encode()).hexdigest():
            raise ValueError('Original retry directory has wrong slot identity')
        carried_allowance = None
        if identity['max_attempts'] == 5:
            # An already resolved extended bank is a cached outcome, not a new
            # service intervention. Preserve its original evidence and budget.
            allowance_path = marker.parent/'recovery-allowance.json'
            if not allowance_path.is_file() or recovery_attempt_budget(marker.parent, inputs) != 5:
                raise ValueError('Extended retry bank lacks a valid original recovery allowance')
            carried_allowance = load_json(allowance_path)
            original = Path(carried_allowance['source_ledger'])
            original_identity = load_json(original)['identity']
            original_attempts = load_bound(original, identity=original_identity)['attempts']
            attempts = state.get('attempts', [])
            if (len(attempts) not in (4, 5) or attempts[:3] != original_attempts
                    or [a.get('index') for a in attempts] != list(range(1, len(attempts)+1))
                    or attempts[-1].get('status') != 'completed'
                    or any(a.get('status') not in {'infrastructure_failed', 'trajectory_failed'} for a in attempts[:-1])):
                raise ValueError('Extended bank must be completed with all original attempts charged')
        result_path = marker.parent/'result.json'
        if (not result_path.exists() and numerical_recovery is not None
                and len(state.get('attempts', [])) == 3
                and all(a['status'] == 'infrastructure_failed' for a in state['attempts'])):
            if episode not in numerical_recovery['episode_ids']:
                raise ValueError('Exhausted slot is not bound to this recovery gate')
            allowance = {'schema_version': 'ifv-psd-numerical-recovery-allowance-v1',
                'max_attempts': 5, 'inputs': inputs, 'source_ledger': str(marker),
                'source_ledger_sha256': sha256_file(marker), 'evidence': numerical_recovery['evidence']}
            recovery_attempt_budget(output/marker.parent.name, inputs, allowance=allowance)
            work.append((marker, None, None, identity, state, allowance, None,
                         {str(marker): sha256_file(marker)}, None))
            continue
        if not result_path.is_file() or not state.get('attempts') or state['attempts'][-1]['status'] != 'completed':
            raise ValueError('In-flight or interrupted slot requires explicit diagnosis before recovery')
        result = load_bound(result_path, identity=identity)
        safe_name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in episode)+'.json'
        trace_path = source/'traces'/safe_name
        if not trace_path.is_file() or load_json(trace_path) != result:
            raise ValueError('Canonical trace differs from durable result')
        evidence = numerical_failure_evidence(result, inputs=inputs, source_slot=marker.parent)
        if carried_allowance is not None and evidence is not None:
            raise ValueError('Completed extended bank has unresolved numerical failure; no further extension')
        if require_complete and evidence is None:
            from .psd_source_completion import source_completion_failure, VERSION as COMPLETION_POLICY
            reason = source_completion_failure(result)
            if reason:
                evidence = {'kind': 'source_completeness', 'reason': reason,
                    'completion_policy': COMPLETION_POLICY,
                    'source_trace': {'path': str(trace_path), 'sha256': sha256_file(trace_path)}}
                if len(state['attempts']) >= identity['max_attempts']:
                    raise ValueError('Incomplete source has no remaining attempts; no automatic budget reset')
        originals = [marker, result_path, trace_path]
        if carried_allowance is not None:
            originals.append(marker.parent/'recovery-allowance.json')
        header = load_json(result_path)
        if header.get('schema_version') == 'ifv-psd-bound-gzip-v1':
            originals.append(result_path.parent/header['archive'])
        compact = {**result, 'state': {'stage_timings': result.get('state', {}).get('stage_timings', {})}}
        work.append((marker, result_path, trace_path, identity, state, evidence, carried_allowance,
                     {str(p): sha256_file(p) for p in originals}, compact))
        # Never retain every multimodal payload until all slots are checked.
        del result
    if not work:
        raise ValueError('No completed prior slots to recover')
    records = []
    for marker, result_path, trace_path, identity, state, evidence, carried_allowance, originals, compact in work:
        if any(sha256_file(Path(p)) != digest for p, digest in originals.items()):
            raise ValueError('Prior source changed during recovery validation')
        target = output/marker.parent.name
        binding = {**identity, 'version': VERSION}
        payload = copy.deepcopy(state)
        if result_path is None:
            from .io import write_json
            write_json(target/'recovery-allowance.json', evidence)
            binding['max_attempts'] = recovery_attempt_budget(target, identity['inputs'])
            save_bound(target/'retry-state.json', identity=binding, payload=payload)
            records.append({'episode_id': identity['inputs']['episode_id'],
                'action': 'two_attempts_after_verified_service_intervention',
                'spent_attempts': 3, 'maximum_attempts': 5,
                'originals': {str(marker): sha256_file(marker)}, 'recovery_allowance': evidence})
            continue
        if evidence is not None:
            payload['attempts'][-1].update(status='trajectory_failed' if evidence.get('kind') == 'source_completeness'
                else 'infrastructure_failed', reason=evidence['reason'], recovery_evidence=evidence)
        if carried_allowance is not None:
            from .io import write_json
            write_json(target/'recovery-allowance.json', carried_allowance)
        cache_header = load_json(result_path)
        save_bound(target/'retry-state.json', identity=binding, payload=payload)
        if evidence is None:
            if binding == identity:
                # Same-filesystem immutable source bank: preserve exact packed
                # bytes and relative archive references without expanding it.
                if cache_header.get('schema_version') == 'ifv-psd-bound-gzip-v1':
                    archive = result_path.parent/cache_header['archive']
                    os.link(archive, target/archive.name)
                os.link(result_path, target/'result.json')
            else:
                save_bound(target/'result.json', identity=binding,
                           payload=load_bound(result_path, identity=identity))
            canonical = destination/'traces'/trace_path.name
            canonical.parent.mkdir(parents=True, exist_ok=True)
            os.link(trace_path, canonical)
        records.append({'episode_id': identity['inputs']['episode_id'],
            'action': 'retry_remaining_budget' if evidence else 'reuse_exact_outcome',
            'spent_attempts': len(state['attempts']), 'maximum_attempts': binding['max_attempts'],
            'originals': originals,
            'cached_result': compact if evidence is None else None,
            'carried_recovery_allowance': carried_allowance is not None,
            'numerical_failure_evidence': evidence if evidence and evidence.get('kind') != 'source_completeness' else None,
            'completion_failure_evidence': evidence if evidence and evidence.get('kind') == 'source_completeness' else None})
    return {'schema_version': 'ifv-psd-slot-recovery-v1', 'slots': len(records),
        'reused': sum(r['action'] == 'reuse_exact_outcome' for r in records),
        'retry_remaining': sum(r['action'] == 'retry_remaining_budget' for r in records),
        'explicit_budget_extensions': sum(r['action'] == 'two_attempts_after_verified_service_intervention' for r in records),
        'preserved_recovery_allowances': sum(r.get('carried_recovery_allowance', False) for r in records),
        'source_unmodified': True, 'selection_by_answer': False, 'records': records}
