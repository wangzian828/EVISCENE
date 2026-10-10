"""Explicit, scoped model handoff for uncached PSD external requests only."""
from __future__ import annotations

from eviscene.environment import get_env

import hashlib
import json
from pathlib import Path


def uncached_model(model, cache_dir):
    path = get_env('EVISCENE_PSD_EXTERNAL_ROUTE')
    if not path:
        return model, None
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != get_env('EVISCENE_PSD_EXTERNAL_ROUTE_SHA256'):
        raise ValueError('PSD external model route changed after launch')
    route = json.loads(raw)
    common = (route.get('preserve_completed_responses') is True
              and route.get('thinking_level') == 'high'
              and Path(str(route.get('search_root', ''))).is_absolute())
    legacy = (route.get('schema_version') == 'ifv-psd-external-model-route-v1'
              and route.get('from_model') == 'gemini-3.7-flash'
              and route.get('to_model') == 'gemini-3.6-flash')
    direct = (route.get('schema_version') == 'ifv-psd-external-model-route-v2'
              and route.get('mode') == 'direct_model'
              and route.get('request_model') == 'gemini-3.7-flash'
              and route.get('model') == 'gemini-3.7-flash')
    if not common or not (legacy or direct):
        raise ValueError('Invalid explicit PSD external model route')
    request_model = route.get('from_model') if legacy else route.get('request_model')
    if model != request_model or cache_dir is None:
        return model, None
    root = Path(route['search_root']).resolve()
    if not Path(cache_dir).resolve().is_relative_to(root):
        return model, None
    actual_model = route['to_model'] if legacy else route['model']
    return actual_model, {'path': str(Path(path).resolve()), 'sha256': digest,
                          'from_model': model, 'to_model': actual_model,
                          **({'mode': 'direct_model'} if direct else {})}
