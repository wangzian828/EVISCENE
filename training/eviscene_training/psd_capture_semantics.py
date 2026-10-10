"""PSD-only metadata adapter; the frozen Agent's actions/prompts stay unchanged."""
RAW_POLICY_LOGPROBS = 'pre_grammar_unprocessed_v1'


def install_capture_semantics():
    import sys
    from eviscene.orchestrator import llm_backend
    current = llm_backend.extract_policy_token_capture
    if getattr(current, '_psd_semantics', False):
        extract = current
        original = current._psd_original_capture
    else:
        original = current

        def extract(payload, **kwargs):
            capture = original(payload, **kwargs)
            marker = payload.get('ifv_policy_logprobs')
            if marker == RAW_POLICY_LOGPROBS:
                capture['teacher_logprob_semantics'] = marker
            return capture

        extract._psd_semantics = True
        extract._psd_original_capture = original
    # stage_runner uses `from llm_backend import ...`; replacing only the
    # defining module misses the already-imported collector/repair call site.
    # Keep the change confined to this PSD process, never edit frozen eviscene.
    stage = sys.modules.get('eviscene.orchestrator.stage_runner')
    if stage is not None:
        alias = stage.extract_policy_token_capture
        if alias is not original and alias is not extract:
            raise ValueError('PSD capture call site has an unexpected replacement')
        stage.extract_policy_token_capture = extract
    llm_backend.extract_policy_token_capture = extract
