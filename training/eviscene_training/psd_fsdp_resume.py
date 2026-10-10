"""Restore the PEFT architecture before Swift's native FSDP2 state loading.

Swift 4.4.2 saves FSDP LoRA tensors as DCP without adapter_config.json. Its
generic from_pretrained path then returns a frozen SwiftModel with zero LoRA
parameters. Recreate the attested adapter architecture only; the unmodified
Trainer subsequently restores model, optimizer, scheduler and RNG from DCP.
"""
from pathlib import Path

from .psd_ms_swift import LEGACY_PSD_MS_SWIFT_NAME, PSD_MS_SWIFT_TEMPLATE


def prepare_fsdp_resume(original, cls, args, model, *, environ, **kwargs):
    resume = getattr(args, 'resume_from_checkpoint', None)
    enabled = (resume and getattr(args, 'template', None) in {PSD_MS_SWIFT_TEMPLATE, LEGACY_PSD_MS_SWIFT_NAME}
        and getattr(args, 'tuner_type', None) == 'lora'
        and getattr(args, 'tuner_backend', None) == 'peft'
        and environ.get('FSDP_VERSION') == '2'
        and environ.get('ACCELERATE_USE_FSDP', '').lower() == 'true')
    if not enabled:
        return original(cls, args, model, **kwargs)
    path = Path(resume).resolve()
    if (path / 'adapter_config.json').is_file():
        return original(cls, args, model, **kwargs)
    if not (path / 'pytorch_model_fsdp_0/.metadata').is_file():
        raise ValueError('PSD FSDP resume lacks distributed model state')
    if not any((parent / 'psd-resume-binding.json').is_file() for parent in list(path.parents)[:10]):
        raise ValueError('PSD FSDP architecture restore requires an attested resume binding')
    adapters = args.adapters
    try:
        args.resume_from_checkpoint, args.adapters = None, []
        prepared = original(cls, args, model, **kwargs)
    finally:
        args.resume_from_checkpoint, args.adapters = resume, adapters
    trainable = [name for name, p in prepared.named_parameters() if p.requires_grad]
    if not getattr(prepared, 'peft_config', None) or not trainable or any('lora_' not in n for n in trainable):
        raise ValueError('PSD FSDP resume did not reconstruct the trainable LoRA architecture')
    return prepared


def install_fsdp_resume_bridge():
    import os
    from swift.pipelines.train.tuner import TunerMixin
    if getattr(TunerMixin, '_eviscene_psd_fsdp_resume', False):
        return
    original = TunerMixin.__dict__['prepare_model'].__func__
    def prepare(cls, args, model, **kwargs):
        return prepare_fsdp_resume(original, cls, args, model, environ=os.environ, **kwargs)
    TunerMixin.prepare_model = classmethod(prepare)
    TunerMixin._eviscene_psd_fsdp_resume = True
