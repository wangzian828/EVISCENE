"""Exact-token multimodal frozen teacher scoring (no decode/re-encode)."""
from __future__ import annotations

from typing import Any, Mapping
from .psd_media import load_media


def enable_cpu_reference_kernels():
    # Transformers selects optional CUDA-only kernels at import time even for
    # a CPU model. Use its own reference implementations in this process.
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    for name in ("FusedRMSNormGated", "causal_conv1d_fn", "causal_conv1d_update",
                 "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
        setattr(module, name, None)
    module.is_fast_path_available = False


class FrozenTeacher:
    def __init__(self, profile: Mapping[str, Any], *, device: str = "cuda:0"):
        import torch
        from transformers import AutoModelForImageTextToText

        self.device = device
        if device == "cpu":
            enable_cpu_reference_kernels()
        path = profile["model_path"]
        base = profile.get("engine_model_path") or path
        self.model = AutoModelForImageTextToText.from_pretrained(
            base, dtype=torch.bfloat16, local_files_only=True,
            attn_implementation="sdpa",
        ).to(device)
        if base != path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, path, is_trainable=False)
        self.model.eval().requires_grad_(False)

    def score(self, scored, topk):
        import torch

        ids = [*scored["teacher_prompt_ids"], *scored["completion_ids"]]
        start = len(scored["teacher_prompt_ids"])
        inputs = {"input_ids": torch.tensor([ids], device=self.device)}
        media = scored["target"].get("psd_media")
        if media:
            tensors = load_media(media, ids)
            inputs.update({key: value.to(self.device) for key, value in tensors.items()})
            inputs["mm_token_type_ids"] = (inputs["input_ids"] == 248056).long()
        positions = torch.arange(start - 1, len(ids) - 1, device=self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs, logits_to_keep=positions, use_cache=False)
            logits = outputs.logits[0]
            if logits.shape[0] != len(scored["completion_ids"]):
                raise ValueError("teacher returned wrong completion position count")
            logprobs = [None] * len(ids)
            # Top-k renormalization cancels the full-vocabulary normalizer.
            # Chunking bounds transient fp32 memory for long actions.
            for offset in range(0, len(positions), 128):
                values, tokens = logits[offset:offset + 128].float().topk(topk, dim=-1)
                values, tokens = values.cpu().tolist(), tokens.cpu().tolist()
                for index, (vs, ts) in enumerate(zip(values, tokens)):
                    logprobs[start + offset + index] = {
                        str(token): {"logprob": value, "rank": rank + 1}
                        for rank, (token, value) in enumerate(zip(ts, vs))
                    }
        return {"choices": [{"prompt_token_ids": ids, "prompt_logprobs": logprobs}]}
