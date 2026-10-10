"""ms-swift 4.4.2 adapter for EviScene sparse top-K PSD training.

The adapter is intentionally loaded through ms-swift's ``--external_plugins``.
It registers a pre-tokenized template and a custom loss while preserving the
framework's normal model, accelerator, FSDP, DeepSpeed, checkpoint, and logging
paths.  It does not train standard SFT labels.
"""

from __future__ import annotations

from eviscene.environment import get_env

from typing import Any, Mapping
from .psd_modality import require_text_only_psd


PSD_MS_SWIFT_TEMPLATE = "eviscene_psd_topk"
PSD_MS_SWIFT_LOSS = "eviscene_psd_topk"
LEGACY_PSD_MS_SWIFT_NAME = "ifv_psd_topk"


def data_parallel_sum_loss(loss: Any, *, data_parallel_size: int) -> Any:
    """Undo DDP/FSDP gradient averaging for Tinker's summed objective.

    Every data-parallel replica owns a different datum. PyTorch averages the
    replica gradients, while the PSD reference objective sums them. Scaling
    each local loss by the DP degree before backward makes the averaged model
    gradient exactly equal to the global datum sum. The returned scalar is a
    DP-scaled local estimator, avoiding a scalar collective per microbatch.
    """

    if not isinstance(data_parallel_size, int) or data_parallel_size < 1:
        raise ValueError("PSD data_parallel_size must be a positive integer")
    return loss if data_parallel_size == 1 else loss * data_parallel_size


def prepare_psd_logits_to_keep(inputs: dict[str, Any]) -> int:
    """Keep model logits only at positions supervised by sparse PSD targets."""

    import torch

    target_tokens = inputs.get("psd_target_tokens")
    weights = inputs.get("psd_weights")
    labels = inputs.get("labels")
    if target_tokens is None or weights is None or labels is None:
        raise ValueError("PSD logits_to_keep requires targets, weights, and labels")
    if labels.ndim != 2 or tuple(labels.shape) != tuple(target_tokens.shape[:2]):
        raise ValueError("PSD labels must match target batch and sequence dimensions")
    if torch.any(labels != -100):
        raise ValueError("PSD logits_to_keep requires supervision only from sparse targets")
    active = _validate_sparse_targets(
        target_tokens,
        weights,
        batch_size=target_tokens.shape[0],
        sequence_length=target_tokens.shape[1],
        require_active_per_datum=True,
    )
    if not torch.any(active):
        raise ValueError("PSD logits_to_keep found no supervised position")
    if target_tokens.shape[0] == 1:
        mask = active[0]
        # Qwen3.5 accepts an arbitrary boolean selection over prediction
        # positions. PSD targets already use causal prediction positions, so
        # Swift's normal SFT label shift must not be applied.
        inputs["logits_to_keep"] = mask
        inputs["labels"] = labels[:, mask]
        inputs["psd_target_tokens"] = target_tokens[:, mask]
        inputs["psd_weights"] = weights[:, mask]
        return int(mask.sum().item())

    # A batched model forward needs one common output shape. Keep the shortest
    # suffix containing every datum's active positions; zero-weight padding in
    # that suffix remains inert in the sparse summed loss.
    earliest_active = int(torch.nonzero(active, as_tuple=False)[:, 1].min().item())
    keep = target_tokens.shape[1] - earliest_active
    inputs["logits_to_keep"] = keep
    inputs["labels"] = labels[:, -keep:]
    inputs["psd_target_tokens"] = target_tokens[:, -keep:]
    inputs["psd_weights"] = weights[:, -keep:]
    return keep


class PsdDatasetPreprocessor:
    """Keep token arrays verbatim; the launcher/template validate each datum.

    AutoPreprocessor otherwise invents an empty messages list and rejects a
    pre-tokenized PSD row before our template is ever reached.
    """

    def __call__(self, dataset: Any, **_: Any) -> Any:
        columns = set(dataset.features)
        required = {"schema_version", "input_ids", "topk", "loss_positions"}
        if not required <= columns or not (
            {"target_tokens", "weights"} <= columns
            or {"sparse_target_tokens", "sparse_weights"} <= columns
        ):
            raise ValueError("registered PSD dataset is missing pre-tokenized targets")
        return dataset


def register_psd_dataset(path: str) -> None:
    """Register only the explicitly selected local PSD path, not all datasets."""
    from pathlib import Path
    from swift.dataset import DatasetMeta, register_dataset

    selected = Path(path).expanduser().resolve(strict=True)
    if not selected.is_file():
        raise ValueError("PSD dataset must be a local file")
    register_dataset(DatasetMeta(dataset_path=str(selected),
        preprocess_func=PsdDatasetPreprocessor()), exist_ok=True)


def _validate_sparse_targets(
    target_tokens: Any,
    weights: Any,
    *,
    batch_size: int,
    sequence_length: int,
    require_active_per_datum: bool,
) -> Any:
    import torch

    if target_tokens.ndim != 3 or tuple(target_tokens.shape[:2]) != (
        batch_size,
        sequence_length,
    ):
        raise ValueError(
            "psd_target_tokens must have shape [batch, sequence, topk]"
        )
    if tuple(weights.shape) != tuple(target_tokens.shape):
        raise ValueError("psd_weights shape must match psd_target_tokens")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("psd_weights must be finite and non-negative")
    active = weights.sum(dim=-1) > 0
    if (
        require_active_per_datum
        and not active.reshape(batch_size, -1).any(dim=1).all()
    ):
        raise ValueError("every PSD datum must contain an active top-k target")
    return active


def _chunked_sparse_position_losses(
    logits: Any,
    target_tokens: Any,
    weights: Any,
    *,
    chunk_size: int,
) -> Any:
    """Compute active-position losses without retaining a full fp32 softmax."""

    import torch

    if chunk_size < 1:
        raise ValueError("PSD loss chunk size must be positive")

    class ChunkedSparseTopKCrossEntropy(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, raw_logits: Any, tokens: Any, values: Any) -> Any:
            ctx.save_for_backward(raw_logits, tokens, values)
            losses = []
            for start in range(0, raw_logits.shape[0], chunk_size):
                end = min(start + chunk_size, raw_logits.shape[0])
                logits_chunk = raw_logits[start:end].float()
                weights_chunk = values[start:end].float()
                mass = weights_chunk.sum(dim=-1)
                selected = torch.gather(
                    logits_chunk,
                    -1,
                    tokens[start:end],
                )
                losses.append(
                    mass * torch.logsumexp(logits_chunk, dim=-1)
                    - (selected * weights_chunk).sum(dim=-1)
                )
            return torch.cat(losses)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, None, None]:
            raw_logits, tokens, values = ctx.saved_tensors
            for start in range(0, raw_logits.shape[0], chunk_size):
                end = min(start + chunk_size, raw_logits.shape[0])
                logits_chunk = (
                    raw_logits[start:end].detach().float().requires_grad_(True)
                )
                weights_chunk = values[start:end].float()
                with torch.enable_grad():
                    mass = weights_chunk.sum(dim=-1)
                    selected = torch.gather(
                        logits_chunk,
                        -1,
                        tokens[start:end],
                    )
                    losses = (
                        mass * torch.logsumexp(logits_chunk, dim=-1)
                        - (selected * weights_chunk).sum(dim=-1)
                    )
                    gradient = torch.autograd.grad(
                        losses,
                        logits_chunk,
                        grad_outputs=grad_output[start:end].float(),
                        retain_graph=False,
                    )[0]
                # This follows ms-swift's ChunkedCrossEntropyLoss: reuse the
                # saved logits buffer for its gradient instead of allocating a
                # second [active_tokens, vocab] tensor at long context.
                with torch.no_grad():
                    raw_logits[start:end].copy_(gradient.to(raw_logits.dtype))
            return raw_logits, None, None

    return ChunkedSparseTopKCrossEntropy.apply(logits, target_tokens, weights)


def sparse_topk_position_losses(
    outputs: Any,
    *,
    psd_target_tokens: Any,
    psd_weights: Any,
    chunk_size: int | None = None,
    require_active_per_datum: bool = True,
) -> Any:
    """Return one sparse teacher cross-entropy value per sequence position."""

    import torch

    logits = outputs["logits"] if isinstance(outputs, Mapping) else outputs.logits
    if logits.ndim != 3:
        raise ValueError("PSD logits must have shape [batch, sequence, vocab]")
    target_tokens = psd_target_tokens.to(
        device=logits.device,
        dtype=torch.long,
        non_blocking=True,
    )
    weights = psd_weights.to(
        device=logits.device,
        dtype=torch.float32,
        non_blocking=True,
    )
    active = _validate_sparse_targets(
        target_tokens,
        weights,
        batch_size=logits.shape[0],
        sequence_length=logits.shape[1],
        require_active_per_datum=require_active_per_datum,
    )
    active_tokens = target_tokens[active]
    if active_tokens.numel() and (
        active_tokens.min() < 0 or active_tokens.max() >= logits.shape[-1]
    ):
        raise ValueError("PSD target token ID is outside model vocabulary")

    # Keep every SP rank connected to the forward graph even when the short
    # PSD completion falls wholly on a different sequence shard.
    position_losses = logits[..., 0].float() * 0.0
    if active.any():
        active_logits = logits[active]
        configured_chunk = chunk_size
        if configured_chunk is None:
            raw_chunk = get_env(
                "EVISCENE_PSD_LOSS_CHUNK_TOKENS",
                get_env("CELOSS_PARALLEL_SIZE", "2048"),
            )
            try:
                configured_chunk = int(raw_chunk)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "EVISCENE_PSD_LOSS_CHUNK_TOKENS must be a positive integer"
                ) from exc
        active_losses = _chunked_sparse_position_losses(
            active_logits,
            active_tokens,
            weights[active],
            chunk_size=configured_chunk,
        )
        position_losses = position_losses.masked_scatter(active, active_losses)
    return position_losses


def sparse_topk_cross_entropy(
    outputs: Any,
    *,
    psd_target_tokens: Any,
    psd_weights: Any,
    chunk_size: int | None = None,
    require_active_per_datum: bool = True,
) -> Any:
    """Return truncated teacher top-K cross entropy over PSD action positions."""
    position_losses = sparse_topk_position_losses(
        outputs,
        psd_target_tokens=psd_target_tokens,
        psd_weights=psd_weights,
        chunk_size=chunk_size,
        require_active_per_datum=require_active_per_datum,
    )
    # Tinker's CE backward sums weighted losses, including across datums.
    # turn_kl.py divides loss:sum by batch size for LOGGING only. Neither a
    # token mean nor a datum mean belongs in the optimizer objective.
    loss = position_losses.sum()
    import torch

    if not torch.isfinite(loss):
        raise ValueError("PSD loss is non-finite")
    return loss


def sequence_parallel_sparse_topk_cross_entropy(
    outputs: Any,
    *,
    psd_target_tokens: Any,
    psd_weights: Any,
    sequence_parallel_instance: Any,
    gather_loss: Any,
) -> Any:
    """Compute the global PSD objective without gathering vocabulary logits."""

    import torch

    position_losses = sparse_topk_position_losses(
        outputs,
        psd_target_tokens=psd_target_tokens,
        psd_weights=psd_weights,
        require_active_per_datum=False,
    )
    position_losses, _ = gather_loss.apply(
        position_losses,
        None,
        1,
        sequence_parallel_instance.real_position_ids,
    )
    loss = position_losses.sum()
    if not torch.isfinite(loss):
        raise ValueError("PSD sequence-parallel loss is non-finite")
    return loss


def split_psd_targets_for_sequence_parallel(
    target_tokens: Any,
    weights: Any,
    *,
    sequence_parallel_instance: Any,
) -> tuple[Any, Any]:
    """Pad/split sparse targets with ms-swift's exact SP/RP ordering."""

    position_ids = sequence_parallel_instance.real_position_ids
    if position_ids is None:
        raise ValueError("PSD sequence parallel requires cached position_ids")
    *_, extra_values = sequence_parallel_instance.pad_and_split_inputs(
        None,
        None,
        None,
        None,
        None,
        None,
        real_position_ids=position_ids,
        extra_split_values=[
            (target_tokens, 0, 1),
            (weights, 0.0, 1),
        ],
    )
    return extra_values[0], extra_values[1]


def install_ms_swift_psd_plugin() -> None:
    """Register the EviScene template/loss and one narrowly scoped trainer bridge."""

    import torch
    from swift.loss.base import BaseLoss
    from swift.loss.mapping import loss_map
    from swift.sequence_parallel import GatherLoss, sequence_parallel
    from swift.template import TemplateMeta, register_template
    from swift.template.templates.qwen import Qwen3_5Template
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer

    dataset_path = get_env("EVISCENE_PSD_DATASET_PATH")
    if dataset_path:
        register_psd_dataset(dataset_path)

    class PsdTopKLoss(BaseLoss):
        def __call__(
            self,
            outputs: Any,
            labels: Any,
            *,
            psd_target_tokens: Any = None,
            psd_weights: Any = None,
            **_: Any,
        ) -> Any:
            if psd_target_tokens is None or psd_weights is None:
                raise ValueError(
                    "eviscene_psd_topk requires psd_target_tokens and psd_weights"
                )
            if sequence_parallel.enabled():
                return sequence_parallel_sparse_topk_cross_entropy(
                    outputs,
                    psd_target_tokens=psd_target_tokens,
                    psd_weights=psd_weights,
                    sequence_parallel_instance=sequence_parallel,
                    gather_loss=GatherLoss,
                )
            return sparse_topk_cross_entropy(
                outputs,
                psd_target_tokens=psd_target_tokens,
                psd_weights=psd_weights,
            )

    class PsdTopKTemplate(Qwen3_5Template):
        """Pass repository-produced token IDs through without re-tokenization."""

        support_padding_free = True

        def encode(
            self,
            inputs: Mapping[str, Any],
            return_template_inputs: bool = False,
            return_length: bool = False,
        ) -> dict[str, Any]:
            if not isinstance(inputs, Mapping):
                raise ValueError("EviScene PSD template requires a mapping input")
            input_ids = inputs.get("input_ids")
            from .psd_datums import expand_datum_targets
            target_tokens, weights = expand_datum_targets(inputs)
            topk = inputs.get("topk")
            if (
                not isinstance(input_ids, list)
                or not input_ids
                or not isinstance(target_tokens, list)
                or not isinstance(weights, list)
            ):
                raise ValueError("EviScene PSD datum is missing token arrays")
            if len(target_tokens) != len(input_ids) or len(weights) != len(input_ids):
                raise ValueError("EviScene PSD datum sequence dimensions do not match")
            require_text_only_psd(inputs, input_ids)
            if len(input_ids) > (getattr(self, "max_length", None) or 131072):
                raise ValueError("PSD datum exceeds max_length; truncation is forbidden")
            if not isinstance(topk, int) or topk < 1:
                raise ValueError("EviScene PSD datum has invalid topk")
            for position, (tokens, values) in enumerate(
                zip(target_tokens, weights, strict=True)
            ):
                if (
                    not isinstance(tokens, list)
                    or len(tokens) != topk
                    or not isinstance(values, list)
                    or len(values) != topk
                ):
                    raise ValueError(
                        f"EviScene PSD datum position {position} has invalid top-k shape"
                    )
            encoded = {
                "input_ids": [int(token) for token in input_ids],
                # The custom loss ignores labels. They make Seq2SeqTrainer take
                # its supported custom-loss route; model.forward therefore
                # never computes the model's built-in label loss.
                "labels": [-100] * len(input_ids),
                "psd_target_tokens": [
                    [int(token) for token in tokens]
                    for tokens in target_tokens
                ],
                "psd_weights": [
                    [float(weight) for weight in values] for values in weights
                ],
            }
            if inputs.get("psd_media"):
                from .psd_media import load_media
                encoded.update(load_media(inputs["psd_media"], input_ids))
                encoded["mm_token_type_ids"] = torch.tensor(
                    [1 if token == 248056 else 0 for token in input_ids],
                    dtype=torch.long,
                )
            if return_length:
                # AddLengthPreprocessor needs the plural lengths field normally supplied
                # by Template.encode; this pre-tokenized override must supply it itself.
                encoded["lengths"] = [len(input_ids)]
            if return_template_inputs:
                raise ValueError(
                    "EviScene PSD pre-tokenized template has no renderable messages"
                )
            return encoded

        def data_collator(
            self,
            batch: list[dict[str, Any]],
            *,
            padding_to: int | None = None,
        ) -> dict[str, Any]:
            if not batch:
                raise ValueError("EviScene PSD data collator received an empty batch")
            if self.sequence_parallel_size > 1 and not self.padding_free:
                raise ValueError(
                    "EviScene PSD sequence parallel requires padding_free=true"
                )
            if self.padding_free and len(batch) != 1:
                raise ValueError(
                    "EviScene PSD padding-free training requires one datum per device"
                )
            topk = len(batch[0]["psd_target_tokens"][0])
            sequence_length = max(len(row["input_ids"]) for row in batch)
            if padding_to is not None:
                if padding_to < sequence_length:
                    raise ValueError("padding_to is smaller than an EviScene PSD datum")
                sequence_length = padding_to
            for index, row in enumerate(batch):
                if (
                    len(row["input_ids"]) != len(row["psd_target_tokens"])
                    or len(row["input_ids"]) != len(row["psd_weights"])
                    or any(
                        len(tokens) != topk
                        for tokens in row["psd_target_tokens"]
                    )
                    or any(len(values) != topk for values in row["psd_weights"])
                ):
                    raise ValueError(f"EviScene PSD datum {index} has incompatible shape")

            batch_size = len(batch)
            input_ids = torch.full(
                (batch_size, sequence_length),
                int(self.tokenizer.pad_token_id),
                dtype=torch.long,
            )
            attention_mask = torch.zeros(
                (batch_size, sequence_length),
                dtype=torch.long,
            )
            labels = torch.full(
                (batch_size, sequence_length),
                -100,
                dtype=torch.long,
            )
            target_tokens = torch.zeros(
                (batch_size, sequence_length, topk),
                dtype=torch.long,
            )
            weights = torch.zeros(
                (batch_size, sequence_length, topk),
                dtype=torch.float32,
            )
            for index, row in enumerate(batch):
                length = len(row["input_ids"])
                input_ids[index, :length] = torch.tensor(
                    row["input_ids"],
                    dtype=torch.long,
                )
                attention_mask[index, :length] = 1
                target_tokens[index, :length] = torch.tensor(
                    row["psd_target_tokens"],
                    dtype=torch.long,
                )
                weights[index, :length] = torch.tensor(
                    row["psd_weights"],
                    dtype=torch.float32,
                )
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "psd_target_tokens": target_tokens,
                "psd_weights": weights,
            }
            if any("pixel_values" in row for row in batch):
                if not all("pixel_values" in row for row in batch):
                    raise ValueError("cannot mix visual and text-only PSD datums")
                result["pixel_values"] = torch.cat(
                    [row["pixel_values"] for row in batch], dim=0
                )
                result["image_grid_thw"] = torch.cat(
                    [row["image_grid_thw"] for row in batch], dim=0
                )
                mm_types = torch.zeros_like(input_ids)
                for index, row in enumerate(batch):
                    mm_types[index, :len(row["input_ids"])] = row[
                        "mm_token_type_ids"
                    ]
                result["mm_token_type_ids"] = mm_types
                positions = self._get_position_ids(result)["position_ids"]
                result["position_ids"] = positions[1:]
                if self.padding_free:
                    result.pop("attention_mask")
                    result["text_position_ids"] = torch.arange(sequence_length).unsqueeze(0)
                return result
            if self.padding_free:
                result.pop("attention_mask")
                result["position_ids"] = torch.arange(
                    sequence_length,
                    dtype=torch.long,
                ).unsqueeze(0)
            return result

    loss_map[PSD_MS_SWIFT_LOSS] = PsdTopKLoss
    loss_map[LEGACY_PSD_MS_SWIFT_NAME] = PsdTopKLoss
    # Existing checkpoint configurations use the original plugin name.
    for name in (PSD_MS_SWIFT_TEMPLATE, LEGACY_PSD_MS_SWIFT_NAME):
        register_template(
            TemplateMeta(
                name,
                prefix=[],
                prompt=["{{QUERY}}"],
                chat_sep=None,
                template_cls=PsdTopKTemplate,
            ),
            exist_ok=True,
        )

    if getattr(Seq2SeqTrainer, "_eviscene_psd_topk_bridge_installed", False):
        return
    original_compute_loss = Seq2SeqTrainer.compute_loss
    original_count_items = Seq2SeqTrainer._get_num_items_in_batch
    original_prepare_logits_to_keep = Seq2SeqTrainer.prepare_logits_to_keep

    def prepare_logits_to_keep_with_psd(
        self: Any,
        inputs: dict[str, Any],
    ) -> Any:
        if "psd_weights" not in inputs:
            return original_prepare_logits_to_keep(self, inputs)
        if self.template.sequence_parallel_size != 1:
            raise ValueError("PSD logits_to_keep is supported only with SP=1")
        prepare_psd_logits_to_keep(inputs)
        return None

    def count_items_with_psd(self: Any, batch_samples: list, device: Any) -> Any:
        sparse = ["psd_weights" in batch for batch in batch_samples]
        if any(sparse):
            if not all(sparse):
                raise ValueError("cannot mix PSD and SFT in one accumulation window")
            # All standard labels are -100. A token count of zero triggers
            # Swift's global-token multiplier despite our summed objective.
            # Returning None also works for a short final accumulation window.
            return None
        return original_count_items(self, batch_samples, device)

    def compute_loss_with_psd_topk(
        self: Any,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> Any:
        target_tokens = inputs.pop("psd_target_tokens", None)
        weights = inputs.pop("psd_weights", None)
        if target_tokens is None and weights is None:
            return original_compute_loss(
                self,
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        if target_tokens is None or weights is None:
            raise ValueError(
                "EviScene PSD batch must contain both psd_target_tokens and psd_weights"
            )
        if getattr(self.accelerator, "gradient_accumulation_steps", 1) != 1:
            raise ValueError("PSD requires Trainer-managed accumulation (Accelerate GAS=1)")
        world_size = getattr(self.accelerator, "num_processes", 1)
        sequence_parallel_size = self.template.sequence_parallel_size
        if (
            not isinstance(sequence_parallel_size, int)
            or sequence_parallel_size < 1
            or world_size % sequence_parallel_size != 0
        ):
            raise ValueError("PSD SP size must be a positive divisor of world size")
        data_parallel_size = world_size // sequence_parallel_size
        _validate_sparse_targets(
            target_tokens,
            weights,
            batch_size=target_tokens.shape[0],
            sequence_length=target_tokens.shape[1],
            require_active_per_datum=True,
        )
        if sequence_parallel.enabled():
            target_tokens, weights = split_psd_targets_for_sequence_parallel(
                target_tokens,
                weights,
                sequence_parallel_instance=sequence_parallel,
            )
        base_loss_func = inputs.get("compute_loss_func")
        if base_loss_func is None:
            raise ValueError(
                "EviScene PSD batch requires --loss_type eviscene_psd_topk"
            )

        def bound_loss(
            outputs: Any,
            labels: Any,
            **kwargs: Any,
        ) -> Any:
            local_group_loss = base_loss_func(
                outputs,
                labels,
                psd_target_tokens=target_tokens,
                psd_weights=weights,
                **kwargs,
            )
            return data_parallel_sum_loss(
                local_group_loss,
                data_parallel_size=data_parallel_size,
            )

        inputs["compute_loss_func"] = bound_loss
        return original_compute_loss(
            self,
            model,
            inputs,
            return_outputs=return_outputs,
            # No HF token normalization, extra world-size factor, or GAS
            # division. GatherLoss already compensates the SP gradient average.
            num_items_in_batch=None,
        )

    Seq2SeqTrainer.compute_loss = compute_loss_with_psd_topk
    Seq2SeqTrainer._get_num_items_in_batch = count_items_with_psd
    Seq2SeqTrainer.prepare_logits_to_keep = prepare_logits_to_keep_with_psd
    Seq2SeqTrainer._eviscene_psd_topk_bridge_installed = True
