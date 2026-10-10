"""Require immutable media for visual Qwen3.5 PSD prompts."""

from typing import Any, Mapping, Sequence


QWEN35_VISUAL_TOKEN_IDS = frozenset({248053, 248054, 248056, 248057})


def require_text_only_psd(
    row: Mapping[str, Any], *token_sequences: Sequence[int]
) -> None:
    media = row.get("psd_media")
    if media:
        from .psd_media import validate_media
        if any(248057 in sequence for sequence in token_sequences):
            raise ValueError("PSD video inputs are not supported by the image adapter")
        found_visual = False
        for sequence in token_sequences:
            # Completions have no images; validate prompt sequences only.
            if any(token in QWEN35_VISUAL_TOKEN_IDS for token in sequence):
                validate_media(media, sequence)
                found_visual = True
        if not found_visual:
            raise ValueError("PSD media supplied without visual prompt tokens")
        return
    visual_fields = (
        "images", "videos", "multi_modal_data", "pixel_values",
        "pixel_values_videos", "image_grid_thw", "video_grid_thw",
        "multimodal_inputs",
    )
    # Avoid truth-testing tensors; an explicitly present payload is unsupported.
    has_media = any(row.get(field) is not None for field in visual_fields)
    has_visual_ids = any(
        token in QWEN35_VISUAL_TOKEN_IDS
        for sequence in token_sequences for token in sequence
    )
    if has_media or has_visual_ids:
        raise ValueError(
            "psd_multimodal_not_supported: unbound visual input; attach psd_media "
            "from the archived request before scoring or training"
        )
