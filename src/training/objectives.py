from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

from src.models.caption_model import build_caption_supervision
from src.training.contrastive import (
    build_alignment_embeddings,
    diagonal_positive_mask,
    exact_caption_positive_mask,
    positive_only_cosine_loss,
    similarity_diagnostics,
    symmetric_multi_positive_info_nce,
)
from src.training.cpr import (
    CPR_NEGATIVE_POLICIES,
    compute_decoder_cpr_objective,
)


ALIGNMENT_OBJECTIVES = ("none", "positive_cosine", "infonce")
FALSE_NEGATIVE_POLICIES = ("diagonal", "exact_caption")


def compute_caption_alignment_objective(
    model: Any,
    video: torch.Tensor,
    caption_ids: torch.Tensor,
    pad_id: int,
    alignment_objective: str = "none",
    alignment_weight: float = 0.0,
    temperature: float = 0.07,
    false_negative_policy: str = "exact_caption",
    cpr_weight: float = 0.0,
    cpr_margin: float = 0.10,
    cpr_negative_policy: str = "cyclic_derangement",
    video_ids: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute caption CE plus optional alignment and prefix-ranking losses."""

    if alignment_objective not in ALIGNMENT_OBJECTIVES:
        raise ValueError(f"unknown alignment_objective: {alignment_objective}")
    if false_negative_policy not in FALSE_NEGATIVE_POLICIES:
        raise ValueError(f"unknown false_negative_policy: {false_negative_policy}")
    if alignment_weight < 0:
        raise ValueError("alignment_weight must be non-negative")
    if cpr_weight < 0:
        raise ValueError("cpr_weight must be non-negative")
    if cpr_margin <= 0:
        raise ValueError("cpr_margin must be positive")
    if cpr_negative_policy not in CPR_NEGATIVE_POLICIES:
        raise ValueError(f"unknown CPR negative policy: {cpr_negative_policy}")

    use_alignment = alignment_objective != "none" and alignment_weight > 0
    use_cpr = cpr_weight > 0
    if use_cpr and video_ids is None:
        raise ValueError("video_ids are required when CPR is enabled")
    attention_mask, labels = build_caption_supervision(caption_ids, pad_id)
    result = model(
        video=video,
        input_ids=caption_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_visual_prefix=use_alignment or use_cpr,
    )
    caption_loss = result["loss"]
    zero = caption_loss.new_zeros(())
    alignment_loss = zero
    diagnostics = {
        "positive_similarity": zero,
        "negative_similarity": zero,
        "similarity_gap": zero,
    }

    if use_alignment:
        visual_prefix = result["visual_prefix"]
        token_embedding_layer = model.decoder.model.transformer.wte
        with torch.no_grad():
            token_embeddings = token_embedding_layer(caption_ids)
        video_embeddings, text_embeddings = build_alignment_embeddings(
            visual_prefix=visual_prefix,
            token_embeddings=token_embeddings,
            caption_ids=caption_ids,
            pad_id=pad_id,
        )
        if false_negative_policy == "exact_caption":
            positive_mask = exact_caption_positive_mask(caption_ids, pad_id)
        else:
            positive_mask = diagonal_positive_mask(caption_ids.size(0), caption_ids.device)

        if alignment_objective == "infonce":
            alignment_loss = symmetric_multi_positive_info_nce(
                video_embeddings,
                text_embeddings,
                temperature=temperature,
                positive_mask=positive_mask,
            )
        else:
            alignment_loss = positive_only_cosine_loss(video_embeddings, text_embeddings)
        diagnostics = similarity_diagnostics(video_embeddings, text_embeddings, positive_mask)

    cpr_loss = zero
    matched_nll_mean = zero
    mismatched_nll_mean = zero
    cpr_gap = zero
    cpr_eligible_fraction = zero
    if use_cpr:
        assert video_ids is not None
        cpr_metrics = compute_decoder_cpr_objective(
            decoder=model.decoder,
            video_embedding=result["video_embedding"],
            visual_prefix=result["visual_prefix"],
            matched_logits=result["logits"],
            caption_ids=caption_ids,
            attention_mask=attention_mask,
            labels=labels,
            pad_id=pad_id,
            video_ids=video_ids,
            margin=cpr_margin,
            negative_policy=cpr_negative_policy,
        )
        cpr_loss = cpr_metrics["cpr_loss"]
        matched_nll_mean = cpr_metrics["matched_nll"]
        mismatched_nll_mean = cpr_metrics["mismatched_nll"]
        cpr_gap = cpr_metrics["cpr_gap"]
        cpr_eligible_fraction = cpr_metrics["cpr_eligible_fraction"]

    return {
        "loss": caption_loss + alignment_weight * alignment_loss + cpr_weight * cpr_loss,
        "caption_loss": caption_loss,
        "alignment_loss": alignment_loss,
        "cpr_loss": cpr_loss,
        "matched_nll": matched_nll_mean,
        "mismatched_nll": mismatched_nll_mean,
        "cpr_gap": cpr_gap,
        "cpr_eligible_fraction": cpr_eligible_fraction,
        **diagnostics,
    }
