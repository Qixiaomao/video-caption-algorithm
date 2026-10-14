from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.training.contrastive import caption_content_mask, exact_caption_positive_mask


CPR_NEGATIVE_POLICIES = ("cyclic_derangement",)


def build_cpr_negative_indices(
    caption_ids: torch.Tensor,
    pad_id: int,
    video_ids: Sequence[str],
    policy: str = "cyclic_derangement",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose one deterministic valid in-batch prefix for every eligible row.

    Candidates with the same video id or exactly the same caption content are
    excluded. Rows without a valid candidate receive index -1 and are skipped.
    """

    if policy not in CPR_NEGATIVE_POLICIES:
        raise ValueError(f"unknown CPR negative policy: {policy}")
    if caption_ids.ndim != 2:
        raise ValueError(f"caption_ids must be [B,L], got {tuple(caption_ids.shape)}")

    batch_size = caption_ids.size(0)
    if len(video_ids) != batch_size:
        raise ValueError("video_ids must contain one id per caption")

    invalid = exact_caption_positive_mask(caption_ids, pad_id)
    same_video = torch.tensor(
        [[left == right for right in video_ids] for left in video_ids],
        dtype=torch.bool,
        device=caption_ids.device,
    )
    invalid = invalid | same_video

    negative_indices = torch.full(
        (batch_size,), -1, dtype=torch.long, device=caption_ids.device
    )
    rows = torch.arange(batch_size, device=caption_ids.device)
    for offset in range(1, batch_size):
        candidates = (rows + offset) % batch_size
        selectable = negative_indices.lt(0) & ~invalid[rows, candidates]
        negative_indices[selectable] = candidates[selectable]

    return negative_indices, negative_indices.ge(0)


def length_normalized_caption_nll(
    logits: torch.Tensor,
    caption_ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Return per-sample NLL averaged over caption content tokens only."""

    if logits.ndim != 3:
        raise ValueError(f"logits must be [B,S,V], got {tuple(logits.shape)}")
    if caption_ids.ndim != 2:
        raise ValueError(f"caption_ids must be [B,L], got {tuple(caption_ids.shape)}")
    if logits.size(0) != caption_ids.size(0):
        raise ValueError("logits and caption_ids must share the batch dimension")

    prefix_len = logits.size(1) - caption_ids.size(1)
    if prefix_len < 1:
        raise ValueError("logits must include at least one visual-prefix position")

    batch_size = caption_ids.size(0)
    ignored_prefix = torch.full(
        (batch_size, prefix_len),
        -100,
        dtype=caption_ids.dtype,
        device=caption_ids.device,
    )
    extended_labels = torch.cat([ignored_prefix, caption_ids], dim=1)

    content = caption_content_mask(caption_ids, pad_id)
    prefix_mask = torch.zeros(
        (batch_size, prefix_len),
        dtype=torch.bool,
        device=caption_ids.device,
    )
    extended_content = torch.cat([prefix_mask, content], dim=1)

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = extended_labels[:, 1:].contiguous()
    shifted_content = extended_content[:, 1:]
    token_losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(batch_size, -1)

    token_counts = shifted_content.sum(dim=1)
    if torch.any(token_counts == 0):
        raise ValueError("every caption must contain at least one content token")
    return (token_losses * shifted_content.to(token_losses.dtype)).sum(dim=1) / token_counts


def conditional_prefix_ranking_loss(
    matched_nll: torch.Tensor,
    mismatched_nll: torch.Tensor,
    margin: float,
    eligible_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Hinge loss requiring a wrong prefix to raise caption NLL by `margin`."""

    if matched_nll.ndim != 1 or mismatched_nll.ndim != 1:
        raise ValueError("matched_nll and mismatched_nll must be one-dimensional")
    if matched_nll.shape != mismatched_nll.shape:
        raise ValueError("matched_nll and mismatched_nll must share shape")
    if margin <= 0:
        raise ValueError("CPR margin must be positive")

    if eligible_mask is None:
        eligible_mask = torch.ones_like(matched_nll, dtype=torch.bool)
    if eligible_mask.shape != matched_nll.shape or eligible_mask.dtype != torch.bool:
        raise ValueError("eligible_mask must be boolean and match the NLL shape")
    if not eligible_mask.any():
        return matched_nll.sum() * 0.0

    gaps = mismatched_nll[eligible_mask] - matched_nll[eligible_mask]
    return F.relu(matched_nll.new_tensor(margin) - gaps).mean()


def compute_decoder_cpr_objective(
    decoder: Any,
    video_embedding: torch.Tensor,
    visual_prefix: torch.Tensor,
    matched_logits: torch.Tensor,
    caption_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    pad_id: int,
    video_ids: Sequence[str],
    margin: float = 0.10,
    negative_policy: str = "cyclic_derangement",
) -> Dict[str, torch.Tensor]:
    """Compute CPR from an already completed matched decoder forward."""

    zero = matched_logits.sum() * 0.0
    metrics = {
        "cpr_loss": zero,
        "matched_nll": zero,
        "mismatched_nll": zero,
        "cpr_gap": zero,
        "cpr_eligible_fraction": zero,
    }
    negative_indices, eligible = build_cpr_negative_indices(
        caption_ids,
        pad_id,
        video_ids,
        policy=negative_policy,
    )
    metrics["cpr_eligible_fraction"] = eligible.to(matched_logits.dtype).mean()
    if not eligible.any():
        return metrics

    eligible_rows = torch.nonzero(eligible, as_tuple=False).squeeze(1)
    matched_nll = length_normalized_caption_nll(
        matched_logits, caption_ids, pad_id
    )[eligible_rows]
    selected_negative_indices = negative_indices[eligible_rows]
    wrong_prefix = visual_prefix[selected_negative_indices]

    def decode_wrong_prefix(
        selected_video_embedding: torch.Tensor,
        selected_caption_ids: torch.Tensor,
        selected_attention_mask: torch.Tensor,
        selected_labels: torch.Tensor,
        selected_wrong_prefix: torch.Tensor,
    ) -> torch.Tensor:
        return decoder(
            selected_video_embedding,
            selected_caption_ids,
            selected_attention_mask,
            selected_labels,
            visual_prefix=selected_wrong_prefix,
        )["logits"]

    wrong_inputs = (
        video_embedding[eligible_rows],
        caption_ids[eligible_rows],
        attention_mask[eligible_rows],
        labels[eligible_rows],
        wrong_prefix,
    )
    fork_devices = []
    if caption_ids.is_cuda:
        device_index = caption_ids.device.index
        fork_devices = [torch.cuda.current_device() if device_index is None else device_index]
    # The CPR-only pass participates in gradients but must not advance the
    # historical caption-training RNG trajectory.
    with torch.random.fork_rng(devices=fork_devices, enabled=True):
        if torch.is_grad_enabled() and decoder.training:
            wrong_logits = checkpoint(
                decode_wrong_prefix,
                *wrong_inputs,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            wrong_logits = decode_wrong_prefix(*wrong_inputs)

    mismatched_nll = length_normalized_caption_nll(
        wrong_logits, caption_ids[eligible_rows], pad_id
    )
    cpr_loss = conditional_prefix_ranking_loss(
        matched_nll,
        mismatched_nll,
        margin=margin,
    )
    metrics.update(
        cpr_loss=cpr_loss,
        matched_nll=matched_nll.mean(),
        mismatched_nll=mismatched_nll.mean(),
        cpr_gap=mismatched_nll.mean() - matched_nll.mean(),
    )
    return metrics
