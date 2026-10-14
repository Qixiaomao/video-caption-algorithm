from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _validate_caption_ids(caption_ids: torch.Tensor) -> None:
    if caption_ids.ndim != 2:
        raise ValueError(f"caption_ids must be [B,L], got {tuple(caption_ids.shape)}")


def caption_content_mask(caption_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Select caption content before the first EOS-as-padding token."""

    _validate_caption_ids(caption_ids)
    reached_padding = caption_ids.eq(pad_id).long().cumsum(dim=1).gt(0)
    return ~reached_padding


def exact_caption_positive_mask(caption_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Return a multi-positive mask for exactly identical token sequences.

    MSVD may place the same caption text on more than one video. Treating those
    texts as negatives would create a known false-negative signal, so identical
    content sequences share the same positive set.
    """

    content_mask = caption_content_mask(caption_ids, pad_id)
    canonical = torch.where(content_mask, caption_ids, torch.full_like(caption_ids, -1))
    positive_mask = canonical[:, None, :].eq(canonical[None, :, :]).all(dim=-1)
    positive_mask.fill_diagonal_(True)
    return positive_mask


def diagonal_positive_mask(batch_size: int, device: torch.device) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return torch.eye(batch_size, dtype=torch.bool, device=device)


def build_alignment_embeddings(
    visual_prefix: torch.Tensor,
    token_embeddings: torch.Tensor,
    caption_ids: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool the frozen stage-0 video/text representations and L2-normalize."""

    if visual_prefix.ndim != 3:
        raise ValueError(f"visual_prefix must be [B,P,H], got {tuple(visual_prefix.shape)}")
    if token_embeddings.ndim != 3:
        raise ValueError(f"token_embeddings must be [B,L,H], got {tuple(token_embeddings.shape)}")
    _validate_caption_ids(caption_ids)
    if token_embeddings.shape[:2] != caption_ids.shape:
        raise ValueError("token_embeddings and caption_ids must share [B,L]")
    if visual_prefix.size(0) != caption_ids.size(0):
        raise ValueError("visual and text batch sizes must match")
    if visual_prefix.size(-1) != token_embeddings.size(-1):
        raise ValueError("visual and text embedding dimensions must match")

    text_mask = caption_content_mask(caption_ids, pad_id)
    valid_tokens = text_mask.sum(dim=1)
    if torch.any(valid_tokens == 0):
        raise ValueError("every caption must contain at least one non-padding token")

    video_embedding = visual_prefix.mean(dim=1)
    text_embedding = (
        token_embeddings * text_mask.unsqueeze(-1).to(token_embeddings.dtype)
    ).sum(dim=1) / valid_tokens.unsqueeze(-1).to(token_embeddings.dtype)
    return F.normalize(video_embedding, dim=-1), F.normalize(text_embedding, dim=-1)


def _multi_positive_cross_entropy(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != positive_mask.shape:
        raise ValueError("logits and positive_mask must have the same [B,B] shape")
    if positive_mask.dtype != torch.bool:
        raise ValueError("positive_mask must be boolean")
    if not torch.all(positive_mask.any(dim=1)):
        raise ValueError("every query must have at least one positive")

    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    return -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()


def symmetric_multi_positive_info_nce(
    video_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07,
    positive_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CLIP-style symmetric in-batch InfoNCE with optional multi-positives."""

    if video_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("video_embeddings and text_embeddings must be [B,H]")
    if video_embeddings.shape != text_embeddings.shape:
        raise ValueError("video_embeddings and text_embeddings must share shape")
    batch_size = video_embeddings.size(0)
    if batch_size < 2:
        raise ValueError("InfoNCE requires a physical batch size of at least 2")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    video_embeddings = F.normalize(video_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    logits = video_embeddings @ text_embeddings.transpose(0, 1) / temperature
    if positive_mask is None:
        positive_mask = diagonal_positive_mask(batch_size, logits.device)
    else:
        positive_mask = positive_mask.to(device=logits.device)

    video_to_text = _multi_positive_cross_entropy(logits, positive_mask)
    text_to_video = _multi_positive_cross_entropy(logits.transpose(0, 1), positive_mask.transpose(0, 1))
    return 0.5 * (video_to_text + text_to_video)


def positive_only_cosine_loss(video_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
    if video_embeddings.shape != text_embeddings.shape or video_embeddings.ndim != 2:
        raise ValueError("video_embeddings and text_embeddings must share [B,H]")
    return 1.0 - F.cosine_similarity(video_embeddings, text_embeddings, dim=-1).mean()


@torch.no_grad()
def similarity_diagnostics(
    video_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    similarities = F.normalize(video_embeddings, dim=-1) @ F.normalize(
        text_embeddings, dim=-1
    ).transpose(0, 1)
    positive_mask = positive_mask.to(device=similarities.device)
    negative_mask = ~positive_mask
    positive_similarity = similarities[positive_mask].mean()
    if negative_mask.any():
        negative_similarity = similarities[negative_mask].mean()
    else:
        negative_similarity = torch.full_like(positive_similarity, float("nan"))
    return {
        "positive_similarity": positive_similarity,
        "negative_similarity": negative_similarity,
        "similarity_gap": positive_similarity - negative_similarity,
    }
