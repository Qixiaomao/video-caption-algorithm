"""Memory-bounded exact ranking for the 2990-by-59794 MSR-VTT test matrix.

Ties are resolved by original candidate order. This module has no data paths
and does not run an evaluation on import.
"""
from __future__ import annotations

import hashlib
import math

import torch

from scripts.eval_retrieval import deterministic_derangement


def _summary(ranks: torch.Tensor) -> dict[str, float]:
    ranks = ranks.to(torch.float32)
    if ranks.numel() == 0 or not torch.isfinite(ranks).all():
        raise ValueError("Missing/nonfinite retrieval ranks")
    return {"r_at_1": float(ranks.le(1).float().mean()),
            "r_at_5": float(ranks.le(5).float().mean()),
            "r_at_10": float(ranks.le(10).float().mean()),
            "median_rank": float(ranks.median()), "mean_rank": float(ranks.mean())}


def _single_target_ranks(scores: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Rank selected column in each row using score descending then index ascending."""
    rows, columns = scores.shape
    if gold.shape != (rows,) or torch.any((gold < 0) | (gold >= columns)):
        raise ValueError("Invalid gold indices")
    target = scores.gather(1, gold[:, None])
    candidate = torch.arange(columns, device=scores.device)[None, :]
    return 1 + (scores > target).sum(dim=1) + ((scores == target) & (candidate < gold[:, None])).sum(dim=1)


def _multi_target_ranks(scores: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    """Rank first of any positive columns, with the same deterministic tie rule."""
    rows, columns = scores.shape
    if positives.shape != scores.shape or positives.dtype != torch.bool or not positives.any(dim=1).all():
        raise ValueError("Invalid multi-positive mask")
    best_score = scores.masked_fill(~positives, -torch.inf).max(dim=1).values
    candidate = torch.arange(columns, device=scores.device)[None, :]
    first_best = torch.where(positives & (scores == best_score[:, None]), candidate,
                             torch.full_like(candidate, columns)).min(dim=1).values
    return 1 + (scores > best_score[:, None]).sum(dim=1) + ((scores == best_score[:, None]) & (candidate < first_best[:, None])).sum(dim=1)


@torch.inference_mode()
def evaluate_blockwise(video_embeddings: torch.Tensor, text_embeddings: torch.Tensor,
                       caption_gold_video: torch.Tensor, *, device: str = "cuda",
                       text_batch_size: int = 256, video_batch_size: int = 16,
                       derangement_seed: int = 123) -> dict:
    if video_embeddings.ndim != 2 or text_embeddings.ndim != 2 or video_embeddings.size(1) != text_embeddings.size(1):
        raise ValueError("Embedding dimensions differ")
    video_count = video_embeddings.size(0)
    caption_count = text_embeddings.size(0)
    if caption_gold_video.shape != (caption_count,) or torch.any((caption_gold_video < 0) | (caption_gold_video >= video_count)):
        raise ValueError("Invalid caption gold IDs")
    if not torch.bincount(caption_gold_video.long(), minlength=video_count).min().item() >= 1:
        raise ValueError("Every video must have at least one caption")
    if not torch.isfinite(video_embeddings).all() or not torch.isfinite(text_embeddings).all():
        raise ValueError("Nonfinite retrieval embedding")
    if text_batch_size < 1 or video_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    videos = video_embeddings.to(device=device, dtype=torch.float32)
    texts = text_embeddings.to(device=device, dtype=torch.float32)
    gold = caption_gold_video.to(device=device, dtype=torch.long)
    derangement = deterministic_derangement(video_count, derangement_seed).to(device)
    shuffled_gold = derangement[gold]
    t2v_true = []
    t2v_shuffled = []
    for start in range(0, caption_count, text_batch_size):
        stop = min(start + text_batch_size, caption_count)
        scores = texts[start:stop] @ videos.T
        t2v_true.append(_single_target_ranks(scores, gold[start:stop]).cpu())
        t2v_shuffled.append(_single_target_ranks(scores, shuffled_gold[start:stop]).cpu())
    v2t_true = []
    v2t_shuffled = []
    positive_sum = 0.0
    total_sum = 0.0
    positive_count = 0
    for start in range(0, video_count, video_batch_size):
        stop = min(start + video_batch_size, video_count)
        scores = videos[start:stop] @ texts.T
        query_ids = torch.arange(start, stop, device=device)[:, None]
        positive = query_ids.eq(gold[None, :])
        shuffled_positive = query_ids.eq(shuffled_gold[None, :])
        v2t_true.append(_multi_target_ranks(scores, positive).cpu())
        v2t_shuffled.append(_multi_target_ranks(scores, shuffled_positive).cpu())
        positive_sum += float(scores[positive].sum())
        total_sum += float(scores.sum())
        positive_count += int(positive.sum())
    if positive_count != caption_count:
        raise RuntimeError("Positive pair count changed")
    negative_count = video_count * caption_count - positive_count
    positive_mean = positive_sum / positive_count
    negative_mean = (total_sum - positive_sum) / negative_count
    if not all(math.isfinite(value) for value in (positive_mean, negative_mean)):
        raise RuntimeError("Nonfinite retrieval similarity")
    true_t2v = torch.cat(t2v_true)
    true_v2t = torch.cat(v2t_true)
    shuffled_t2v = torch.cat(t2v_shuffled)
    shuffled_v2t = torch.cat(v2t_shuffled)
    return {"implementation": "blockwise_exact_rank_score_desc_id_asc_v1",
            "candidate_videos": video_count, "caption_queries": caption_count,
            "score_units": "cosine similarity of L2-normalized representations",
            "recall_units": "fraction; multiply by 100 for percent display",
            "true_labels": {"text_to_video": _summary(true_t2v),
                            "video_to_text": _summary(true_v2t)},
            "shuffled_label_control": {"text_to_video": _summary(shuffled_t2v),
                                       "video_to_text": _summary(shuffled_v2t)},
            "similarity": {"positive_mean": positive_mean,
                           "negative_mean": negative_mean,
                           "gap": positive_mean - negative_mean},
            "ranks": {"t2v_true": true_t2v.tolist(), "v2t_true": true_v2t.tolist(),
                      "t2v_shuffled": shuffled_t2v.tolist(),
                      "v2t_shuffled": shuffled_v2t.tolist()},
            "rank_hashes": {"t2v_true_sha256": hashlib.sha256(true_t2v.numpy().tobytes()).hexdigest(),
                            "v2t_true_sha256": hashlib.sha256(true_v2t.numpy().tobytes()).hexdigest()}}
