#!/usr/bin/env python3
"""Evaluate the stage-0 visual-prefix/WTE retrieval diagnostic on MSVD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from src.data.data_loader import build_dataloader
from src.models.caption_model import VideoCaptionModel
from src.training.contrastive import caption_content_mask


RECALL_KS = (1, 5, 10)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint_path: Path, device: str) -> tuple[VideoCaptionModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        checkpoint = {"model_state": checkpoint}
    checkpoint_args = checkpoint.get("args") or {}
    model = VideoCaptionModel(
        vit_name=checkpoint_args.get("vit_name", "vit_base_patch16_224"),
        gpt2_name=checkpoint_args.get("gpt2_name", "gpt2"),
        cond_mode=checkpoint_args.get("cond_mode", "prefix"),
        prefix_len=int(checkpoint_args.get("prefix_len", 4)),
        freeze_vit=True,
        vit_enable_torch_compile=False,
    ).to(device).eval()
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"incompatible checkpoint: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model, checkpoint


def _rank_summary(ranks: torch.Tensor) -> dict[str, float]:
    ranks = ranks.float()
    metrics = {
        f"r_at_{k}": float(ranks.le(k).float().mean())
        for k in RECALL_KS
    }
    metrics.update(
        {
            "median_rank": float(ranks.median()),
            "mean_rank": float(ranks.mean()),
        }
    )
    return metrics


def single_target_retrieval_metrics(
    scores: torch.Tensor,
    gold_indices: torch.Tensor,
) -> dict[str, float]:
    """Rank one target per query for text-to-video retrieval."""

    if scores.ndim != 2 or gold_indices.ndim != 1 or scores.size(0) != gold_indices.numel():
        raise ValueError("scores must be [Q,C] with one gold index per query")
    order = scores.argsort(dim=1, descending=True)
    matches = order.eq(gold_indices[:, None])
    if not torch.all(matches.any(dim=1)):
        raise ValueError("every query must have a valid target")
    ranks = matches.float().argmax(dim=1) + 1
    return _rank_summary(ranks)


def multi_target_retrieval_metrics(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
) -> dict[str, float]:
    """Rank the first matching target for video-to-text retrieval."""

    if scores.shape != positive_mask.shape or positive_mask.dtype != torch.bool:
        raise ValueError("scores and boolean positive_mask must share [Q,C]")
    if not torch.all(positive_mask.any(dim=1)):
        raise ValueError("every query must have at least one target")
    order = scores.argsort(dim=1, descending=True)
    ordered_positives = positive_mask.gather(1, order)
    ranks = ordered_positives.float().argmax(dim=1) + 1
    return _rank_summary(ranks)


def deterministic_derangement(size: int, seed: int) -> torch.Tensor:
    if size < 2:
        raise ValueError("at least two videos are required for a shuffled-label control")
    generator = torch.Generator().manual_seed(seed)
    identity = torch.arange(size)
    for _ in range(100):
        permutation = torch.randperm(size, generator=generator)
        if not permutation.eq(identity).any():
            return permutation
    # Guaranteed derangement fallback, still deterministic.
    return identity.roll(1)


@torch.no_grad()
def encode_videos(
    model: VideoCaptionModel,
    loader,
    device: str,
    limit: int,
) -> tuple[list[str], torch.Tensor]:
    video_ids: list[str] = []
    embeddings: list[torch.Tensor] = []
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        video_embedding = model.proj(model.encoder(video))
        visual_prefix = model.decoder.build_visual_prefix(video_embedding)
        pooled = F.normalize(visual_prefix.mean(dim=1), dim=-1)
        batch_ids = [str(video_id) for video_id in batch["video_id"]]
        if limit > 0 and len(video_ids) + len(batch_ids) > limit:
            keep = limit - len(video_ids)
            batch_ids = batch_ids[:keep]
            pooled = pooled[:keep]
        video_ids.extend(batch_ids)
        embeddings.append(pooled.cpu())
        if limit > 0 and len(video_ids) >= limit:
            break
    if not embeddings:
        raise RuntimeError("no valid video frames were loaded")
    return video_ids, torch.cat(embeddings, dim=0)


@torch.no_grad()
def encode_texts(
    model: VideoCaptionModel,
    captions: list[str],
    max_length: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    tokenizer = model.decoder.tokenizer
    encoded_batches: list[torch.Tensor] = []
    for start in range(0, len(captions), batch_size):
        tokenized = tokenizer(
            captions[start : start + batch_size],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        caption_ids = tokenized.input_ids.to(device)
        token_embeddings = model.decoder.model.transformer.wte(caption_ids)
        mask = caption_content_mask(caption_ids, tokenizer.pad_token_id)
        valid_tokens = mask.sum(dim=1)
        if torch.any(valid_tokens == 0):
            raise ValueError("retrieval captions must contain at least one content token")
        pooled = (
            token_embeddings * mask.unsqueeze(-1).to(token_embeddings.dtype)
        ).sum(dim=1) / valid_tokens.unsqueeze(-1).to(token_embeddings.dtype)
        encoded_batches.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(encoded_batches, dim=0)


def evaluate_scores(
    video_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    caption_gold_video: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    video_to_text_scores = video_embeddings @ text_embeddings.T
    text_to_video_scores = video_to_text_scores.T
    video_count = video_embeddings.size(0)
    positive_mask = torch.arange(video_count)[:, None].eq(caption_gold_video[None, :])

    true_metrics = {
        "text_to_video": single_target_retrieval_metrics(
            text_to_video_scores,
            caption_gold_video,
        ),
        "video_to_text": multi_target_retrieval_metrics(
            video_to_text_scores,
            positive_mask,
        ),
    }

    derangement = deterministic_derangement(video_count, seed)
    shuffled_gold = derangement[caption_gold_video]
    shuffled_mask = torch.arange(video_count)[:, None].eq(shuffled_gold[None, :])
    shuffled_metrics = {
        "text_to_video": single_target_retrieval_metrics(
            text_to_video_scores,
            shuffled_gold,
        ),
        "video_to_text": multi_target_retrieval_metrics(
            video_to_text_scores,
            shuffled_mask,
        ),
    }

    positive_similarity = video_to_text_scores[positive_mask].mean()
    negative_similarity = video_to_text_scores[~positive_mask].mean()
    return {
        "true_labels": true_metrics,
        "shuffled_label_control": shuffled_metrics,
        "similarity": {
            "positive_mean": float(positive_similarity),
            "negative_mean": float(negative_similarity),
            "gap": float(positive_similarity - negative_similarity),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    annotation_path = args.annotations.resolve()
    records = json.loads(annotation_path.read_text(encoding="utf-8"))
    records_by_id = {str(record["video_id"]): record for record in records}
    model, checkpoint = load_model(checkpoint_path, args.device)
    loader = build_dataloader(
        str(annotation_path),
        model.decoder.tokenizer,
        batch_size=args.batch_size,
        max_len=args.max_length,
        num_frame=args.num_frames,
        image_size=args.image_size,
        shuffle=False,
        num_wokers=0,
        random_caption=False,
    )
    video_ids, video_embeddings = encode_videos(model, loader, args.device, args.limit)

    captions: list[str] = []
    caption_gold: list[int] = []
    first_caption_indices: list[int] = []
    for video_index, video_id in enumerate(video_ids):
        record_captions = records_by_id[video_id].get("captions") or []
        if isinstance(record_captions, str):
            record_captions = [record_captions]
        record_captions = [str(caption) for caption in record_captions if str(caption).strip()]
        if not record_captions:
            raise ValueError(f"video {video_id} has no non-empty captions")
        first_caption_indices.append(len(captions))
        captions.extend(record_captions)
        caption_gold.extend([video_index] * len(record_captions))

    text_embeddings = encode_texts(
        model,
        captions,
        args.max_length,
        args.text_batch_size,
        args.device,
    )
    caption_gold_tensor = torch.tensor(caption_gold, dtype=torch.long)
    all_caption_results = evaluate_scores(
        video_embeddings,
        text_embeddings,
        caption_gold_tensor,
        args.seed,
    )

    canonical_indices = torch.tensor(first_caption_indices, dtype=torch.long)
    canonical_results = evaluate_scores(
        video_embeddings,
        text_embeddings[canonical_indices],
        torch.arange(len(video_ids)),
        args.seed,
    )
    result = {
        "protocol": "prefix-wte-retrieval-diagnostic-v1",
        "paper_ready": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_contract": checkpoint.get("training_contract"),
        "annotations": str(annotation_path),
        "annotations_sha256": sha256_file(annotation_path),
        "video_count": len(video_ids),
        "caption_count": len(captions),
        "representation": {
            "video": "mean_visual_prefix_tokens_l2_normalized",
            "text": "frozen_gpt2_wte_masked_mean_l2_normalized",
        },
        "all_captions": all_caption_results,
        "canonical_first_caption": canonical_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
