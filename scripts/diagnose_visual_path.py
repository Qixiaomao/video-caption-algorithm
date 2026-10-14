#!/usr/bin/env python3
"""Measure where cross-video variation collapses along the visual caption path."""

from __future__ import annotations

import argparse
from collections import Counter
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

from core.preprocessing.frame_loader import load_video_tensor
from src.models.caption_model import VideoCaptionModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_frames_dir(record: dict[str, Any], annotation_path: Path) -> Path:
    configured = record.get("frames_dir")
    if configured:
        return Path(configured)
    return annotation_path.parent / "frames" / str(record["video_id"])


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    rows = tensor.detach().float().reshape(tensor.shape[0], -1).cpu()
    centered = rows - rows.mean(dim=0, keepdim=True)
    normalized = F.normalize(rows, dim=1)
    cosine = normalized @ normalized.T
    upper = cosine[torch.triu_indices(rows.shape[0], rows.shape[0], offset=1).unbind()]
    l2 = torch.pdist(rows, p=2)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    effective_rank = (
        float(energy.sum().square() / energy.square().sum())
        if energy.numel() and float(energy.square().sum()) > 0
        else 0.0
    )
    return {
        "shape": list(rows.shape),
        "row_norm_mean": float(rows.norm(dim=1).mean()),
        "row_norm_std": float(rows.norm(dim=1).std(unbiased=False)),
        "feature_std_l2": float(rows.std(dim=0, unbiased=False).norm()),
        "pairwise_cosine_mean": float(upper.mean()) if upper.numel() else 1.0,
        "pairwise_cosine_std": float(upper.std(unbiased=False)) if upper.numel() else 0.0,
        "pairwise_cosine_min": float(upper.min()) if upper.numel() else 1.0,
        "pairwise_cosine_max": float(upper.max()) if upper.numel() else 1.0,
        "pairwise_l2_mean": float(l2.mean()) if l2.numel() else 0.0,
        "pairwise_l2_std": float(l2.std(unbiased=False)) if l2.numel() else 0.0,
        "effective_rank": effective_rank,
    }


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
    state_dict = checkpoint.get("model_state", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"incompatible checkpoint: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model, checkpoint


@torch.no_grad()
def extract_stages(
    model: VideoCaptionModel,
    frames_dir: Path,
    device: str,
    num_frames: int,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    video = load_video_tensor(
        frames_dir,
        num_frames=num_frames,
        image_size=image_size,
        device=device,
    )
    encoder = model.encoder
    batch, timesteps, channels, height, width = video.shape
    flattened = video.reshape(batch * timesteps, channels, height, width)
    if encoder.cfg.prefer_channels_last and flattened.device.type == "cuda":
        flattened = flattened.contiguous(memory_format=torch.channels_last)
    backbone_features = encoder._forward_backbone_impl(flattened)
    pooled = encoder._spatial_temporal_pool(backbone_features, batch, timesteps)
    projected_pre_norm = encoder.proj(encoder.dropout(pooled))
    encoder_output = (
        F.normalize(projected_pre_norm, dim=-1)
        if encoder.cfg.l2norm
        else projected_pre_norm
    )
    video_embedding = model.proj(encoder_output)
    visual_prefix = model.decoder.build_visual_prefix(video_embedding)
    attention_mask = torch.ones(
        visual_prefix.shape[:2], dtype=torch.long, device=visual_prefix.device
    )
    first_token_logits = model.decoder.model(
        inputs_embeds=visual_prefix,
        attention_mask=attention_mask,
    ).logits[:, -1, :]
    return pooled, projected_pre_norm, encoder_output, visual_prefix, first_token_logits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    annotation_path = args.annotations.resolve()
    records = json.loads(annotation_path.read_text(encoding="utf-8"))
    model, checkpoint = load_model(checkpoint_path, args.device)

    collected: dict[str, list[torch.Tensor]] = {
        "backbone_pooled": [],
        "encoder_projected_pre_norm": [],
        "encoder_output": [],
        "visual_prefix": [],
        "first_token_logits": [],
    }
    video_ids: list[str] = []
    for record in records:
        frames_dir = resolve_frames_dir(record, annotation_path)
        if not frames_dir.exists():
            continue
        stages = extract_stages(
            model,
            frames_dir,
            args.device,
            args.num_frames,
            args.image_size,
        )
        for name, tensor in zip(collected, stages):
            collected[name].append(tensor.cpu())
        video_ids.append(str(record.get("video_id", frames_dir.name)))
        if len(video_ids) >= args.limit:
            break

    if len(video_ids) < 2:
        raise RuntimeError("At least two valid videos are required")
    stacked = {name: torch.cat(values, dim=0) for name, values in collected.items()}
    stage_stats = {name: tensor_stats(tensor) for name, tensor in stacked.items()}

    logits = stacked["first_token_logits"].float()
    probabilities = logits.softmax(dim=-1)
    top_ids = probabilities.argmax(dim=-1)
    top_tokens = [model.decoder.tokenizer.decode([int(token_id)]) for token_id in top_ids]
    top_counts = Counter(token.strip() or repr(token) for token in top_tokens)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)

    zero_prefix = torch.zeros_like(stacked["visual_prefix"][:1]).to(args.device)
    zero_attention = torch.ones(
        zero_prefix.shape[:2], dtype=torch.long, device=args.device
    )
    with torch.no_grad():
        zero_logits = model.decoder.model(
            inputs_embeds=zero_prefix,
            attention_mask=zero_attention,
        ).logits[:, -1, :].cpu().float()
    zero_probabilities = zero_logits.softmax(dim=-1)
    zero_top_id = int(zero_probabilities.argmax(dim=-1).item())
    zero_top_token = model.decoder.tokenizer.decode([zero_top_id])
    kl_to_zero = (
        probabilities
        * (probabilities.clamp_min(1e-12).log() - zero_probabilities.clamp_min(1e-12).log())
    ).sum(dim=-1)

    result = {
        "protocol": "visual-path-gate-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_contract": checkpoint.get("training_contract"),
        "annotations": str(annotation_path),
        "sample_count": len(video_ids),
        "video_ids": video_ids,
        "stages": stage_stats,
        "first_token": {
            "top1_unique_count": len(set(top_tokens)),
            "top1_unique_rate": len(set(top_tokens)) / len(top_tokens),
            "top1_counts": dict(top_counts.most_common()),
            "entropy_mean": float(entropy.mean()),
            "entropy_std": float(entropy.std(unbiased=False)),
            "zero_prefix_top1_token": zero_top_token,
            "top1_matches_zero_prefix_rate": float(top_ids.eq(zero_top_id).float().mean()),
            "kl_to_zero_prefix_mean": float(kl_to_zero.mean()),
            "kl_to_zero_prefix_std": float(kl_to_zero.std(unbiased=False)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "video_ids"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
