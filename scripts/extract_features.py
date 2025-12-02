#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract video features using your ViT encoder (with src/model/video_encoder.py)
Outputs: .npy feature files per video
"""

import torch
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from src.models.video_encoder import build_vit_encoder

@torch.no_grad()
def extract_video_feature(model, frames_dir, num_frames=8, image_size=224, device="cpu"):
    frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
    if not frame_files:
        return None
    step = max(len(frame_files) // num_frames, 1)
    frames = [Image.open(frame_files[i]).convert("RGB") for i in range(0, len(frame_files), step)[:num_frames]]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    frames = torch.stack([transform(f) for f in frames]).to(device)  # [T,3,224,224]
    feat = model(frames.unsqueeze(0))  # [1, D]
    return feat.squeeze(0).cpu().numpy()  # [D]


def main():
    ckpt = "checkpoints/msvd_vit_freeze_best.pt"
    ann_path = "data/processed/msvd/train/annotations.json"
    out_dir = Path("data/processed/msvd/features")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 构建ViT模型
    model = build_vit_encoder(
        model_name="vit_base_patch16_224",
        out_dim=256,
        pretrained=True,
        pool="cls",
        l2norm=True,
        freeze=True,         # 全冻结做特征提取
        unfreeze_last=0
    ).to(device).eval()

    # 加载权重（如果有）
    if Path(ckpt).exists():
        state = torch.load(ckpt, map_location=device)
        if "vit" in state:
            model.backbone.load_state_dict(state["vit"], strict=False)
            print(f"[OK] Loaded checkpoint from {ckpt}")
        else:
            print("[WARN] No vit key found in checkpoint, using pretrained weights.")
    else:
        print(f"[WARN] Checkpoint not found: {ckpt}, using ImageNet pretrained weights.")

    # 读取 annotations
    with open(ann_path, "r", encoding="utf-8") as f:
        anns = json.load(f)

    print(f"[INFO] Extracting features for {len(anns)} videos...")
    for rec in tqdm(anns, desc="Extracting"):
        vid = rec["video_id"]
        frames_dir = Path(rec["frames_dir"])
        feat = extract_video_feature(model, frames_dir, num_frames=8, image_size=224, device=device)
        if feat is None:
            continue
        np.save(out_dir / f"{vid}.npy", feat)

    print(f"[OK] Saved features to {out_dir}")

if __name__ == "__main__":
    main()