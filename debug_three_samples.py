#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick debug: run caption generation on 3 distinct validation items and print outputs.
Goal:
- Verify frames_dir differs per sample
- Verify model outputs differ across samples and stages
- Surface common pitfalls: fixed input, cached features, hard length caps, etc.
"""

import os
import json
import random
import hashlib
from pathlib import Path
from typing import List
from PIL import Image

# ====== TODO: 替换为你项目里的真实推理入口 ======
# 例如 from your_project.infer import generate_caption_stage1, ...
def generate_caption_stage1(frames_dir: Path) -> str:
    # TODO: 替换为真实逻辑
    return "STUB: please replace with real stage1 inference"

def generate_caption_stage2(frames_dir: Path) -> str:
    # TODO: 替换为真实逻辑
    return "STUB: please replace with real stage2 inference"

def generate_caption_stage3(frames_dir: Path) -> str:
    # TODO: 替换为真实逻辑
    return "STUB: please replace with real stage3 inference"
# ===============================================

VAL_JSON = Path(r"./data/processed/msvd/val/annotations.json")  # TODO: 按需修改
RANDOM_SEED = 42

def list_frames(frames_dir: Path, limit: int = 5) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    frames = [p for p in sorted(frames_dir.glob("*")) if p.suffix.lower() in exts]
    return frames[:limit]

def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 16)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:12]

def main():
    assert VAL_JSON.exists(), f"Not found: {VAL_JSON}"

    with open(VAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 过滤掉没有 frames_dir 的项
    data = [d for d in data if "video_id" in d and "frames_dir" in d]
    assert len(data) >= 3, "Validation set has fewer than 3 items."

    random.Random(RANDOM_SEED).shuffle(data)
    sample3 = data[:3]

    print(f"🧪 Selected {len(sample3)} items for quick debug.\n")

    for i, rec in enumerate(sample3, 1):
        vid = str(rec["video_id"])
        frames_dir = Path(rec["frames_dir"])

        print("="*80)
        print(f"[Item {i}] video_id: {vid}")
        print(f"frames_dir: {frames_dir.resolve()}")
        if not frames_dir.exists():
            print("❌ frames_dir not found!")
            continue

        frs = list_frames(frames_dir, limit=5)
        print(f"First frames ({len(frs)} shown): {[p.name for p in frs]}")
        if frs:
            # 用前两张图的哈希做“不同输入”的粗检
            hashes = [sha1_of_file(p) for p in frs[:2]]
            print(f"Frame hashes (first 1-2): {hashes}")

        # -------- 真正推理（请确保函数内部使用了 frames_dir 的图像/特征）--------
        cap1 = generate_caption_stage1(frames_dir)
        cap2 = generate_caption_stage2(frames_dir)
        cap3 = generate_caption_stage3(frames_dir)

        # 简单规整
        def norm(s: str) -> str:
            s = (s or "").strip()
            if s and s[-1] not in ".!?":
                s += "."
            return s

        cap1, cap2, cap3 = norm(cap1), norm(cap2), norm(cap3)

        print("\n--- Captions ---")
        print(f"Stage-1: {cap1}")
        print(f"Stage-2: {cap2}")
        print(f"Stage-3: {cap3}")

        # 高亮“可疑情况”
        suspicious = []
        if len(cap1.split()) <= 8: suspicious.append("S1<=8w")
        if len(cap2.split()) <= 8: suspicious.append("S2<=8w")
        if len(cap3.split()) <= 10: suspicious.append("S3<=10w")
        if (cap1 == cap2 == cap3):  suspicious.append("AllEqual")
        if suspicious:
            print(f"\n⚠️ Suspicious flags: {', '.join(suspicious)}")
        print()

    print("="*80)
    print("Done. If captions look identical across items, likely input/features are constant,")
    print("or generation length is hard-capped / post-truncated. See checklist in comments.\n")

if __name__ == "__main__":
    main()
