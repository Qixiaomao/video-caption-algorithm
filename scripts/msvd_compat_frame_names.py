#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为每个 frames_dir 补充一份兼容命名的帧文件：
- 已有：00001.jpg, 00002.jpg, ...
- 生成：image_00001.jpg / img_00001.jpg 两套别名（优先硬链接，失败则复制）
"""

import os
from pathlib import Path
import shutil
import json

PROC = Path("data/processed/msvd")
RAW  = Path("data/raw/msvd")
SPLITS = ["train", "val", "test"]

def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)  # 硬链接，节省空间（NTFS 需要同一分区）
    except Exception:
        shutil.copy2(src, dst)

def process_split(split):
    ann = PROC / split / "annotations_frames.json"
    if not ann.exists():
        print(f"[WARN] skip {split}: {ann} not found")
        return
    items = json.loads(ann.read_text(encoding="utf-8"))
    fixed_dirs = 0
    for it in items:
        frames_dir = Path(it.get("frames_dir", ""))
        if not frames_dir.is_dir():
            continue
        # 找数字命名的帧
        jpgs = sorted(frames_dir.glob("[0-9][0-9][0-9][0-9][0-9].jpg"))
        if not jpgs:
            continue
        # 为前 N 张生成兼容名字（全部也可以，只是更慢；这里全量生成）
        for src in jpgs:
            stem = src.stem  # e.g., "00001"
            for prefix in ("image_", "img_"):
                dst = src.with_name(f"{prefix}{stem}.jpg")
                link_or_copy(src, dst)
        fixed_dirs += 1
    print(f"[{split}] processed frame dirs: {fixed_dirs}")

if __name__ == "__main__":
    for sp in SPLITS:
        process_split(sp)
    print("[DONE] compatibility names created.")
