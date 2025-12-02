#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描 MSVD 视频文件夹，打印前几个视频文件名，检查与 annotations.txt 是否对得上
"""

from pathlib import Path

# 修改为你的实际路径
RAW_DIR = Path("data/raw/msvd")

# 递归查找视频文件
VIDEO_EXTS = [".avi", ".mp4", ".webm", ".mkv", ".mov"]
video_files = []
for sub in ["train", "validation", "testing"]:
    folder = RAW_DIR / sub
    if not folder.exists():
        continue
    for ext in VIDEO_EXTS:
        video_files.extend(folder.rglob(f"*{ext}"))

print(f"总共找到视频文件: {len(video_files)}")
print("前20个文件示例：")
for f in sorted(video_files)[:20]:
    print("文件名:", f.name, " → stem:", f.stem)
