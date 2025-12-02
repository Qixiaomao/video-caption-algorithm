#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract frames from MSVD videos on Windows
Usage:
  python scripts/extract_frames.py --limit 50 --fps 2 --image-size 224
"""

import argparse, subprocess, shutil
from pathlib import Path

RAW_V_DIR = Path("data/raw/msvd/_full")
FRAMES_DIR = Path("data/processed/msvd/frames")

def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def extract_frames(limit: int, fps: int, image_size: int):
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    vids = [p for p in RAW_V_DIR.glob("*") if p.suffix.lower() in (".mp4",".mkv",".webm") and not p.name.endswith(".part")]
    if limit and limit > 0:
        vids = vids[:limit]

    print(f"[INFO] found {len(vids)} videos, processing {len(vids)}")
    for v in vids:
        vid_id = v.stem.lstrip("-_")  # 去掉开头的 - 或 _
        out_dir = FRAMES_DIR / vid_id
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y", "-i", str(v),
            "-vf", f"fps={fps},scale={image_size}:{image_size}",
            str(out_dir / "%04d.jpg")
        ]
        print("[CMD]", " ".join(cmd))
        subprocess.call(cmd)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50, help="抽多少个视频 (0 = 全部)")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    if not have_ffmpeg():
        print("[ERROR] ffmpeg 未安装或未在 PATH 里，请先安装。")
        exit(1)

    extract_frames(args.limit, args.fps, args.image_size)
    print("[DONE] 抽帧完成，帧保存在 data/processed/msvd/frames/")
