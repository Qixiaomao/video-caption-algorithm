#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 annotations.json 批量抽帧到记录中的 frames_dir。
- 对每个样本：若 frames_dir 下已存在 frame_*.jpg 则跳过
- 使用 ffmpeg 抽帧，默认 2 FPS，你可按需调整
"""
import json, subprocess, sys
from pathlib import Path

def extract_split(split_json: Path, fps: int = 2, max_videos: int = None):
    recs = json.loads(split_json.read_text(encoding="utf-8"))
    done = 0
    for i, r in enumerate(recs):
        if max_videos and i >= max_videos:
            break
        v = r.get("video")
        fdir = Path(r["frames_dir"])
        if not v or not Path(v).exists():
            continue
        fdir.mkdir(parents=True, exist_ok=True)
        # 若已有帧则跳过
        if any(fdir.glob("frame_*.jpg")):
            done += 1
            continue
        # 抽帧命令（2FPS，可改成 1/4/8）
        cmd = [
            "ffmpeg", "-y", "-i", v,
            "-vf", f"fps={fps}",
            str(fdir / "frame_%06d.jpg")
        ]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)
        done += 1
    print(f"[OK] {split_json} -> processed: {done}/{len(recs)}")

def main():
    base = Path("data/processed/msvd")
    for split in ["train","val","test"]:
        p = base / split / "annotations.json"
        if p.exists():
            extract_split(p, fps=2)  # 你也可以传 max_videos=50 做小规模测试
        else:
            print("[WARN]", p, "not found")

if __name__ == "__main__":
    try:
        subprocess.run(["ffmpeg","-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        sys.exit("请先安装并配置 ffmpeg 到 PATH（命令行可运行 `ffmpeg -version`）。")
    main()
