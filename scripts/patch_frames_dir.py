#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json
from pathlib import Path
import difflib

FRAMES_DIR = Path(os.getenv("FRAMES_DIR", "data/processed/msvd/frames")).resolve()

def _norm_vid(vid: str) -> str:
    # 抽帧时我们去掉了前缀 '-' / '_'
    return vid.lstrip("-_") if vid else vid

def _best_match(name: str, candidates: list[str]) -> str | None:
    m = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    return m[0] if m else None

def patch_one(split_path: Path):
    ann = json.load(open(split_path, "r", encoding="utf-8"))
    changed = 0

    # 准备候选目录名列表用于模糊匹配
    cand = []
    if FRAMES_DIR.exists():
        cand = [p.name for p in FRAMES_DIR.iterdir() if p.is_dir()]

    for rec in ann:
        # 已有 frames_dir 就不动
        if rec.get("frames_dir"):
            continue

        # 1) 优先 video_id
        vid = rec.get("video_id") or rec.get("id") or rec.get("video") or rec.get("vid") or ""
        vid = vid.lstrip("-_")

        # 优先 exact
        exact = Path(FRAMES_DIR) / vid
        if exact.exists():
            rec["frames_dir"] = exact.as_posix()
        else:
            # 再尝试模糊匹配（防止有尾巴 _224 或 _xx_yy）
            from difflib import get_close_matches
            cand = [p.name for p in Path(FRAMES_DIR).iterdir() if p.is_dir()]
            m = get_close_matches(vid, cand, n=1, cutoff=0.6)
            if m:
                rec["frames_dir"] = (Path(FRAMES_DIR) / m[0]).as_posix()

        # 3) 再不行就从 frames 列表推断父目录
        if target is None:
            frames = rec.get("frames") or rec.get("frame_paths") or []
            if frames:
                p = Path(frames[0]).parent
                if p.exists():
                    target = p

        # 4) 只有当目录真实存在时才写入；否则保持缺失，交给过滤脚本处理
        if target is not None and target.exists():
            rec["frames_dir"] = target.as_posix()
            changed += 1

    json.dump(ann, open(split_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[OK] {split_path} patched: {changed} records updated.")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        p = Path(f"data/processed/msvd/{split}/annotations.json")
        if p.exists():
            patch_one(p)
        else:
            print(f"[WARN] {p} not found, skip.")
