#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 processed/msvd/{train,val,test}/annotations.json 里 video 为 None 的项，
尽量补上可用的视频路径：
1) 优先使用已裁剪片段：data/raw/msvd/{split}/{video_id}.mp4
2) 若没有片段，尝试用完整视频：data/raw/msvd/_full/{ytid}.*  （ytid 来自 video_id 前缀）
同时生成两份文件：
- annotations_patched.json   （尽量补齐 path，仍可能有少量 None）
- annotations_filtered.json  （只保留有 video 路径的样本，适合直接训练）
"""

import json, re
from pathlib import Path

RAW = Path("data/raw/msvd")
PROC = Path("data/processed/msvd")
SPLITS = ["train","val","test"]
VIDEO_EXTS = [".mp4",".mkv",".webm",".mov",".avi"]
pat = re.compile(r"^([A-Za-z0-9_-]+)_(\d+)_(\d+)$")  # ytid_start_end

def find_cut(split, vid):
    p = RAW / split / f"{vid}.mp4"
    if p.exists(): return p.as_posix()
    # 容错：试试其它后缀
    for ext in VIDEO_EXTS:
        p = RAW / split / f"{vid}{ext}"
        if p.exists(): return p.as_posix()
    return None

def find_full(vid):
    m = pat.match(vid)
    if not m: return None
    ytid = m.group(1)
    for ext in VIDEO_EXTS:
        p = RAW / "_full" / f"{ytid}{ext}"
        if p.exists(): return p.as_posix()
    return None

def process_split(split):
    ann = PROC / split / "annotations.json"
    if not ann.exists():
        print(f"[WARN] missing {ann}")
        return
    items = json.loads(ann.read_text(encoding="utf-8"))
    patched, filtered = [], []
    fixed, kept = 0, 0

    # 支持 grouped 和 flat 两种结构
    key_caption_list = "captions"  # grouped
    key_caption = "caption"        # flat

    for it in items:
        vid = it.get("video_id")
        vpath = it.get("video", None)

        if not vpath or vpath == "null":
            # 先找裁剪片段
            vpath = find_cut(split, vid)
            if not vpath:
                # 用完整视频兜底
                vpath = find_full(vid)
            if vpath:
                it["video"] = vpath
                fixed += 1

        patched.append(it)
        if it.get("video"):
            filtered.append(it); kept += 1

    (PROC / split / "annotations_patched.json").write_text(
        json.dumps(patched, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (PROC / split / "annotations_filtered.json").write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = len(items)
    print(f"[{split}] total:{total}  fixed:{fixed}  kept_with_video:{kept}")

if __name__ == "__main__":
    for sp in SPLITS:
        process_split(sp)
    print("[DONE] wrote annotations_* for all splits.")
