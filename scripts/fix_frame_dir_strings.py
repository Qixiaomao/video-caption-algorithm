#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, re
from pathlib import Path

ROOT = Path(".").resolve()
FRAMES_ROOT = ROOT / "data/processed/msvd/frames"

LEADING = re.compile(r"^[-_]+")        # 去掉开头 - 或 _
TAIL    = re.compile(r"(_\d+_\d+)$")   # 去掉尾部 _start_end

def norm_name(name: str) -> str:
    if not name: return name
    name = LEADING.sub("", name)
    name = TAIL.sub("", name)
    return name

def fix_split(split: str):
    ann_path = ROOT / f"data/processed/msvd/{split}/annotations.json"
    if not ann_path.exists():
        print(f"[WARN] {ann_path} not found, skip")
        return
    ann = json.load(open(ann_path, "r", encoding="utf-8"))
    changed, ok = 0, 0
    for rec in ann:
        fd = rec.get("frames_dir", "")
        if not fd:
            # 试着用 video_id 推断
            vid = rec.get("video_id") or rec.get("id") or rec.get("video") or rec.get("vid") or ""
            vid = norm_name(vid)
            cand = FRAMES_ROOT / vid
        else:
            # 从 frames_dir 字符串里提取最后一段目录名并归一化
            tail = Path(fd).name
            tail = norm_name(tail)
            cand = FRAMES_ROOT / tail

        if cand.exists():
            new_fd = cand.as_posix()
            if rec.get("frames_dir") != new_fd:
                rec["frames_dir"] = new_fd
                changed += 1
            ok += 1
        else:
            # 留空让过滤脚本丢弃
            rec["frames_dir"] = rec.get("frames_dir", "")

    out = ann_path.with_name("annotations.patched.json")
    json.dump(ann, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[OK] {split}: total={len(ann)}  fixed={changed}  valid_now={ok} -> {out}")

if __name__ == "__main__":
    for s in ["train", "val", "test"]:
        fix_split(s)
