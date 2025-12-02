#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, re, subprocess, sys
from pathlib import Path

RAW = Path("data/raw/msvd")
PROC = Path("data/processed/msvd")
SPLITS = ["train", "val", "test"]
FPS = 8  # 每秒抽多少帧，可根据 num_frame 调整
VIDEO_EXTS = [".mp4",".mkv",".webm",".mov",".avi"]
PAT = re.compile(r"^([A-Za-z0-9_-]+)_(\d+)_(\d+)$")  # ytid_start_end

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)

def find_clip(split, vid):
    # data/raw/msvd/train/<video_id>.*
    for ext in VIDEO_EXTS:
        p = (RAW / split / f"{vid}{ext}")
        if p.exists():
            return p
    return None

def find_full(vid):
    m = PAT.match(vid)
    if not m: return None, None, None
    ytid, s, e = m.group(1), int(m.group(2)), int(m.group(3))
    for ext in VIDEO_EXTS:
        p = RAW / "_full" / f"{ytid}{ext}"
        if p.exists():
            return p, s, e
    return None, s, e

def ensure_frames(clip_path: Path, out_dir: Path, start=None, end=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if start is None or end is None:
        # 直接从片段抽帧
        cmd = f'ffmpeg -hide_banner -loglevel error -y -i "{clip_path}" -vf fps={FPS} "{(out_dir / "%05d.jpg").as_posix()}"'
    else:
        dur = max(0, end - start)
        cmd = f'ffmpeg -hide_banner -loglevel error -y -ss {start} -i "{clip_path}" -t {dur} -vf fps={FPS} "{(out_dir / "%05d.jpg").as_posix()}"'
    r = run(cmd)
    # 返回是否生成了一些帧
    generated = any(out_dir.glob("*.jpg"))
    return generated

def process_split(split):
    ann = PROC / split / "annotations_filtered.json"
    if not ann.exists():
        print(f"[WARN] missing {ann}, skip {split}")
        return
    items = json.loads(ann.read_text(encoding="utf-8"))
    out_items = []
    made, total = 0, 0
    for it in items:
        vid = it.get("video_id")
        if not vid:
            continue
        total += 1
        frames_dir = RAW / "frames" / split / vid
        # 已有帧就跳过
        if any(frames_dir.glob("*.jpg")):
            it["frames_dir"] = frames_dir.as_posix()
            out_items.append(it)
            continue

        clip = find_clip(split, vid)
        if clip:
            ok = ensure_frames(clip, frames_dir)
        else:
            full, s, e = find_full(vid)
            ok = False
            if full and s is not None and e is not None and e > s:
                ok = ensure_frames(full, frames_dir, start=s, end=e)

        if ok:
            made += 1
            it["frames_dir"] = frames_dir.as_posix()
            out_items.append(it)
        else:
            # 没生成出来就先不写入，避免 DataLoader 报错
            pass

    # 写出仅包含有 frames_dir 的样本
    out_file = PROC / split / "annotations_frames.json"
    out_file.write_text(json.dumps(out_items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{split}] kept:{len(out_items)}/{total}  newly_made_frames:{made}  → {out_file}")

if __name__ == "__main__":
    for sp in SPLITS:
        process_split(sp)
    print("[DONE] frames annotations generated.")
