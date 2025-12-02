#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, re, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

RAW_ROOT   = Path("data/raw/msvd")
PROC_ROOT  = Path("data/processed/msvd")
FULL_VROOT = RAW_ROOT / "_full"       # 原视频下载目录
SPLITS     = ["train","val","test"]   # 读取你已生成的 splits
MAX_WORKERS = 4                        # 视网速/磁盘调节

YT_TMPL = "https://www.youtube.com/watch?v={}"

pat = re.compile(r"^([A-Za-z0-9_-]+)_(\d+)_(\d+)$")

def parse_vid(vid: str):
    m = pat.match(vid)
    if not m: return None
    ytid, s, e = m.group(1), int(m.group(2)), int(m.group(3))
    if e <= s: return None
    return ytid, s, e

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def download_full_video(ytid: str) -> Path | None:
    ensure_dir(FULL_VROOT)
    # 统一 mp4 容器（yt-dlp 会选最优编码）
    out_tpl = FULL_VROOT / f"{ytid}.%(ext)s"
    # 若已有 mp4/mkv/webm 任意一种，直接复用
    for ext in (".mp4",".mkv",".webm",".mov",".avi"):
        cand = FULL_VROOT / f"{ytid}{ext}"
        if cand.exists():
            return cand
    cmd = f'yt-dlp -f "bv*+ba/b" -o "{out_tpl.as_posix()}" "{YT_TMPL.format(ytid)}"'
    res = run(cmd)
    # 查找实际输出文件
    for ext in (".mp4",".mkv",".webm",".mov",".avi"):
        cand = FULL_VROOT / f"{ytid}{ext}"
        if cand.exists():
            return cand
    return None

def cut_segment(src: Path, dst: Path, start: int, end: int):
    ensure_dir(dst.parent)
    dur = end - start
    # 直接流拷贝可能在部分封装上对齐出问题，这里用重新编码更稳（慢一些）
    cmd = f'ffmpeg -hide_banner -loglevel error -ss {start} -i "{src}" -t {dur} -c:v libx264 -c:a aac -y "{dst}"'
    return run(cmd)

def collect_targets():
    targets = []  # (split, video_id, ytid, start, end, out_path)
    for sp in SPLITS:
        ann = PROC_ROOT / sp / "annotations.json"
        if not ann.exists():
            print(f"[WARN] missing {ann}, skip {sp}")
            continue
        items = json.loads(ann.read_text(encoding="utf-8"))
        # 支持 grouped/flat 两种结构
        vids = set()
        for it in items:
            vid = it.get("video_id")
            if not vid: continue
            vids.add(vid)
        for vid in sorted(vids):
            parsed = parse_vid(vid)
            if not parsed: 
                continue
            ytid, s, e = parsed
            out_path = RAW_ROOT / sp / f"{vid}.mp4"
            targets.append((sp, vid, ytid, s, e, out_path))
    return targets

def worker(task):
    sp, vid, ytid, s, e, outp = task
    if outp.exists():
        return (vid, "skip_exists")
    full = download_full_video(ytid)
    if full is None:
        return (vid, "download_fail")
    r = cut_segment(full, outp, s, e)
    if outp.exists() and outp.stat().st_size > 0:
        return (vid, "ok")
    return (vid, "cut_fail")

def main():
    tasks = collect_targets()
    print(f"[INFO] total clips to prepare: {len(tasks)}")
    ok = 0
    fails = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        for fu in as_completed(futs):
            vid, status = fu.result()
            if status == "ok" or status == "skip_exists":
                ok += 1
            else:
                fails.append((vid, status))
                print(f"[WARN] {vid}: {status}")
    print(f"[DONE] prepared: {ok}/{len(tasks)}, fails: {len(fails)}")
    if fails:
        (RAW_ROOT/"fetch_failed.json").write_text(
            json.dumps(fails, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[INFO] wrote fail list: {RAW_ROOT/'fetch_failed.json'}")

if __name__ == "__main__":
    main()
