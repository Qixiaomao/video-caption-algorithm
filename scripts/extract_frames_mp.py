#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多进程抽帧到 annotations.json 里的 frames_dir
- 默认只处理“缺帧”的样本（--only-missing）
- 可选择 split: train/val/test/all
- 可调并发 --workers、帧率 --fps
- 已有帧将跳过，安全可重复执行
"""
import argparse, json, subprocess, sys, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def has_frames(frames_dir: Path) -> bool:
    return frames_dir.exists() and any(frames_dir.glob("frame_*.jpg"))

def ffmpeg_exists() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def extract_one(video: Path, frames_dir: Path, fps: int = 2, overwrite: bool = False) -> tuple[bool,str]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    # 如果已有帧，且不覆盖则跳过
    if not overwrite and has_frames(frames_dir):
        return True, f"skip(has frames) -> {frames_dir}"
    cmd = ["ffmpeg", "-y" if overwrite else "-n", "-i", str(video), "-vf", f"fps={fps}", str(frames_dir / "frame_%06d.jpg")]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode == 0 and has_frames(frames_dir):
            return True, f"ok -> {frames_dir}"
        else:
            return False, f"ffmpeg failed({r.returncode}) {video} :: {r.stderr.splitlines()[-1] if r.stderr else ''}"
    except Exception as e:
        return False, f"exception {video}: {e}"

def load_split(ann_path: Path) -> list[dict]:
    return json.loads(ann_path.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/processed/msvd", help="processed 根目录")
    ap.add_argument("--splits", default="train", help="train,val,test,all")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only-missing",dest="only_missing", action="store_true", default=True)
    ap.add_argument("--overwrite", action="store_true", help="强制重抽")
    args = ap.parse_args()

    base = Path(args.base)
    if not ffmpeg_exists():
        sys.exit("请先安装 ffmpeg 并确保命令行可用（ffmpeg -version）。")

    splits = (["train","val","test"] if args.splits=="all"
              else [s.strip() for s in args.splits.split(",") if s.strip()])

    for split in splits:
        ann = base / split / "annotations.json"
        if not ann.exists():
            print(f"[WARN] {ann} not found, skip.")
            continue
        recs = load_split(ann)
        print(f"[INFO] split={split} total={len(recs)}")

        # 仅处理缺帧的样本
        tasks = []
        for r in recs:
            v = Path(r.get("video",""))
            fdir = Path(r.get("frames_dir",""))
            if not v.exists():
                continue
            if args.only_missing and has_frames(fdir) and not args.overwrite:
                continue
            tasks.append((v, fdir))

        print(f"[INFO] pending={len(tasks)} (only-missing={args.only_missing}) fps={args.fps} workers={args.workers}")
        ok = fail = 0
        if not tasks:
            print("[OK] no tasks to run.")
            continue

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(extract_one, v, fdir, args.fps, args.overwrite) for v, fdir in tasks]
            for fut in as_completed(futs):
                success, msg = fut.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                if (ok+fail) % 20 == 0:
                    print(f" .. progress {ok+fail}/{len(tasks)} (ok={ok}, fail={fail})")
        print(f"[DONE] split={split} ok={ok} fail={fail}")

        # 覆盖率报告
        have = sum(1 for r in recs if has_frames(Path(r.get("frames_dir",""))))
        print(f"[COVERAGE] {split}: with_frames={have}/{len(recs)} ({have/len(recs):.1%})")

if __name__ == "__main__":
    main()
