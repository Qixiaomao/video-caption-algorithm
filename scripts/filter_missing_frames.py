#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}

def count_frames(frames_dir: Path) -> int:
    # 更稳：不用 glob(as_posix())，直接用 pathlib 遍历
    if not frames_dir.exists():
        return 0
    n = 0
    for p in frames_dir.iterdir():  # 非递归；帧都在该目录下
        if p.is_file() and p.suffix in IMG_EXTS:
            n += 1
    return n

def filter_split(in_path: Path, out_path: Path, min_frames: int = 8, debug: int = 0):
    ann = json.load(open(in_path, "r", encoding="utf-8"))
    keep, drop = [], []
    printed = 0

    for idx, rec in enumerate(ann):
        fd = rec.get("frames_dir", "") or ""
        # 统一 & 解析为绝对路径
        frames_dir = Path(fd.strip()).resolve() if fd else None

        if not frames_dir or not frames_dir.exists():
            drop.append({"rec": rec, "reason": f"dir_not_exists:{fd}"})
            if debug and printed < debug:
                print(f"[DBG] MISS-DIR idx={idx} frames_dir='{fd}'")
                printed += 1
            continue

        n = count_frames(frames_dir)
        if n >= min_frames:
            keep.append(rec)
        else:
            drop.append({"rec": rec, "reason": f"too_few_frames({n})<{min_frames} in {frames_dir}"})
            if debug and printed < debug:
                print(f"[DBG] FEW-FRAMES idx={idx} n={n} dir={frames_dir}")
                printed += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(keep, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    drop_log = out_path.with_suffix(".dropped.json")
    json.dump(drop, open(drop_log, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[OK] {in_path} -> {out_path}")
    print(f" kept: {len(keep)}  dropped: {len(drop)}")
    print(f" dropped details -> {drop_log}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--min_frames", type=int, default=8)
    ap.add_argument("--debug", type=int, default=0, help="print first N dropped details inline")
    args = ap.parse_args()

    filter_split(Path(args.ann), Path(args.out), args.min_frames, args.debug)

if __name__ == "__main__":
    main()
