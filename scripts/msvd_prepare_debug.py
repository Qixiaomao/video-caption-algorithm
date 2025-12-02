#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEBUG 版 MSVD 预处理：
- 兼容 Python 3.7+（不使用 list[dict]，改用 typing.List/Dict）
- 全程打印进度，异常时打印完整堆栈
- 生成 grouped/flat 的 annotations.json，并写 manifest.json
- 记录 frames_dir 字段，便于 Dataloader
"""

import argparse, json, re, sys, traceback, random
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

VIDEO_EXTS = [".mp4",".avi",".webm",".mkv",".mov"]

def log(msg: str):
    print(msg, flush=True)

def load_annotations_txt(txt_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\S+)\s+(.*)$", line)
            if not m:
                continue
            vid, cap = m.group(1).strip(), m.group(2).strip()
            if vid and cap:
                rows.append({"video_id": vid, "caption": cap})
    if not rows:
        raise SystemExit(f"[ERROR] 未能从 {txt_path} 解析出任何 (video_id, caption) 行。")
    return rows

def index_videos(split_dirs: List[Path]) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for sd in split_dirs:
        if not sd.exists():
            continue
        for ext in VIDEO_EXTS:
            for p in sd.rglob(f"*{ext}"):
                idx[p.stem] = p
    return idx

def grouped_records(group: Dict[str, List[str]], video_map: Dict[str, Path], frames_root: Path) -> List[Dict]:
    items: List[Dict] = []
    for vid, caps in group.items():
        vpath = video_map.get(vid)
        frames_dir = (frames_root / vid)
        items.append({
            "video_id": vid,
            "video": str(vpath.as_posix()) if vpath else None,
            "frames_dir": str(frames_dir.as_posix()),
            "captions": caps
        })
    return items

def flat_records(group: Dict[str, List[str]], video_map: Dict[str, Path], frames_root: Path) -> List[Dict]:
    items: List[Dict] = []
    for vid, caps in group.items():
        vpath = video_map.get(vid)
        frames_dir = (frames_root / vid)
        for cap in caps:
            items.append({
                "video_id": vid,
                "video": str(vpath.as_posix()) if vpath else None,
                "frames_dir": str(frames_dir.as_posix()),
                "caption": cap
            })
    return items

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--format", choices=["grouped","flat"], default="grouped")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow_missing_video", action="store_true")
    args = ap.parse_args()

    log(f"[DEBUG] Python: {sys.version.split()[0]}  cwd: {Path.cwd()}")
    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    anno_txt = raw / "annotations.txt"
    log(f"[DEBUG] raw_dir={raw}  out_dir={out}  format={args.format}  seed={args.seed}")

    if not anno_txt.exists():
        raise SystemExit(f"[ERROR] {anno_txt} 不存在。")
    log("[STEP] 1/6 读取 annotations.txt ...")
    rows = load_annotations_txt(anno_txt)
    log(f"[INFO] loaded caption rows: {len(rows)}")

    log("[STEP] 2/6 聚合 captions 按 video_id ...")
    grouped_caps: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        grouped_caps[r["video_id"]].append(r["caption"])
    log(f"[INFO] unique videos in annotations: {len(grouped_caps)}")

    log("[STEP] 3/6 建立本地视频索引 ...")
    split_roots = [raw/"train", raw/"validation", raw/"testing"]
    video_map = index_videos(split_roots)
    log(f"[INFO] indexed local video files: {len(video_map)}")

    log("[STEP] 3.5/6 过滤可用 video_id ...")
    all_vids = sorted(grouped_caps.keys())
    if args.allow_missing_video:
        vids = all_vids
        log("[WARN] allow_missing_video=True，允许保留 video=None 的样本")
    else:
        vids = [vid for vid in all_vids if vid in video_map]
    if not vids:
        raise SystemExit("[ERROR] 没有可用的 video_id（可能未解压视频或路径不对）。")
    log(f"[INFO] usable video_ids: {len(vids)}")

    log("[STEP] 4/6 划分 train/val/test (8/1/1) ...")
    random.seed(args.seed)
    random.shuffle(vids)
    n = len(vids)
    n_tr = int(n*0.8); n_va = int(n*0.1)
    tr_ids = vids[:n_tr]; va_ids = vids[n_tr:n_tr+n_va]; te_ids = vids[n_tr+n_va:]
    log(f"[INFO] split sizes -> train={len(tr_ids)} val={len(va_ids)} test={len(te_ids)}")

    def pick(ids: List[str], split_name: str) -> List[Dict]:
        g = {vid: grouped_caps[vid] for vid in ids}
        frames_root = out / split_name / "frames"
        frames_root.mkdir(parents=True, exist_ok=True)
        return grouped_records(g, video_map, frames_root) if args.format == "grouped" \
               else flat_records(g, video_map, frames_root)

    log("[STEP] 5/6 生成记录并写出 annotations.json ...")
    splits = {"train": pick(tr_ids,"train"), "val": pick(va_ids,"val"), "test": pick(te_ids,"test")}
    for name, recs in splits.items():
        out_file = out / name / "annotations.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        log(f"[OK] {name}: {len(recs)} -> {out_file}")

    log("[STEP] 6/6 写出 manifest.json ...")
    manifest = {
        "total_ids_after_filter": n,
        "counts": {k: len(v) for k, v in splits.items()},
        "with_video_counts": {k: sum(1 for r in v if r.get("video")) for k, v in splits.items()},
        "seed": args.seed,
        "allow_missing_video": args.allow_missing_video,
        "format": args.format,
        "raw_dir": str(raw.as_posix()),
        "out_dir": str(out.as_posix()),
    }
    (out/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[OK] wrote manifest -> {(out/'manifest.json').as_posix()}")

if __name__ == "__main__":
    try:
        run()
    except SystemExit as e:
        # 打印 SystemExit 消息，避免 PowerShell 静默
        log(f"{e}")
        raise
    except Exception:
        # 打印完整堆栈，避免“无输出”
        traceback.print_exc()
        sys.exit(1)
