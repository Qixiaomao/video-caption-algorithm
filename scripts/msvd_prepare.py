#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSVD (Kaggle) with annotations.txt → annotations.json
- 读取 data/raw/msvd/annotations.txt （两列：video_id, caption）
- 递归在 train/validation/testing 目录中查找对应视频文件
- 输出到 data/processed/msvd/{train,val,test}/annotations.json
- 支持 grouped/flat 两种格式
"""

import argparse, json, csv, re
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import random

VIDEO_EXTS = [".mp4",".avi",".webm",".mkv",".mov"]

def sniff_delimiter(txt_path: Path) -> str:
    # 依次尝试常见分隔符：\t, ',', '|', ':'
    candidates = ["\t", ",", "|", ":"]
    head = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:5]
    for delim in candidates:
        ok = True
        for line in head:
            if not line.strip():
                continue
            if delim not in line:
                ok = False
                break
        if ok:
            return delim
    # 回退：用制表符
    return "\t"



def load_annotations_txt(txt_path: Path) -> list[dict]:
    """
    解析形如：
    <video_id><space(s)><caption...>
    例如：
    -4wsuPCjDBc_5_15 a squirrel is eating a peanut in it s shell
    """
    rows = []
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 用正则按“第一处空白”切分
            m = re.match(r"^(\S+)\s+(.*)$", line)
            if not m:
                continue
            vid, cap = m.group(1).strip(), m.group(2).strip()
            if vid and cap:
                rows.append({"video_id": vid, "caption": cap})
    if not rows:
        raise SystemExit(f"[ERROR] 未能从 {txt_path} 解析出任何 (video_id, caption) 行，请打开看下格式。")
    return rows

def index_videos(split_dirs: List[Path]) -> Dict[str, Path]:
    """
    递归索引三个 split 目录下所有视频，key=不带扩展名的文件名（video_id）
    """
    idx = {}
    for sd in split_dirs:
        if not sd.exists(): 
            continue
        for ext in VIDEO_EXTS:
            for p in sd.rglob(f"*{ext}"):
                idx[p.stem] = p
    return idx

def grouped_records(group: Dict[str, List[str]], video_map: Dict[str, Path], frames_root: Path) -> List[Dict]:
    items = []
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
    items = []
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

def main():
    

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="原始MSVD根目录，如 data/raw/msvd")
    ap.add_argument("--out_dir", required=True, help="处理后输出目录，如 data/processed/msvd")
    ap.add_argument("--format", choices=["grouped", "flat"], default="grouped",
                    help="annotations.json 的记录格式")
    ap.add_argument("--seed", type=int, default=42, help="划分随机种子")
    ap.add_argument("--allow_missing_video", action="store_true",
                    help="允许保留找不到本地视频文件的样本（video=None）")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    anno_txt = raw / "annotations.txt"

    if not anno_txt.exists():
        raise SystemExit(f"[ERROR] {anno_txt} 不存在，请确认 Kaggle 包是否解压完全。")

    # 1) 读 annotations.txt（video_id 与 caption）
    rows = load_annotations_txt(anno_txt)
    print(f"[INFO] loaded caption rows: {len(rows)}")

    # 2) 按 video_id 聚合 captions
    grouped_caps = defaultdict(list)
    for r in rows:
        grouped_caps[r["video_id"]].append(r["caption"])
    print(f"[INFO] unique videos in annotations: {len(grouped_caps)}")

    # 3) 建视频文件索引（从 train/validation/testing 递归查找）
    split_roots = [raw / "train", raw / "validation", raw / "testing"]
    video_map = index_videos(split_roots)
    print(f"[INFO] indexed local video files: {len(video_map)}")

    # 3.5) 过滤：默认仅保留“有本地视频文件”的 video_id
    all_vids = sorted(grouped_caps.keys())
    if args.allow_missing_video:
        vids = all_vids
        print("[WARN] allow_missing_video=True，数据集中将保留找不到视频文件的样本（video=None）")
    else:
        vids = [vid for vid in all_vids if vid in video_map]
    if not vids:
        raise SystemExit("[ERROR] 没有可用的 video_id（可能未解压视频或目录不对）。")

    # 4) 划分（先打乱再 8/1/1）
    random.seed(args.seed)
    random.shuffle(vids)
    n = len(vids)
    n_tr = int(n * 0.8)
    n_va = int(n * 0.1)
    tr_ids = vids[:n_tr]
    va_ids = vids[n_tr:n_tr + n_va]
    te_ids = vids[n_tr + n_va:]

    def pick(ids: List[str], split_name: str) -> List[Dict]:
        """把指定 split 的 video_id 列表转成记录列表，并约定 frames_dir 根路径。"""
        g = {vid: grouped_caps[vid] for vid in ids}
        frames_root = out / split_name / "frames"
        frames_root.mkdir(parents=True, exist_ok=True)
        if args.format == "grouped":
            return grouped_records(g, video_map, frames_root)
        return flat_records(g, video_map, frames_root)

    # 5) 组装三个 splits
    splits = {
        "train": pick(tr_ids, "train"),
        "val":   pick(va_ids, "val"),
        "test":  pick(te_ids, "test"),
    }

    # 6) 写出 annotations.json
    for name, recs in splits.items():
        out_file = out / name / "annotations.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        print(f"[OK] {name}: {len(recs)} -> {out_file}")

    # 7) 写出 manifest.json（统计信息，便于自检）
    manifest = {
        "total_ids_after_filter": n,
        "counts": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "with_video_counts": {
            "train": sum(1 for r in splits["train"] if r.get("video")),
            "val":   sum(1 for r in splits["val"] if r.get("video")),
            "test":  sum(1 for r in splits["test"] if r.get("video")),
        },
        "seed": args.seed,
        "allow_missing_video": args.allow_missing_video,
        "format": args.format,
        "raw_dir": str(raw.as_posix()),
        "out_dir": str(out.as_posix()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote manifest -> {(out / 'manifest.json').as_posix()}")
