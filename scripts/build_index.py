#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build FAISS index with captions for MSVD features
"""

import json
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm

def main():
    feat_dir = Path("data/processed/msvd/features")
    ann_path = Path("data/processed/msvd/train/annotations.json")
    out_dir = Path("data/processed/msvd/faiss_index")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_path, "r", encoding="utf-8") as f:
        anns = json.load(f)

    feats, metas = [], []
    for rec in tqdm(anns, desc="Loading features"):
        vid = rec["video_id"]
        fpath = feat_dir / f"{vid}.npy"
        if not fpath.exists():
            continue
        feats.append(np.load(fpath))
        # 带上caption（第一个或随机一条）
        caption = rec["captions"][0] if isinstance(rec["captions"], list) and rec["captions"] else ""
        metas.append({
            "video_id": vid,
            "caption": caption
        })

    feats = np.stack(feats).astype("float32")
    dim = feats.shape[1]

    print(f"[INFO] Building FAISS index with {len(feats)} vectors, dim={dim}")
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(feats)
    index.add(feats)

    faiss.write_index(index, str(out_dir / "video.index"))
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print(f"[OK] index -> {out_dir/'video.index'}")
    print(f"[OK] meta  -> {out_dir/'meta.json'}")
    print(f"[OK] num vectors: {len(feats)} dim={dim}")

if __name__ == "__main__":
    main()