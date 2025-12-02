#./tools/batch_hybrid_infer.py
#-- coding: utf-8 -*-

from __future__ import annotations
import json,csv,argparse
from pathlib import Path
from hybrid_infer import hybrid_caption

def iter_frame_dirs(root: Path):
    for p in sorted(root.iterdir()):
        if p.is_dir() and list(p.glob("frame_*.jpg")):
            yield p

def main():
    ap = argparse.ArgumentParser("Batch hybrid inference")
    ap.add_argument("--root", required=True, help="根目录（包含多个帧子目录）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_csv", default="data/human_eval_samples/hybrid_batch.csv")
    ap.add_argument("--out_json", default="data/human_eval_samples/hybrid_batch.json")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    root = Path(args.root)
    out_rows, out_json = [], []
    for i, frames_dir in enumerate(iter_frame_dirs(root)):
        if i >= args.limit: break
        out = hybrid_caption(frames_dir, ckpt=args.ckpt, blip_fallback=True)
        row = {
            "id": frames_dir.name,
            "S1": out["S1"],
            "S2": out["S2"],
            "S3": out["S3"],
            "BEST": out["BEST"]["text"],
            "USED": out["USED"]["source"]
        }
        out_rows.append(row)
        out_json.append({"id": frames_dir.name, **out})

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved {len(out_rows)} rows -> {args.out_csv}")
    print(f"[OK] saved JSON -> {args.out_json}")

if __name__ == "__main__":
    main()