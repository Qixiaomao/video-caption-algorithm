#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decode ablation on a single model:
- 固定视觉+文本权重，遍历多组解码参数（beam/temperature/top_p/no_repeat...）
- 在 val 上跑若干样本，统计 corpus BLEU，导出 CSV
"""

import argparse, json, logging, sys, itertools
from pathlib import Path
from typing import List, Dict, Any

import torch
import sacrebleu
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.models.caption_model import VideoCaptionModel


def setup_logging(level="INFO", log_file=None):
    level = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=handlers,
                        force=True)
    return logging.getLogger("ablate_decode")

def load_ann(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def resolve_frames_dir(rec, frames_root: Path):
    if rec.get("frames_dir"):
        return Path(rec["frames_dir"])
    return frames_root / rec["video_id"]

def sample_frames(frames_dir: Path, num_frames: int, image_size: int, device: str):
    files = sorted(frames_dir.glob("frame_*.jpg"))
    if not files:
        return None
    step = max(len(files) // num_frames, 1)
    picks = files[::step][:num_frames]
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    imgs = []
    for p in picks:
        with Image.open(p) as im:
            imgs.append(tfm(im.convert("RGB")))
    video = torch.stack(imgs, dim=0).unsqueeze(0).to(device)
    return video

def bleu_corpus(refs_list: List[List[str]], hyps: List[str]) -> float:
    max_r = max(len(r) for r in refs_list)
    ref_groups = []
    for r_idx in range(max_r):
        group = []
        for refs in refs_list:
            group.append(refs[r_idx] if r_idx < len(refs) and refs[r_idx].strip() else (refs[0] if refs else ""))
        ref_groups.append(group)
    return float(sacrebleu.corpus_bleu(hyps, ref_groups).score)

def main():
    ap = argparse.ArgumentParser("Ablate decoding params for a single model")
    ap.add_argument("--ann", required=True)
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--limit", type=int, default=60)

    ap.add_argument("--vit_name", default="vit_base_patch16_224")
    ap.add_argument("--gpt2_name", default="gpt2")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--cond_mode", choices=["prefix","bos"], default="prefix")
    ap.add_argument("--prefix_len", type=int, default=4)
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)

    # 搜索空间（可以改小/改大）
    ap.add_argument("--grid_beams", nargs="+", type=int, default=[1,3,5])
    ap.add_argument("--grid_temp",  nargs="+", type=float, default=[0.7,0.8,1.0])
    ap.add_argument("--grid_topp",  nargs="+", type=float, default=[0.8,0.9,0.95])
    ap.add_argument("--grid_ngram", nargs="+", type=int,   default=[2,3,4])

    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--min_new_tokens", type=int, default=8)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)

    ap.add_argument("--out_dir", default="eval_results/ablate")
    ap.add_argument("--log_level", default="INFO")
    ap.add_argument("--log_file", default=None)
    args = ap.parse_args()

    logger = setup_logging(args.log_level, args.log_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ann = load_ann(Path(args.ann))
    if args.limit > 0:
        ann = ann[:args.limit]
    frames_root = Path(args.frames_root)

    logger.info(f"Samples: {len(ann)}  frames_root={frames_root}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # build model
    model = VideoCaptionModel(
        vit_name=args.vit_name,
        gpt2_name=args.gpt2_name,
        cond_mode=args.cond_mode,
        prefix_len=args.prefix_len,
        freeze_vit=True,
        unfreeze_last=0
    ).to(device).eval()
    if args.ckpt and Path(args.ckpt).exists():
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded checkpoint: {args.ckpt}")
    else:
        logger.warning("No ckpt loaded (use default weights).")

    grid = list(itertools.product(args.grid_beams, args.grid_temp, args.grid_topp, args.grid_ngram))
    logger.info(f"Grid size: {len(grid)}")

    results = []
    for beams, temp, topp, ngram in grid:
        refs_list, hyps = [], []
        for rec in ann:
            fdir = resolve_frames_dir(rec, frames_root)
            if not fdir.exists():
                continue
            video = sample_frames(fdir, args.num_frames, args.image_size, device)
            if video is None:
                continue

            with torch.no_grad():
                hyp = model.generate(
                    video,
                    prompt="",
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    num_beams=beams,
                    temperature=temp,
                    top_p=topp,
                    no_repeat_ngram_size=ngram,
                    repetition_penalty=args.repetition_penalty,
                )[0].strip()
            hyps.append(hyp)
            refs = [c for c in rec.get("captions", []) if isinstance(c, str) and c.strip()]
            refs = refs if refs else [""]
            refs_list.append(refs)

        bleu = bleu_corpus(refs_list, hyps) if hyps else 0.0
        results.append({
            "num_beams": beams,
            "temperature": temp,
            "top_p": topp,
            "no_repeat_ngram_size": ngram,
            "BLEU": round(bleu, 2),
            "N": len(hyps)
        })
        logger.info(f"beams={beams} temp={temp} top_p={topp} ngram={ngram} -> BLEU={bleu:.2f}")

    df = pd.DataFrame(results).sort_values(by="BLEU", ascending=False)
    csv_path = out_dir / "ablate_decode.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("\n===== TOP CONFIGS =====")
    print(df.head(10).to_string(index=False))
    logger.info(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()