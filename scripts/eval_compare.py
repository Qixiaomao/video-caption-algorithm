#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two captioning setups on MSVD val/test with BLEU:
- A/B 可分别指定 ckpt 与 gpt2_name（比如 A=stage2+base GPT-2，B=stage2+stage3微调GPT-2）
- 直接使用已抽好的帧：<split>/frames/<video_id>/frame_*.jpg
- 输出: results.csv + summary.txt
"""

import argparse, json, logging, sys
from pathlib import Path
from typing import List, Dict, Any

import torch
import sacrebleu
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.models.caption_model import VideoCaptionModel


# ---------- Logging ----------
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
    return logging.getLogger("eval_compare")


# ---------- Data Utils ----------
def load_ann(ann_path: Path) -> List[Dict[str, Any]]:
    rows = json.loads(ann_path.read_text(encoding="utf-8"))
    return rows

def resolve_frames_dir(rec: Dict[str, Any], frames_root: Path) -> Path:
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
    video = torch.stack(imgs, dim=0).unsqueeze(0).to(device)  # [1,T,3,H,W]
    return video, len(files), len(picks)

def build_model(vit_name: str, gpt2_name: str, cond_mode: str, prefix_len: int,
                ckpt: str | None, device: str, logger: logging.Logger):
    model = VideoCaptionModel(
        vit_name=vit_name,
        gpt2_name=gpt2_name,
        cond_mode=cond_mode,
        prefix_len=prefix_len,
        freeze_vit=True,
        unfreeze_last=0
    ).to(device).eval()
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"[{ckpt}] missing keys: {sorted(missing)}")
        if unexpected:
            logger.warning(f"[{ckpt}] unexpected keys: {sorted(unexpected)}")
        logger.info(f"Loaded checkpoint: {ckpt}")
    else:
        logger.warning(f"No checkpoint loaded for {ckpt} (use default weights).")
    return model


# ---------- BLEU ----------
def bleu_corpus(refs_list: List[List[str]], hyps: List[str]) -> float:
    """
    refs_list: list of reference lists per sample (N x R)
    hyps: list of hypotheses (N)
    """
    # sacrebleu expects refs grouped by reference: R x N
    # transform to that shape
    max_r = max(len(r) for r in refs_list)
    ref_groups = []
    for r_idx in range(max_r):
        group = []
        for refs in refs_list:
            if r_idx < len(refs) and refs[r_idx].strip():
                group.append(refs[r_idx])
            else:
                # fallback to first ref if missing
                group.append(refs[0] if refs else "")
        ref_groups.append(group)
    bleu = sacrebleu.corpus_bleu(hyps, ref_groups)
    return float(bleu.score)


def main():
    ap = argparse.ArgumentParser("Compare two captioners (A vs B) with BLEU on MSVD split")
    ap.add_argument("--ann", required=True, help="annotations.json (val/test)")
    ap.add_argument("--frames_root", required=True, help="e.g. data/processed/msvd/val/frames")
    ap.add_argument("--limit", type=int, default=0, help="limit samples for quick run (0=all)")

    # shared
    ap.add_argument("--vit_name", default="vit_base_patch16_224")
    ap.add_argument("--cond_mode", choices=["prefix","bos"], default="prefix")
    ap.add_argument("--prefix_len", type=int, default=4)
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)

    # decoding params (both use same for fairness)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--min_new_tokens", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)

    # A setup
    ap.add_argument("--name_a", default="stage2+basegpt2")
    ap.add_argument("--ckpt_a", default="", help="checkpoint for A")
    ap.add_argument("--gpt2_name_a", default="gpt2", help="HF id or local path for A")

    # B setup
    ap.add_argument("--name_b", default="stage2+stage3lm")
    ap.add_argument("--ckpt_b", default="", help="checkpoint for B")
    ap.add_argument("--gpt2_name_b", default="gpt2", help="HF id or local path for B (e.g. checkpoints/gpt2_lm_stage3/best)")

    # io/log
    ap.add_argument("--out_dir", default="eval_results/compare")
    ap.add_argument("--log_level", default="INFO")
    ap.add_argument("--log_file", default=None)
    args = ap.parse_args()

    logger = setup_logging(args.log_level, args.log_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ann = load_ann(Path(args.ann))
    frames_root = Path(args.frames_root)
    if args.limit > 0:
        ann = ann[:args.limit]

    logger.info(f"Samples: {len(ann)} | frames_root={frames_root}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # build models
    logger.info(f"Building A: {args.name_a} | ckpt={args.ckpt_a} | gpt2={args.gpt2_name_a}")
    model_a = build_model(args.vit_name, args.gpt2_name_a, args.cond_mode, args.prefix_len,
                          args.ckpt_a, device, logger)
    logger.info(f"Building B: {args.name_b} | ckpt={args.ckpt_b} | gpt2={args.gpt2_name_b}")
    model_b = build_model(args.vit_name, args.gpt2_name_b, args.cond_mode, args.prefix_len,
                          args.ckpt_b, device, logger)

    rows = []
    refs_all_a, hyps_a = [], []
    refs_all_b, hyps_b = [], []

    for i, rec in enumerate(ann, 1):
        vid = rec.get("video_id", "")
        caps = rec.get("captions", [])
        fdir = resolve_frames_dir(rec, frames_root)
        sample = {"idx": i, "video_id": vid, "frames_dir": str(fdir)}

        if not fdir.exists():
            logger.warning(f"[skip] frames not found: {fdir}")
            continue

        try:
            video, total_frames, used_frames = sample_frames(
                fdir, args.num_frames, args.image_size, device)
        except Exception as e:
            logger.warning(f"[skip] load frames failed for {vid}: {e}")
            continue

        # generate A/B
        with torch.no_grad():
            gen_kwargs = dict(
                prompt="",
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                num_beams=args.num_beams,
                temperature=args.temperature,
                top_p=args.top_p,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
            )
            text_a = model_a.generate(video, **gen_kwargs)[0].strip()
            text_b = model_b.generate(video, **gen_kwargs)[0].strip()

        # keep simple reference set
        refs = [c for c in caps if isinstance(c, str) and c.strip()]
        if not refs:
            refs = [""]  # fallback to avoid crash

        # record per-sample BLEU-1 (quick sanity), final we compute corpus BLEU
        # (单样本 BLEU-1 仅用于参考，真正以 corpus BLEU 为准)
        bleu1_a = sacrebleu.sentence_bleu(text_a, refs, smooth_method="exp", smooth_value=0.0,
                                          use_effective_order=True).precisions[0]
        bleu1_b = sacrebleu.sentence_bleu(text_b, refs, smooth_method="exp", smooth_value=0.0,
                                          use_effective_order=True).precisions[0]

        sample.update({
            "ref_0": refs[0] if refs else "",
            "hyp_a": text_a,
            "hyp_b": text_b,
            "frames_total": total_frames,
            "frames_used": used_frames,
            "bleu1_a": round(float(bleu1_a), 2),
            "bleu1_b": round(float(bleu1_b), 2),
        })
        rows.append(sample)

        refs_all_a.append(refs)
        refs_all_b.append(refs)
        hyps_a.append(text_a)
        hyps_b.append(text_b)

        if i % 20 == 0:
            logger.info(f"progress {i}/{len(ann)}")

    # corpus BLEU
    corpus_bleu_a = bleu_corpus(refs_all_a, hyps_a) if hyps_a else 0.0
    corpus_bleu_b = bleu_corpus(refs_all_b, hyps_b) if hyps_b else 0.0

    df = pd.DataFrame(rows)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = (
        f"Samples evaluated: {len(rows)}\n"
        f"A ({args.name_a}) corpus BLEU: {corpus_bleu_a:.2f}\n"
        f"B ({args.name_b}) corpus BLEU: {corpus_bleu_b:.2f}\n"
    )
    print("\n===== SUMMARY =====")
    print(summary)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    logger.info(f"Saved: {csv_path}")
    logger.info(f"Saved: {out_dir / 'summary.txt'}")

if __name__ == "__main__":
    main()