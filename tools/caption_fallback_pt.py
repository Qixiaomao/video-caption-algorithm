#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caption fallback (PyTorch only, no TensorFlow).
- Prefer safetensors to avoid torch.load vulnerability restrictions.
- Support:
    * Salesforce/blip-image-captioning-base (or -large)
    * nlpconnect/vit-gpt2-image-captioning  (safetensors available)
- Public API:
    caption_blip_from_frames_dir(frames_dir, num_frames=8, model="Salesforce/blip-image-captioning-base", device=None)
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image
from torchvision import transforms

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoImageProcessor,
    BlipForConditionalGeneration,
    VisionEncoderDecoderModel,
)


# ---------------- I/O & frames ----------------
def _list_frames(frames_dir: Path) -> List[Path]:
    files = sorted(frames_dir.glob("frame_*.jpg"))
    return files


def _pick_indices(n: int, k: int) -> List[int]:
    if n <= k:
        return list(range(n))
    step = max(n // k, 1)
    picks = list(range(0, n, step))[:k]
    return picks


def _load_sampled_images(frames_dir: str, num_frames: int = 8, image_size: int = 224) -> List[Image.Image]:
    frames_dir = Path(frames_dir)
    files = _list_frames(frames_dir)
    if not files:
        raise FileNotFoundError(f"[caption_fallback] no frames found under {frames_dir}")

    idxs = _pick_indices(len(files), max(1, num_frames))
    picks = [files[i] for i in idxs]

    imgs = []
    for p in picks:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    return imgs


# --------------- small cleaner & scorer ---------------
def _ensure_period_and_caps(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s[0].isalpha():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".!?":
        s += "."
    return s


def _dedup_tokens(s: str) -> str:
    s = re.sub(r"(?i)\b(\w+)\b(?:\s+\1\b)+", r"\1", s)  # 连续重复词
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _strip_webby(s: str) -> str:
    if re.search(r"(https?://|www\.|<a\b|reddit\.com|pastebin|mailto:)", s, re.I):
        return ""
    if re.match(r'^\s*(©|copyright\b)', s, re.I):
        return ""
    return s


def _score_sentence(s: str) -> float:
    if not s:
        return -1e9
    toks = s.split()
    n = len(toks)
    score = 0.0
    # 长度高斯偏好
    mu, sigma = 10.0, 4.0
    score += -((n - mu) ** 2) / (2 * sigma * sigma)
    # 动词 / be
    if re.search(r"\b\w+ing\b", s): score += 0.8
    if re.search(r"\b(?:is|are|was|were)\b", s): score += 0.3
    # 结尾标点
    if s.endswith((".", "!", "?")): score += 0.2
    # 噪声扣分
    if re.search(r"\b(?:[A-Z]\.){2,}\b", s): score -= 1.0
    if re.search(r"(?i)\b(click here|subscribe|report abuse)\b", s): score -= 1.5
    if n < 4: score -= 1.0
    return score


def _clean_caption(s: str) -> str:
    s = (s or "").strip()
    s = _strip_webby(s)
    s = _dedup_tokens(s)
    s = _ensure_period_and_caps(s)
    return s


def _select_best(cands: List[str]) -> str:
    cands = [ _clean_caption(x) for x in cands if x and _strip_webby(x) != "" ]
    if not cands:
        return "Someone is in the scene."
    return max(cands, key=_score_sentence)


# --------------- core captioners ---------------
@torch.no_grad()
def _caption_with_blip(
    model_name: str,
    images: List[Image.Image],
    device: Optional[str] = None,
    gen_kwargs: Optional[dict] = None,
) -> str:
    """
    Use BLIP ForConditionalGeneration. Prefer safetensors; if not available, raise ValueError to let caller fallback.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen_kwargs = gen_kwargs or dict(num_beams=3, max_new_tokens=30)

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    # 强制优先 safetensors
    blip = BlipForConditionalGeneration.from_pretrained(
        model_name, use_safetensors=True, low_cpu_mem_usage=True
    ).to(device).eval()

    # 批量处理若干帧，逐帧生成后挑最优一句
    captions = []
    for im in images:
        inputs = processor(images=im, return_tensors="pt").to(device)
        out_ids = blip.generate(**inputs, **gen_kwargs)
        text = processor.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        captions.append(text)

    return _select_best(captions)


@torch.no_grad()
def _caption_with_vit_gpt2(
    model_name: str,
    images: List[Image.Image],
    device: Optional[str] = None,
    gen_kwargs: Optional[dict] = None,
) -> str:
    """
    Use VisionEncoderDecoderModel (nlpconnect/vit-gpt2-image-captioning). This repo ships safetensors.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen_kwargs = gen_kwargs or dict(num_beams=3, max_new_tokens=30)

    # 该模型通常需要分开 processor
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained(
        model_name, use_safetensors=True, low_cpu_mem_usage=True
    ).to(device).eval()

    captions = []
    for im in images:
        pixel = image_processor(images=im, return_tensors="pt").pixel_values.to(device)
        out_ids = model.generate(pixel, **gen_kwargs)
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        captions.append(text)

    return _select_best(captions)


# --------------- public API ---------------
@torch.no_grad()
def caption_blip_from_frames_dir(
    frames_dir: str,
    num_frames: int = 8,
    model: str = "Salesforce/blip-image-captioning-base",
    device: Optional[str] = None,
    image_size: int = 224,
    gen_kwargs: Optional[dict] = None,
) -> str:
    """
    Public API for Chainlit / backend:
    - frames_dir: directory containing frame_*.jpg
    - model: "Salesforce/blip-image-captioning-base" | "Salesforce/blip-image-captioning-large" | "nlpconnect/vit-gpt2-image-captioning"
    - prefer safetensors; if current model has no safetensors, fallback to nlpconnect/vit-gpt2-image-captioning
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    images = _load_sampled_images(frames_dir, num_frames=num_frames, image_size=image_size)

    model_l = model.lower()
    gen_kwargs = gen_kwargs or dict(num_beams=3, max_new_tokens=30)

    try:
        if model_l.startswith("salesforce/blip"):
            return _caption_with_blip(model, images, device=device, gen_kwargs=gen_kwargs)
        elif model_l.startswith("nlpconnect/vit-gpt2-image-captioning"):
            return _caption_with_vit_gpt2(model, images, device=device, gen_kwargs=gen_kwargs)
        else:
            # 未知模型名：优先尝试 BLIP 接口
            return _caption_with_blip(model, images, device=device, gen_kwargs=gen_kwargs)

    except ValueError as e:
        # 典型场景：无 safetensors 或被 torch>=2.6 检查挡住
        alt = "nlpconnect/vit-gpt2-image-captioning"
        return _caption_with_vit_gpt2(alt, images, device=device, gen_kwargs=gen_kwargs)


# --------------- CLI ---------------
def parse_args():
    p = argparse.ArgumentParser("Fallback Image Captioning (PyTorch-only)")
    p.add_argument("--frames_dir", required=True, help="Directory that contains frame_*.jpg")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument(
        "--model",
        default="Salesforce/blip-image-captioning-base",
        choices=[
            "Salesforce/blip-image-captioning-base",
            "Salesforce/blip-image-captioning-large",
            "nlpconnect/vit-gpt2-image-captioning",
        ],
        help="Use BLIP or ViT-GPT2 (has safetensors)."
    )
    p.add_argument("--emit_json", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cap = caption_blip_from_frames_dir(
        frames_dir=args.frames_dir,
        num_frames=args.num_frames,
        model=args.model,
        device=args.device,
        image_size=args.image_size,
    )
    if args.emit_json:
        print(json.dumps({"model": args.model, "caption": cap}, ensure_ascii=False))
    else:
        print(cap)


if __name__ == "__main__":
    main()