# hybrid_infer.py
#-- coding: utf-8 -*-
from __future__ import annotations
import re,json
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from torchvision import transforms

# --force pytorch-only block TF/Flax at import time

import os,sys
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

# 避免有的版本尝试import tensorflow
sys.modules.setdefault("tensorflow",None)
sys.modules.setdefault("tf_keras",None)
sys.modules.setdefault("keras",None)

## === 复用diy 推理模块
from inference import run_one_video


## BLIP (纯Pytorch,避免TF/Keras) 兜底
from transformers import AutoProcessor, BlipForConditionalGeneration

_BLIP_SINGLETON = {"proc": None, "model": None} 

def _blip_load(model_name: str = "Salesforce/blip-image-captioning-base", device: str = "cuda"):
    if _BLIP_SINGLETON["proc"] is None:
        _BLIP_SINGLETON["proc"] = AutoProcessor.from_pretrained(model_name, use_fast=True)
        _BLIP_SINGLETON["model"] = BlipForConditionalGeneration.from_pretrained(model_name).to(device).eval()
    return _BLIP_SINGLETON["proc"], _BLIP_SINGLETON["model"]

def _sample_frames(frames_dir: Path, num_frames: int = 8, image_size: int = 224):
    files = sorted(frames_dir.glob("frame_*.jpg"))
    if not files:
        raise FileNotFoundError(f"No frames in {frames_dir}")
    step = max(len(files) // num_frames, 1)
    picks = files[::step][:num_frames]
    tfm = transforms.Compose([transforms.Resize((image_size, image_size))])
    ims = []
    for p in picks:
        with Image.open(p) as im:
            ims.append(tfm(im.convert("RGB")))
    return ims

@torch.no_grad()
def blip_caption(frames_dir: Path, device: str = "cuda") -> str:
    proc, model = _blip_load(device=device)
    images = _sample_frames(frames_dir)
    # 取中位帧（或平均五张投票，简单起见先取中位）
    mid = images[len(images)//2]
    inputs = proc(images=mid, return_tensors="pt").to(device)
    out_ids = model.generate(**inputs, max_new_tokens=30, num_beams=3)
    text = proc.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    return text

# ==== 质量判定（简单但好用）

_URL_RE = re.compile(r"https?://|www\.", re.I)
_BAD_TOKENS = (
    "copyright", "©", "click here", "report abuse",
    "reddit", "youtube", "facebook", "login", "sign up"
)

def _alpha_ratio(s: str) -> float:
    if not s: return 0.0
    letters = sum(ch.isalpha() for ch in s)
    return letters / max(1, len(s))

def _too_repetitive(s: str) -> bool:
    # 连续词复读 或 “xxx xxx xxx”
    if re.search(r"\b(\w+)(\s+\1){2,}\b", s, flags=re.I): return True
    # 字符级复读
    if re.search(r"(.)\1{4,}", s): return True
    return False

def is_bad_caption(s: str) -> Tuple[bool, str]:
    if not s or not s.strip():
        return True, "empty"
    s_strip = s.strip()
    if len(s_strip) < 8:
        return True, "too_short"
    if _URL_RE.search(s_strip):
        return True, "url_like"
    low = s_strip.lower()
    if any(tok in low for tok in _BAD_TOKENS):
        return True, "boilerplate"
    if _alpha_ratio(s_strip) < 0.6:
        return True, "low_alpha_ratio"
    if _too_repetitive(s_strip):
        return True, "repetition"
    # 句子尾标点太奇怪
    if len(s_strip) > 0 and s_strip[-1] not in ".?!":
        s_strip += "."
    return False, "ok"

def _pick_best_from_ours(d: Dict[str, str]) -> Tuple[str, str]:
    # 简单打分：优先不坏的 + 越自然越好（S3> S2> S1）
    order = ["S3", "S2", "S1"]
    for k in order:
        bad, _ = is_bad_caption(d.get(k, ""))
        if not bad:
            return k, d[k].strip()
    # 都不行就返回最短可用的一个（再兜底 BLIP）
    for k in order:
        s = d.get(k, "").strip()
        if s:
            return k, s
    return "NONE", ""


# 对外主函数：优先使用自研模型，必要时使用 BLIP 兜底
@torch.no_grad()
def hybrid_caption(
    frames_dir: str | Path,
    ckpt: str,
    prefix_len: int = 4,
    ln_scale: float = 0.6,
    in_weight: float = 0.4,
    preset1: str = "natural",
    preset2: str = "precise",
    preset3: str = "detailed",
    prompt1: str = " ",
    prompt2: str = "Describe the visible action in one short sentence:",
    prompt3: str = "Write a short, natural caption about the scene:",
    device: str = "cuda",
    blip_fallback: bool = True
) -> Dict:
    frames_dir = Path(frames_dir)
    ours = run_one_video(
        frames_dir=frames_dir,
        ckpt=ckpt,
        prefix_len=prefix_len,
        ln_scale=ln_scale,
        in_weight=in_weight,
        presets=(preset1, preset2, preset3),
        prompts=(prompt1, prompt2, prompt3),
        emit_json=True
    )
    # ours: {"S1","S2","S3","BEST":{"key","text"}}（按你当前实现）
    # 重新做一次稳健选择
    key, text = _pick_best_from_ours(ours)
    bad, reason = is_bad_caption(text)

    used = {"source": "ours", "detail": key, "fallback_reason": ""}
    if (bad or key == "NONE") and blip_fallback:
        try:
            text_blip = blip_caption(frames_dir, device=device)
            bad2, _ = is_bad_caption(text_blip)
            if not bad2:
                text = text_blip
                used = {"source": "blip", "detail": "Salesforce/blip-image-captioning-base", "fallback_reason": reason}
        except Exception as e:
            # BLIP 也挂了就还是用自研的（为了不中断演示）
            used["fallback_reason"] = f"blip_failed:{e}"

    return {
        "S1": ours.get("S1", ""),
        "S2": ours.get("S2", ""),
        "S3": ours.get("S3", ""),
        "BEST": {"key": key, "text": text},
        "USED": used
    }
# ==== 结束 ====
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser("Hybrid caption: ours-first with BLIP fallback")
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prefix_len", type=int, default=4)
    p.add_argument("--ln_scale", type=float, default=0.6)
    p.add_argument("--in_weight", type=float, default=0.4)
    p.add_argument("--emit_json", action="store_true")
    args = p.parse_args()

    out = hybrid_caption(
        frames_dir=args.frames_dir,
        ckpt=args.ckpt,
        prefix_len=args.prefix_len,
        ln_scale=args.ln_scale,
        in_weight=args.in_weight
    )
    if args.emit_json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"[BEST] ({out['USED']['source']}) {out['BEST']['text']}")
        print(f"S1: {out['S1']}\nS2: {out['S2']}\nS3: {out['S3']}")