#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, sys, logging, json, re
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from transformers import GPT2TokenizerFast

# 你的模型外壳（保持和项目一致）
from src.models.caption_model import VideoCaptionModel

def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

log = logging.getLogger("debug")

def load_frames(frames_dir: Path, num_frames=8, image_size=224, device="cpu"):
    files = sorted(frames_dir.glob("*.jpg"))
    if not files:
        raise SystemExit(f"[FATAL] no jpg frames in: {frames_dir}")

    step = max(len(files) // num_frames, 1)
    picks = files[::step][:num_frames]

    tfm = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    imgs = []
    for p in picks:
        with Image.open(p) as im:
            imgs.append(tfm(im.convert("RGB")))
    v = torch.stack(imgs, dim=0).unsqueeze(0).to(device)  # [1,T,3,H,W]
    return v, len(files), [p.name for p in picks]

def _maybe_remap_keys(sd: dict) -> dict:
    # 兼容旧 ckpt 的 vit.* 前缀
    if any(k.startswith("vit.") for k in sd):
        out, moved = {}, 0
        for k,v in sd.items():
            if k.startswith("vit."):
                out["encoder.backbone." + k[len("vit."):]] = v
                moved += 1
            else:
                out[k] = v
        log.info("remap: moved %d 'vit.*' -> 'encoder.backbone.*'", moved)
        return out
    return sd

def load_ckpt(model: nn.Module, ckpt_path: Path, device: str):
    state = torch.load(ckpt_path, map_location=device)
    sd = state.get("model_state", state)
    sd = _maybe_remap_keys(sd)
    total   = len(sd)
    decoder = sum(1 for k in sd if k.startswith("decoder."))
    mapper  = sum(1 for k in sd if "decoder.mapper" in k)
    log.info("[ckpt] keys=%d | decoder=%d | mapper=%d", total, decoder, mapper)
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss:  log.warning("missing keys: %d (sample: %s)", len(miss), miss[:6])
    if unexp: log.warning("unexpected keys: %d (sample: %s)", len(unexp), unexp[:6])

def prefix_to_hidden(model: VideoCaptionModel, prefix: torch.Tensor, hidden: int) -> torch.Tensor:
    B,P,Dp = prefix.shape
    if Dp == hidden:
        return prefix

    mapper = getattr(model.decoder, "mapper", None)
    if isinstance(mapper, nn.Module):
        try:
            out = mapper(prefix)  # 期望 [B,P,H]
            if out.dim()==3 and out.shape[-1]==hidden:
                return out
        except Exception:
            pass
        try:
            out = mapper(prefix.reshape(B, P*Dp))  # 期望 [B,P*H] or [B,H]
            if out.dim()==2 and out.shape[-1]==hidden:
                return out.unsqueeze(1)
            if out.dim()==2 and out.shape[-1] % hidden == 0:
                P2 = out.shape[-1] // hidden
                return out.reshape(B, P2, hidden)
        except Exception:
            pass

    # 兜底线性适配（只在推理用，不会保存）
    if not hasattr(model, "_adapter") or not isinstance(model._adapter, nn.Linear):
        model._adapter = nn.Linear(Dp, hidden, bias=True).to(prefix.device)
        log.info("[adapter] created Linear(%d->%d) for inference", Dp, hidden)
    return model._adapter(prefix)

@torch.no_grad()
def quick_generate(model: VideoCaptionModel, video: torch.Tensor,
                   prompt: str, ln_scale: float, in_weight: float, device: str):
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    gpt2 = getattr(model.decoder, "model", model.decoder)

    vfeat = model.encoder(video)              # [1,*,D]
    log.info("encoder out: %s", tuple(vfeat.shape))

    prefix = model.proj(vfeat)                # [1,P,Dp] or [1,Dp]
    if prefix.dim()==2: prefix = prefix.unsqueeze(1)
    log.info("proj out:    %s", tuple(prefix.shape))

    hidden = int(getattr(gpt2.config, "n_embd", 768))
    prefix_h = prefix_to_hidden(model, prefix, hidden)  # [1,P,H]
    log.info("mapper out:  %s", tuple(prefix_h.shape))

    # 轻量归一/缩放
    if not hasattr(model, "_infer_ln") or not isinstance(model._infer_ln, nn.LayerNorm):
        model._infer_ln = nn.LayerNorm(prefix_h.shape[-1]).to(prefix_h.device)
    prefix_h = model._infer_ln(prefix_h) * ln_scale
    prefix_h = prefix_h * in_weight

    # 组装输入
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    txt_emb = gpt2.transformer.wte(ids)
    inputs_embeds = torch.cat([prefix_h, txt_emb], dim=1)
    attn_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=device)

    # 用束搜索，尽量稳定
    gen_kwargs = dict(num_beams=4, max_new_tokens=20, min_new_tokens=8,
                      no_repeat_ngram_size=6, repetition_penalty=1.30)
    out_ids = gpt2.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
        **gen_kwargs
    )
    text = tok.decode(out_ids[0], skip_special_tokens=True).strip()
    # 简单清洗
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+|\S+\.(com|net|org)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?": text += "."
    return text

def main():
    ap = argparse.ArgumentParser("End-to-end debug for inference chain")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prefix_len", type=int, default=4)
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--ln_scale", type=float, default=0.8)
    ap.add_argument("--in_weight", type=float, default=0.8)
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        raise SystemExit(f"[FATAL] frames_dir not found: {frames_dir}")

    # ① 帧检查
    video, total_frames, picked = load_frames(frames_dir, args.num_frames, args.image_size, device)
    log.info("frames total=%d | sampled=%d | picks=%s", total_frames, len(picked), picked[:4])

    # ②/③/④ 模型各段检查
    model = VideoCaptionModel(
        vit_name="vit_base_patch16_224",
        gpt2_name="gpt2",
        cond_mode="prefix",
        prefix_len=args.prefix_len,
        freeze_vit=True, unfreeze_last=0
    ).to(device).eval()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"[FATAL] ckpt not found: {ckpt_path}")
    load_ckpt(model, ckpt_path, device)

    # mapper 参数量（是否存在）
    mapper = getattr(model.decoder, "mapper", None)
    n_mapper = sum(p.numel() for p in mapper.parameters()) if isinstance(mapper, nn.Module) else 0
    log.info("mapper params: %d", n_mapper)

    # 小规模生成，验证端到端
    prompt = "Describe the visible action and objects in one short sentence:"
    caption = quick_generate(model, video, prompt, args.ln_scale, args.in_weight, device)
    print("\n===== SUMMARY =====")
    print(json.dumps({
        "device": device,
        "frames_total": total_frames,
        "prefix_len_arg": args.prefix_len,
        "ln_scale": args.ln_scale,
        "in_weight": args.in_weight,
        "mapper_params": n_mapper,
        "caption_preview": caption[:120]
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()