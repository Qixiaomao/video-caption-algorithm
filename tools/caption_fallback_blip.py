# tools/batch_caption_blip.py
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import argparse, json, csv, random
from pathlib import Path
from typing import List
from PIL import Image

import torch
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

# -------- utilities --------
def find_video_dirs(root: Path) -> List[Path]:
    # 子目录中含有 frame_*.jpg 的才算一个视频
    vids = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and any(p.glob("frame_*.jpg")):
            vids.append(p)
    return vids

def load_pil_samples(frames_dir: Path, num_frames: int = 8) -> List[Image.Image]:
    files = sorted(frames_dir.glob("frame_*.jpg"))
    if not files:
        return []
    step = max(len(files) // max(1, num_frames), 1)
    picks = files[::step][:num_frames]
    imgs = []
    for p in picks:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    return imgs

def clean_text(s: str) -> str:
    s = (s or "").strip().strip('"').strip()
    # 简单清洗：多空格合并、句末补句号
    s = " ".join(s.split())
    if s and s[-1] not in ".!?":
        s += "."
    return s

def pick_best(cands: List[str]) -> str:
    # 去重保序 + 选“看起来完整”的最长句子
    cands = [clean_text(c) for c in cands if c and c.strip()]
    seen, uniq = set(), []
    for c in cands:
        if c not in seen:
            uniq.append(c); seen.add(c)
    if not uniq:
        return ""
    full = [c for c in uniq if c[-1] in ".!?" and len(c.split()) >= 5]
    if full:
        return max(full, key=len)
    return max(uniq, key=len)

# -------- caption backends --------
@torch.no_grad()
def caption_blip(model_name: str, device: str, images: List[Image.Image], gen_kwargs):
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else None,
        use_safetensors=True,
    ).to(device).eval()

    # 用首/中/尾三帧投票
    idxs = [0, max(0, len(images)//2), len(images)-1]
    caps = []
    for i in idxs:
        inputs = processor(images=images[i], return_tensors="pt").to(device)
        out = model.generate(**inputs, **gen_kwargs)
        text = processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        caps.append(text)
    return pick_best(caps)

@torch.no_grad()
def caption_vit_gpt2(model_name: str, device: str, images: List[Image.Image], gen_kwargs):
    fea = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else None,
        use_safetensors=True,
    ).to(device).eval()

    idxs = [0, max(0, len(images)//2), len(images)-1]
    caps = []
    for i in idxs:
        pix = fea(images=images[i], return_tensors="pt").pixel_values.to(device)
        out = model.generate(pixel_values=pix, **gen_kwargs)
        text = tok.decode(out[0], skip_special_tokens=True).strip()
        caps.append(text)
    return pick_best(caps)

def build_argparse():
    p = argparse.ArgumentParser("Batch caption with BLIP/VED for human eval")
    p.add_argument("--frames_root", required=True, help="含有各视频帧子目录的根路径")
    p.add_argument("--output", required=True, help="输出 JSON 文件路径")
    p.add_argument("--emit_csv", action="store_true", help="同时导出 CSV（同名 .csv）")
    p.add_argument("--model", default="Salesforce/blip-image-captioning-base",
                   help="如 Salesforce/blip-image-captioning-base 或 nlpconnect/vit-gpt2-image-captioning")
    p.add_argument("--num_videos", type=int, default=30)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # 生成解码参数（保守自然）
    p.add_argument("--max_new_tokens", type=int, default=30)
    p.add_argument("--num_beams", type=int, default=3)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)

    # 过滤长度（可选）
    p.add_argument("--min_words", type=int, default=6)
    p.add_argument("--max_words", type=int, default=25)
    return p

def main():
    args = build_argparse().parse_args()
    random.seed(args.seed)

    root = Path(args.frames_root)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    vids = find_video_dirs(root)
    if not vids:
        raise SystemExit(f"[ERR] no video frame folders found under: {root}")

    # 随机采样固定顺序
    random.shuffle(vids)
    vids = vids[:args.num_videos]

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    results = []
    is_blip = "blip" in args.model.lower()
    for idx, vdir in enumerate(vids):
        imgs = load_pil_samples(vdir, args.num_frames)
        if not imgs:
            print(f"[skip] no frames -> {vdir.name}")
            continue

        try:
            if is_blip:
                cap = caption_blip(args.model, device, imgs, gen_kwargs)
            else:
                cap = caption_vit_gpt2(args.model, device, imgs, gen_kwargs)
        except Exception as e:
            print(f"[err] {vdir.name}: {e}")
            cap = ""

        # 过滤过短/过长
        wc = len(cap.split())
        if wc < args.min_words or wc > args.max_words:
            # 不合格就略缩：取最长的子句或直接跳过
            if wc == 0:
                print(f"[drop] empty -> {vdir.name}")
                continue

        results.append({"video_id": vdir.name, "caption": clean_text(cap)})
        print(f"[{idx:02d}] {vdir.name} -> {results[-1]['caption']}")

    # 写 JSON
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved JSON -> {outp}")

    # 写 CSV（问卷星友好）
    if args.emit_csv:
        csv_path = outp.with_suffix(".csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "caption"])
            for r in results:
                w.writerow([r["video_id"], r["caption"]])
        print(f"[OK] saved CSV  -> {csv_path}")

if __name__ == "__main__":
    main()