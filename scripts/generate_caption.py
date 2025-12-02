#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import shutil
import subprocess
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.models.caption_model import VideoCaptionModel

# === Logging setup ===
import logging, sys, os

def setup_logging(level: str = "INFO", log_file: str | None = None):
    level = level.upper()
    numeric_level = getattr(logging, level, logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )

logger = logging.getLogger(__name__)
# =====================

def extract_frames(video_path: Path, out_dir: Path, fps: int = 2):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(out_dir / "frame_%06d.jpg"),
    ]
    logger.info(f"Extracting frames via ffmpeg (fps={fps}) …")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logger.exception(f"ffmpeg extraction failed: {e}")
        raise

def load_frames(frames_dir: Path, num_frames=8, image_size=224, device="cpu"):
    files = sorted(frames_dir.glob("frame_*.jpg"))
    if not files:
        msg = f"No frames found in {frames_dir}"
        logger.error(msg)
        raise SystemExit(msg)

    total = len(files)
    step = max(total // num_frames, 1)
    picks = files[::step][:num_frames]
    logger.info(f"Frames found: {total} | sampled: {len(picks)} (step={step})")

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    imgs = []
    for i, p in enumerate(picks):
        try:
            with Image.open(p) as im:
                imgs.append(tfm(im.convert("RGB")))
        except Exception as e:
            logger.warning(f"Failed to load frame {p.name}: {e}")

    if not imgs:
        msg = f"All frame loads failed in {frames_dir}"
        logger.error(msg)
        raise SystemExit(msg)

    video = torch.stack(imgs, dim=0).unsqueeze(0).to(device)  # [1,T,3,224,224]
    logger.debug(f"Video tensor shape: {tuple(video.shape)} device={device}")
    return video

def clean_caption(s: str) -> str:
    s = s.strip()
    for t in ["a.k.a", "aka", "AKA"]:
        s = s.replace(t, "")
    s = " ".join(s.split())
    if s and s[-1] not in ".!?":
        s += "."
    return s

def build_argparser():
    p = argparse.ArgumentParser("Video → Caption generation")
    p.add_argument("--video", required=True, help="输入视频路径")
    p.add_argument("--ckpt", default="", help="端到端训练后的 checkpoint（可选）")
    p.add_argument("--vit_name", default="vit_base_patch16_224")
    p.add_argument("--gpt2_name", default="gpt2")
    p.add_argument("--cond_mode", choices=["prefix", "bos"], default="prefix")
    p.add_argument("--prefix_len", type=int, default=4)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--prompt", default="", help="轻提示，如 'Describe the video in one sentence:'")

    # 生成相关
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--min_new_tokens", type=int, default=8)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3)
    p.add_argument("--repetition_penalty", type=float, default=1.15)

    # 日志相关
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
                   help="logging level (default: INFO)")
    p.add_argument("--log_file", default=None,
                   help="optional log file path, e.g. ./logs/generate_caption.log")
    return p

def main():
    args = build_argparser().parse_args()
    setup_logging(level=args.log_level, log_file=args.log_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("🚀 Start Video → Caption generation")
    logger.info(f"Args: video={args.video} ckpt={args.ckpt or '(none)'} "
                f"gpt2_name={args.gpt2_name} vit_name={args.vit_name} "
                f"cond_mode={args.cond_mode} prefix_len={args.prefix_len} "
                f"num_frames={args.num_frames} image_size={args.image_size} "
                f"decode: beams={args.num_beams} max_new={args.max_new_tokens} "
                f"temp={args.temperature} top_p={args.top_p} ngram={args.no_repeat_ngram_size} "
                f"rept_penalty={args.repetition_penalty} device={device}")

    tmp = Path("outputs/tmp_cap")
    try:
        # 清理临时目录
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)

        # 1) 抽帧
        logger.info(f"Extracting frames from: {args.video}")
        extract_frames(Path(args.video), tmp, fps=2)
        n_frames = len(list(tmp.glob("frame_*.jpg")))
        logger.info(f"Frames extracted: {n_frames}")

        # 2) 读帧 → 张量
        logger.info("Loading & sampling frames …")
        video = load_frames(tmp, num_frames=args.num_frames,
                            image_size=args.image_size, device=device)

        # 3) 构建模型
        logger.info("Building model (VideoCaptionModel) …")
        model = VideoCaptionModel(
            vit_name=args.vit_name,
            gpt2_name=args.gpt2_name,
            cond_mode=args.cond_mode,
            prefix_len=args.prefix_len,
            freeze_vit=True,
            unfreeze_last=0,
        ).to(device).eval()
        logger.info("Model ready.")

        # 4) 可选加载 checkpoint
        if args.ckpt and Path(args.ckpt).exists():
            logger.info(f"Loading checkpoint: {args.ckpt}")
            state = torch.load(args.ckpt, map_location=device)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                logger.warning(f"Missing keys: {sorted(missing)}")
            if unexpected:
                logger.warning(f"Unexpected keys: {sorted(unexpected)}")
            logger.info("Checkpoint loaded.")
        else:
            logger.warning("No checkpoint provided or path not found, using default weights.")

        # 5) 生成
        logger.info("Decoding with GPT-2 …")
        texts = model.generate(
            video,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_p=args.top_p,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
        )
        caption = clean_caption(texts[0])
        logger.info("✅ Caption generated.")
        print(f"\n[CAPTION] {caption}")

    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        raise
    finally:
        # 结束可选清理（如需保留帧，注释掉即可）
        try:
            if tmp.exists():
                shutil.rmtree(tmp)
                logger.debug(f"Temp dir removed: {tmp}")
        except Exception as ce:
            logger.warning(f"Failed to cleanup temp dir {tmp}: {ce}")

        logger.info("Done.")

if __name__ == "__main__":
    main()