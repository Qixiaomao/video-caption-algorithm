#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query a video against pre-built FAISS index (with captions).
Usage:
    python -m scripts.query_video --video ./some_test.mp4 --k 5
"""
import argparse, json, subprocess, time
from pathlib import Path

import numpy as np
import torch
import faiss
from PIL import Image
import torchvision.transforms as T
from transformers import GPT2TokenizerFast

from src.models.vit_text_align import ViTTextAlignModel

def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def extract_frames_ffmpeg(video_path: Path, out_dir: Path, fps=2):
    """优先用 ffmpeg 抽帧；失败则抛异常（外层负责fallback）"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(out_dir / "frame_%06d.jpg")
    ]
    # 捕获stderr，便于报错时打印
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{cp.stderr.strip()}")

def extract_frames_cv2(video_path: Path, out_dir: Path, fps=2, max_frames=32):
    """OpenCV 兜底抽帧：按 fps 近似采样"""
    import cv2, math
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV failed to open video.")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(int(round(vfps / fps)), 1)
    count = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if count % interval == 0:
            out = out_dir / f"frame_{saved+1:06d}.jpg"
            cv2.imwrite(str(out), frame)
            saved += 1
            if saved >= max_frames: break
        count += 1
    cap.release()
    if saved == 0:
        raise RuntimeError("OpenCV extracted 0 frames.")

def encode_dir(frames_dir: Path, model, device, num_frames=8, image_size=224):
    imgs = sorted(frames_dir.glob("frame_*.jpg"))
    if not imgs:
        raise SystemExit(f"[ERROR] No frames found in {frames_dir}")
    imgs = imgs[:num_frames]
    tfm = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tensor = torch.stack([tfm(Image.open(p).convert("RGB")) for p in imgs], dim=0)
    with torch.no_grad():
        v = tensor.unsqueeze(0).to(device)  # [1,T,3,H,W]
        emb = model.encode_video(v).cpu().numpy()  # [1,D]
    return l2norm(emb.astype("float32"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index", default="data/processed/msvd/faiss_index/video.index")
    parser.add_argument("--meta",  default="data/processed/msvd/faiss_index/meta.json")
    parser.add_argument("--ckpt",  default="checkpoints/msvd_vit_freeze_best.pt")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    idx_path, meta_path = Path(args.index), Path(args.meta)
    if not idx_path.exists() or not meta_path.exists():
        raise SystemExit(f"[ERROR] Missing index/meta under {idx_path.parent}")
    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"[ERROR] Video file not found: {video_path.resolve()}")

    print(f"[INFO] Loading index from {idx_path}")
    index = faiss.read_index(str(idx_path))
    meta  = json.loads(meta_path.read_text("utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    model = ViTTextAlignModel(vocab_size=tok.vocab_size, pad_id=tok.pad_token_id).to(device).eval()

    ckpt = Path(args.ckpt)
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state.get("model_state", state))
        print(f"[INFO] Loaded checkpoint: {ckpt}")
    else:
        print(f"[WARN] No checkpoint found: {ckpt}")

    tmp = Path("./outputs/tmp_query"); tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("frame_*.jpg"): f.unlink()

    print(f"[INFO] Extracting frames from {video_path}")
    try:
        extract_frames_ffmpeg(video_path, tmp, fps=args.fps)
    except Exception as e:
        print(f"[WARN] ffmpeg failed: {e}\n[INFO] Fallback to OpenCV...")
        extract_frames_cv2(video_path, tmp, fps=args.fps)

    print("[INFO] Encoding query video ...")
    t0 = time.time()
    q_emb = encode_dir(tmp, model, device, num_frames=args.num_frames, image_size=args.image_size)
    elapsed = time.time() - t0

    D, I = index.search(q_emb, args.k)
    print(f"[INFO] Search done in {elapsed:.2f}s\n")

    print("[TOP-K RESULTS]")
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        item = meta[idx]
        # 兼容两种情况：有 "captions"（列表）或 "caption"（单字符串）
        if "captions" in item and isinstance(item["captions"], list) and item["captions"]:
            caption = item["captions"][0]
        elif "caption" in item and isinstance(item["caption"], str):
            caption = item["caption"]
        else:
            caption = "(no caption)"
        print(f"[{rank}] score={score:.4f} | id={item.get('video_id','?')} | caption={caption}")
    print("\n✅ Done.")

if __name__ == "__main__":
    main()