#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build FAISS index (with captions in meta)
- 在线编码视频：用当前模型 encode_video → 向量
- 从 annotations.json 读取每条记录的 captions（取第一条）
- 写出：video.index（FAISS） + meta.json（含 video_id 与 caption）

用法示例：
python -m scripts.build_index_with_captions ^
  --ann_path data/processed/msvd/train/annotations.json ^
  --out_dir  data/processed/msvd/faiss_index ^
  --ckpt     checkpoints/msvd_vit_freeze_best.pt ^
  --num_frame 8 --image_size 224
"""
import json
import faiss
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from src.data.data_loader import build_dataloader
from src.models.vit_text_align import ViTTextAlignModel


def l2norm(x: np.ndarray):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n


def build_faiss(X: np.ndarray, index_type: str = "Flat", nlist: int = 50):
    """构建 FAISS 索引；MSVD 体量用 Flat 足够"""
    d = X.shape[1]
    if index_type == "Flat":
        index = faiss.IndexFlatIP(d)  # 单位化后用点积≈余弦
    elif index_type == "IVF_FLAT":
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(X)
    else:
        raise ValueError(f"Unsupported index_type: {index_type}")
    index.add(X)
    return index


def load_caption_map(ann_path: Path) -> dict[str, str]:
    """从 annotations.json 构建 video_id → 第一条 caption 的映射"""
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    cap_map = {}
    for r in data:
        vid = r.get("video_id")
        cap = None
        if isinstance(r.get("captions"), list) and r["captions"]:
            cap = r["captions"][0]
        elif isinstance(r.get("caption"), str):
            cap = r["caption"]
        if vid:
            cap_map[vid] = cap or ""
    return cap_map


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_path", required=True, help="e.g. data/processed/msvd/train/annotations.json")
    ap.add_argument("--out_dir", default="data/processed/msvd/faiss_index", help="输出目录")
    ap.add_argument("--ckpt", default="checkpoints/msvd_vit_freeze_best.pt", help="模型权重路径")
    ap.add_argument("--num_frame", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--index_type", choices=["Flat","IVF_FLAT"], default="Flat")
    ap.add_argument("--nlist", type=int, default=50, help="IVF_FLAT 的簇数")
    args = ap.parse_args()

    ann_path = Path(args.ann_path)
    out_dir  = Path(args.out_dir)
    ckpt     = Path(args.ckpt)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] ann_path={ann_path}")

    # 0) 准备 caption 映射（用于写 meta.json）
    cap_map = load_caption_map(ann_path)

    # 1) dataloader（在线编码）
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    loader = build_dataloader(
        ann_path=str(ann_path),
        tokenizer=tok,
        batch_size=1,
        num_frame=args.num_frame,
        image_size=args.image_size,
        shuffle=False
    )

    # 2) 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTTextAlignModel(vocab_size=tok.vocab_size, pad_id=tok.pad_token_id).to(device).eval()
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state.get("model_state", state))
        print(f"[INFO] loaded checkpoint: {ckpt}")
    else:
        print(f"[WARN] checkpoint not found: {ckpt} (使用随机初始化权重)")

    # 3) 编码 & 收集向量+meta
    embs, metas = [], []
    with torch.no_grad():
        for b in tqdm(loader, desc="encode videos"):
            vid = b["video_id"][0]
            v   = b["video"].to(device)                       # [1,T,3,224,224]
            e   = model.encode_video(v).cpu().numpy().astype("float32")  # [1,D]
            embs.append(e)
            metas.append({
                "video_id": vid,
                "caption": cap_map.get(vid, "")
            })

    X = np.vstack(embs)
    X = l2norm(X)

    # 4) 建索引 & 保存
    index = build_faiss(X, index_type=args.index_type, nlist=args.nlist)
    faiss.write_index(index, str(out_dir / "video.index"))
    (out_dir / "meta.json").write_text(json.dumps(metas, ensure_ascii=False, indent=2), "utf-8")

    print(f"[OK] index -> {out_dir / 'video.index'}")
    print(f"[OK] meta  -> {out_dir / 'meta.json'}")
    print(f"[OK] num vectors: {len(metas)} | dim: {X.shape[1]} | type={args.index_type}")


if __name__ == "__main__":
    main()