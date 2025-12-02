#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, random
from pathlib import Path
import torch, numpy as np
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.data.data_loader import build_dataloader
from src.models.simple_vc import SimpleVideoCaptioner  # 如要用 TinyCaptioner 可换
# 可以切换为TinyCaptioner：
# from src.models.tiny_captioner import TinyCaptioner

# ---------- utils ----------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_tokenizer():
    try:
        from transformers import BertTokenizerFast
        tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
        if tok.pad_token_id is None: tok.pad_token_id = 0
        return tok
    except Exception:
        class MinimalTokenizer:
            def __init__(self):
                self.vocab = {"[PAD]":0, "[UNK]":1, "[CLS]":2, "[SEP]":3}
                self.pad_token_id = 0
                self.vocab_size = 4
        return MinimalTokenizer()

def shift_for_lm(labels, pad_id=0):
    inp = labels[:, :-1].contiguous()
    tgt = labels[:, 1:].contiguous()
    return inp, tgt

# ---------- core builders ----------
def build_model(model_name: str, vocab_size: int, d_model: int, max_len: int):
    if model_name == "simple_vc":
        # 平均池化 + Linear（你已实现的 SimpleVideoCaptioner）
        return SimpleVideoCaptioner(vocab_size=vocab_size, hidden_size=d_model, max_len=max_len)
    elif model_name == "tiny_gru":
        # 如果你想保留你贴的 GRU 版，就把 TinyCaptioner 挪到 src.models.tiny_captioner 并引入
        from src.models.tiny_captioner import TinyCaptioner
        return TinyCaptioner(vocab_size=vocab_size, d_model=d_model)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def build_loader(train_ann, tokenizer, batch_size, num_wokers, num_frame, image_size):
    return build_dataloader(
        train_ann, tokenizer,
        batch_size=batch_size, shuffle=True,
        num_wokers=num_wokers,   # 注意：与你项目的参数拼写保持一致
        num_frame=num_frame, image_size=image_size,
    )

# ---------- train one epoch ----------
def train_one_epoch(model, loader, optimizer, criterion, device, pad_id, max_batches, log_interval=10):
    model.train()
    pbar = tqdm(enumerate(loader), total=min(len(loader), max_batches), desc="train")
    global_step = 0
    for i, batch in pbar:
        if i >= max_batches: break
        video = batch.get("video"); cap_ids = batch.get("caption_ids")
        if video is None or cap_ids is None: continue

        video = video.to(device, non_blocking=True)
        labels = cap_ids.to(device, non_blocking=True).long()
        inputs, targets = shift_for_lm(labels, pad_id=pad_id)

        logits = model(video, inputs)  # [B, L, V]
        minL = min(logits.size(1), targets.size(1))
        loss = criterion(
            logits[:, :minL, :].reshape(-1, logits.size(-1)),
            targets[:, :minL].reshape(-1)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1
        if global_step % log_interval == 0:
            pbar.set_postfix(loss=float(loss.detach().cpu()), step=global_step)
    return global_step

# ---------- main pipeline ----------
def train_pipeline(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    tokenizer = get_tokenizer()
    vocab_size = getattr(tokenizer, "vocab_size", 30522)
    pad_id = getattr(tokenizer, "pad_token_id", 0)

    loader = build_loader(
        args.train_ann, tokenizer,
        args.batch_size, args.num_wokers, args.num_frame, args.image_size
    )

    model = build_model(args.model_name, vocab_size, args.d_model, args.max_len).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    steps = train_one_epoch(
        model, loader, optimizer, criterion, device,
        pad_id=pad_id, max_batches=args.max_batches, log_interval=args.log_interval
    )

    ckpt_path = Path(args.save_dir) / f"{args.model_name}_smoke.pt"
    torch.save(model.state_dict(), ckpt_path.as_posix())
    print(f"[DONE] Training smoke test complete. Steps={steps}. Saved -> {ckpt_path}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ann", type=str, default="data/processed/msvd/train/annotations.json")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_wokers", type=int, default=0)   # 和loader 参数拼写一致
    ap.add_argument("--num_frame", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--max_len", type=int, default=32)

    ap.add_argument("--epochs", type=int, default=1)       # 预留，将来多 epoch
    ap.add_argument("--max_batches", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--model_name", type=str, default="simple_vc", choices=["simple_vc","tiny_gru"])
    ap.add_argument("--save_dir", type=str, default="checkpoints/msvd_debug")
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_pipeline(args)
