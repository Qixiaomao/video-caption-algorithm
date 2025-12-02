#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.data.data_loader import build_dataloader
from src.models.simple_vc import SimpleVideoCaptioner

def set_seed(seed=42):
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

def shift_for_lm(labels):
    # teacher forcing: input =[:-1], target =[1:]
    return labels[:, :-1].contiguous(), labels[:, 1:].contiguous()

def train_one_epoch(model, loader, optimizer, criterion, device, pad_id, max_batches, log_interval):
    model.train()
    pbar = tqdm(enumerate(loader), total=min(len(loader), max_batches), desc="train")
    step = 0
    for i, batch in pbar:
        if i >= max_batches: break
        video = batch.get("video")
        cap_ids = batch.get("caption_ids")
        if video is None or cap_ids is None: continue

        video = video.to(device, non_blocking=True)
        labels = cap_ids.to(device, non_blocking=True).long()

        inp, tgt = shift_for_lm(labels)
        logits = model(video, inp)  # logits: [B, L, V]

        L = min(logits.size(1), tgt.size(1))
        loss = criterion(
            logits[:, :L, :].reshape(-1, logits.size(-1)),
            tgt[:, :L].reshape(-1)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        if step % log_interval == 0:
            pbar.set_postfix(loss=float(loss.detach().cpu()), step=step)
    return step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ann", type=str, default="data/processed/msvd/train/annotations.json")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_wokers", type=int, default=0)  # 与现有的loader命名一致
    ap.add_argument("--num_frame", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_batches", type=int, default=50)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--save_dir", type=str, default="checkpoints/msvd_debug")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    tok = get_tokenizer()
    vocab_size = getattr(tok, "vocab_size", 30522)
    pad_id = getattr(tok, "pad_token_id", 0)

    loader = build_dataloader(
        args.train_ann, tok,
        batch_size=args.batch_size, shuffle=True,
        num_wokers=args.num_wokers, num_frame=args.num_frame, image_size=args.image_size
    )

    model = SimpleVideoCaptioner(
        vocab_size=vocab_size, hidden_size=args.d_model, max_len=args.max_len, pad_id=pad_id
    ).to(device)

    os.makedirs(args.save_dir, exist_ok=True)
    opt = AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=pad_id)

    steps = train_one_epoch(model, loader, opt, crit, device, pad_id, args.max_batches, args.log_interval)
    ckpt = Path(args.save_dir) / "simple_vc_smoke.pt"
    torch.save(model.state_dict(), ckpt.as_posix())
    print(f"[DONE] steps={steps}, saved -> {ckpt}")

if __name__ == "__main__":
    main()
