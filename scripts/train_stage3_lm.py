#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-3: GPT-2 语言风格微调（仅用 captions，纯 PyTorch 训练循环）
- 不导入 transformers.Trainer，也不触发 TensorFlow/Keras
- 输入: data/processed/msvd/train/annotations.json
- 输出: checkpoints/gpt2_lm_stage3/best/  (可供 --gpt2_name 直接加载)
"""

import os
# 保险禁用 TF/Flax
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import argparse, json, random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup


def load_captions(ann_path: Path, max_items: int = 0) -> List[str]:
    rows = json.loads(ann_path.read_text(encoding="utf-8"))
    caps = []
    for r in rows:
        c = r.get("captions")
        if isinstance(c, list):
            caps.extend([x for x in c if isinstance(x, str) and x.strip()])
        elif isinstance(c, str) and c.strip():
            caps.append(c.strip())
    random.shuffle(caps)
    if max_items and len(caps) > max_items:
        caps = caps[:max_items]
    return caps


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_len: int = 48):
        self.texts = texts
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        labels = item["input_ids"].clone()
        labels[labels == self.tok.pad_token_id] = -100  # 忽略 pad
        item["labels"] = labels
        return item


def main():
    ap = argparse.ArgumentParser(description="Stage-3 LM (PyTorch loop)")
    ap.add_argument("--ann_train", default="data/processed/msvd/train/annotations.json")
    ap.add_argument("--gpt2_name", default="gpt2")
    ap.add_argument("--out_dir", default="checkpoints/gpt2_lm_stage3")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_items", type=int, default=0)  # 0=全量
    ap.add_argument("--max_len", type=int, default=48)
    ap.add_argument("--grad_accum", type=int, default=1) # 累计步数
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    caps = load_captions(Path(args.ann_train), max_items=args.max_items)
    if not caps:
        raise SystemExit(f"[ERROR] No captions found in {args.ann_train}")
    print(f"[INFO] loaded captions: {len(caps)}")

    tok = AutoTokenizer.from_pretrained(args.gpt2_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = TextDataset(caps, tok, max_len=args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.gpt2_name)
    model.to(device)

    # 优化器 & 计划
    num_update_steps_per_epoch = max(1, len(loader) // args.grad_accum)
    t_total = num_update_steps_per_epoch * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(t_total * args.warmup_ratio),
        num_training_steps=t_total
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())

    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast(enabled=torch.cuda.is_available()):
                out = model(**batch)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()
            running += loss.item()

            if step % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            if global_step % 50 == 0:
                print(f"epoch {epoch} | step {global_step}/{t_total} | loss {running:.4f}")
                running = 0.0

        # 每个 epoch 后保存一次（覆盖 latest）
        out_dir = Path(args.out_dir)
        (out_dir / "latest").mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir / "latest")
        tok.save_pretrained(out_dir / "latest")
        print(f"[SAVE] epoch {epoch} -> {out_dir / 'latest'}")

    # 训练结束，保存 best（这里直接用 latest 作为 best）
    best_dir = Path(args.out_dir) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_dir)
    tok.save_pretrained(best_dir)
    print(f"[OK] Stage-3 LM saved to: {best_dir}")


if __name__ == "__main__":
    main()