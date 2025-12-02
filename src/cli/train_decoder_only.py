# -*- coding: utf-8 -*-
"""
Train GPT-2 decoder as a pure language model on MSVD captions.
Usage:
  python -m src.cli.train_decoder_only --ann_train data/processed/msvd/train/annotations.json --ann_val data/processed/msvd/val/annotations.json --save_dir checkpoints/gpt2_lm_stage3
"""
import argparse, json, math, os
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup
from tqdm import tqdm


def load_captions(ann_path: str, min_len: int = 3) -> List[str]:
    rows = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    caps: List[str] = []
    for r in rows:
        c = r.get("captions") or r.get("caption") or []
        if isinstance(c, list):
            for s in c:
                s = (s or "").strip()
                if len(s.split()) >= min_len:
                    caps.append(s)
        elif isinstance(c, str):
            s = c.strip()
            if len(s.split()) >= min_len:
                caps.append(s)
    return caps


class CaptionLMDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: GPT2TokenizerFast, max_len: int = 64):
        self.texts = texts
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = enc.input_ids.squeeze(0)
        attn = enc.attention_mask.squeeze(0)
        # causal LM：labels = input_ids（padding 位置忽略）
        labels = input_ids.clone()
        labels[attn == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def build_loader(texts: List[str], tok: GPT2TokenizerFast, max_len: int, batch_size: int, shuffle: bool) -> DataLoader:
    ds = CaptionLMDataset(texts, tok, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def parse_args():
    ap = argparse.ArgumentParser("Decoder-only LM finetune on captions")
    ap.add_argument("--ann_train", required=True)
    ap.add_argument("--ann_val",   required=True)
    ap.add_argument("--gpt2_name", default="gpt2")
    ap.add_argument("--epochs",    type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len",   type=int, default=64)
    ap.add_argument("--lr",        type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--save_dir",  default="checkpoints/gpt2_lm_stage3")
    ap.add_argument("--seed",      type=int, default=123)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) data
    train_texts = load_captions(args.ann_train)
    val_texts   = load_captions(args.ann_val)
    print(f"[DATA] train caps={len(train_texts)}  val caps={len(val_texts)}")

    tok = GPT2TokenizerFast.from_pretrained(args.gpt2_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_loader = build_loader(train_texts, tok, args.max_len, args.batch_size, shuffle=True)
    val_loader   = build_loader(val_texts, tok, args.max_len, batch_size=args.batch_size, shuffle=False)

    # 2) model/optim
    model = GPT2LMHeadModel.from_pretrained(args.gpt2_name).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    # 3) train
    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"[E{ep}] train", ncols=100)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            running += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(1, len(train_loader))

        # 4) eval
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[E{ep}]  val", ncols=100):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                val_loss += out.loss.item()
        val_loss /= max(1, len(val_loader))
        ppl = math.exp(min(val_loss, 20))
        print(f"[E{ep}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ppl={ppl:.2f}")

        # 5) save best
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        # 始终保存最近一次，且保存 best
        model.save_pretrained(args.save_dir)
        tok.save_pretrained(args.save_dir)
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(os.path.join(args.save_dir, "best"))
            tok.save_pretrained(os.path.join(args.save_dir, "best"))
            print(f"[SAVE] best -> {os.path.join(args.save_dir, 'best')}")

    print(f"[DONE] saved dir: {args.save_dir}  (and {args.save_dir}/best)")


if __name__ == "__main__":
    main()