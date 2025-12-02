# src/cli/train.py
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2TokenizerFast
from src.data.data_loader import build_dataloader
import random

# ============ 极简可跑模型 ============
class SimpleAlignModel(nn.Module):
    """
    用于 dry run 的最小可跑模型：
    - 视频特征：对 [B,T,3,H,W] 在 T,H,W 上做均值 -> [B,3]，线性投到 d 维后做归一化
    - 文本特征：token 嵌入均值（忽略 pad）-> 线性投到 d 维后做归一化
    - 损失：CosineEmbeddingLoss，目标都为 +1
    """
    def __init__(self, vocab_size: int, pad_id: int, d: int = 256):
        super().__init__()
        self.pad_id = pad_id
        self.txt_emb = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.txt_proj = nn.Linear(d, d)
        self.vid_proj = nn.Linear(3, d)  # 视频均值后只剩 3 通道
        self.loss_fn = nn.CosineEmbeddingLoss()

    def forward(self, video: torch.Tensor, caption_ids: torch.Tensor) -> torch.Tensor:
        # video: [B,T,3,H,W] -> [B,3]
        v = video.mean(dim=(1, 3, 4))
        v = self.vid_proj(v)
        v = nn.functional.normalize(v, dim=-1)

        # text: [B,L] -> mask pad -> mean -> proj
        mask = (caption_ids != self.pad_id).float()  # [B,L]
        x = self.txt_emb(caption_ids)                # [B,L,d]
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        x = (x * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B,d]
        x = self.txt_proj(x)
        x = nn.functional.normalize(x, dim=-1)

        target = torch.ones(video.size(0), device=video.device)
        loss = self.loss_fn(v, x, target)
        return loss

# ============ 参数解析 ============
def parse_args():
    p = argparse.ArgumentParser()
    # 数据 & Loader
    p.add_argument("--ann_path", default="data/processed/msvd/train/annotations.json")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_frame", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument("--shuffle", action="store_true", default=True)
    # 训练
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=50, help="到达该步数提前停止；None 表示按 epochs 完整跑")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=123)
    # 日志
    p.add_argument("--events_csv", default="runs/dryrun_day7/events.csv")
    return p.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(Path(args.events_csv).parent.as_posix(), exist_ok=True)

    # Tokenizer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    # Debug 路径确认
    print("[DEBUG] ann_path used by train:", args.ann_path, flush=True)

    # DataLoader
    train_loader = build_dataloader(
        ann_path=args.ann_path,
        tokenizer=tok,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_frame=args.num_frame,
        image_size=args.image_size,
        shuffle=args.shuffle,
        num_wokers=0,  # 注意：保持与你 data_loader.py 中的参数名一致
    )

    device = torch.device(args.device)
    model = SimpleAlignModel(vocab_size=tok.vocab_size, pad_id=tok.pad_token_id, d=256).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    # 训练循环（dry run）
    step = 0
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            video = batch["video"].to(device)           # [B,T,3,H,W]
            caps  = batch["caption_ids"].to(device)     # [B,L]
            loss  = model(video, caps)

            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

            # 打印 & 记录
            if step == 1 or step % 10 == 0:
                print(f"step {step:04d} | loss {loss.item():.4f}")
            with open(args.events_csv, "a", encoding="utf-8") as f:
                f.write(f"{step},{loss.item():.6f}\n")

            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    print("[OK] training dry run finished. events ->", args.events_csv)

if __name__ == "__main__":
    main()
