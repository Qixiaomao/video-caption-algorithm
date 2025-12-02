# src/cli/train_full.py
import argparse, os
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from transformers import GPT2TokenizerFast
from src.data.data_loader import build_dataloader

# 修复
from src.models.vit_text_align import ViTTextAlignModel


# —— 简洁稳健的对齐基线模型（延续你dry run的思路）——
class SimpleAlignModel(nn.Module):
    """
    视频：平均 [T,H,W] -> [B,3] -> 线性 -> L2 归一
    文本：Embedding(GPT2 词表) 求均值(忽略pad) -> 线性 -> L2 归一
    损失：CosineEmbeddingLoss（同类目标=+1）
    """
    def __init__(self, vocab_size: int, pad_id: int, d: int = 256):
        super().__init__()
        self.pad_id = pad_id
        self.txt_emb = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.txt_proj = nn.Linear(d, d)
        self.vid_proj = nn.Linear(3, d)
        self.loss_fn = nn.CosineEmbeddingLoss()

    def forward(self, video, caption_ids):
        # video: [B,T,3,H,W] -> [B,3]
        v = video.mean(dim=(1,3,4))
        v = nn.functional.normalize(self.vid_proj(v), dim=-1)

        mask = (caption_ids != self.pad_id).float()
        x = self.txt_emb(caption_ids)                             # [B,L,d]
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        x = (x * mask.unsqueeze(-1)).sum(dim=1) / denom          # [B,d]
        x = nn.functional.normalize(self.txt_proj(x), dim=-1)

        target = torch.ones(video.size(0), device=video.device)
        loss = self.loss_fn(v, x, target)
        return loss

def parse_args():
    p = argparse.ArgumentParser()
    # 添加的参数
    p.add_argument("--model",choices=["simple","vit"],default="simple")
    p.add_argument("--vit_name",default="vit_base_patch16_224")
    p.add_argument("--freeze_vit",action="store_true",default=False)
    # 数据
    p.add_argument("--ann_train", default="./data/processed/msvd/train/annotations.json")
    p.add_argument("--ann_val",   default="./data/processed/msvd/val/annotations.json")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_frame",  type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_len",    type=int, default=32)
    p.add_argument("--shuffle", action="store_true", default=True)
    # 训练
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=0, help=">0 则在达到该步数提前停止")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=123)
    # 日志 & 权重
    p.add_argument("--run_dir", default="runs/2025-10-05_day8")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--ckpt_name", default="msvd_simplealign_best.pt")
    p.add_argument("--val_every", type=int, default=50, help="每多少 step 做一次验证")
    return p.parse_args()

def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        v = batch["video"].to(device)
        t = batch["caption_ids"].to(device)
        loss = model(v, t)
        total += loss.item()
        n += 1
        if n >= 50:  # 验证不必太长，节省时间
            break
    model.train()
    return total / max(n,1)

def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    events_csv = Path(args.run_dir) / "events.csv"
    
    # 初始化 tokenizer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    # loaders
    train_loader = build_dataloader(
        ann_path=args.ann_train, tokenizer=tok,
        batch_size=args.batch_size, max_len=args.max_len,
        num_frame=args.num_frame, image_size=args.image_size,
        shuffle=args.shuffle, num_wokers=0
    )
    val_loader = build_dataloader(
        ann_path=args.ann_val, tokenizer=tok,
        batch_size=args.batch_size, max_len=args.max_len,
        num_frame=args.num_frame, image_size=args.image_size,
        shuffle=False, num_wokers=0
    )

    # 创建模型
    device = torch.device(args.device)
    
    if args.model == "vit":
        model = ViTTextAlignModel(
            vocab_size=tok.vocab_size,
            pad_id = tok.pad_token_id,
            vit_name = args.vit_name,
            freeze_vit=args.freeze_vit,
            proj_dim=256,txt_dim=512,txt_layers=2,txt_nhead=8
        ).to(device)
    else:
       model = SimpleAlignModel(tok.vocab_size, tok.pad_token_id, d=256).to(device)
    # 优化器
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    # 训练循环
    best_val = float("inf")
    step = 0

    for epoch in range(args.epochs):
        for batch in train_loader:
            video = batch["video"].to(device)
            caps  = batch["caption_ids"].to(device)
            loss  = model(video, caps)

            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

            # 训练日志
            if step == 1 or step % 10 == 0:
                print(f"[train] step {step:05d} | loss {loss.item():.4f}")
            with open(events_csv, "a", encoding="utf-8") as f:
                f.write(f"{step},{loss.item():.6f}\n")

            # 验证 & 保存最好模型
            if args.val_every > 0 and step % args.val_every == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"[val]   step {step:05d} | val_loss {val_loss:.4f}")
                with open(Path(args.run_dir) / "val.csv", "a", encoding="utf-8") as f:
                    f.write(f"{step},{val_loss:.6f}\n")
                if val_loss < best_val:
                    best_val = val_loss
                    ckpt_path = Path(args.ckpt_dir) / args.ckpt_name
                    torch.save({
                        "step": step,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "opt_state": opt.state_dict(),
                        "best_val": best_val,
                        "args": vars(args),
                    }, ckpt_path)
                    print(f"[ckpt] saved best -> {ckpt_path} (val={best_val:.4f})")

            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    print("[OK] training finished.")

if __name__ == "__main__":
    main()
