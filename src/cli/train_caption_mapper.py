# src/cli/train_caption_mapper.py
import argparse, os
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2TokenizerFast

from src.data.data_loader import build_dataloader
from src.models.caption_model import VideoCaptionModel


# ---------- Utils ----------
def set_seed(seed: int):
    import random
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: VideoCaptionModel, loader, pad_id: int, device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        v = batch["video"].to(device, non_blocking=True)
        t = batch["caption_ids"].to(device, non_blocking=True)
        loss = compute_loss_local(model, v, t, pad_id)
        total += float(loss.item()); n += 1
        if n >= 50:  # 验证不必太久
            break
    model.train()
    return total / max(n, 1)


def _prefix_to_hidden(model: VideoCaptionModel, prefix: torch.Tensor, hidden: int) -> torch.Tensor:
    """
    目标输出：[B, P, hidden]。
    兼容 mapper 的多种输入习惯，若不通则自动创建 model.adapter: Linear(Dp->hidden)。
    优先级：decoder.mapper([B,Dp]/[B,P,Dp]/[B,P*Dp]) -> reshape(Dp整除hidden) -> adapter
    """
    B, P, Dp = prefix.shape
    device = prefix.device
    if Dp == hidden:
        return prefix

    mapper = getattr(model.decoder, "mapper", None)
    if isinstance(mapper, nn.Module):
        # 1) 常见：只接 [B, Dp]
        try:
            out2 = mapper(prefix.mean(dim=1))  # [B,hidden]?
            if out2.dim() == 2 and out2.shape[-1] == hidden:
                return out2.unsqueeze(1)
        except Exception:
            pass
        # 2) 支持 3D：[B,P,Dp] -> [B,P,hidden]
        try:
            out3 = mapper(prefix)
            if out3.dim() == 3 and out3.shape[-1] == hidden:
                return out3
        except Exception:
            pass
        # 3) 支持扁平：[B,P*Dp] -> [B,hidden] / [B,P*hidden]
        try:
            flat = prefix.reshape(B, P * Dp)
            outf = mapper(flat)
            if outf.dim() == 2 and outf.shape[-1] == hidden:
                return outf.unsqueeze(1)
            if outf.dim() == 2 and outf.shape[-1] % hidden == 0:
                P2 = outf.shape[-1] // hidden
                return outf.reshape(B, P2, hidden)
            if outf.dim() == 3 and outf.shape[-1] == hidden:
                return outf
        except Exception:
            pass

    # 4) 直接 reshape（当 Dp 是 hidden 的整数倍）
    if Dp % hidden == 0:
        P2 = Dp // hidden
        return prefix.reshape(B, P * P2, hidden)

    # 5) 兜底：自动挂 adapter
    if not hasattr(model, "adapter") or not isinstance(model.adapter, nn.Module):
        model.adapter = nn.Linear(Dp, hidden, bias=True).to(device)
        print(f"[adapter] created Linear({Dp} -> {hidden})")
    return model.adapter(prefix)  # [B,P,hidden]


def compute_loss_local(model: VideoCaptionModel, video: torch.Tensor, caption_ids: torch.Tensor, pad_id: int):
    """
    与 generate() 同路径：encoder → proj → (mapper/adapter) → GPT-2 （teacher-forcing）
    """
    device = video.device

    # 1) 视觉前缀
    video_emb = model.encoder(video)          # 形状依 encoder
    prefix = model.proj(video_emb)            # [B,Dp] 或 [B,P,Dp]
    if prefix.dim() == 2:
        prefix = prefix.unsqueeze(1)          # [B,1,Dp]
    B, P, Dp = prefix.shape

    # 2) GPT-2
    gpt2 = getattr(model.decoder, "model", model.decoder)
    hidden = int(gpt2.config.n_embd)          # gpt2-base=768

    # 3) 前缀对齐到 [B,P,hidden]
    prefix = _prefix_to_hidden(model, prefix, hidden)  # [B,P,hidden]
    B, P, _ = prefix.shape

    # 4) teacher-forcing：inputs/labels 错一位
    inp_ids = caption_ids[:, :-1].contiguous()
    labels  = caption_ids[:, 1:].contiguous()

    # 5) 前缀 + token embedding
    tok_emb = gpt2.transformer.wte(inp_ids)           # [B,L-1,hidden]
    inputs_embeds = torch.cat([prefix, tok_emb], dim=1)  # [B,P+L-1,hidden]

    # 6) mask & labels（前缀不计损）
    att_txt   = (inp_ids != pad_id).long()
    attn_mask = torch.cat([torch.ones((B, P), device=device, dtype=torch.long),
                           att_txt], dim=1)
    ignore    = -100
    pad_prefix = torch.full((B, P), ignore, dtype=torch.long, device=device)
    lm_labels  = torch.cat([pad_prefix, labels], dim=1)

    out = gpt2(inputs_embeds=inputs_embeds,
               attention_mask=attn_mask,
               labels=lm_labels,
               use_cache=False)
    return out.loss


# ---------- Main ----------
def build_argparser():
    p = argparse.ArgumentParser("Fine-tune VideoCaptionModel mapper (and optional GPT-2 tail)")
    # 数据
    p.add_argument("--ann_train", default="./data/processed/msvd/train/annotations.json")
    p.add_argument("--ann_val",   default="./data/processed/msvd/val/annotations.json")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_frame",  type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_len",    type=int, default=48)
    p.add_argument("--shuffle", action="store_true", default=True)
    # 模型
    p.add_argument("--vit_name", default="vit_base_patch16_224")
    p.add_argument("--gpt2_name", default="gpt2")
    p.add_argument("--cond_mode", choices=["prefix","bos"], default="prefix")
    p.add_argument("--prefix_len", type=int, default=4)
    # 冻结/解冻
    p.add_argument("--freeze_vit", action="store_true", default=True)
    p.add_argument("--unfreeze_gpt2_last", type=int, default=0, help="unfreeze last N GPT-2 blocks (e.g., 2)")
    # 训练
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4, help="lr for proj/mapper/adapter")
    p.add_argument("--lr_gpt2", type=float, default=1e-5, help="lr for unfrozen GPT-2 tail")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--val_every", type=int, default=20)
    # 日志 & 权重
    p.add_argument("--run_dir", default="runs/mapper_ft")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--ckpt_name", default="msvd_mapper_finetune.pt")
    return p


def main():
    ap = build_argparser(); args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    events_csv = Path(args.run_dir) / "events.csv"

    # tokenizer
    tok = GPT2TokenizerFast.from_pretrained(args.gpt2_name)
    tok.pad_token = tok.eos_token

    # dataloaders
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

    device = args.device
    model = VideoCaptionModel(
        vit_name=args.vit_name, gpt2_name=args.gpt2_name,
        cond_mode=args.cond_mode, prefix_len=args.prefix_len,
        freeze_vit=True, unfreeze_last=0
    ).to(device).train()

    # ===== 冻结/解冻设置 =====
    # 冻结 ViT
    if args.freeze_vit:
        for p in model.encoder.parameters():
            p.requires_grad = False

    # 冻结 GPT-2（默认全冻）
    gpt2 = getattr(model.decoder, "model", model.decoder)
    for p in gpt2.parameters():
        p.requires_grad = False

    # —— 预热一次前向，必要时触发 adapter 的创建（无梯度） —— #
    with torch.no_grad():
        probe_batch = next(iter(train_loader))
        v_probe = probe_batch["video"].to(device)
        t_probe = probe_batch["caption_ids"].to(device)
        _ = compute_loss_local(model, v_probe, t_probe, pad_id=tok.pad_token_id)

    # 组装可训练块
    train_blocks: List[tuple] = []  # (name, module, lr)

    # 1) proj
    if isinstance(model.proj, nn.Module):
        for p in model.proj.parameters():
            p.requires_grad = True
        train_blocks.append(("proj", model.proj, args.lr))

    # 2) decoder.mapper（若存在）
    mapper = getattr(model.decoder, "mapper", None)
    if isinstance(mapper, nn.Module):
        for p in mapper.parameters(): p.requires_grad = True
        train_blocks.append(("decoder.mapper", mapper, args.lr))

    # 3) adapter（若上一步创建了）
    if hasattr(model, "adapter") and isinstance(model.adapter, nn.Module):
        for p in model.adapter.parameters(): p.requires_grad = True
        train_blocks.append(("adapter", model.adapter, args.lr))

    # 4) 可选：解冻 GPT-2 尾层
    tail_params = []
    if args.unfreeze_gpt2_last > 0:
        blocks = gpt2.transformer.h
        for block in blocks[-args.unfreeze_gpt2_last:]:
            for p in block.parameters():
                p.requires_grad = True
                tail_params.append(p)

    # ===== 优化器参数组 =====
    groups = []
    for name, module, lr_ in train_blocks:
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr_})
    if tail_params:
        groups.append({"params": tail_params, "lr": args.lr_gpt2})

    total_trainable = sum(p.numel() for g in groups for p in g["params"])
    print("[trainable groups]")
    for name, module, lr_ in train_blocks:
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if n: print(f"  - {name:18s} | params={n:,} | lr={lr_}")
    if tail_params:
        print(f"  - gpt2_tail           | params={sum(p.numel() for p in tail_params):,} | lr={args.lr_gpt2}")
    print(f"[trainable total] {total_trainable:,} params")

    if not groups:
        names = [n for n, p in model.named_parameters()]
        hint = [n for n in names if "mapper" in n or "proj" in n or "prefix" in n][:20]
        raise RuntimeError(
            "没有可训练参数。请检查映射层命名（proj/decoder.mapper/prefix_*）。\n"
            f"可疑参数名样例（最多20个）：{hint}"
        )

    optimizer = optim.AdamW(groups, weight_decay=0.01)

    # ===== 训练 =====
    best_val = float("inf")
    step = 0
    for epoch in range(args.epochs):
        for batch in train_loader:
            video = batch["video"].to(device, non_blocking=True)
            caps  = batch["caption_ids"].to(device, non_blocking=True)

            loss = compute_loss_local(model, video, caps, pad_id=tok.pad_token_id)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step == 1 or step % 10 == 0:
                print(f"[train] step {step:05d} | loss {loss.item():.4f}")
            with open(events_csv, "a", encoding="utf-8") as f:
                f.write(f"{step},{loss.item():.6f}\n")

            if args.val_every > 0 and step % args.val_every == 0:
                val_loss = evaluate(model, val_loader, tok.pad_token_id, device)
                print(f"[val]   step {step:05d} | val_loss {val_loss:.4f}")
                with open(Path(args.run_dir) / "val.csv", "a", encoding="utf-8") as f:
                    f.write(f"{step},{val_loss:.6f}\n")
                if val_loss < best_val:
                    best_val = val_loss
                    ckpt_path = Path(args.ckpt_dir) / args.ckpt_name
                    torch.save({
                        "model_state": model.state_dict(),
                        "step": step, "epoch": epoch, "best_val": best_val,
                        "args": vars(args)
                    }, ckpt_path)
                    print(f"[ckpt] saved best -> {ckpt_path} (val={best_val:.4f})")

        # —— 每个 epoch 结束兜底验证 + 保存 —— #
        val_loss = evaluate(model, val_loader, tok.pad_token_id, device)
        print(f"[val][epoch end] epoch {epoch} | val_loss {val_loss:.4f}")
        with open(Path(args.run_dir) / "val.csv", "a", encoding="utf-8") as f:
            f.write(f"{step},{val_loss:.6f}\n")
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = Path(args.ckpt_dir) / args.ckpt_name
            torch.save({
                "model_state": model.state_dict(),
                "step": step, "epoch": epoch, "best_val": best_val,
                "args": vars(args)
            }, ckpt_path)
            print(f"[ckpt][epoch end] saved best -> {ckpt_path} (val={best_val:.4f})")

    print("[OK] mapper fine-tuning finished.")


if __name__ == "__main__":
    main()