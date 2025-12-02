# src/models/caption_model.py
# -*- coding: utf-8 -*-
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch, torch.nn.functional as F

from src.models.video_encoder import build_vit_encoder
from src.models.text_decoder import GPT2TextDecoder

class VideoCaptionModel(nn.Module):
    """
    视频帧 -> ViT 编码 -> (可选 MLP) -> GPT-2 条件生成
    只依赖 text_decoder，不要形成反向 import。
    """
    def __init__(self,
                 vit_name: str = "vit_base_patch16_224",
                 video_dim: int = 256,
                 gpt2_name: str = "gpt2",
                 cond_mode: str = "prefix",
                 prefix_len: int = 4,
                 proj_hidden: int = 0,
                 freeze_vit: bool = True,
                 unfreeze_last: int = 0):
        super().__init__()
        self.encoder = build_vit_encoder(
            model_name=vit_name,
            out_dim=video_dim,
            pretrained=True,
            pool="cls",
            l2norm=False,
            freeze=freeze_vit,
            unfreeze_last=unfreeze_last
        )
        if proj_hidden > 0:
            self.proj = nn.Sequential(
                nn.Linear(video_dim, proj_hidden),
                nn.ReLU(),
                nn.Linear(proj_hidden, video_dim),
            )
        else:
            self.proj = nn.Identity()

        self.decoder = GPT2TextDecoder(
            gpt2_name=gpt2_name,
            cond_mode=cond_mode,
            prefix_len=prefix_len,
            video_dim=video_dim,
        )

    def forward(self,
                video: torch.Tensor,                # [B,T,3,224,224]
                input_ids: torch.Tensor,            # [B,L]
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        video_emb = self.encoder(video)             # [B, D]
        video_emb = self.proj(video_emb)            # [B, D]
        out = self.decoder(video_emb, input_ids, attention_mask, labels)
        return out

    @torch.no_grad()
    def generate(self,
                 video: torch.Tensor,
                 prompt: str = "",
                 **gen_kwargs):
        video_emb = self.encoder(video)
        video_emb = self.proj(video_emb)
        return self.decoder.generate(video_emb, prompt=prompt, **gen_kwargs)
    
    # --- 在 VideoCaptionModel 内部新增（或作为同文件的辅助函数也行）---
    def compute_loss(self, video: torch.Tensor, caption_ids: torch.Tensor, pad_id: int):
        """
        计算视频-文本对的交叉熵损失。
        video: [B,T,3,H,W]
        caption_ids: [B,L]，包含 <bos> 和 <eos>，padding 用 pad_id
        返回: 标量损失
        """
        device = video.device
        # 1) 视觉前缀（和 generate() 同步）
        video_emb = self.encoder(video)     # 形状依你的 encoder
        prefix    = self.proj(video_emb)    # 可能是 [B, Dp] 或 [B, P, Dp]
        if prefix.dim() == 2:
            prefix = prefix.unsqueeze(1)    # -> [B, 1, Dp]

        # 2) GPT-2 配置
        gpt2 = getattr(self.decoder, "model", self.decoder)   # 兼容 wrapper
        hidden = gpt2.config.n_embd                           # 通常 768
        B, P, Dp = prefix.shape

        # 3) 让 prefix 最后一维对齐到 GPT-2 隐藏维
        if Dp != hidden:
            # 3a) 优先使用 decoder.mapper（若存在）
            if hasattr(self.decoder, "mapper") and self.decoder.mapper is not None:
                # 常见 mapper 支持 [B,P,Dp] 或 [B,Dp]
                try:
                    prefix = self.decoder.mapper(prefix)      # 期望 -> [B,P,hidden]
                except Exception:
                    # 尝试先合并 P 维
                    prefix_flat = prefix.reshape(B, P*Dp)
                    prefix = self.decoder.mapper(prefix_flat).reshape(B, -1, hidden)
            # 3b) 如果 proj 输出已经是 [B, P*hidden]，则直接 reshape
            if prefix.shape[-1] != hidden:
                if prefix.shape[-1] % hidden == 0:
                    P2 = prefix.shape[-1] // hidden
                    prefix = prefix.reshape(B, P*P2, hidden)
                    P = prefix.shape[1]
                else:
                    raise RuntimeError(
                        f"[compute_loss] prefix dim mismatch: got Dp={Dp}, but GPT2 hidden={hidden}. "
                        "请把 self.proj 的 out_features 设为 hidden * prefix_len（例如 768*P），"
                        "或提供 decoder.mapper 将 Dp→hidden 的映射。"
                    )

        # 4) teacher-forcing：输入/标签错一位
        inp_ids = caption_ids[:, :-1].contiguous()  # [B,L-1]
        labels  = caption_ids[:, 1:].contiguous()   # [B,L-1]

        # 5) 拼接输入嵌入：prefix + token embedding
        wte  = gpt2.transformer.wte                 # 词嵌入层
        tok_emb = wte(inp_ids)                      # [B,L-1,hidden]
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)  # [B,P+L-1,hidden]

        # 6) attention mask / labels 对齐
        att_txt   = (inp_ids != pad_id).long()                   # [B,L-1]
        attn_mask = torch.cat([torch.ones((B, P), device=device, dtype=torch.long),
                            att_txt], dim=1)                  # [B,P+L-1]
        ignore    = -100
        pad_prefix = torch.full((B, P), ignore, dtype=torch.long, device=device)
        lm_labels  = torch.cat([pad_prefix, labels], dim=1)      # [B,P+L-1]

        out = gpt2(inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                labels=lm_labels,
                use_cache=False)
        return out.loss

    