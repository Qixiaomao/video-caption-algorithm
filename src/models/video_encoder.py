# src/model/video_encoder.py
# -*- coding: utf-8 -*-
"""
Generic ViT-based video frame encoder.
- backend: timm (preferred) or torchvision (fallback)
- input: [B,T,3,H,W] or [B,3,H,W]
- temporal pooling: mean over frames
- spatial pooling: 'cls' or 'gap'
- fine-tune control: freeze or unfreeze last N layers
- timm (at first) or torchvision (fallback)
- GLS / GAP 池化
- 冻结/解冻策略，微调友好
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn

# --------- backend loader (timm preferred) ----------
_HAVE_TIMM = True
try:
    import timm
except Exception:
    _HAVETIMM = False
    _HAVE_TIMM = False
from torchvision.models import vit_b_16, ViT_B_16_Weights


@dataclass
class ViTConfig:
    model_name: str = "vit_base_patch16_224"
    out_dim: int = 256
    pretrained: bool = True
    pool: Literal["cls", "gap"] = "cls"
    l2norm: bool = True
    freeze_all: bool = False
    unfreeze_last: int = 0  # 解冻最后 N 个 block（当 freeze_all=False 时才生效）
    drop: float = 0.0       # 额外 dropout（投影层前）


class ViTFrameEncoder(nn.Module):
    """
    ViT video frame encoder:
      forward(x): x ∈ [B,T,3,H,W] or [B,3,H,W] -> [B, D]
    """
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg

        if _HAVE_TIMM:
            self.backbone = timm.create_model(
                cfg.model_name,
                pretrained=cfg.pretrained,
                num_classes=0,        # 返回特征，不要分类头
                global_pool="",       # 手动做池化
            )
            self.embed_dim = self.backbone.num_features
            self.blocks = getattr(self.backbone, "blocks", None)  # 用于解冻控制
        else:
            # 兜底：torchvision 的 ViT-B/16，输出是1000维分类 logits；我们去掉 head 前层做特征
            tv = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if cfg.pretrained else None)
            # 去分类头，拿到 encoder 输出（利用 forward_features）
            class _TorchVisionVit(nn.Module):
                def __init__(self, tv_model):
                    super().__init__()
                    self.model = tv_model
                    # torchvision vit_b_16 内部 embed_dim 固定为 768
                    self.num_features = 768
                def forward_features(self, x):
                    # 参考 torchvision 实现
                    # 返回 (cls_token, patch_tokens) 的拼接：取 cls（index 0）
                    x = self.model._process_input(x)
                    n = x.shape[0]
                    batch_class_token = self.model.class_token.expand(n, -1, -1)
                    x = torch.cat([batch_class_token, x], dim=1)
                    x = self.model.encoder(x)
                    return x  # [B, 1+N, 768]
                def forward(self, x):
                    feats = self.forward_features(x)  # [B, 1+N, 768]
                    # 下游手动做池化，不做 head
                    return feats
            self.backbone = _TorchVisionVit(tv)
            self.embed_dim = 768
            self.blocks = None

        # 投影到统一维度
        proj_in = self.embed_dim
        self.dropout = nn.Dropout(cfg.drop) if cfg.drop > 1e-6 else nn.Identity()
        self.proj = nn.Linear(proj_in, cfg.out_dim)

        # 冻结/解冻策略
        self._apply_finetune_policy(cfg.freeze_all, cfg.unfreeze_last)

    # ---------- finetune control ----------
    def _apply_finetune_policy(self, freeze_all: bool, unfreeze_last: int):
        # 先全冻结/解冻
        for p in self.backbone.parameters():
            p.requires_grad = not freeze_all

        # 如果使用 timm 并且需要解冻最后 N 个 block
        if _HAVE_TIMM and self.blocks is not None and (not freeze_all) and unfreeze_last > 0:
            # 先默认冻结，再解冻尾部
            for p in self.backbone.parameters():
                p.requires_grad = False
            n = len(self.blocks)
            for blk in self.blocks[max(0, n - unfreeze_last):]:
                for p in blk.parameters():
                    p.requires_grad = True
            # 同时解冻最后的 norm/head 层（如果存在）
            for name in ["norm", "pre_logits"]:
                mod = getattr(self.backbone, name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = True

    # ---------- pooling ----------
    def _spatial_pool(self, feat_2d: torch.Tensor) -> torch.Tensor:
        """
        feat_2d: 若来自 timm.forward_features → [B, C] 或 [B, tokens, C]
                 若来自 torchvision 兜底 → [B, 1+N, C]
        返回: [B, C]
        """
        if feat_2d.ndim == 2:
            # 已是 [B, C]
            return feat_2d

        # [B, tokens, C]
        if self.cfg.pool == "cls":
            return feat_2d[:, 0]  # CLS token
        else:
            # GAP over patch tokens（忽略 cls）
            return feat_2d[:, 1:].mean(dim=1)

    # ---------- forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,3,H,W] or [B,3,H,W]
        return: [B, out_dim]
        """
        if x.ndim == 4:
            # [B,3,H,W] -> expand T=1
            x = x.unsqueeze(1)
        assert x.ndim == 5, f"expect [B,T,3,H,W], got {x.shape}"

        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        # timm: forward_features 返回 [B,C] 或 [B, tokens, C]
        if _HAVE_TIMM:
            if hasattr(self.backbone, "forward_features"):
                feat = self.backbone.forward_features(x)
            else:
                feat = self.backbone(x)  # 兜底
        else:
            feat = self.backbone(x)  # torchvision 兜底 → [B, 1+N, C]

        feat = self._spatial_pool(feat)  # [B*T, C]
        feat = feat.reshape(B, T, -1).mean(dim=1)  # temporal mean → [B, C]

        feat = self.dropout(feat)
        feat = self.proj(feat)  # [B, out_dim]

        if self.cfg.l2norm:
            feat = torch.nn.functional.normalize(feat, dim=-1)

        return feat


# ---------- builder API ----------
def build_vit_encoder(
    model_name: str = "vit_base_patch16_224",
    out_dim: int = 256,
    pretrained: bool = True,
    pool: str = "cls",
    l2norm: bool = True,
    freeze: bool = False,
    unfreeze_last: int = 0,
    drop: float = 0.0,
) -> ViTFrameEncoder:
    """
    Factory to build ViTFrameEncoder with a simple signature.
    - If freeze=True, whole backbone is frozen (proj trainable)
    - If freeze=False and unfreeze_last>0, unfreeze last N blocks (timm only)
    """
    cfg = ViTConfig(
        model_name=model_name,
        out_dim=out_dim,
        pretrained=pretrained,
        pool=pool, l2norm=l2norm,
        freeze_all=freeze,
        unfreeze_last=unfreeze_last,
        drop=drop
    )
    return ViTFrameEncoder(cfg)