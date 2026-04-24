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
"""

from __future__ import annotations
import logging
import importlib.util
from contextlib import nullcontext
from dataclasses import dataclass
from types import MethodType
from typing import Literal

import torch
import torch.nn as nn

from core.operators.cupy_vit_pool import vit_fused_pool_temporal

log = logging.getLogger(__name__)

_HAVE_TIMM = True
try:
    import timm
except Exception:
    _HAVETIMM = False
    _HAVE_TIMM = False
from torchvision.models import ViT_B_16_Weights, vit_b_16


@dataclass
class ViTConfig:
    model_name: str = "vit_base_patch16_224"
    out_dim: int = 256
    pretrained: bool = True
    pool: Literal["cls", "gap"] = "cls"
    l2norm: bool = True
    freeze_all: bool = False
    unfreeze_last: int = 0
    drop: float = 0.0
    enable_attention_fastpath: bool = True
    prefer_channels_last: bool = True
    enable_torch_compile: bool = True
    torch_compile_mode: str = "reduce-overhead"
    enable_amp_fp16: bool = False
    enable_mlp_bias_gelu_fusion: bool = True
    enable_residual_layernorm_fusion: bool = True
    enable_inplace_residual_add_fusion: bool = True
    enable_cupy_fused_pool: bool = False
    cupy_pool_force_fp16: bool = True


class ViTFrameEncoder(nn.Module):
    """
    ViT video frame encoder:
      forward(x): x -> [B,T,3,H,W] or [B,3,H,W] -> [B, D]
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg

        if _HAVE_TIMM:
            self.backbone = timm.create_model(
                cfg.model_name,
                pretrained=cfg.pretrained,
                num_classes=0,
                global_pool="",
            )
            self.embed_dim = self.backbone.num_features
            self.blocks = getattr(self.backbone, "blocks", None)
            self._enable_attention_fastpath()
            self._enable_mlp_bias_gelu_fusion()
            self._enable_residual_layernorm_fusion()
        else:
            tv = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if cfg.pretrained else None)

            class _TorchVisionVit(nn.Module):
                def __init__(self, tv_model):
                    super().__init__()
                    self.model = tv_model
                    self.num_features = 768

                def forward_features(self, x):
                    x = self.model._process_input(x)
                    n = x.shape[0]
                    batch_class_token = self.model.class_token.expand(n, -1, -1)
                    x = torch.cat([batch_class_token, x], dim=1)
                    x = self.model.encoder(x)
                    return x

                def forward(self, x):
                    return self.forward_features(x)

            self.backbone = _TorchVisionVit(tv)
            self.embed_dim = 768
            self.blocks = None

        self.dropout = nn.Dropout(cfg.drop) if cfg.drop > 1e-6 else nn.Identity()
        self.proj = nn.Linear(self.embed_dim, cfg.out_dim)
        self._backbone_forward = self._forward_backbone_impl
        self._compile_attempted = False
        self._compile_disabled = False
        self._apply_finetune_policy(cfg.freeze_all, cfg.unfreeze_last)

    def _enable_attention_fastpath(self):
        """Prefer timm fused attention so PyTorch can dispatch SDPA fast kernels."""

        if not self.cfg.enable_attention_fastpath or self.blocks is None:
            return

        for block in self.blocks:
            attn = getattr(block, "attn", None)
            if attn is not None and hasattr(attn, "fused_attn"):
                attn.fused_attn = True

    def _enable_mlp_bias_gelu_fusion(self):
        """Make timm MLP activation fusion-friendly for inference kernels."""

        if not self.cfg.enable_mlp_bias_gelu_fusion or self.blocks is None:
            return

        for block in self.blocks:
            mlp = getattr(block, "mlp", None)
            act = getattr(mlp, "act", None)
            if isinstance(act, nn.GELU):
                # tanh-approx GELU is commonly fused with Linear+bias on CUDA.
                act.approximate = "tanh"

    def _enable_residual_layernorm_fusion(self):
        """Enable residual+layernorm fusion hints when supported by the model."""

        if not self.cfg.enable_residual_layernorm_fusion or self.blocks is None:
            return

        for block in self.blocks:
            # Some timm variants expose explicit fused add-norm toggles.
            if hasattr(block, "fused_add_norm"):
                block.fused_add_norm = True
            if hasattr(block, "fused_dropout_add_ln"):
                block.fused_dropout_add_ln = True
            if self.cfg.enable_inplace_residual_add_fusion and hasattr(block, "norm1") and hasattr(block, "norm2"):
                self._patch_block_inplace_residual_forward(block)

    def _patch_block_inplace_residual_forward(self, block: nn.Module):
        """Patch timm ViT block forward for inference-only in-place residual adds.

        The training/grad path stays on the original implementation.
        """

        if hasattr(block, "_orig_forward_inference_fusion"):
            return

        block._orig_forward_inference_fusion = block.forward

        def _forward_fused(this, x: torch.Tensor, attn_mask=None):
            if this.training or torch.is_grad_enabled():
                return this._orig_forward_inference_fusion(x, attn_mask=attn_mask)

            # In eval mode, drop_path is identity. Use in-place residual adds to
            # reduce temporary tensors and elementwise traffic.
            attn_out = this.ls1(this.attn(this.norm1(x), attn_mask=attn_mask))
            x = x.add_(attn_out)
            mlp_out = this.ls2(this.mlp(this.norm2(x)))
            x = x.add_(mlp_out)
            return x

        block.forward = MethodType(_forward_fused, block)
    def _apply_finetune_policy(self, freeze_all: bool, unfreeze_last: int):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze_all

        if _HAVE_TIMM and self.blocks is not None and (not freeze_all) and unfreeze_last > 0:
            for p in self.backbone.parameters():
                p.requires_grad = False
            n = len(self.blocks)
            for blk in self.blocks[max(0, n - unfreeze_last):]:
                for p in blk.parameters():
                    p.requires_grad = True
            for name in ["norm", "pre_logits"]:
                mod = getattr(self.backbone, name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = True

    def _forward_backbone_impl(self, x: torch.Tensor) -> torch.Tensor:
        if _HAVE_TIMM:
            if hasattr(self.backbone, "forward_features"):
                return self.backbone.forward_features(x)
            return self.backbone(x)
        return self.backbone(x)

    def _maybe_enable_compile(self, x: torch.Tensor):
        if self._compile_attempted:
            return self._backbone_forward

        self._compile_attempted = True
        if (
            self._compile_disabled
            or not self.cfg.enable_torch_compile
            or self.training
            or x.device.type != "cuda"
            or not hasattr(torch, "compile")
            or importlib.util.find_spec("triton") is None
        ):
            return self._backbone_forward

        try:
            # Inductor can fold adjacent pointwise ops around ViT blocks, which
            # helps expose fewer memory round-trips during inference.
            self._backbone_forward = torch.compile(
                self._forward_backbone_impl,
                mode=self.cfg.torch_compile_mode,
                fullgraph=False,
                dynamic=False,
            )
        except Exception:
            self._backbone_forward = self._forward_backbone_impl
        return self._backbone_forward

    def _spatial_pool(self, feat_2d: torch.Tensor) -> torch.Tensor:
        if feat_2d.ndim == 2:
            return feat_2d
        if self.cfg.pool == "cls":
            return feat_2d[:, 0]
        return feat_2d[:, 1:].mean(dim=1)

    def _spatial_temporal_pool(self, feat: torch.Tensor, bsz: int, timesteps: int) -> torch.Tensor:
        if feat.ndim == 2:
            return feat.reshape(bsz, timesteps, -1).mean(dim=1)

        if self.cfg.enable_cupy_fused_pool and feat.device.type == "cuda":
            fused = vit_fused_pool_temporal(
                feat=feat,
                bsz=bsz,
                timesteps=timesteps,
                pool=self.cfg.pool,
                force_fp16=self.cfg.cupy_pool_force_fp16,
            )
            if fused is not None:
                return fused

        # Pure PyTorch fast path to keep pooling on a single graph/runtime path
        # and avoid Python/CuPy bridge overhead for small/medium workloads.
        bt, tokens, channels = feat.shape
        if bt != bsz * timesteps:
            spatial = self._spatial_pool(feat)
            return spatial.reshape(bsz, timesteps, -1).mean(dim=1)

        feat_bt = feat.reshape(bsz, timesteps, tokens, channels)
        if self.cfg.pool == "cls":
            return feat_bt[:, :, 0, :].mean(dim=1)
        # GAP over patch tokens and temporal dimension in one reduction path.
        return feat_bt[:, :, 1:, :].mean(dim=(1, 2))
    def _amp_context(self, x: torch.Tensor):
        if self.training or not self.cfg.enable_amp_fp16 or x.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _sdpa_context(self, x: torch.Tensor):
        if (
            not self.cfg.enable_attention_fastpath
            or x.device.type != "cuda"
            or not hasattr(torch.backends, "cuda")
            or not hasattr(torch.backends.cuda, "sdp_kernel")
        ):
            return nullcontext()
        if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
            return torch.nn.attention.sdpa_kernel([
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
                torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
            ])
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True,
            enable_cudnn=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        assert x.ndim == 5, f"expect [B,T,3,H,W], got {x.shape}"

        bsz, timesteps, channels, height, width = x.shape
        x = x.reshape(bsz * timesteps, channels, height, width)
        if self.cfg.prefer_channels_last and x.device.type == "cuda":
            # Patch embedding starts from conv-like memory access; channels_last often
            # reduces layout traffic on Ampere during fp16 inference.
            x = x.contiguous(memory_format=torch.channels_last)

        backbone_forward = self._maybe_enable_compile(x)
        with self._amp_context(x):
            with self._sdpa_context(x):
                try:
                    feat = backbone_forward(x)
                except Exception as exc:
                    if self.cfg.enable_torch_compile and not self._compile_disabled:
                        log.warning("torch.compile fallback disabled for ViT encoder; reverting to eager path: %s", exc)
                        self._compile_disabled = True
                        self._backbone_forward = self._forward_backbone_impl
                        feat = self._backbone_forward(x)
                    else:
                        raise

            feat = self._spatial_temporal_pool(feat, bsz=bsz, timesteps=timesteps)
            feat = self.dropout(feat)
            feat = self.proj(feat)

        if self.cfg.l2norm:
            feat = torch.nn.functional.normalize(feat, dim=-1)

        # Keep downstream caption generation numerically stable while only
        # optimizing the ViT encoder path in fp16/autocast.
        if feat.dtype != torch.float32:
            feat = feat.float()

        return feat


def build_vit_encoder(
    model_name: str = "vit_base_patch16_224",
    out_dim: int = 256,
    pretrained: bool = True,
    pool: str = "cls",
    l2norm: bool = True,
    freeze: bool = False,
    unfreeze_last: int = 0,
    drop: float = 0.0,
    enable_attention_fastpath: bool = True,
    prefer_channels_last: bool = True,
    enable_torch_compile: bool = True,
    torch_compile_mode: str = "reduce-overhead",
    enable_amp_fp16: bool = False,
    enable_mlp_bias_gelu_fusion: bool = True,
    enable_residual_layernorm_fusion: bool = True,
    enable_inplace_residual_add_fusion: bool = True,
    enable_cupy_fused_pool: bool = False,
    cupy_pool_force_fp16: bool = True,
) -> ViTFrameEncoder:
    cfg = ViTConfig(
        model_name=model_name,
        out_dim=out_dim,
        pretrained=pretrained,
        pool=pool,
        l2norm=l2norm,
        freeze_all=freeze,
        unfreeze_last=unfreeze_last,
        drop=drop,
        enable_attention_fastpath=enable_attention_fastpath,
        prefer_channels_last=prefer_channels_last,
        enable_torch_compile=enable_torch_compile,
        torch_compile_mode=torch_compile_mode,
        enable_amp_fp16=enable_amp_fp16,
        enable_mlp_bias_gelu_fusion=enable_mlp_bias_gelu_fusion,
        enable_residual_layernorm_fusion=enable_residual_layernorm_fusion,
        enable_inplace_residual_add_fusion=enable_inplace_residual_add_fusion,
        enable_cupy_fused_pool=enable_cupy_fused_pool,
        cupy_pool_force_fp16=cupy_pool_force_fp16,
    )
    return ViTFrameEncoder(cfg)










