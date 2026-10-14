# src/models/caption_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.text_decoder import GPT2TextDecoder
from src.models.video_encoder import build_vit_encoder


def build_caption_supervision(caption_ids: torch.Tensor, pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build causal-LM supervision without dropping the first caption token.

    GPT-2 shifts labels internally. Passing the full caption sequence means the
    final visual-prefix position is trained to predict the first caption token.
    MSVD uses GPT-2 EOS as padding. The first EOS after caption tokens is kept as
    the sequence terminator; repeated EOS padding after it is ignored.
    """

    is_eos_padding = caption_ids.eq(pad_id)
    first_eos = is_eos_padding & is_eos_padding.long().cumsum(dim=1).eq(1)
    attention_mask = ((~is_eos_padding) | first_eos).long()
    labels = caption_ids.clone()
    labels.masked_fill_(attention_mask == 0, -100)
    return attention_mask, labels


class VideoCaptionModel(nn.Module):
    """ViT video encoder conditioned GPT-2 caption model."""

    def __init__(
        self,
        vit_name: str = "vit_base_patch16_224",
        video_dim: int = 256,
        gpt2_name: str = "gpt2",
        cond_mode: str = "prefix",
        prefix_len: int = 4,
        proj_hidden: int = 0,
        freeze_vit: bool = True,
        unfreeze_last: int = 0,
        vit_enable_fp16: bool = False,
        vit_enable_attention_fastpath: bool = True,
        vit_prefer_channels_last: bool = True,
        vit_enable_torch_compile: bool = True,
        vit_torch_compile_mode: str = "reduce-overhead",
        vit_enable_mlp_bias_gelu_fusion: bool = True,
        vit_enable_residual_layernorm_fusion: bool = True,
        vit_enable_inplace_residual_add_fusion: bool = True,
        vit_enable_cupy_fused_pool: bool = False,
        vit_cupy_pool_force_fp16: bool = True,
        vit_enable_cuda_graph: bool = False,
        use_cupy_prefix_projector: bool = False,
        cupy_prefix_force_fp16: bool = True,
    ):
        super().__init__()
        self.encoder = build_vit_encoder(
            model_name=vit_name,
            out_dim=video_dim,
            pretrained=True,
            pool="cls",
            l2norm=False,
            freeze=freeze_vit,
            unfreeze_last=unfreeze_last,
            enable_attention_fastpath=vit_enable_attention_fastpath,
            prefer_channels_last=vit_prefer_channels_last,
            enable_torch_compile=vit_enable_torch_compile and freeze_vit,
            torch_compile_mode=vit_torch_compile_mode,
            enable_amp_fp16=vit_enable_fp16,
            enable_mlp_bias_gelu_fusion=vit_enable_mlp_bias_gelu_fusion,
            enable_residual_layernorm_fusion=vit_enable_residual_layernorm_fusion,
            enable_inplace_residual_add_fusion=vit_enable_inplace_residual_add_fusion,
            enable_cupy_fused_pool=vit_enable_cupy_fused_pool,
            cupy_pool_force_fp16=vit_cupy_pool_force_fp16,
            enable_cuda_graph=vit_enable_cuda_graph,
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
            use_cupy_prefix_projector=use_cupy_prefix_projector,
            cupy_prefix_force_fp16=cupy_prefix_force_fp16,
        )

    def forward(
        self,
        video: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_visual_prefix: bool = False,
    ) -> Dict[str, Any]:
        video_emb = self.proj(self.encoder(video))
        if not return_visual_prefix:
            return self.decoder(video_emb, input_ids, attention_mask, labels)

        visual_prefix = self.decoder.build_visual_prefix(video_emb)
        result = self.decoder(
            video_emb,
            input_ids,
            attention_mask,
            labels,
            visual_prefix=visual_prefix,
        )
        result["visual_prefix"] = visual_prefix
        result["video_embedding"] = video_emb
        return result

    @torch.no_grad()
    def generate(self, video: torch.Tensor, prompt: str = "", **gen_kwargs):
        video_emb = self.proj(self.encoder(video))
        return self.decoder.generate(video_emb, prompt=prompt, **gen_kwargs)

    def compute_loss(self, video: torch.Tensor, caption_ids: torch.Tensor, pad_id: int):
        attention_mask, labels = build_caption_supervision(caption_ids, pad_id)
        result = self.forward(
            video=video,
            input_ids=caption_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return result["loss"]
