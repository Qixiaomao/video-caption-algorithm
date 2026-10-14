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
from typing import Any, Literal

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
    enable_cuda_graph: bool = False


class ViTFrameEncoder(nn.Module):
    """
    ViT video frame encoder:
      forward(x): x -> [B,T,3,H,W] or [B,3,H,W] -> [B, D]
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self._optimization_status: dict[str, dict[str, Any]] = {}

        if _HAVE_TIMM:
            self.backbone = timm.create_model(
                cfg.model_name,
                pretrained=cfg.pretrained,
                num_classes=0,
                global_pool="",
            )
            self.embed_dim = self.backbone.num_features
            self.blocks = getattr(self.backbone, "blocks", None)
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

        self._configure_attention_fastpath()
        self._configure_mlp_gelu_approximation()
        self._configure_residual_layernorm_fusion()
        self._configure_inplace_residual_add()
        self.dropout = nn.Dropout(cfg.drop) if cfg.drop > 1e-6 else nn.Identity()
        self.proj = nn.Linear(self.embed_dim, cfg.out_dim)
        self._backbone_forward = self._forward_backbone_impl
        self._compile_attempted = False
        self._compile_disabled = False
        self._apply_finetune_policy(cfg.freeze_all, cfg.unfreeze_last)
        self._cuda_graph = None
        self._static_input = None
        self._static_output = None
        self._cuda_graph_signature = None
        self._cuda_graph_replays = 0
        self._initialize_runtime_status()

    def _configure_attention_fastpath(self):
        """Set timm fused attention explicitly instead of relying on its default."""

        supported = 0
        active = 0
        for block in self.blocks or []:
            attn = getattr(block, "attn", None)
            if attn is not None and hasattr(attn, "fused_attn"):
                supported += 1
                attn.fused_attn = bool(self.cfg.enable_attention_fastpath)
                active += int(bool(attn.fused_attn))

        requested = bool(self.cfg.enable_attention_fastpath)
        self._optimization_status["attention_fastpath"] = {
            "requested": requested,
            "supported_blocks": supported,
            "active_blocks": active,
            "active": active > 0,
            "request_honored": (not requested) or (supported > 0 and active == supported),
            "validated": True,
            "reason": None if supported > 0 else "backbone_does_not_expose_fused_attn",
        }

    def _configure_mlp_gelu_approximation(self):
        """Configure tanh GELU approximation; this is not claimed as a fused kernel."""

        supported = 0
        active = 0
        for block in self.blocks or []:
            mlp = getattr(block, "mlp", None)
            act = getattr(mlp, "act", None)
            if isinstance(act, nn.GELU):
                supported += 1
                act.approximate = "tanh" if self.cfg.enable_mlp_bias_gelu_fusion else "none"
                active += int(act.approximate == "tanh")

        requested = bool(self.cfg.enable_mlp_bias_gelu_fusion)
        self._optimization_status["mlp_gelu_tanh_approximation"] = {
            "requested": requested,
            "supported_blocks": supported,
            "active_blocks": active,
            "active": active > 0,
            "request_honored": (not requested) or (supported > 0 and active == supported),
            "validated": True,
            "reason": None if supported > 0 else "backbone_does_not_use_torch_gelu",
        }

    def _configure_residual_layernorm_fusion(self):
        """Set add-norm flags only when the installed backbone exposes them."""

        supported = 0
        active = 0
        requested = bool(self.cfg.enable_residual_layernorm_fusion)
        for block in self.blocks or []:
            for attr in ("fused_add_norm", "fused_dropout_add_ln"):
                if hasattr(block, attr):
                    supported += 1
                    setattr(block, attr, requested)
                    active += int(bool(getattr(block, attr)))

        self._optimization_status["residual_layernorm_fusion"] = {
            "requested": requested,
            "supported_flags": supported,
            "active_flags": active,
            "active": active > 0,
            "request_honored": (not requested) or (supported > 0 and active == supported),
            "validated": True,
            "reason": None if supported > 0 else "installed_backbone_has_no_add_norm_fusion_flag",
        }

    def _configure_inplace_residual_add(self):
        """Configure inference-only in-place residual adds independently of add-norm flags."""

        requested = bool(self.cfg.enable_inplace_residual_add_fusion)
        supported = 0
        patched = 0
        for block in self.blocks or []:
            if hasattr(block, "norm1") and hasattr(block, "norm2"):
                supported += 1
                if requested:
                    self._patch_block_inplace_residual_forward(block)
                    patched += int(hasattr(block, "_orig_forward_inference_fusion"))

        self._optimization_status["inplace_residual_add"] = {
            "requested": requested,
            "supported_blocks": supported,
            "patched_blocks": patched,
            "active": patched > 0,
            "request_honored": (not requested) or (supported > 0 and patched == supported),
            "validated": True,
            "reason": None if supported > 0 else "backbone_blocks_do_not_expose_norm1_norm2",
        }

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

    def _initialize_runtime_status(self):
        triton_available = importlib.util.find_spec("triton") is not None
        self._optimization_status.update(
            {
                "fp16_autocast": {
                    "requested": bool(self.cfg.enable_amp_fp16),
                    "active": False,
                    "request_honored": None,
                    "validated": False,
                    "reason": "requires_cuda_forward" if self.cfg.enable_amp_fp16 else "disabled_by_config",
                },
                "channels_last": {
                    "requested": bool(self.cfg.prefer_channels_last),
                    "active": False,
                    "request_honored": None,
                    "validated": False,
                    "reason": "requires_cuda_forward" if self.cfg.prefer_channels_last else "disabled_by_config",
                },
                "torch_compile": {
                    "requested": bool(self.cfg.enable_torch_compile),
                    "triton_available": triton_available,
                    "attempted": False,
                    "active": False,
                    "request_honored": False if self.cfg.enable_torch_compile and not triton_available else None,
                    "validated": bool(self.cfg.enable_torch_compile and not triton_available),
                    "reason": "triton_not_installed" if self.cfg.enable_torch_compile and not triton_available else (
                        "disabled_by_config" if not self.cfg.enable_torch_compile else "requires_cuda_forward"
                    ),
                },
                "cupy_fused_pool": {
                    "requested": bool(self.cfg.enable_cupy_fused_pool),
                    "active": False,
                    "request_honored": None,
                    "validated": False,
                    "reason": "requires_cuda_forward" if self.cfg.enable_cupy_fused_pool else "disabled_by_config",
                },
                "cuda_graph": {
                    "requested": bool(self.cfg.enable_cuda_graph),
                    "captured": False,
                    "replays": 0,
                    "active": False,
                    "request_honored": None,
                    "validated": False,
                    "reason": "requires_cuda_forward" if self.cfg.enable_cuda_graph else "disabled_by_config",
                },
            }
        )

    def get_optimization_status(self) -> dict[str, dict[str, Any]]:
        """Return JSON-serializable requested/effective optimization state."""

        return {name: dict(values) for name, values in self._optimization_status.items()}

    def get_optimization_issues(self) -> list[str]:
        """Return requested optimizations that were validated but not honored."""

        issues = []
        for name, status in self._optimization_status.items():
            if status.get("requested") and status.get("validated") and status.get("request_honored") is False:
                reason = status.get("reason") or "requested_state_not_active"
                issues.append(f"{name}: {reason}")
        return issues

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
        status = self._optimization_status["torch_compile"]
        status["attempted"] = True
        if (
            self._compile_disabled
            or not self.cfg.enable_torch_compile
            or self.training
            or x.device.type != "cuda"
            or not hasattr(torch, "compile")
            or importlib.util.find_spec("triton") is None
        ):
            status["validated"] = True
            status["active"] = False
            status["request_honored"] = not self.cfg.enable_torch_compile
            if not self.cfg.enable_torch_compile:
                status["reason"] = "disabled_by_config"
            elif self.training:
                status["reason"] = "training_mode"
            elif x.device.type != "cuda":
                status["reason"] = "requires_cuda_input"
            elif importlib.util.find_spec("triton") is None:
                status["reason"] = "triton_not_installed"
            else:
                status["reason"] = "compile_unavailable"
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
            status.update({"active": True, "request_honored": True, "validated": True, "reason": None})
        except Exception as exc:
            self._backbone_forward = self._forward_backbone_impl
            status.update(
                {
                    "active": False,
                    "request_honored": False,
                    "validated": True,
                    "reason": f"compile_failed:{type(exc).__name__}",
                }
            )
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
                self._optimization_status["cupy_fused_pool"].update(
                    {"active": True, "request_honored": True, "validated": True, "reason": None}
                )
                return fused
            self._optimization_status["cupy_fused_pool"].update(
                {
                    "active": False,
                    "request_honored": False,
                    "validated": True,
                    "reason": "cupy_kernel_returned_fallback",
                }
            )

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

    def _record_cuda_graph(self, x: torch.Tensor):
        """Record a CUDA graph for the backbone forward pass."""
        self._static_input = torch.empty_like(x)
        self._static_input.copy_(x)
        current_stream = torch.cuda.current_stream(x.device)
        warmup_stream = torch.cuda.Stream(device=x.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                self._backbone_forward(self._static_input)
        current_stream.wait_stream(warmup_stream)
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            self._static_output = self._backbone_forward(self._static_input)
        self._cuda_graph_signature = (tuple(x.shape), str(x.dtype), str(x.device))
        self._optimization_status["cuda_graph"].update(
            {"captured": True, "active": True, "request_honored": True, "validated": True, "reason": None}
        )

    def _cuda_graph_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Replay the recorded CUDA graph."""
        self._static_input.copy_(x)
        self._cuda_graph.replay()
        self._cuda_graph_replays += 1
        self._optimization_status["cuda_graph"]["replays"] = self._cuda_graph_replays
        return self._static_output.clone()

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
            x = x.contiguous(memory_format=torch.channels_last)
        channels_last_active = x.device.type == "cuda" and x.is_contiguous(memory_format=torch.channels_last)
        self._optimization_status["channels_last"].update(
            {
                "active": channels_last_active,
                "request_honored": channels_last_active if self.cfg.prefer_channels_last else True,
                "validated": True,
                "reason": None if channels_last_active else (
                    "disabled_by_config" if not self.cfg.prefer_channels_last else "channels_last_not_active"
                ),
            }
        )
        amp_active = bool(self.cfg.enable_amp_fp16 and x.device.type == "cuda" and not self.training)
        self._optimization_status["fp16_autocast"].update(
            {
                "active": amp_active,
                "request_honored": amp_active if self.cfg.enable_amp_fp16 else True,
                "validated": True,
                "reason": None if amp_active else (
                    "disabled_by_config" if not self.cfg.enable_amp_fp16 else "fp16_requires_cuda_eval"
                ),
            }
        )

        # CUDA Graph path: record once, replay on subsequent calls
        if (
            self.cfg.enable_cuda_graph
            and not self.training
            and x.device.type == "cuda"
        ):
            signature = (tuple(x.shape), str(x.dtype), str(x.device))
            try:
                if self._cuda_graph is None or self._cuda_graph_signature != signature:
                    self._maybe_enable_compile(x)
                    with self._amp_context(x):
                        with self._sdpa_context(x):
                            self._record_cuda_graph(x)
                with self._amp_context(x):
                    feat = self._cuda_graph_forward(x)
            except Exception as exc:
                self._optimization_status["cuda_graph"].update(
                    {
                        "captured": False,
                        "active": False,
                        "request_honored": False,
                        "validated": True,
                        "reason": f"capture_failed:{type(exc).__name__}",
                    }
                )
                raise RuntimeError("CUDA Graph was requested but capture/replay failed") from exc
        else:
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
                            self._optimization_status["torch_compile"].update(
                                {
                                    "active": False,
                                    "request_honored": False,
                                    "validated": True,
                                    "reason": f"compiled_forward_failed:{type(exc).__name__}",
                                }
                            )
                            feat = self._backbone_forward(x)
                        else:
                            raise

        with self._amp_context(x):
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
    enable_cuda_graph: bool = False,
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
        enable_cuda_graph=enable_cuda_graph,
    )
    return ViTFrameEncoder(cfg)










