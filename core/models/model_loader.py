from __future__ import annotations

import logging
from pathlib import Path

import torch

from core.config import InferenceConfig

log = logging.getLogger(__name__)


def load_caption_model(config: InferenceConfig):
    """Load the configured caption model runtime.

    Product inference code should call this function instead of importing a
    concrete model implementation directly. The current runtime is the
    PyTorch baseline; TensorRT is kept as an explicit future backend boundary.
    """

    backend = config.backend.lower()
    if config.tensorrt.enabled or backend == "tensorrt":
        # TODO(tensorrt): route to core.trt.runtime once encoder/full graph
        # engines are exported and accuracy-aligned with this PyTorch loader.
        raise NotImplementedError("TensorRT backend hook is reserved but not implemented yet.")
    if backend != "torch":
        raise ValueError(f"Unsupported inference backend: {config.backend}")
    return load_torch_caption_model(config)


def _load_checkpoint(path: Path, device: str):
    """Load checkpoints safely when the installed torch version supports it."""

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception as exc:
        log.warning("weights_only checkpoint load failed; falling back to compatibility mode: %s", exc)
        return torch.load(path, map_location=device, weights_only=False)


def load_torch_caption_model(config: InferenceConfig):
    """Load the PyTorch baseline model.

    TODO(tensorrt): keep this as the reference loader when TensorRT engines
    are introduced for accuracy alignment.
    """

    from core.models.caption_model import VideoCaptionModel

    model = VideoCaptionModel(
        vit_name=config.vit_name,
        gpt2_name=config.gpt2_name,
        cond_mode="prefix",
        prefix_len=config.prefix_len,
        freeze_vit=True,
        unfreeze_last=0,
        vit_enable_fp16=config.vit_opt.enable_fp16,
        vit_enable_attention_fastpath=config.vit_opt.enable_attention_fastpath,
        vit_prefer_channels_last=config.vit_opt.prefer_channels_last,
        vit_enable_torch_compile=config.vit_opt.enable_torch_compile,
        vit_torch_compile_mode=config.vit_opt.torch_compile_mode,
        vit_enable_mlp_bias_gelu_fusion=config.vit_opt.enable_mlp_bias_gelu_fusion,
        vit_enable_residual_layernorm_fusion=config.vit_opt.enable_residual_layernorm_fusion,
        vit_enable_inplace_residual_add_fusion=config.vit_opt.enable_inplace_residual_add_fusion,
        vit_enable_cupy_fused_pool=config.vit_opt.enable_cupy_fused_pool,
        vit_cupy_pool_force_fp16=config.vit_opt.cupy_pool_force_fp16,
        use_cupy_prefix_projector=config.use_cupy_prefix_projector,
        cupy_prefix_force_fp16=config.cupy_prefix_force_fp16,
    ).to(config.device).eval()

    state = _load_checkpoint(Path(config.ckpt), config.device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning("missing keys (<=6): %s", missing[:6])
    if unexpected:
        log.warning("unexpected keys (<=6): %s", unexpected[:6])
    return model











