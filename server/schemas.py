from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field

from server.settings import default_inference_config

_DEFAULT = default_inference_config()


class InferRequest(BaseModel):
    frames_dir: str = Field(..., description="Directory containing frame_*.jpg")
    ckpt: str = Field(default=_DEFAULT.ckpt, description="Path to PyTorch checkpoint")
    stage: str = Field(default=_DEFAULT.stage, description="Reserved for compatibility")
    vit_name: str = Field(default=_DEFAULT.vit_name)
    gpt2_name: str = Field(default=_DEFAULT.gpt2_name)
    prefix_len: int = Field(default=_DEFAULT.prefix_len)
    num_frames: int = Field(default=_DEFAULT.num_frames)
    image_size: int = Field(default=_DEFAULT.image_size)
    ln_scale: float = Field(default=_DEFAULT.ln_scale)
    in_weight: float = Field(default=_DEFAULT.in_weight)
    preset1: str = Field(default=_DEFAULT.preset1)
    preset2: str = Field(default=_DEFAULT.preset2)
    preset3: str = Field(default=_DEFAULT.preset3)
    prompt1: str = Field(default=_DEFAULT.prompt1)
    prompt2: str = Field(default=_DEFAULT.prompt2)
    prompt3: str = Field(default=_DEFAULT.prompt3)
    device: str = Field(default=_DEFAULT.device)
    backend: str = Field(default=_DEFAULT.backend)
    vit_enable_fp16: bool = Field(default=_DEFAULT.vit_opt.enable_fp16)
    vit_enable_attention_fastpath: bool = Field(default=_DEFAULT.vit_opt.enable_attention_fastpath)
    vit_prefer_channels_last: bool = Field(default=_DEFAULT.vit_opt.prefer_channels_last)
    vit_enable_torch_compile: bool = Field(default=_DEFAULT.vit_opt.enable_torch_compile)
    vit_torch_compile_mode: str = Field(default=_DEFAULT.vit_opt.torch_compile_mode)
    vit_enable_mlp_bias_gelu_fusion: bool = Field(default=_DEFAULT.vit_opt.enable_mlp_bias_gelu_fusion)
    vit_enable_residual_layernorm_fusion: bool = Field(default=_DEFAULT.vit_opt.enable_residual_layernorm_fusion)
    vit_enable_cupy_fused_pool: bool = Field(default=_DEFAULT.vit_opt.enable_cupy_fused_pool)
    vit_cupy_pool_force_fp16: bool = Field(default=_DEFAULT.vit_opt.cupy_pool_force_fp16)
    use_cupy_prefix_projector: bool = Field(default=_DEFAULT.use_cupy_prefix_projector)
    cupy_prefix_force_fp16: bool = Field(default=_DEFAULT.cupy_prefix_force_fp16)


class InferResponse(BaseModel):
    S1: str
    S2: str
    S3: str
    BEST: Dict[str, str]


class HealthResponse(BaseModel):
    status: str





