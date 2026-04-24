from __future__ import annotations

from dataclasses import dataclass

import backend_config
from core.config import InferenceConfig, MemoryConfig, TensorRTConfig, ViTOptimizeConfig


@dataclass(frozen=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8001
    api_prefix: str = "/api/v1"
    allow_origins: tuple[str, ...] = ("*",)


def default_inference_config() -> InferenceConfig:
    return InferenceConfig(
        ckpt=str(backend_config.CKPT_PATH),
        vit_name=backend_config.VIT_NAME,
        gpt2_name=backend_config.GPT2_NAME,
        prefix_len=backend_config.PREFIX_LEN,
        num_frames=backend_config.NUM_FRAMES,
        image_size=backend_config.IMAGE_SIZE,
        ln_scale=backend_config.LN_SCALE,
        in_weight=backend_config.IN_WEIGHT,
        preset1=backend_config.PRESET1,
        preset2=backend_config.PRESET2,
        preset3=backend_config.PRESET3,
        prompt1=backend_config.PROMPT1,
        prompt2=backend_config.PROMPT2,
        prompt3=backend_config.PROMPT3,
        device=backend_config.DEVICE,
        memory=MemoryConfig(),
        tensorrt=TensorRTConfig(),
        vit_opt=ViTOptimizeConfig(
            enable_fp16=backend_config.VIT_ENABLE_FP16,
            enable_attention_fastpath=backend_config.VIT_ENABLE_ATTENTION_FASTPATH,
            prefer_channels_last=backend_config.VIT_PREFER_CHANNELS_LAST,
            enable_torch_compile=backend_config.VIT_ENABLE_TORCH_COMPILE,
            torch_compile_mode=backend_config.VIT_TORCH_COMPILE_MODE,
            enable_mlp_bias_gelu_fusion=backend_config.VIT_ENABLE_MLP_BIAS_GELU_FUSION,
            enable_residual_layernorm_fusion=backend_config.VIT_ENABLE_RESIDUAL_LAYERNORM_FUSION,
            enable_cupy_fused_pool=backend_config.VIT_ENABLE_CUPY_FUSED_POOL,
            cupy_pool_force_fp16=backend_config.VIT_CUPY_POOL_FORCE_FP16,
        ),
        use_cupy_prefix_projector=backend_config.USE_CUPY_PREFIX_PROJECTOR,
        cupy_prefix_force_fp16=backend_config.CUPY_PREFIX_FORCE_FP16,
    )


SETTINGS = ServerSettings()










