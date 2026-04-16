from __future__ import annotations

from dataclasses import dataclass

import backend_config
from core.config import InferenceConfig, MemoryConfig, TensorRTConfig


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
    )


SETTINGS = ServerSettings()

