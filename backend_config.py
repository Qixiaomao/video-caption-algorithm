# backend_config.py
from pathlib import Path
import importlib.util
import os

# Environment variables: avoid TensorFlow branch and HF telemetry noise.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Model path and default runtime config.
CKPT_PATH = Path("./checkpoints/msvd_mapper_finetune_v2.pt")
VIT_NAME = "vit_base_patch16_224"
GPT2_NAME = "gpt2"


def _default_device() -> str:
    # Do not hard-import torch at module import time. The formal dependency
    # check runs before model loading in server/services/model_registry.py.
    if importlib.util.find_spec("torch") is None:
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


DEVICE = os.getenv("VIDEO_CAPTION_DEVICE", _default_device())

# Default inference parameters.
PREFIX_LEN = 4
NUM_FRAMES = 8
IMAGE_SIZE = 224
LN_SCALE = 0.6
IN_WEIGHT = 0.4

# ViT optimization switches.
VIT_ENABLE_FP16 = _env_bool("VIDEO_CAPTION_VIT_FP16", True)
VIT_ENABLE_ATTENTION_FASTPATH = _env_bool("VIDEO_CAPTION_VIT_ATTN_FASTPATH", True)
VIT_PREFER_CHANNELS_LAST = _env_bool("VIDEO_CAPTION_VIT_CHANNELS_LAST", True)
VIT_ENABLE_TORCH_COMPILE = _env_bool("VIDEO_CAPTION_VIT_COMPILE", True)
VIT_TORCH_COMPILE_MODE = os.getenv("VIDEO_CAPTION_VIT_COMPILE_MODE", "reduce-overhead")
VIT_ENABLE_MLP_BIAS_GELU_FUSION = _env_bool("VIDEO_CAPTION_VIT_MLP_GELU_FUSION", True)
VIT_ENABLE_RESIDUAL_LAYERNORM_FUSION = _env_bool("VIDEO_CAPTION_VIT_RESIDUAL_LN_FUSION", True)
VIT_ENABLE_INPLACE_RESIDUAL_ADD_FUSION = _env_bool("VIDEO_CAPTION_VIT_INPLACE_RESIDUAL_ADD_FUSION", True)
VIT_ENABLE_CUPY_FUSED_POOL = _env_bool("VIDEO_CAPTION_VIT_CUPY_FUSED_POOL", False)
VIT_CUPY_POOL_FORCE_FP16 = _env_bool("VIDEO_CAPTION_VIT_CUPY_POOL_FORCE_FP16", True)

# Cross-modal projection operator switches.
USE_CUPY_PREFIX_PROJECTOR = _env_bool("VIDEO_CAPTION_USE_CUPY_PREFIX_PROJECTOR", False)
CUPY_PREFIX_FORCE_FP16 = _env_bool("VIDEO_CAPTION_CUPY_PREFIX_FORCE_FP16", True)

# Three decode presets.
PRESET1 = "precise"
PRESET2 = "detailed"
PRESET3 = "natural"

PROMPT1 = ""
PROMPT2 = "State the main action in one short sentence:"
PROMPT3 = "Write a short, natural caption:"

