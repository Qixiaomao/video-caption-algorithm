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


DEVICE = os.getenv("VIDEO_CAPTION_DEVICE", _default_device())

# Default inference parameters.
PREFIX_LEN = 4
NUM_FRAMES = 8
IMAGE_SIZE = 224
LN_SCALE = 0.6
IN_WEIGHT = 0.4

# Three decode presets.
PRESET1 = "precise"
PRESET2 = "detailed"
PRESET3 = "natural"

PROMPT1 = ""
PROMPT2 = "State the main action in one short sentence:"
PROMPT3 = "Write a short, natural caption:"
