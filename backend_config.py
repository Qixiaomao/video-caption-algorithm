# backend_config.py
from pathlib import Path
import torch
import os

# —— 环境变量：强制禁用 TF 分支（避免 TensorFlow 报错）——
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# —— 模型路径 & 默认配置（按需修改）——
CKPT_PATH = Path("./checkpoints/msvd_mapper_finetune_v2.pt")  # 改成你实际的 .pt 权重路径
VIT_NAME  = "vit_base_patch16_224"
GPT2_NAME = "gpt2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# —— 默认推理参数 —— 
PREFIX_LEN  = 4
NUM_FRAMES  = 8
IMAGE_SIZE  = 224
LN_SCALE    = 0.6
IN_WEIGHT   = 0.4

# —— 三路解码预设 —— 
PRESET1 = "precise"
PRESET2 = "detailed"
PRESET3 = "natural"

PROMPT1 = ""
PROMPT2 = "State the main action in one short sentence:"
PROMPT3 = "Write a short, natural caption:"