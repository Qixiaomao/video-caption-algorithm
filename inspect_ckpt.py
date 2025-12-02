import torch
from pathlib import Path
from src.models.caption_model import VideoCaptionModel

ckpt_path = Path("./checkpoints/msvd_mapper_finetune.pt")  # 换成你想看的
state = torch.load(ckpt_path, map_location="cpu")

def pick_state_dict(obj):
    for k in ("model_state","model","state_dict"):
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            return obj[k]
    return obj if isinstance(obj, dict) else {}

sd = pick_state_dict(state)
print(f"ckpt keys sample: {list(sd.keys())[:20]}")

m = VideoCaptionModel(
    vit_name="vit_base_patch16_224", gpt2_name="gpt2",
    cond_mode="prefix", prefix_len=4, freeze_vit=True, unfreeze_last=0
)
print(f"model keys sample: {list(m.state_dict().keys())[:20]}")