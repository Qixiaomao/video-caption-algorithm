#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from pathlib import Path
from src.data.data_loader import build_dataloader  # 按你的项目路径
from transformers import BertTokenizerFast  # 你之前使用的tokenizer，按需替换

def main():
    ann = "data/processed/msvd/train/annotations.json"
    assert Path(ann).exists(), f"{ann} not found"

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    loader = build_dataloader(
        ann_path=ann,
        tokenizer=tokenizer,
        batch_size=2,
        max_len=32,
        num_frame=8,
        image_size=224,
        shuffle=False,
        num_wokers=0
    )
    for i, batch in enumerate(loader):
        print(f"---- Batch {i} ----")
        video = batch["video"]           # [B, T, C, H, W]
        caption_ids = batch["caption_ids"]
        print("video:", type(video), video.shape)
        print("caption_ids:", type(caption_ids), caption_ids.shape)
        print("video_id:", batch.get("video_id"))
        if i == 2:
            break

if __name__ == "__main__":
    main()
