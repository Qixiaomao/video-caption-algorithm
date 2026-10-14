# src/data/data_loader.py
import os
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random


def _resolve_frames_dir(rec, split_root=None):
    # 优先用 annotations.json 的 frames_dir
    if "frames_dir" in rec and rec["frames_dir"]:
        return Path(rec["frames_dir"])
    # 兼容旧格式（无 frames_dir 字段就用 <split_root>/<video_id>）
    if split_root:
        return Path(split_root) / rec["video_id"]
    return Path(rec["video_id"])


def _has_frames(frames_dir: Path) -> bool:
    return frames_dir.exists() and any(frames_dir.glob("frame_*.jpg"))


class VideoCaptionDataset(Dataset):
    """
    从预处理后的 MSVD 数据集加载帧和字幕
    annotations.json 每条记录包含:
      { video_id, video, frames_dir, captions }
    """

    def __init__(self,
                 ann_path: str,
                 tokenizer,
                 max_len: int = 32,
                 transform=None,
                 num_frames: int = 12,
                 require_frames: bool = True,
                 random_caption: bool = True):
        super().__init__()

        with open(ann_path, "r", encoding="utf-8") as f:
            all_anns: List[Dict[str, Any]] = json.load(f)

        # 初始化时过滤无帧样本
        filtered = []
        dropped = 0
        for r in all_anns:
            fdir = _resolve_frames_dir(r, split_root=str(Path(ann_path).parent / "frames"))
            if _has_frames(fdir):
                r["_frames_dir"] = str(fdir)
                filtered.append(r)
            else:
                dropped += 1
        if dropped > 0:
            print(f"[WARN] Dropped {dropped} samples without frames. kept={len(filtered)}")

        self.anns = filtered if require_frames else all_anns
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        self.num_frames = num_frames
        self.random_caption = random_caption

    def __len__(self):
        return len(self.anns)

    def __sample_indices(self, total: int) -> List[int]:
        """
        始终返回 self.num_frames 个索引：
        - 如果 total >= num_frames：均匀采样
        - 如果 total <  num_frames：循环重复补齐
        """
        nf = int(self.num_frames)
        if total <= 0:
            return [0] * nf
        if total >= nf:
            import torch
            pos = torch.linspace(0, total - 1, steps=nf)
            return [int(round(p.item())) for p in pos]
        reps = (nf + total - 1) // total
        idxs = (list(range(total)) * reps)[:nf]
        return idxs



    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.anns[idx]
        frames_dir = Path(rec.get("_frames_dir") or rec.get("frames_dir", ""))
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        if not frame_files:
            # 如果极少数漏掉帧 → 让 DataLoader 跳过
            raise IndexError(f"No frames in: {frames_dir}")

        # 均匀采样帧
        idxs = self.__sample_indices(len(frame_files))
        imgs = []
        for i in idxs:
            with Image.open(frame_files[i]) as im:
                imgs.append(im.convert("RGB"))

        if self.transform:
            imgs = [self.transform(img) for img in imgs]

        video = torch.stack(imgs, dim=0)

        T = video.shape[0]
        nf = self.num_frames
        if T < nf:
            # 用最后一帧补齐
            pad = video[-1:].repeat(nf - T, 1, 1, 1)
            video = torch.cat([video, pad], dim=0)
        elif T > nf:
            # 再保险：截断
            video = video[:nf]

        # 随机取一条 caption
        caps = rec.get("captions")
        if isinstance(caps, list) and caps:
            cap = random.choice(caps) if self.random_caption else caps[0]
        elif isinstance(caps, str):
            cap = caps
        else:
            cap = ""

        enc = self.tokenizer(
            cap,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        caption_ids = enc.input_ids.squeeze(0)

        return {
            "video": video,                     # [T, C, H, W]
            "caption_ids": caption_ids,         # [L]
            "video_id": rec.get("video_id", "")
        }


def collate_fn(batch):
    """组 batch"""
    videos = torch.stack([b["video"] for b in batch], dim=0)
    captions = torch.stack([b["caption_ids"] for b in batch], dim=0)
    video_ids = [b["video_id"] for b in batch]
    return {
        "video": videos,
        "caption_ids": captions,
        "video_id": video_ids
    }


def build_dataloader(ann_path: str,
                     tokenizer,
                     batch_size: int = 2,
                     max_len: int = 32,
                     num_frame: int = 8,
                     image_size: int = 224,
                     shuffle: bool = False,
                     num_wokers: int = 0,
                     random_caption: bool = True,
                     drop_last: bool = False,
                     generator: torch.Generator | None = None):
    """构建 DataLoader"""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = VideoCaptionDataset(
        ann_path,
        tokenizer,
        max_len=max_len,
        transform=transform,
        num_frames=num_frame,
        random_caption=random_caption,
    )

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_wokers,
                        pin_memory=True,
                        collate_fn=collate_fn,
                        drop_last=drop_last,
                        generator=generator)
    return loader
