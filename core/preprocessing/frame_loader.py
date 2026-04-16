from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

log = logging.getLogger(__name__)


def list_frames(frames_dir: Path):
    """Return frame files in the canonical preprocessed dataset format."""

    return sorted(frames_dir.glob("frame_*.jpg"))


def load_video_tensor(frames_dir: str | Path, num_frames: int, image_size: int, device: str):
    """frames_dir -> [1,T,3,H,W].

    INFO(tensor-flow): this is the first explicit tensor boundary used by
    the product runtime, benchmark scripts, and patent flow diagrams.
    """

    frames_dir = Path(frames_dir)
    files = list_frames(frames_dir)
    if not files:
        raise FileNotFoundError(f"No frame_*.jpg files found under {frames_dir}")

    step = max(len(files) // num_frames, 1)
    picks = files[::step][:num_frames]

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    imgs = []
    for path in picks:
        with Image.open(path) as image:
            imgs.append(transform(image.convert("RGB")))

    video = torch.stack(imgs, dim=0).unsqueeze(0).to(device)
    log.info("frames_dir=%s total=%s sampled=%s", frames_dir, len(files), len(picks))
    return video

