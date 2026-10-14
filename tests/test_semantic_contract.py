from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from core.preprocessing.frame_loader import load_video_tensor
from src.models.caption_model import build_caption_supervision


class CaptionSupervisionTests(unittest.TestCase):
    def test_first_token_is_supervised_and_padding_is_ignored(self):
        caption_ids = torch.tensor([[11, 12, 99, 99], [21, 22, 23, 99]])

        attention_mask, labels = build_caption_supervision(caption_ids, pad_id=99)

        self.assertEqual(labels[0, 0].item(), 11)
        self.assertEqual(labels[1, 0].item(), 21)
        self.assertEqual(labels.tolist(), [[11, 12, 99, -100], [21, 22, 23, 99]])
        self.assertEqual(attention_mask.tolist(), [[1, 1, 1, 0], [1, 1, 1, 1]])


class FrameSamplingTests(unittest.TestCase):
    def test_short_video_repeats_frames_to_exact_temporal_length(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            frames_dir = Path(temp_dir)
            for index in range(3):
                image = Image.new("RGB", (10, 10), color=(index * 40, 0, 0))
                image.save(frames_dir / f"frame_{index:04d}.jpg")

            video = load_video_tensor(frames_dir, num_frames=8, image_size=8, device="cpu")

            self.assertEqual(tuple(video.shape), (1, 8, 3, 8, 8))


if __name__ == "__main__":
    unittest.main()
