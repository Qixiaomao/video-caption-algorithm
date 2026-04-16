from __future__ import annotations

import torch
import torch.nn as nn


class TemporalMeanPool(nn.Module):
    """Explicit temporal pooling operator.

    This mirrors the current mean pooling behavior and gives us a stable
    replacement point for CUDA/TensorRT plugin experiments.
    """

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        return frame_features.mean(dim=1)

