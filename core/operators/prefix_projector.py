from __future__ import annotations

import torch
import torch.nn as nn


class PrefixProjector(nn.Module):
    """Visual embedding -> language prefix projection hook."""

    def __init__(self, video_dim: int, hidden_dim: int, prefix_len: int):
        super().__init__()
        self.prefix_len = prefix_len
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(video_dim, hidden_dim * prefix_len)

    def forward(self, video_emb: torch.Tensor) -> torch.Tensor:
        batch = video_emb.shape[0]
        return self.proj(video_emb).view(batch, self.prefix_len, self.hidden_dim)

