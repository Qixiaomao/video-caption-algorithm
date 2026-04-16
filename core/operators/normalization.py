from __future__ import annotations

import torch


def apply_prefix_norm(prefix: torch.Tensor, ln_scale: float, in_weight: float) -> torch.Tensor:
    """Current prefix normalization/scaling logic as an isolated operator."""

    if ln_scale is not None and ln_scale > 0:
        prefix = torch.nn.functional.layer_norm(prefix, prefix.shape[-1:]) * ln_scale
    if in_weight is not None and in_weight > 0:
        prefix = prefix * in_weight
    return prefix

