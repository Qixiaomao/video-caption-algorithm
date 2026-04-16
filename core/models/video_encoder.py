"""Compatibility wrapper for the in-house ViT video encoder."""

from src.models.video_encoder import ViTFrameEncoder, build_vit_encoder

__all__ = ["ViTFrameEncoder", "build_vit_encoder"]

