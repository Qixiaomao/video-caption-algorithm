"""Core model import boundary.

The actual in-house model implementation remains in `src.models` for now.
Product runtime modules should import loaders through explicit submodules, for
example `core.models.model_loader`, so package import stays lightweight and
server startup does not import heavy ML dependencies before model loading.
"""

__all__ = [
    "caption_model",
    "text_decoder",
    "video_encoder",
    "model_loader",
]
