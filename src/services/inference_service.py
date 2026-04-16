"""Compatibility wrapper for the new core/server split.

New code should import:
- core.engine.InferenceEngine for standalone model execution.
- server.services.inference_service.INFERENCE_SERVICE for FastAPI orchestration.
"""

from core.config import InferenceConfig
from core.engine import InferenceEngine
from server.services.inference_service import INFERENCE_SERVICE, InferenceService, request_to_config
from server.services.model_registry import MODEL_REGISTRY


def run_one_video_with_config(frames_dir: str, config: InferenceConfig):
    return MODEL_REGISTRY.get_engine(config).infer(frames_dir).to_api_dict()


__all__ = [
    "InferenceConfig",
    "InferenceEngine",
    "InferenceService",
    "INFERENCE_SERVICE",
    "MODEL_REGISTRY",
    "request_to_config",
    "run_one_video_with_config",
]
