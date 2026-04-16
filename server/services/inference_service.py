from __future__ import annotations

from pathlib import Path

from core.config import InferenceConfig
from server.schemas import InferRequest
from server.services.model_registry import MODEL_REGISTRY
from server.services.task_manager import GPU_TASK_MANAGER


def request_to_config(req: InferRequest) -> InferenceConfig:
    return InferenceConfig(
        ckpt=req.ckpt,
        stage=req.stage,
        vit_name=req.vit_name,
        gpt2_name=req.gpt2_name,
        prefix_len=req.prefix_len,
        num_frames=req.num_frames,
        image_size=req.image_size,
        ln_scale=req.ln_scale,
        in_weight=req.in_weight,
        preset1=req.preset1,
        preset2=req.preset2,
        preset3=req.preset3,
        prompt1=req.prompt1,
        prompt2=req.prompt2,
        prompt3=req.prompt3,
        device=req.device,
        backend=req.backend,
    )


class InferenceService:
    """Application service: request DTO -> singleton core engine -> API DTO."""

    def infer(self, req: InferRequest):
        frames_dir = Path(req.frames_dir)
        ckpt_path = Path(req.ckpt)
        if not frames_dir.exists() or not frames_dir.is_dir():
            raise FileNotFoundError(f"frames_dir not found: {frames_dir}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

        config = request_to_config(req)
        engine = MODEL_REGISTRY.get_engine(config)
        with GPU_TASK_MANAGER.acquire():
            return engine.infer(req.frames_dir).to_api_dict()


INFERENCE_SERVICE = InferenceService()

