from __future__ import annotations

from pathlib import Path

from core.config import InferenceConfig, ViTOptimizeConfig
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
        vit_opt=ViTOptimizeConfig(
            enable_fp16=req.vit_enable_fp16,
            enable_attention_fastpath=req.vit_enable_attention_fastpath,
            prefer_channels_last=req.vit_prefer_channels_last,
            enable_torch_compile=req.vit_enable_torch_compile,
            torch_compile_mode=req.vit_torch_compile_mode,
            enable_mlp_bias_gelu_fusion=req.vit_enable_mlp_bias_gelu_fusion,
            enable_residual_layernorm_fusion=req.vit_enable_residual_layernorm_fusion,
            enable_cupy_fused_pool=req.vit_enable_cupy_fused_pool,
            cupy_pool_force_fp16=req.vit_cupy_pool_force_fp16,
        ),
        use_cupy_prefix_projector=req.use_cupy_prefix_projector,
        cupy_prefix_force_fp16=req.cupy_prefix_force_fp16,
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










