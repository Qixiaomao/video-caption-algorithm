from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryConfig:
    """GPU memory policy for small cards such as RTX 3050 4GB."""

    max_gpu_mem_mb: int = 3800
    allow_cuda_empty_cache: bool = True
    allow_cpu_fallback: bool = False
    max_concurrent_gpu_tasks: int = 1


@dataclass(frozen=True)
class TensorRTConfig:
    """Reserved TensorRT integration settings.

    The current implementation runs the PyTorch backend. These fields define
    the future handoff point for ONNX/TensorRT engines and custom plugins.
    """

    enabled: bool = False
    engine_path: str = ""
    precision: str = "fp16"
    workspace_mb: int = 512
    plugin_namespace: str = "video_caption_plugins"


@dataclass(frozen=True)
class InferenceConfig:
    """Stateless core inference configuration."""

    ckpt: str
    stage: str = "all"
    vit_name: str = "vit_base_patch16_224"
    gpt2_name: str = "gpt2"
    prefix_len: int = 4
    num_frames: int = 8
    image_size: int = 224
    ln_scale: float = 0.6
    in_weight: float = 0.4
    preset1: str = "precise"
    preset2: str = "precise"
    preset3: str = "natural"
    prompt1: str = ""
    prompt2: str = "State the main action in one short sentence:"
    prompt3: str = "Write a short, natural caption:"
    device: str = "cpu"
    backend: str = "torch"
    memory: MemoryConfig = MemoryConfig()
    tensorrt: TensorRTConfig = TensorRTConfig()

