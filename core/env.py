from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Iterable


class RuntimeDependencyError(RuntimeError):
    """Raised before model loading when required runtime packages are missing."""


@dataclass(frozen=True)
class DependencySpec:
    module: str
    package: str
    purpose: str
    required: bool = True


CORE_RUNTIME_DEPENDENCIES = (
    DependencySpec("torch", "torch", "PyTorch tensor runtime and checkpoint loading"),
    DependencySpec("torchvision", "torchvision", "image transforms and ViT fallback models"),
    DependencySpec("PIL", "Pillow", "frame image loading"),
    DependencySpec("transformers", "transformers", "GPT-2 tokenizer and language decoder"),
    DependencySpec("timm", "timm", "ViT backbone provider"),
    DependencySpec("cupy", "cupy-cuda12x", "CUDA array runtime for future custom operators / TensorRT plugin alignment"),
)

SERVER_DEPENDENCIES = (
    DependencySpec("fastapi", "fastapi", "REST API server"),
    DependencySpec("uvicorn", "uvicorn", "ASGI server"),
    DependencySpec("pydantic", "pydantic", "request/response schema validation"),
)

FRONTEND_DEPENDENCIES = (
    DependencySpec("chainlit", "chainlit", "Chainlit frontend UI"),
    DependencySpec("httpx", "httpx", "frontend REST client"),
)


def _missing_dependencies(specs: Iterable[DependencySpec]) -> list[DependencySpec]:
    return [spec for spec in specs if spec.required and importlib.util.find_spec(spec.module) is None]


def assert_dependencies(specs: Iterable[DependencySpec], group_name: str):
    missing = _missing_dependencies(specs)
    if not missing:
        return

    details = "\n".join(
        f"- module `{spec.module}` from package `{spec.package}`: {spec.purpose}"
        for spec in missing
    )
    raise RuntimeDependencyError(
        f"Missing required {group_name} dependencies before model loading:\n"
        f"{details}\n\n"
        "Install the unified environment with:\n"
        "  python -m pip install -r requirements.txt"
    )


def assert_core_runtime_ready(device: str = "cpu", require_cupy: bool = True):
    specs = list(CORE_RUNTIME_DEPENDENCIES)
    if not require_cupy:
        specs = [spec for spec in specs if spec.module != "cupy"]
    assert_dependencies(specs, "core runtime")

    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeDependencyError(
            "The inference config requested CUDA, but torch.cuda.is_available() is False.\n"
            "Check your NVIDIA driver, CUDA-compatible PyTorch wheel, and GPU visibility."
        )


def assert_server_runtime_ready():
    assert_dependencies(SERVER_DEPENDENCIES, "server")


def assert_frontend_runtime_ready():
    assert_dependencies(FRONTEND_DEPENDENCIES, "frontend")
