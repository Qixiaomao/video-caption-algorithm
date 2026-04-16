from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from core.config import MemoryConfig


@dataclass(frozen=True)
class GpuMemorySnapshot:
    allocated_mb: float
    reserved_mb: float
    free_mb: float | None = None
    total_mb: float | None = None


class MemoryManager:
    """Lightweight GPU memory guard for 4GB deployment targets."""

    def __init__(self, config: MemoryConfig):
        self.config = config

    def snapshot(self) -> GpuMemorySnapshot:
        if not torch.cuda.is_available():
            return GpuMemorySnapshot(allocated_mb=0.0, reserved_mb=0.0)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return GpuMemorySnapshot(
            allocated_mb=torch.cuda.memory_allocated() / 1024**2,
            reserved_mb=torch.cuda.memory_reserved() / 1024**2,
            free_mb=free_bytes / 1024**2,
            total_mb=total_bytes / 1024**2,
        )

    def cleanup(self):
        if self.config.allow_cuda_empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def oom_guard(self):
        try:
            yield
        except torch.cuda.OutOfMemoryError:
            self.cleanup()
            raise

