from __future__ import annotations

import json
import threading
from dataclasses import asdict
from typing import Dict

from core.config import InferenceConfig
from core.env import assert_core_runtime_ready


def _engine_key(config: InferenceConfig) -> str:
    """Stable key for one model/runtime configuration."""

    return json.dumps(asdict(config), sort_keys=True, default=str)


class ModelRegistry:
    """Single-card model registry.

    INFO(memory): On RTX 3050 4GB, keep one engine instance per config and
    serialize access at the server/task layer to avoid accidental duplicate
    ViT + GPT-2 loads.
    """

    def __init__(self):
        self._engines: Dict[str, object] = {}
        self._lock = threading.Lock()

    def get_engine(self, config: InferenceConfig):
        key = _engine_key(config)
        with self._lock:
            engine = self._engines.get(key)
            if engine is None:
                # Check heavy runtime dependencies before importing/loading the model.
                assert_core_runtime_ready(device=config.device, require_cupy=config.tensorrt.enabled)
                from core.engine import InferenceEngine

                engine = InferenceEngine.from_config(config)
                self._engines[key] = engine
            return engine


MODEL_REGISTRY = ModelRegistry()

