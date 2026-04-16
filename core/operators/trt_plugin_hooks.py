from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TrtPluginHook:
    """Metadata for replacing a PyTorch op with a TensorRT plugin later."""

    name: str
    torch_path: str
    plugin_name: str
    enabled: bool = False


_PLUGIN_HOOKS: Dict[str, TrtPluginHook] = {}


def register_plugin_hook(hook: TrtPluginHook):
    _PLUGIN_HOOKS[hook.name] = hook


def get_plugin_hook(name: str) -> TrtPluginHook | None:
    return _PLUGIN_HOOKS.get(name)


register_plugin_hook(
    TrtPluginHook(
        name="temporal_mean_pool",
        torch_path="core.operators.temporal_pool.TemporalMeanPool",
        plugin_name="TemporalMeanPoolPlugin",
    )
)

