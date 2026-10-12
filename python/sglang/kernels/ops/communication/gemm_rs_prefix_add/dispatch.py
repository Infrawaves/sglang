"""Exact-M dispatch table for fused GEMM + reduce-scatter + prefix addition.

Selection is exact in M: an unmeasured, baseline or disabled shape keeps the
framework's native GEMM + RS + prefix path. The checked-in GB300 table keeps
its historical 4K push candidate disabled because the sender changed after
measurement.
"""

from __future__ import annotations

import json
import os
from typing import NamedTuple, Optional

from .workspace import KernelConfig

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "configs", "gemm_rs_prefix_add"
)


class DispatchConfig(NamedTuple):
    mode: str
    kernel: KernelConfig
    tile_order: str


class DispatchTable(NamedTuple):
    k: int
    configs: dict[int, DispatchConfig]


def load_dispatch_table(
    world_size: int, hidden_size: int, device_name: str
) -> Optional[DispatchTable]:
    """Return the enabled fused shapes, or None without a table for this device."""
    path = os.path.join(
        _CONFIG_DIR,
        f"world={world_size},H={hidden_size},device_name={device_name}.json",
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        table = json.load(f)
    configs = {
        int(m): DispatchConfig(
            config["mode"],
            KernelConfig(*config["tile_shape"], config["cluster_hidden"]),
            config["tile_order"],
        )
        for m, config in table["configs"].items()
        if config["enabled"] and config["mode"] != "baseline"
    }
    return DispatchTable(table["match"]["k"], configs)
