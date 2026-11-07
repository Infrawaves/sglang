"""Floor-M dispatch table for fused GEMM + reduce-scatter + prefix addition.

Selection takes the largest measured M not exceeding the live M, the same rule
as the SP collective tables. A baseline or disabled bucket, or an M below every
bucket, keeps the framework's native GEMM + RS + prefix path. The checked-in
GB300 table keeps its historical 4K push candidate disabled because the sender
changed after measurement.
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
    kernel: Optional[KernelConfig] = None
    tile_order: Optional[str] = None


class DispatchTable(NamedTuple):
    k: int
    configs: dict[int, DispatchConfig]

    def select(self, m: int) -> Optional[DispatchConfig]:
        # Return the fused config of the largest bucket <= m, None for native.
        selected = None
        for bucket in sorted(self.configs):
            if bucket > m:
                break
            selected = self.configs[bucket]
        if selected is None or selected.mode == "baseline":
            return None
        return selected


def load_dispatch_table(
    world_size: int, hidden_size: int, device_name: str
) -> Optional[DispatchTable]:
    """Return every measured bucket, or None without a table for this device."""
    path = os.path.join(
        _CONFIG_DIR,
        f"world={world_size},H={hidden_size},device_name={device_name}.json",
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        table = json.load(f)
    configs = {}
    for m, config in table["configs"].items():
        if config["enabled"] and config["mode"] != "baseline":
            configs[int(m)] = DispatchConfig(
                config["mode"],
                KernelConfig(*config["tile_shape"], config["cluster_hidden"]),
                config["tile_order"],
            )
        else:
            # A disabled candidate keeps the native path, like a baseline.
            configs[int(m)] = DispatchConfig("baseline")
    return DispatchTable(table["match"]["k"], configs)
