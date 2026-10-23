"""Floor-M dispatch table for fused all-gather + GEMM.

Selection takes the largest measured M not exceeding the live M, the same rule
as the GEMM + RS + prefix table. A baseline or disabled bucket, or an M below
every bucket, keeps the framework's native all-gather + GEMM path. One table
covers one (K, N) projection, so each consumer of the gathered rows has its own.
"""

from __future__ import annotations

import json
import os
from typing import NamedTuple, Optional

from .workspace import KernelConfig

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "configs", "ag_gemm"
)


class DispatchConfig(NamedTuple):
    mode: str
    kernel: Optional[KernelConfig] = None


class DispatchTable(NamedTuple):
    k: int
    n: int
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
    world_size: int, k: int, n: int, device_name: str
) -> Optional[DispatchTable]:
    path = os.path.join(
        _CONFIG_DIR,
        f"world={world_size},K={k},N={n},device_name={device_name}.json",
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        table = json.load(f)
    configs = {}
    for m, config in table["configs"].items():
        if config["enabled"] and config["mode"] != "baseline":
            tile_m, tile_n = config["tile_shape"]
            if tile_n != 128:
                raise ValueError(f"{path}: SM100 AG GEMM compiles only tile_n=128")
            configs[int(m)] = DispatchConfig(
                config["mode"], KernelConfig(tile_m, config["cluster_hidden"])
            )
        else:
            # A disabled candidate keeps the native path, like a baseline.
            configs[int(m)] = DispatchConfig("baseline")
    return DispatchTable(table["match"]["k"], table["match"]["n"], configs)
