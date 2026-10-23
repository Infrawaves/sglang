from __future__ import annotations

import msgspec

SUPPORTED_WORLD_SIZES = (2, 4, 8)
# Output hidden columns per CTA, a compile-time constant of every instance.
TILE_N = 128


class KernelConfig(msgspec.Struct, frozen=True):
    """One compiled kernel: CTA token tile and hidden cluster."""

    tile_m: int = 128
    cluster_hidden: int = 1

    def __post_init__(self):
        # dispatch_sm100_ag_config compiles only these; a 256-token tile
        # requires the 2-SM MMA.
        if (self.tile_m, self.cluster_hidden) not in ((128, 1), (128, 2), (256, 2)):
            raise ValueError(
                "uncompiled SM100 AG configuration: (tile_m, cluster_hidden) "
                "must be (128, 1), (128, 2) or (256, 2)"
            )

    @property
    def key(self):
        return (self.tile_m, self.cluster_hidden)


SUPPORTED_CONFIGS = (KernelConfig(128, 1), KernelConfig(128, 2), KernelConfig(256, 2))


class WorkspaceLayout(msgspec.Struct, frozen=True):
    max_m: int
    k: int
    world_size: int
    configs: tuple[KernelConfig, ...] = SUPPORTED_CONFIGS

    def __post_init__(self):
        configs = tuple(
            sorted(
                {
                    config
                    if isinstance(config, KernelConfig)
                    else KernelConfig(*config)
                    for config in self.configs
                },
                key=lambda config: config.key,
            )
        )
        if not configs:
            raise ValueError("configs must declare at least one compiled kernel")
        msgspec.structs.force_setattr(self, "configs", configs)
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError("world_size must be 2, 4, or 8")
        if self.max_m <= 0 or self.max_m % self.world_size:
            raise ValueError("max_m must be positive and divisible by world_size")
        if self.k <= 0 or self.k % 128:
            raise ValueError("K must be positive and divisible by 128")
        if max(self.max_m, self.k) > 2**31 - 1:
            raise ValueError("dimensions must fit int32")

    @property
    def gathered_input_numel(self):
        # Shards are packed by the live shard size, so one call uses only the
        # [world_size, M/world_size, K] prefix of this capacity.
        return self.max_m * self.k

    def can_fuse(self, m, n, config=KernelConfig()):
        if config not in self.configs:
            raise ValueError(
                f"configuration {config.key} was not declared for this workspace"
            )
        if m <= 0 or m % self.world_size or n <= 0:
            raise ValueError("M and N must be positive and M divisible by world_size")
        if max(m, n) > 2**31 - 1:
            raise ValueError("dimensions must fit int32")
        return m <= self.max_m and n % 8 == 0

    def cluster_work(self, m, n, config=KernelConfig()):
        num_shard_token_tiles = (
            m // self.world_size + config.tile_m - 1
        ) // config.tile_m
        num_hidden_tiles = (n + TILE_N - 1) // TILE_N
        num_hidden_clusters = (
            num_hidden_tiles + config.cluster_hidden - 1
        ) // config.cluster_hidden
        return self.world_size * num_shard_token_tiles * num_hidden_clusters
