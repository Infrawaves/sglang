from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_WORLD_SIZES = (2, 4, 8)


@dataclass(frozen=True)
class KernelConfig:
    """One compiled kernel: logical CTA token tile, hidden tile, hidden cluster."""

    tile_m: int = 128
    tile_n: int = 128
    cluster_hidden: int = 1

    def __post_init__(self):
        if (
            self.tile_m not in (16, 32, 64, 128, 256)
            or self.tile_n not in (64, 128)
            or self.cluster_hidden not in (1, 2)
        ):
            raise ValueError(
                "uncompiled SM100 configuration: token tile must be 16/32/64/128/256, "
                "hidden tile 64/128, and cluster_hidden 1/2"
            )

    @property
    def key(self):
        return (self.tile_m, self.tile_n, self.cluster_hidden)


SUPPORTED_CONFIGS = tuple(
    KernelConfig(tile_m, tile_n, cluster_hidden)
    for tile_m in (16, 32, 64, 128, 256)
    for tile_n in (64, 128)
    for cluster_hidden in (1, 2)
)


@dataclass(frozen=True)
class WorkspaceLayout:
    max_m: int
    n: int
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
        object.__setattr__(self, "configs", configs)
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError("world_size must be 2, 4, or 8")
        if self.max_m <= 0 or self.max_m % self.world_size:
            raise ValueError("max_m must be positive and divisible by world_size")
        if self.n <= 0 or self.n % 8:
            raise ValueError("N must be positive and divisible by 8")
        if max(self.max_m, self.n, self.ready_numel) > 2**31 - 1:
            raise ValueError("dimensions and tile capacity must fit int32")

    @property
    def tile_n_capacity(self):
        # Physical tile IDs use this fixed stride for every configured tile.
        min_tile_n = min(config.tile_n for config in self.configs)
        return (self.n + min_tile_n - 1) // min_tile_n

    @property
    def ready_numel(self):
        min_tile_m = min(config.tile_m for config in self.configs)
        return (self.max_m + min_tile_m - 1) // min_tile_m * self.tile_n_capacity

    @property
    def partial_numel(self):
        return self.max_m * self.n

    @property
    def plane_stride_elements(self):
        # One receive plane per source rank, sized by capacity so the stride is
        # fixed across active M. N % 8 == 0 keeps it 16-byte aligned for uint4.
        return self.max_m // self.world_size * self.n

    @property
    def recv_data_numel(self):
        return self.world_size * self.plane_stride_elements

    @property
    def arena_numel(self):
        return self.recv_data_numel + self.partial_numel

    def can_fuse(self, m, k, config=KernelConfig()):
        if config not in self.configs:
            raise ValueError(
                f"configuration {config.key} was not declared for this workspace"
            )
        if m <= 0 or m % self.world_size or k <= 0:
            raise ValueError("M and K must be positive and M divisible by world_size")
        if max(m, self.n, k, self.ready_numel) > 2**31 - 1:
            raise ValueError("dimensions and tile capacity must fit int32")
        return (
            m <= self.max_m
            and k % 128 == 0
            and (
                config.cluster_hidden == 1
                or self.n % (config.tile_n * config.cluster_hidden) == 0
            )
        )

    def cluster_work(self, m, config=KernelConfig()):
        num_token_tiles = (m + config.tile_m - 1) // config.tile_m
        num_hidden_tiles = (self.n + config.tile_n - 1) // config.tile_n
        num_hidden_clusters = (
            num_hidden_tiles + config.cluster_hidden - 1
        ) // config.cluster_hidden
        return num_token_tiles * num_hidden_clusters
