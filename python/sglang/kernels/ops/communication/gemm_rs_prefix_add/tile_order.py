from __future__ import annotations


def build_tile_m_order(
    num_shard_tokens: int, world_size: int, tile_m: int, rank: int | None
) -> tuple[int, ...]:
    """Physical M-tile order: ``rank=None`` gives the common order, which starts
    each round at shard 0 on every source; a rank gives the rotated order, which
    starts at that source's own shard, then visits every destination before the
    next row."""
    if num_shard_tokens <= 0:
        raise ValueError("num_shard_tokens must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if tile_m <= 0:
        raise ValueError("tile_m must be positive")
    if rank is not None and not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    # A tile crossing a shard boundary belongs to both shards' ranges.
    shard_tile_m_ranges = [
        range(
            shard_rank * num_shard_tokens // tile_m,
            ((shard_rank + 1) * num_shard_tokens - 1) // tile_m + 1,
        )
        for shard_rank in range(world_size)
    ]
    num_m_tiles = (world_size * num_shard_tokens + tile_m - 1) // tile_m
    tile_already_scheduled = [False] * num_m_tiles
    tile_m_order = []
    first_shard_rank = 0 if rank is None else rank
    for shard_tile_idx in range(max(map(len, shard_tile_m_ranges))):
        for rank_offset in range(world_size):
            shard_tiles = shard_tile_m_ranges[
                (first_shard_rank + rank_offset) % world_size
            ]
            if shard_tile_idx >= len(shard_tiles):
                continue
            physical_tile_m_idx = shard_tiles[shard_tile_idx]
            if not tile_already_scheduled[physical_tile_m_idx]:
                tile_m_order.append(physical_tile_m_idx)
                tile_already_scheduled[physical_tile_m_idx] = True

    assert len(tile_m_order) == num_m_tiles
    return tuple(tile_m_order)
