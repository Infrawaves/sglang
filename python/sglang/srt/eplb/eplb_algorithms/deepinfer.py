# Ported from DeepInfer (deepseek-ai/DeepInfer,
# deepinfer/model_executor/eplb/policy.py).
#
# Replica selection is the same per-step vector greedy as deepseek_vec's
# decode path.  The difference is placement: instead of sorting replicas by
# their *total* load and snaking them across ranks, every replica is placed on
# the rank that minimizes  sum_{step} max_{rank} load(rank, step)  given what is
# already placed.  A rank that is hot in some steps and cold in others is no
# longer treated like a rank that is uniformly warm, which is what decode
# imbalance actually looks like.
from typing import Tuple

import torch


def _batch_greedy_indices(
    tokens_per_expert: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    logical_count: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    """Place physical experts on ranks with a per-step greedy.

    Args:
        tokens_per_expert: [num_steps, num_layers, num_logical]
        physical_to_logical_map: [num_layers, num_physical]
        logical_count: [num_layers, num_logical], replicas per logical expert
        capacity: physical experts per rank

    Returns:
        indices: [num_layers, num_physical], a permutation of the old physical
        slots per layer; each consecutive ``capacity`` entries form one rank.
    """
    weights = tokens_per_expert.sum(dim=0, dtype=torch.float32) / logical_count
    physical_experts = physical_to_logical_map.to(torch.int64)
    # Heaviest replica first; stable so equal loads keep their slot order.
    physical_order = weights.gather(-1, physical_experts).argsort(
        dim=-1, descending=True, stable=True
    )
    expert_order = physical_experts.gather(-1, physical_order)
    # [layers, logical, steps]: per-replica load vector of each logical expert.
    vectors = tokens_per_expert.permute(1, 2, 0).to(
        dtype=torch.float32, memory_format=torch.contiguous_format, copy=True
    )
    vectors.div_(logical_count.unsqueeze(-1))

    num_layers, _, num_steps = vectors.shape
    num_physical = physical_order.size(1)
    assert num_physical % capacity == 0, f"{num_physical=} {capacity=}"
    num_ranks = num_physical // capacity

    loads = vectors.new_zeros((num_layers, num_ranks, num_steps))
    peaks = vectors.new_zeros((num_layers, num_steps))
    occupancy = torch.zeros(
        (num_layers, num_ranks), dtype=torch.int64, device=vectors.device
    )
    candidates = torch.empty_like(loads)
    scores = vectors.new_empty((num_layers, num_ranks))
    indices = torch.empty_like(physical_order)
    layers = torch.arange(num_layers, device=vectors.device)
    for job_index in range(num_physical):
        vector = vectors[layers, expert_order[:, job_index]]
        # candidates[l, r, s]: rank r's load at step s if this replica lands
        # there; max with peaks gives the per-step max-rank load of that choice.
        torch.add(loads, vector.unsqueeze(1), out=candidates)
        torch.maximum(candidates, peaks.unsqueeze(1), out=candidates)
        torch.sum(candidates, dim=-1, out=scores)
        scores.masked_fill_(occupancy == capacity, float("inf"))
        rank = scores.argmin(dim=-1)  # ties -> lowest rank id
        occupied = occupancy[layers, rank]
        indices[layers, rank * capacity + occupied] = physical_order[:, job_index]
        occupancy[layers, rank] = occupied + 1
        updated = loads[layers, rank] + vector
        loads[layers, rank] = updated
        torch.maximum(peaks, updated, out=peaks)
    return indices


def rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flat (non-hierarchical) rebalance over all EP ranks.

    Args:
        tokens_per_expert: [num_steps, num_layers, num_logical] (2-D is
            treated as a single step)

    Returns:
        physical_to_logical_map: [num_layers, num_physical]
        logical_to_physical_map: [num_layers, num_logical, num_redundant + 1],
            -1 padded
        logical_count: [num_layers, num_logical]
    """
    if tokens_per_expert.dim() == 2:
        tokens_per_expert = tokens_per_expert.unsqueeze(0)
    tokens_per_expert = tokens_per_expert.float()
    _, num_layers, num_logical = tokens_per_expert.shape
    num_redundant = num_physical_experts - num_logical
    assert num_redundant >= 0, f"{num_physical_experts=} {num_logical=}"
    device = tokens_per_expert.device

    physical_to_logical_map = torch.empty(
        num_layers, num_physical_experts, dtype=torch.int, device=device
    )
    logical_to_physical_map = torch.full(
        (num_layers, num_logical, num_redundant + 1), -1, dtype=torch.int, device=device
    )
    logical_count = torch.ones(num_layers, num_logical, dtype=torch.int, device=device)

    arange_layers = torch.arange(num_layers, dtype=torch.int, device=device)
    arange_logical = torch.arange(num_logical, dtype=torch.int, device=device)
    physical_to_logical_map[:, :num_logical] = arange_logical
    logical_to_physical_map[:, :, 0] = arange_logical

    # One redundant replica per round.  For candidate j, values[..., j] is the
    # max per-replica load if j gets the replica (unchanged unless j is the
    # current argmax); summed over steps, argmin picks the replica.
    tokens_all_diff = tokens_per_expert + arange_logical * 1e-4
    for i in range(num_redundant):
        score = tokens_all_diff / logical_count  # values must be pairwise distinct
        score_added = tokens_per_expert / (logical_count + 1)
        values, idx = score.max(-1, keepdim=True)
        values = values.expand_as(score).contiguous()
        score.scatter_(-1, idx, score_added.gather(-1, idx))
        values.scatter_(-1, idx, score.max(-1, keepdim=True).values)
        redundancy_indices = values.sum(0).argmin(-1)  # [num_layers]

        physical_index = num_logical + i
        physical_to_logical_map[:, physical_index] = redundancy_indices
        redundancy_count = logical_count.gather(
            -1, redundancy_indices.unsqueeze(-1)
        ).squeeze(-1)
        logical_to_physical_map[arange_layers, redundancy_indices, redundancy_count] = (
            physical_index
        )
        logical_count[arange_layers, redundancy_indices] += 1

    if num_local_physical_experts > 1:
        assert num_physical_experts % num_local_physical_experts == 0
        indices = _batch_greedy_indices(
            tokens_per_expert,
            physical_to_logical_map,
            logical_count,
            num_local_physical_experts,
        )
        physical_to_logical_map = physical_to_logical_map.gather(-1, indices)
        # indices.argsort maps old slot -> new slot; remap log2phy through it.
        mask = logical_to_physical_map == -1
        logical_to_physical_map[mask] = 0
        logical_to_physical_map = (
            indices.argsort(-1)
            .gather(-1, logical_to_physical_map.view(num_layers, -1).to(torch.int64))
            .view_as(logical_to_physical_map)
            .to(torch.int)
        )
        logical_to_physical_map[mask] = -1

    return physical_to_logical_map, logical_to_physical_map, logical_count
