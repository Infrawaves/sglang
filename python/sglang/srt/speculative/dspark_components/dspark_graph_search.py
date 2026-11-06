"""CPU-only SPS search over the shared, captured verify graph grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.speculative.dspark_components.dspark_graph_budget import (
    GraphBudgetShape,
    GraphBudgetShapes,
)
from sglang.srt.speculative.dspark_components.dspark_sps import graph_step_time

UTILITY_SCALE = 1 << 20


@dataclass(frozen=True)
class SharedGraphDecision:
    shape: GraphBudgetShape
    step_seconds: float
    expected_tokens: float


class SharedGraphBudgetSearch:
    def __init__(self, *, shapes: GraphBudgetShapes, cost_table) -> None:
        self.shapes = shapes
        self.cost_table = cost_table
        self.grid = shapes.capture_num_tokens
        self._grid_tensor = torch.tensor(self.grid, dtype=torch.int64)
        self._cost_cache: dict[int, tuple] = {}
        self._budget_indices: dict[int, torch.Tensor] = {}

    @property
    def payload_size(self) -> int:
        # The last value scores uniform fallback, which bypasses survival_eps.
        return len(self.grid) + 1

    def utilities(self, survival: torch.Tensor, *, survival_eps: float) -> list[int]:
        """One sort/prefix sum, then O(K) reads for every aligned graph budget.

        Input is the existing lagged host confidence. Fixed-point values fit
        the int64 MLP metadata row and give every rank identical utility sums.
        """
        bs = survival.shape[0]
        if not bool(torch.isfinite(survival).all()) or bool(
            ((survival < 0) | (survival > 1)).any()
        ):
            return [-1] * self.payload_size
        candidates = survival.flatten()
        valid = candidates[candidates >= survival_eps].to(torch.float64)
        prefix = torch.cat(
            (
                torch.zeros(1, dtype=torch.float64),
                valid.sort(descending=True).values.cumsum(0),
            )
        )
        indices = self._budget_indices.get(bs)
        if indices is None:
            indices = (
                self._grid_tensor.clamp(max=bs * self.shapes.verify_width) - bs
            ).clamp(min=0)
            self._budget_indices[bs] = indices
        aligned = bs + prefix[indices.clamp(max=valid.numel())]
        uniform = bs + candidates.to(torch.float64).sum().reshape(1)
        return (
            (torch.cat((aligned, uniform)) * UTILITY_SCALE)
            .round()
            .to(torch.int64)
            .tolist()
        )

    def select(
        self, *, batch_sizes: list[int], utilities: Optional[list[list[int]]]
    ) -> Optional[SharedGraphDecision]:
        if (
            utilities is None
            or len(utilities) != len(batch_sizes)
            or not batch_sizes
            or min(batch_sizes) < 0
            or any(len(row) != self.payload_size or min(row) < 0 for row in utilities)
        ):
            return None
        max_bs = max(batch_sizes)
        if max_bs <= 0:
            return None
        candidates = self._cost_cache.get(max_bs)
        if candidates is None:
            candidates = tuple(
                (
                    shape,
                    graph_step_time(
                        self.cost_table,
                        num_reqs=max_bs,
                        graph_num_tokens=shape.graph_num_tokens,
                        graph_slots=shape.slots,
                    )
                    if shape is not None
                    else None,
                )
                for shape in self.shapes.for_batch(max_bs)
            )
            self._cost_cache[max_bs] = candidates
        best = None
        best_score = -1.0
        for index, (shape, cost) in enumerate(candidates):
            if shape is None or cost is None:
                continue
            utility_index = len(self.grid) if shape.uniform else index
            total = sum(row[utility_index] for row in utilities)
            score = total / cost
            # The grid is increasing: strict improvement keeps the smaller G
            # on ties and every rank performs the same deterministic reduction.
            if score > best_score:
                best_score = score
                best = SharedGraphDecision(shape, cost, total / UTILITY_SCALE)
        return best
