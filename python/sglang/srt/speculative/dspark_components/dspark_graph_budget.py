"""Host-only graph geometry shared by DSpark budgeting and graph replay."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence


def full_coverage_graph_slots(
    max_live_bs: int, graph_num_tokens: int, verify_width: int
) -> int:
    """Slots needed when each live request consumes only its anchor token."""
    padding_tokens = max(0, graph_num_tokens - max_live_bs)
    return max_live_bs + (padding_tokens + verify_width - 1) // verify_width


def requires_full_width_ragged_graph(
    *,
    max_live_bs: int,
    graph_num_tokens: int,
    verify_width: int,
    max_graph_slots: int,
    reserve_padding_slot: bool,
) -> bool:
    return reserve_padding_slot and (
        full_coverage_graph_slots(max_live_bs, graph_num_tokens, verify_width)
        > max_graph_slots
    )


def resolve_ragged_graph_slots(
    slots_by_tokens: Mapping[int, Sequence[int]],
    *,
    graph_num_tokens: int,
    max_live_bs: int,
    verify_width: int,
    reserve_padding_slot: bool,
) -> Optional[int]:
    """Resolve the smallest captured slot bucket using the replay padding rule."""
    slots = slots_by_tokens.get(graph_num_tokens, ())
    required_slots = (
        full_coverage_graph_slots(max_live_bs, graph_num_tokens, verify_width)
        if reserve_padding_slot
        else max_live_bs
    )
    index = bisect_left(slots, required_slots)
    return slots[index] if 0 < max_live_bs and index < len(slots) else None


@dataclass(frozen=True)
class GraphBudgetShape:
    slots: int
    graph_num_tokens: int
    uniform: bool = False


class GraphBudgetShapes:
    """Cache feasible replay shapes in the common token-grid order.

    A compact tier whose padding exceeds the captured slot limit causes replay
    to switch to full-width verification. Its only candidate is the rounded
    full-width tier, with uniform utility. Keeping that distinction here makes
    the budget search cost the graph that replay actually uses.
    """

    def __init__(
        self,
        *,
        capture_num_tokens: Sequence[int],
        slots_by_tokens: Mapping[int, Sequence[int]],
        verify_width: int,
        max_bs: int,
        reserve_padding_slot: bool,
    ):
        self.capture_num_tokens = tuple(capture_num_tokens)
        if (
            not self.capture_num_tokens
            or self.capture_num_tokens[0] <= 0
            or tuple(sorted(set(self.capture_num_tokens))) != self.capture_num_tokens
        ):
            raise ValueError(
                "capture_num_tokens must be positive and strictly increasing"
            )
        if verify_width <= 0 or max_bs <= 0:
            raise ValueError("verify_width and max_bs must be positive")
        self.verify_width = verify_width
        self.max_bs = max_bs
        self.reserve_padding_slot = reserve_padding_slot
        self.slots_by_tokens = {
            tokens: tuple(slots_by_tokens.get(tokens, ()))
            for tokens in self.capture_num_tokens
        }
        for tokens, slots in self.slots_by_tokens.items():
            if slots != tuple(sorted(set(slots))) or any(
                slot <= 0
                or slot > max_bs
                or slot > tokens
                or tokens > slot * verify_width
                for slot in slots
            ):
                raise ValueError(
                    f"invalid captured slot buckets for G={tokens}: {slots}"
                )
        self._empty = (None,) * len(self.capture_num_tokens)
        self._cache: dict[int, tuple[Optional[GraphBudgetShape], ...]] = {}

    @classmethod
    def from_runner(cls, runner) -> GraphBudgetShapes:
        return cls(
            capture_num_tokens=runner.capture_num_tokens,
            slots_by_tokens=runner._ragged_slots_by_tokens,
            verify_width=runner.captured_req_width,
            max_bs=runner.max_bs,
            reserve_padding_slot=runner.reserve_ragged_padding_slot,
        )

    def for_batch(self, max_live_bs: int) -> tuple[Optional[GraphBudgetShape], ...]:
        if not 0 < max_live_bs <= self.max_bs:
            return self._empty
        cached = self._cache.get(max_live_bs)
        if cached is not None:
            return cached
        full_index = bisect_left(
            self.capture_num_tokens, max_live_bs * self.verify_width
        )
        shapes: list[Optional[GraphBudgetShape]] = []
        for index, tokens in enumerate(self.capture_num_tokens):
            shape = None
            # The planner clamps the shared token hint to B * W before
            # rounding. Larger candidates would replay this full-width tier.
            if tokens >= max_live_bs and index <= full_index:
                uniform = requires_full_width_ragged_graph(
                    max_live_bs=max_live_bs,
                    graph_num_tokens=tokens,
                    verify_width=self.verify_width,
                    max_graph_slots=self.max_bs,
                    reserve_padding_slot=self.reserve_padding_slot,
                )
                if uniform:
                    slots = self.slots_by_tokens[tokens]
                    if index == full_index and slots:
                        shape = GraphBudgetShape(slots[0], tokens, uniform=True)
                else:
                    slots = resolve_ragged_graph_slots(
                        self.slots_by_tokens,
                        graph_num_tokens=tokens,
                        max_live_bs=max_live_bs,
                        verify_width=self.verify_width,
                        reserve_padding_slot=self.reserve_padding_slot,
                    )
                    if slots is not None:
                        shape = GraphBudgetShape(slots, tokens)
            shapes.append(shape)
        result = tuple(shapes)
        self._cache[max_live_bs] = result
        return result
