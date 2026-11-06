from __future__ import annotations

import bisect
import math
from typing import Optional

import msgspec


def floor_probe_index(edges: list[int], batch_tokens: int) -> int:
    idx = bisect.bisect_right(edges, batch_tokens) - 1
    return max(0, min(idx, len(edges) - 1))


class SpsCostTable(msgspec.Struct, frozen=True):
    sample_batch_tokens: list[int]
    sample_steps_per_sec: list[float]
    max_batch_tokens: int

    def __post_init__(self) -> None:
        if not self.sample_batch_tokens:
            raise ValueError("SpsCostTable requires at least one probe.")
        if self.sample_batch_tokens != sorted(set(self.sample_batch_tokens)):
            raise ValueError(
                "sample_batch_tokens must be strictly increasing (monotone-sorted "
                f"invariant), got {self.sample_batch_tokens}."
            )
        if len(self.sample_batch_tokens) != len(self.sample_steps_per_sec):
            raise ValueError(
                "sample_batch_tokens and sample_steps_per_sec must have equal length, "
                f"got {len(self.sample_batch_tokens)} vs {len(self.sample_steps_per_sec)}."
            )
        if self.max_batch_tokens < self.sample_batch_tokens[-1]:
            raise ValueError(
                "max_batch_tokens must be >= the largest probe, got "
                f"{self.max_batch_tokens} < {self.sample_batch_tokens[-1]}."
            )

    def lookup(self, batch_tokens: int) -> float:
        return self.sample_steps_per_sec[
            floor_probe_index(self.sample_batch_tokens, batch_tokens)
        ]

    def to_json(self) -> str:
        return msgspec.json.encode(self).decode("utf-8")

    @classmethod
    def from_json(cls, data: str) -> SpsCostTable:
        return msgspec.json.decode(data.encode("utf-8"), type=cls)


def _interp_clamped(xs: list[int], ys: list[float], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    hi = bisect.bisect_right(xs, x)
    lo = hi - 1
    frac = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + frac * (ys[hi] - ys[lo])


class SpsAdditiveCostTable(msgspec.Struct, frozen=True):
    bias_seconds: float
    bs_probes: list[int]
    alpha_seconds: list[float]
    m_probes: list[int]
    theta_seconds: list[float]

    def __post_init__(self) -> None:
        for name, probes, values in (
            ("bs", self.bs_probes, self.alpha_seconds),
            ("m", self.m_probes, self.theta_seconds),
        ):
            if not probes:
                raise ValueError(f"SpsAdditiveCostTable requires {name}_probes.")
            if probes != sorted(set(probes)):
                raise ValueError(
                    f"{name}_probes must be strictly increasing, got {probes}."
                )
            if len(probes) != len(values):
                raise ValueError(
                    f"{name}_probes and its values must have equal length, got "
                    f"{len(probes)} vs {len(values)}."
                )
        if self.bias_seconds <= 0:
            raise ValueError(f"bias_seconds must be > 0, got {self.bias_seconds}.")

    def step_time(self, *, num_reqs: int, budget: int) -> float:
        return (
            self.bias_seconds
            + _interp_clamped(self.bs_probes, self.alpha_seconds, float(num_reqs))
            + _interp_clamped(
                self.m_probes, self.theta_seconds, float(num_reqs + budget)
            )
        )

    def to_json(self) -> str:
        return msgspec.json.encode(self).decode("utf-8")

    @classmethod
    def from_json(cls, data: str) -> SpsAdditiveCostTable:
        return msgspec.json.decode(data.encode("utf-8"), type=cls)


class SpsGraphCostSample(msgspec.Struct, frozen=True):
    """A whole-DP-group round measured with align and the actual replay shape."""

    num_reqs: int
    graph_slots: int
    graph_num_tokens: int
    step_seconds: float


class SpsGraphCostTable(msgspec.Struct, frozen=True):
    """Discrete graph costs; only the request-count axis is interpolated.

    ``num_reqs`` is the maximum local request count across the DP group.
    Samples measure a complete group round for the configured deployment.
    ``graph_slots`` and ``graph_num_tokens`` are the actual CUDA graph key,
    including padding. Graph shapes are never interpolated.
    """

    verify_width: int
    capture_num_tokens: list[int]
    samples: list[SpsGraphCostSample]
    schema: str = "dspark_graph_v1"
    align: bool = True

    def __post_init__(self) -> None:
        if self.schema != "dspark_graph_v1":
            raise ValueError(f"Unsupported graph SPS schema: {self.schema}.")
        if self.verify_width < 1 or not self.align:
            raise ValueError("Graph SPS requires verify_width >= 1 and align=true.")
        grid = self.capture_num_tokens
        if not grid or grid != sorted(set(grid)) or grid[0] < 1:
            raise ValueError("capture_num_tokens must be positive and increasing.")
        if not self.samples:
            raise ValueError("Graph SPS requires at least one measured sample.")
        seen = set()
        for sample in self.samples:
            key = (sample.num_reqs, sample.graph_slots, sample.graph_num_tokens)
            if key in seen:
                raise ValueError(f"Duplicate graph SPS sample: {key}.")
            seen.add(key)
            if not (
                0 < sample.num_reqs <= sample.graph_slots
                and sample.num_reqs <= sample.graph_num_tokens
                and sample.graph_num_tokens <= sample.graph_slots * self.verify_width
                and sample.graph_num_tokens in grid
            ):
                raise ValueError(f"Invalid graph SPS sample shape: {key}.")
            if not math.isfinite(sample.step_seconds) or sample.step_seconds <= 0:
                raise ValueError("Graph SPS step_seconds must be finite and positive.")

    def validate_capture(
        self, *, verify_width: int, capture_num_tokens: list[int]
    ) -> None:
        if self.verify_width != verify_width:
            raise ValueError(
                f"Graph SPS verify_width={self.verify_width} differs from "
                f"runtime verify_width={verify_width}."
            )
        if self.capture_num_tokens != list(capture_num_tokens):
            raise ValueError("Graph SPS capture_num_tokens differs from runtime grid.")

    def step_time(
        self, *, num_reqs: int, graph_slots: int, graph_num_tokens: int
    ) -> Optional[float]:
        samples = sorted(
            (
                sample
                for sample in self.samples
                if sample.graph_slots == graph_slots
                and sample.graph_num_tokens == graph_num_tokens
            ),
            key=lambda sample: sample.num_reqs,
        )
        if not samples or not samples[0].num_reqs <= num_reqs <= samples[-1].num_reqs:
            return None
        return _interp_clamped(
            [sample.num_reqs for sample in samples],
            [sample.step_seconds for sample in samples],
            float(num_reqs),
        )

    def to_json(self) -> str:
        return msgspec.json.encode(self).decode("utf-8")

    @classmethod
    def from_json(cls, data: str) -> SpsGraphCostTable:
        return msgspec.json.decode(data.encode("utf-8"), type=cls)


def graph_step_time(
    table: SpsCostTable | SpsAdditiveCostTable | SpsGraphCostTable,
    *,
    num_reqs: int,
    graph_num_tokens: int,
    graph_slots: int,
) -> Optional[float]:
    """Return a candidate graph's cost, or None if the cost is unavailable.

    Existing additive tables model planned tokens. Evaluating their token
    axis at the replay tier provides a compatibility estimate; it does not
    add calibration for the slot count. New measurements can use the graph
    schema to provide costs for each actual (slots, tokens) shape.
    """
    if min(num_reqs, graph_num_tokens, graph_slots) < 1:
        return None
    if isinstance(table, SpsGraphCostTable):
        seconds = table.step_time(
            num_reqs=num_reqs,
            graph_num_tokens=graph_num_tokens,
            graph_slots=graph_slots,
        )
    elif isinstance(table, SpsAdditiveCostTable):
        seconds = table.step_time(num_reqs=num_reqs, budget=graph_num_tokens - num_reqs)
    else:
        sps = table.lookup(graph_num_tokens)
        seconds = 1.0 / sps if sps > 0 else None
    return (
        seconds
        if seconds is not None and math.isfinite(seconds) and seconds > 0
        else None
    )


def profile_sps_table(
    *,
    probes: list[tuple[int, float]],
    max_batch_tokens: Optional[int] = None,
) -> SpsCostTable:
    if not probes:
        raise ValueError("profile_sps_table requires at least one probe.")

    sorted_probes = sorted(probes, key=lambda probe: probe[0])

    sample_batch_tokens: list[int] = []
    sample_steps_per_sec: list[float] = []
    for batch_tokens, steps_per_sec in sorted_probes:
        batch_tokens = int(batch_tokens)
        if batch_tokens < 1:
            raise ValueError(
                f"profile_sps_table requires batch_tokens >= 1, got {batch_tokens}."
            )
        if sample_batch_tokens and batch_tokens == sample_batch_tokens[-1]:
            raise ValueError(
                "profile_sps_table requires unique batch_tokens per probe; "
                f"batch_tokens={batch_tokens} appears more than once. Median the "
                "repeated samples per batch_tokens before calling the assembler."
            )
        sample_batch_tokens.append(batch_tokens)
        sample_steps_per_sec.append(float(steps_per_sec))

    resolved_max = (
        int(max_batch_tokens)
        if max_batch_tokens is not None
        else sample_batch_tokens[-1]
    )
    return SpsCostTable(
        sample_batch_tokens=sample_batch_tokens,
        sample_steps_per_sec=sample_steps_per_sec,
        max_batch_tokens=resolved_max,
    )


def load_sps_table_from_path(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    fields = msgspec.json.decode(data)
    if fields.get("schema") == "dspark_graph_v1":
        return SpsGraphCostTable.from_json(data)
    if "schema" in fields:
        raise ValueError(f"Unsupported SPS schema: {fields['schema']}.")
    if "bias_seconds" in fields:
        return SpsAdditiveCostTable.from_json(data)
    return SpsCostTable.from_json(data)


def build_uninitialized_sps_table(*, max_batch_tokens: int) -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1],
        sample_steps_per_sec=[1.0],
        max_batch_tokens=max_batch_tokens,
    )


def is_uninitialized_sps_table(
    table: SpsCostTable | SpsAdditiveCostTable | SpsGraphCostTable,
) -> bool:
    if isinstance(table, (SpsAdditiveCostTable, SpsGraphCostTable)):
        return False
    return len(table.sample_batch_tokens) <= 1
