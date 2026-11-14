"""CUDA graphs for the DFLASH decode aux append: target hidden states into the
draft's context KV."""

import bisect
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch

from sglang.kernels.ops.memory.small_copy import try_small_copy

# hidden, cache_loc, cache_loc_2d, positions, commit_lens
AuxInputs = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class DFlashAuxGraphs:
    """One graph per decode batch-size bucket, each reading a prefix of one set of
    fixed-address inputs, all sharing one private pool. A batch replays the
    smallest bucket that holds it; its padded requests carry commit_lens 0, so
    ``append`` must write only each request's first commit_lens rows."""

    def __init__(
        self,
        *,
        buckets: Sequence[int],
        block_size: int,
        hidden_width: int,
        hidden_dtype: torch.dtype,
        device: Union[str, torch.device],
    ) -> None:
        self._buckets = tuple(sorted(set(buckets)))
        self._block_size = block_size
        max_bs = self._buckets[-1]
        rows = max_bs * block_size
        self._hidden = torch.zeros(
            (rows, hidden_width), dtype=hidden_dtype, device=device
        )
        self._cache_loc = torch.zeros((rows,), dtype=torch.int64, device=device)
        self._positions = torch.zeros((rows,), dtype=torch.int64, device=device)
        self._commit_lens = torch.zeros((max_bs,), dtype=torch.int32, device=device)
        self._no_commit_lens = torch.zeros_like(self._commit_lens)
        self._graphs: Dict[int, torch.cuda.CUDAGraph] = {}

    @property
    def buckets(self) -> Tuple[int, ...]:
        return self._buckets

    @property
    def input_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self._hidden,
                self._cache_loc,
                self._positions,
                self._commit_lens,
                self._no_commit_lens,
            )
        )

    def _inputs(self, bs: int) -> AuxInputs:
        rows = bs * self._block_size
        cache_loc = self._cache_loc[:rows]
        return (
            self._hidden[:rows],
            cache_loc,
            cache_loc.view(bs, self._block_size),
            self._positions[:rows],
            self._commit_lens[:bs],
        )

    def capture(self, append: Callable[..., None], *, small_copy: bool) -> None:
        """Capture every bucket. The inputs hold commit_lens 0, so nothing is
        written to the KV cache."""
        device = self._hidden.device
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(device=device)
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        # Largest first, so each smaller graph reuses the shared pool's blocks.
        for bs in reversed(self._buckets):
            inputs = self._inputs(bs)
            # Warm cuBLAS and Triton autotuning outside the capture.
            with torch.cuda.stream(stream):
                append(*inputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph, pool=pool, stream=stream, capture_error_mode="thread_local"
            ):
                append(*inputs)
            self._graphs[bs] = graph
        current.wait_stream(stream)
        if small_copy:
            # Tuple sizes and padded-tail pointer alignment affect Triton
            # specializations. Warm every supported load before serving.
            for bs in range(1, self._buckets[-1] + 1):
                hidden, loc, _, positions, commit_lens = self._inputs(bs)
                self.load(
                    target_hidden=hidden,
                    cache_loc=loc,
                    positions=positions,
                    commit_lens=commit_lens,
                    small_copy=True,
                )

    def load(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        commit_lens: torch.Tensor,
        small_copy: bool,
    ) -> Optional[torch.cuda.CUDAGraph]:
        """Stage the batch and return its graph, or None for eager append."""
        bs = int(commit_lens.shape[0])
        index = bisect.bisect_left(self._buckets, bs)
        if (
            index == len(self._buckets)
            or target_hidden.dtype != self._hidden.dtype
            or target_hidden.shape[1] != self._hidden.shape[1]
        ):
            return None
        graph_bs = self._buckets[index]
        rows = bs * self._block_size
        self._hidden[:rows].copy_(target_hidden)
        destinations = [
            self._cache_loc[:rows],
            self._positions[:rows],
            self._commit_lens[:bs],
        ]
        sources = [cache_loc, positions, commit_lens]
        if graph_bs > bs:
            # A padded request still holding an earlier batch's commit_lens would
            # rewrite that batch's KV slots, which may since belong to other requests.
            destinations.append(self._commit_lens[bs:graph_bs])
            sources.append(self._no_commit_lens[: graph_bs - bs])
        if not (small_copy and try_small_copy(destinations, sources)):
            for destination, source in zip(destinations, sources):
                destination.copy_(source)
        return self._graphs[graph_bs]
