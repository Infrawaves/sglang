"""Interleave BCG adapter for hybrid DSA/KDA text prefills.

Keep full-sequence KPool positions separate from the local DSA positions.
Graph row capacity is fixed while live token counts and prefix state can change.
"""

import torch

from sglang.srt.arg_groups.model_override_base import (
    model_config_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.layers.cp.bcg.bcg import PrefillCPBCGInput
from sglang.srt.layers.cp.interleave import (
    position_owned_rows_per_request,
    run_indexer_with_kv_gather,
)
from sglang.srt.layers.cp.padding import physical_rank_len
from sglang.srt.layers.cp.utils import cp_interleave_input_ids
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)
from sglang.srt.runtime_context import get_parallel


def _indexer_with_kv(
    indexer, x, q_lora, positions, output, layer_id, kv_local, kv_output
):
    # Resolve counts, prefix-cache state and destinations on every replay.
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    n, logical = cp_interleave_indexer_rows(forward_batch)
    if (
        not 0
        <= logical
        <= n
        <= min(x.shape[0], q_lora.shape[0], positions.shape[0], kv_local.shape[0])
    ):
        raise ValueError("Invalid CP indexer input token counts")
    topk, kv = run_indexer_with_kv_gather(
        indexer, x[:n], q_lora[:n], positions[:n], forward_batch, layer_id, kv_local[:n]
    )
    kv_rows = sum(forward_batch.extend_seq_lens_cpu)
    if kv.shape[0] != kv_rows or kv_rows > kv_output.shape[0]:
        raise ValueError("CP KV gather returned an unexpected token count")
    if (
        topk is None
        or topk.ndim != 2
        or not logical <= topk.shape[0] <= n
        or topk.shape[1] != output.shape[1]
    ):
        raise ValueError("CP indexer returned an unexpected top-k shape")
    kv_output[:kv_rows].copy_(kv)
    kv_output[kv_rows:].zero_()
    output[:logical].copy_(topk[:logical])
    output[logical:].fill_(-1)


bcg_indexer_with_kv = eager_on_graph(True)(_indexer_with_kv)


def position_layout_max_rank_rows(extend_lens, prefix_lens, size: int) -> int:
    """Most rows one CP rank holds when prefill CP with DCP shards by position.

    Page-aligned prefixes start every extend on rank 0, so short extends pile
    onto the low ranks and exceed the even split ceil(total / size).
    """
    extend = [int(length) for length in extend_lens]
    if prefix_lens is None:
        # Upper bound: any rank holds at most ceil(extend / size) of a request.
        return sum((length + size - 1) // size for length in extend)
    prefix = [int(length) for length in prefix_lens]
    return max(
        sum(position_owned_rows_per_request(prefix, extend, rank=rank, size=size))
        for rank in range(size)
    )


def restride_gather_index(
    gather_index: torch.Tensor, *, old_stride: int, new_stride: int
) -> torch.Tensor:
    """Re-address rank-major slots (owner * stride + slot) for a wider shard."""
    if old_stride == new_stride:
        return gather_index
    return gather_index // old_stride * new_stride + gather_index % old_stride


def fill_graph_gather_index(
    out: torch.Tensor, live: torch.Tensor, *, rank_rows, stride: int
) -> None:
    """Fill a fixed-address graph gather index from the live one.

    Rows past the live tokens point at distinct padding slots (a rank's rows
    beyond its live count), so the captured index_copy_ never writes a slot
    twice and the captured index_select reads only padding there.
    """
    total = live.shape[0]
    need = out.shape[0] - total
    if need < 0:
        raise RuntimeError(
            f"Graph gather index holds {out.shape[0]} rows, live batch has {total}"
        )
    out[:total].copy_(live)
    if need == 0:
        return
    rows = torch.arange(stride)
    owned = torch.tensor([int(n) for n in rank_rows])
    slots = torch.arange(len(rank_rows))[:, None] * stride + rows[None, :]
    spare = slots[rows[None, :] >= owned[:, None]]
    if spare.numel() < need:
        raise RuntimeError(
            f"Graph gather index needs {need} padding slots, the captured shards "
            f"have {spare.numel()} (stride={stride}, rank_rows={list(rank_rows)})"
        )
    out[total:].copy_(spare[:need])


class InterleaveCPBCGInput(PrefillCPBCGInput):
    @classmethod
    def create(cls, runner):
        result = super().create(runner)
        # Each local shard is CP-aligned before rank-major MoE all-gather.
        alignment = get_parallel().attn_cp_size ** 2
        capacity = (runner.max_num_tokens + alignment - 1) // alignment * alignment
        result.input_ids_global = torch.zeros(
            capacity, dtype=torch.int64, device=runner.device
        )
        # Graphs must read the live count from the same address on each replay.
        result.moe_local_token_count = torch.zeros(
            (), dtype=torch.int32, device=runner.device
        )
        # Prefill CP with DCP reorders by a per-batch gather_index inside the
        # captured KDA segments; replays copy it here instead of rebinding it.
        result.graph_gather_index = (
            torch.zeros(capacity, dtype=torch.int64, device=runner.device)
            if get_parallel().attn_dcp_size > 1
            else None
        )
        return result

    def allows_replay(self, batch_size, num_tokens, prefix_lens, contains_mm_inputs):
        return batch_size > 0 and not contains_mm_inputs and prefix_lens is not None

    # GLM-5.3 Flash (KDA hybrid), the dense-DSA GLM MoE family (GLM-5.2 /
    # GLM-5.3 and their fine-tunes) and Kimi-K3.1 share the interleave BCG path.
    _SUPPORTED_ARCHITECTURES = (
        "Glm5NextForConditionalGeneration",
        "GlmMoeDsaForCausalLM",
        "KimiK31ForConditionalGeneration",
    )

    @staticmethod
    def supports(server_args):
        cfg = resolving_view(server_args)
        view = resolved_view(server_args)
        architectures = model_config_of(server_args).hf_config.architectures
        is_kimi_k31 = "KimiK31ForConditionalGeneration" in architectures
        # MegaMoE EP over the CP group: each rank owns its CP shard's rows.
        # Pipeline stages exchange CP-local rows (see execute_prefill_cp_bcg).
        kimi_ep = (
            view.ep_size == cfg.tp_size == view.attn_cp_size
            and view.moe_a2a_backend == "megamoe"
            and cfg.moe_dp_size == 1
        )
        return (
            cfg.dp_size == 1
            and (cfg.pp_size == 1 or is_kimi_k31)
            and (not is_kimi_k31 or view.ep_size == 1 or kimi_ep)
            and view.attn_cp_size > 1
            and any(
                arch in architectures
                for arch in InterleaveCPBCGInput._SUPPORTED_ARCHITECTURES
            )
        )

    def _max_local_tokens(self, extend_seq_lens, prefix_lens=None):
        size = get_parallel().attn_cp_size
        if get_parallel().attn_dcp_size > 1:
            return position_layout_max_rank_rows(extend_seq_lens, prefix_lens, size)
        return (sum(int(length) for length in extend_seq_lens) + size - 1) // size

    def model_positions(self, forward_batch):
        return self.positions[: self.live_local_tokens]

    def prepare(self, runner, forward_batch, *, static_num_tokens, capture):
        global_positions = forward_batch.positions
        out_cache_loc = forward_batch.out_cache_loc
        super().prepare(
            runner, forward_batch, static_num_tokens=static_num_tokens, capture=capture
        )
        metadata = forward_batch.attn_cp_metadata
        # KDA projections/reorders retain the captured shape. Eager KPool
        # compression, KV writes and the output gather use the live lengths
        # produced by prepare_cp_forward, never the bucket's padded tail.
        metadata.graph_num_tokens = static_num_tokens
        if metadata.gather_index is not None:
            # super().prepare widened every shard to the captured rows, so the
            # rank-major slots must use that stride before anything reads them.
            metadata.gather_index = restride_gather_index(
                metadata.gather_index,
                old_stride=physical_rank_len(metadata.per_rank_logical_token),
                new_stride=metadata.per_rank_actual_token[0],
            )
        self.moe_local_token_count.fill_(
            metadata.per_rank_logical_token[get_parallel().attn_cp_rank]
        )
        metadata.moe_local_token_count = self.moe_local_token_count
        forward_batch.out_cache_loc = out_cache_loc
        # KPool compresses consecutive global tokens; only the model argument
        # (DSA RoPE) is local. Do not replace ForwardBatch's global positions.
        forward_batch.positions = global_positions
        ids = cp_interleave_input_ids(
            forward_batch.input_ids[: forward_batch.extend_num_tokens], forward_batch
        )
        self.input_ids_global[: ids.shape[0]].copy_(ids)
        forward_batch.input_ids_global = self.input_ids_global[: ids.shape[0]]
        if metadata.gather_index is not None:
            # After the input-id reorder above, which reads the live index.
            graph_index = self.graph_gather_index[:static_num_tokens]
            fill_graph_gather_index(
                graph_index,
                metadata.gather_index,
                rank_rows=metadata.per_rank_logical_token,
                stride=metadata.per_rank_actual_token[0],
            )
            metadata.graph_gather_index = graph_index


def cp_interleave_indexer_rows(forward_batch):
    metadata = forward_batch.attn_cp_metadata
    rank = get_parallel().attn_cp_rank
    logical = metadata.per_rank_logical_token or metadata.per_rank_actual_token
    return metadata.per_rank_actual_token[rank], logical[rank]
