"""MegaMoE on T-Head PPU via DeepGEMM-for-sail (W4A4 / MXFP4, mma_type fp4xfp4).

Only reached when SGLANG_USE_SAIL is set together with --moe-a2a-backend megamoe.
"""

from __future__ import annotations

import torch

from sglang.srt.environ import envs

_SAIL_MMA_TYPE = "fp4xfp4"

_SAIL_SYMM_BUFFER: dict = {}


def sail_activation(sglang_activation: str) -> str:
    # sglang names gated SiLU "silu"; sail calls it "swiglu". "situ" matches as is.
    return "swiglu" if sglang_activation == "silu" else sglang_activation


def build_sail_mega_moe_weights(layer: torch.nn.Module) -> None:
    import deep_gemm

    activation = sail_activation(layer.moe_runner_config.activation)

    # Contiguous uint8 packed e2m1 [E, N, K/2] with uint8 UE8M0 scales [E, N, K/32]; 
    # w13 rows must be [gate; up].
    l1_pair, l2_pair = deep_gemm.transform_weights_for_mega_moe(
        (layer.w13_weight.data.contiguous(), layer.w13_weight_scale.data.contiguous()),
        (layer.w2_weight.data.contiguous(), layer.w2_weight_scale.data.contiguous()),
        activation=activation,
    )
    layer.mega_l1_weights = l1_pair
    layer.mega_l2_weights = l2_pair
    # Repoint params so the checkpoint layout is freed instead of kept alongside.
    layer.w13_weight.data = l1_pair[0]
    layer.w13_weight_scale.data = l1_pair[1]
    layer.w2_weight.data = l2_pair[0]
    layer.w2_weight_scale.data = l2_pair[1]
    layer._mega_moe_weights_built = True
    layer._mxfp4_backend = "deep_gemm"


def _get_sail_symm_buffer(
    *,
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation: str,
):
    import deep_gemm

    key = (
        id(group),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        activation,
    )
    buf = _SAIL_SYMM_BUFFER.get(key)
    if buf is None:
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            mma_type=_SAIL_MMA_TYPE,
            activation=activation,
        )
        _SAIL_SYMM_BUFFER[key] = buf
    return buf


def forward_sail_mega_experts(
    *,
    experts: torch.nn.Module,
    routed_input: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden: int,
    intermediate_hidden: int,
    top_k: int,
    activation: str,
) -> torch.Tensor:
    import deep_gemm

    from sglang.srt.runtime_context import get_parallel

    num_tokens = routed_input.shape[0]
    num_max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    assert num_tokens <= num_max_tokens_per_rank, (
        f"sail mega MoE: num_tokens={num_tokens} exceeds "
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK={num_max_tokens_per_rank}"
    )
    buf = _get_sail_symm_buffer(
        group=get_parallel().moe_ep_group.device_group,
        num_experts=experts.num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=top_k,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        activation=activation,
    )

    if num_tokens > 0:
        topk_ids_in = topk_ids.to(torch.int64)
        topk_weights_in = topk_weights.to(torch.float32)
    else:
        topk_ids_in = routed_input.new_empty((0, top_k), dtype=torch.int64)
        topk_weights_in = routed_input.new_empty((0, top_k), dtype=torch.float32)

    deep_gemm.mega_moe_pre_dispatch(
        routed_input, topk_ids_in, topk_weights_in, buf, num_tokens=num_tokens
    )
    # Exactly num_tokens rows: sail consumes the input prefix selected by y.shape[0],
    # so a padded row would dispatch a stale route left in the buffer.
    y = torch.empty(
        (num_tokens, hidden), dtype=torch.bfloat16, device=routed_input.device
    )
    deep_gemm.fp4_fp4_mega_moe(
        y,
        experts.mega_l1_weights,
        experts.mega_l2_weights,
        buf,
        recipe=(1, 1, 32),
        activation=activation,
        fast_math=True,
    )
    return y
