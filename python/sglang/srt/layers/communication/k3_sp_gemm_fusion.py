"""Kimi-K3's optional o_proj + reduce-scatter + prefix-add adapter."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.runtime_context import get_disagg, get_exec, get_forward, get_lora

if TYPE_CHECKING:
    from sglang.kernels.ops.communication.gemm_rs_prefix_add import DispatchTable
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)


class FusedOProjOutput(NamedTuple):
    output: torch.Tensor
    shard_start_row: int


@dataclass
class OProjFusionCall:
    prefix_sum: Optional[torch.Tensor]
    num_global_tokens: int
    # Filled only when o_proj executes GEMM + RS + prefix; None means native.
    fused_output: Optional[FusedOProjOutput] = None


_CURRENT_O_PROJ_FUSION_CALL: ContextVar[Optional[OProjFusionCall]] = ContextVar(
    "k3_sp_o_proj_fusion_call", default=None
)


@contextmanager
def use_o_proj_fusion_call(
    prefix_sum: Optional[torch.Tensor],
    num_global_tokens: int,
):
    """Pass the decoder's prefix to o_proj for the enclosed attention call.

    ``try_fused_o_proj`` fills ``fused_output`` when fusion runs; the decoder
    reads it after the block to skip the native RS and prefix add.
    """
    fusion_call = OProjFusionCall(prefix_sum, num_global_tokens)
    token = _CURRENT_O_PROJ_FUSION_CALL.set(fusion_call)
    try:
        yield fusion_call
    finally:
        _CURRENT_O_PROJ_FUSION_CALL.reset(token)


def create_gemm_fusion_manager(
    group: GroupCoordinator, hidden_size: int
) -> Optional[K3SPGemmFusionManager]:
    execution_config = get_exec()
    unsupported_runtime_features = [
        name
        for name, enabled in (
            ("PDMux", get_disagg().enable_pdmux),
            ("two-batch overlap", execution_config.overlap.enable_two_batch_overlap),
            (
                "single-batch overlap",
                execution_config.overlap.enable_single_batch_overlap,
            ),
            ("torch.compile", execution_config.graph.enable_torch_compile),
            ("LoRA", get_lora().enable_lora or get_lora().lora_paths),
        )
        if enabled
    ]
    if unsupported_runtime_features:
        logger.warning(
            "K3 SP GEMM fusion disabled; unsupported execution configuration: %s",
            ", ".join(unsupported_runtime_features),
        )
        return None

    # The image registers the SM100 operators; this module never loads binaries.
    missing_ops = [
        name
        for name in (
            "sm100_device_uuid",
            "sm100_check_peer_access",
            "sm100_resident_cluster_capacity",
            "sm100_gemm_rs_prefix_add_pull",
            "sm100_gemm_rs_prefix_add_push",
        )
        if not hasattr(torch.ops.mega_ops, name)
    ]
    if missing_ops:
        logger.warning(
            "K3 SP GEMM fusion disabled; torch.ops.mega_ops is missing: %s",
            ", ".join(missing_ops),
        )
        return None

    from sglang.kernels.ops.communication.gemm_rs_prefix_add import (
        load_dispatch_table,
    )

    device_name = (
        torch.cuda.get_device_name(group.device).replace(" ", "_").replace("/", "_")
    )
    dispatch_table = load_dispatch_table(group.world_size, hidden_size, device_name)
    if dispatch_table is None:
        logger.warning(
            "K3 SP GEMM fusion has no table for TP%d on %s; using native o_proj.",
            group.world_size,
            device_name,
        )
        return None
    return K3SPGemmFusionManager(group, dispatch_table, hidden_size)


class K3SPGemmFusionManager:
    def __init__(
        self, group: GroupCoordinator, dispatch_table: DispatchTable, hidden_size: int
    ):
        self.group = group
        self.dispatch_table = dispatch_table
        self.hidden_size = hidden_size
        self.device = torch.device(group.device)
        self.gemm_rs_operator = None
        self.zero_prefix_buffer = None
        # M -> DispatchConfig for every shape prepared on the operator.
        self._prepared_dispatch = {}

    def prepare(self, max_m: int) -> None:
        """Allocate the workspace, i.e. the symmetric memory."""
        dispatch_configs = {
            m: dispatch
            for m, dispatch in self.dispatch_table.configs.items()
            if m <= max_m
        }
        if not dispatch_configs:
            return

        from sglang.kernels.ops.communication.gemm_rs_prefix_add import (
            GemmRSPrefixAdd,
        )

        max_fused_m = max(dispatch_configs)
        k = self.dispatch_table.k
        with torch.inference_mode(False), torch.cuda.device(self.device):
            self.gemm_rs_operator = GemmRSPrefixAdd(
                self.group.cpu_group,
                self.group.rank_in_group,
                self.group.world_size,
                max_fused_m,
                self.hidden_size,
                self.device,
                {dispatch.kernel for dispatch in dispatch_configs.values()},
            )
            for m, dispatch in sorted(dispatch_configs.items()):
                self.gemm_rs_operator.prepare(m, k, dispatch)
            self.zero_prefix_buffer = torch.zeros(
                (max_fused_m // self.group.world_size, self.hidden_size),
                dtype=torch.bfloat16,
                device=self.device,
            )
            torch.cuda.current_stream(self.device).synchronize()
        self._prepared_dispatch = dispatch_configs
        logger.info(
            "K3 SP GEMM fusion prepared (TP%d, M=%s)",
            self.group.world_size,
            sorted(dispatch_configs),
        )

    def supports(self, o_proj) -> bool:
        from sglang.srt.layers.linear import RowParallelLinear
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        return (
            type(o_proj) is RowParallelLinear
            and type(o_proj.quant_method) is UnquantizedLinearMethod
            and o_proj.input_is_parallel
            and not o_proj.skip_bias_add
            and o_proj.bias is None
            and o_proj.scheme is None
            and o_proj.tp_size == self.group.world_size
            and o_proj.output_size == self.hidden_size
        )

    def try_fused_o_proj(self, o_proj, attention_output: torch.Tensor, **kwargs):
        fusion_call = _CURRENT_O_PROJ_FUSION_CALL.get()
        if (
            fusion_call is None
            or fusion_call.fused_output is not None
            or any(key != "forward_batch" for key in kwargs)
            or get_forward().sp_active
            or o_proj.use_decode_attn_tp
        ):
            return None
        m, k = attention_output.shape
        dispatch = self._prepared_dispatch.get(m)
        if (
            dispatch is None
            or m != fusion_call.num_global_tokens
            or k != self.dispatch_table.k
            or attention_output.dtype != torch.bfloat16
            or not attention_output.is_contiguous()
        ):
            return None
        num_local_tokens = m // self.group.world_size
        prefix_sum = fusion_call.prefix_sum
        if prefix_sum is None:
            prefix_sum = self.zero_prefix_buffer[:num_local_tokens]
        elif prefix_sum.dtype != torch.bfloat16 or not prefix_sum.is_contiguous():
            return None
        output = torch.empty(
            (num_local_tokens, self.hidden_size),
            device=self.device,
            dtype=attention_output.dtype,
        )
        # Directly writes the output tensor to avoid redundant copy.
        self.gemm_rs_operator(
            attention_output, o_proj.weight, prefix_sum, output, dispatch
        )
        fusion_call.fused_output = FusedOProjOutput(
            output, self.group.rank_in_group * num_local_tokens
        )
        return output, None
