"""Kimi-K3's optional o_proj + reduce-scatter + prefix-add and all-gather +
qkv_proj adapters."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.runtime_context import get_disagg, get_exec, get_forward, get_lora

if TYPE_CHECKING:
    from sglang.kernels.ops.communication import ag_gemm
    from sglang.kernels.ops.communication.gemm_rs_prefix_add import DispatchTable
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)


class FusedOProjOutput(NamedTuple):
    output: torch.Tensor
    shard_start_row: int


class FusedQKVProjOutput(NamedTuple):
    # [M, N] qkv_proj output and the [M, K] all-gathered input it consumed.
    output: torch.Tensor
    hidden_states: torch.Tensor


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


_CURRENT_QKV_PROJ_FUSION_OUTPUT: ContextVar[Optional[FusedQKVProjOutput]] = ContextVar(
    "k3_sp_qkv_proj_fusion_output", default=None
)


@contextmanager
def use_qkv_proj_fusion_output(fused_output: FusedQKVProjOutput):
    """Hand the decoder's fused AG + GEMM result to qkv_proj for the enclosed
    attention call; ``try_fused_qkv_proj`` returns it instead of the GEMM."""
    token = _CURRENT_QKV_PROJ_FUSION_OUTPUT.set(fused_output)
    try:
        yield
    finally:
        _CURRENT_QKV_PROJ_FUSION_OUTPUT.reset(token)


def _load_fusion_ops(feature: str, op_names: tuple[str, ...]) -> bool:
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
            "%s disabled; unsupported execution configuration: %s",
            feature,
            ", ".join(unsupported_runtime_features),
        )
        return False

    try:
        # Importing the package loads its extension, which registers
        # torch.ops.mega_ops in this TP worker.
        import mega_ops  # noqa: F401
    except ImportError as exc:
        logger.warning("%s disabled; cannot import mega_ops: %s", feature, exc)
        return False
    missing_ops = [name for name in op_names if not hasattr(torch.ops.mega_ops, name)]
    if missing_ops:
        logger.warning(
            "%s disabled; torch.ops.mega_ops is missing: %s",
            feature,
            ", ".join(missing_ops),
        )
        return False
    return True


def _device_name(group: GroupCoordinator) -> str:
    return torch.cuda.get_device_name(group.device).replace(" ", "_").replace("/", "_")


def create_gemm_fusion_manager(
    group: GroupCoordinator, hidden_size: int
) -> Optional[K3SPGemmFusionManager]:
    if not _load_fusion_ops(
        "K3 SP GEMM fusion",
        (
            "sm100_device_uuid",
            "sm100_check_peer_access",
            "sm100_resident_cluster_capacity",
            "sm100_gemm_rs_prefix_add_pull",
            "sm100_gemm_rs_prefix_add_push",
        ),
    ):
        return None

    from sglang.kernels.ops.communication.gemm_rs_prefix_add import (
        load_dispatch_table,
    )

    device_name = _device_name(group)
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
            if m <= max_m and dispatch.mode != "baseline"
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

    def _dispatch_for(self, m: int):
        dispatch = self._prepared_dispatch.get(m)
        if dispatch is not None:
            return dispatch
        if self.gemm_rs_operator is None:
            return None
        dispatch = self.dispatch_table.select(m)
        # Skip stream capture and baseline path.
        if dispatch is None or torch.cuda.is_current_stream_capturing():
            return None
        with torch.inference_mode(False), torch.cuda.device(self.device):
            self.gemm_rs_operator.prepare(m, self.dispatch_table.k, dispatch)
        self._prepared_dispatch[m] = dispatch
        return dispatch

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
        num_rows = fusion_call.num_global_tokens
        if (
            m > num_rows
            or k != self.dispatch_table.k
            or attention_output.dtype != torch.bfloat16
            or not attention_output.is_contiguous()
        ):
            return None
        dispatch = self._dispatch_for(num_rows)
        if dispatch is None:
            return None
        num_local_tokens = num_rows // self.group.world_size
        prefix_sum = fusion_call.prefix_sum
        if prefix_sum is None:
            prefix_sum = self.zero_prefix_buffer[:num_local_tokens]
        elif prefix_sum.dtype != torch.bfloat16 or not prefix_sum.is_contiguous():
            return None
        if m < num_rows:
            # Padded M to the next multiple of world_size for the fused kernel.
            padded = attention_output.new_zeros(num_rows, k)
            padded[:m] = attention_output
            attention_output = padded
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


def create_ag_gemm_fusion_manager(
    group: GroupCoordinator, hidden_size: int
) -> Optional[K3SPAgGemmFusionManager]:
    if not _load_fusion_ops(
        "K3 SP AG GEMM fusion",
        (
            "sm100_device_uuid",
            "sm100_check_peer_access",
            "sm100_ag_gemm_warp_pull_resident_cluster_capacity",
            "sm100_ag_gemm_warp_pull",
        ),
    ):
        return None
    return K3SPAgGemmFusionManager(group, hidden_size)


class K3SPAgGemmFusionManager:
    def __init__(self, group: GroupCoordinator, hidden_size: int):
        self.group = group
        self.hidden_size = hidden_size
        self.device = torch.device(group.device)
        # qkv_proj N -> DispatchTable; None caches a missing table.
        self.dispatch_tables: dict[int, Optional[ag_gemm.DispatchTable]] = {}
        self.ag_gemm_operator = None
        # (M, N) -> DispatchConfig for every shape prepared on the operator.
        self._prepared_dispatch = {}

    def supports(self, qkv_proj) -> bool:
        from sglang.srt.layers.linear import (
            MergedColumnParallelLinear,
            ReplicatedLinear,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        return (
            type(qkv_proj) in (MergedColumnParallelLinear, ReplicatedLinear)
            and type(qkv_proj.quant_method) is UnquantizedLinearMethod
            and not qkv_proj.skip_bias_add
            and qkv_proj.bias is None
            and qkv_proj.scheme is None
            and qkv_proj.input_size == self.hidden_size
            and qkv_proj.weight.dtype == torch.bfloat16
        )

    def register(self, qkv_proj) -> bool:
        """Load the table for this projection's N; layers register at build time."""
        if not self.supports(qkv_proj):
            return False
        n = qkv_proj.weight.shape[0]
        if n not in self.dispatch_tables:
            from sglang.kernels.ops.communication.ag_gemm import load_dispatch_table

            device_name = _device_name(self.group)
            self.dispatch_tables[n] = load_dispatch_table(
                self.group.world_size, self.hidden_size, n, device_name
            )
            if self.dispatch_tables[n] is None:
                logger.warning(
                    "K3 SP AG GEMM fusion has no table for TP%d N=%d on %s; "
                    "using native all-gather + qkv_proj.",
                    self.group.world_size,
                    n,
                    device_name,
                )
        return self.dispatch_tables[n] is not None

    def prepare(self, max_m: int) -> None:
        dispatch_configs = {
            (m, n): dispatch
            for n, dispatch_table in self.dispatch_tables.items()
            if dispatch_table is not None
            for m, dispatch in dispatch_table.configs.items()
            if m <= max_m and dispatch.mode != "baseline"
        }
        if not dispatch_configs:
            return

        from sglang.kernels.ops.communication.ag_gemm import AgGemm

        max_fused_m = max(m for m, _ in dispatch_configs)
        with torch.inference_mode(False), torch.cuda.device(self.device):
            self.ag_gemm_operator = AgGemm(
                self.group.cpu_group,
                self.group.rank_in_group,
                self.group.world_size,
                max_fused_m,
                self.hidden_size,
                self.device,
                {dispatch.kernel for dispatch in dispatch_configs.values()},
            )
            for (m, n), dispatch in sorted(dispatch_configs.items()):
                self.ag_gemm_operator.prepare(m, n, dispatch)
            torch.cuda.current_stream(self.device).synchronize()
        self._prepared_dispatch = dispatch_configs
        logger.info(
            "K3 SP AG GEMM fusion prepared (TP%d, (M, N)=%s)",
            self.group.world_size,
            sorted(dispatch_configs),
        )

    def _dispatch_for(self, m: int, n: int):
        dispatch = self._prepared_dispatch.get((m, n))
        if dispatch is not None:
            return dispatch
        if self.ag_gemm_operator is None or m > self.ag_gemm_operator.layout.max_m:
            return None
        dispatch_table = self.dispatch_tables.get(n)
        dispatch = None if dispatch_table is None else dispatch_table.select(m)
        # Skip stream capture and baseline path.
        if dispatch is None or torch.cuda.is_current_stream_capturing():
            return None
        with torch.inference_mode(False), torch.cuda.device(self.device):
            self.ag_gemm_operator.prepare(m, n, dispatch)
        self._prepared_dispatch[m, n] = dispatch
        return dispatch

    def local_input_shard(
        self, qkv_proj, num_shard_tokens: int
    ) -> Optional[torch.Tensor]:
        if get_forward().sp_active:
            return None
        m = num_shard_tokens * self.group.world_size
        if self._dispatch_for(m, qkv_proj.weight.shape[0]) is None:
            return None
        return self.ag_gemm_operator.local_input_shard(m)

    def fused_qkv_proj(self, qkv_proj, local_input: torch.Tensor) -> FusedQKVProjOutput:
        m = local_input.shape[0] * self.group.world_size
        n = qkv_proj.weight.shape[0]
        dispatch = self._prepared_dispatch[m, n]
        output = torch.empty((m, n), device=self.device, dtype=local_input.dtype)
        self.ag_gemm_operator(local_input, qkv_proj.weight, output, dispatch)
        return FusedQKVProjOutput(output, self.ag_gemm_operator.gathered_input(m))

    def try_fused_qkv_proj(self, hidden_states: torch.Tensor):
        fused_output = _CURRENT_QKV_PROJ_FUSION_OUTPUT.get()
        # Only the gathered input itself, or the real-row prefix attention
        # trims it to, was multiplied by the fused kernel.
        if (
            fused_output is None
            or hidden_states.data_ptr() != fused_output.hidden_states.data_ptr()
            or hidden_states.shape[0] > fused_output.output.shape[0]
        ):
            return None
        return fused_output.output[: hidden_states.shape[0]], None
