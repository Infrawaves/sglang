"""SM90 mega_ops fusion of the TP layer boundary into its neighbouring GEMMs.

GemmRSNormAG runs a row-parallel linear (o_proj or down_proj) together with
the reduce-scatter, residual add, RMSNorm and multimem all-gather after it.
RsNormAgGemm runs the reduce-scatter, residual add, RMSNorm and all-gather
together with the column-parallel linear after them (gate_up_proj or qkv_proj).

Dense (Qwen3, Llama): attn -> FFN is RsNormAgGemm(gate_up), FFN -> next attn
is GemmRSNormAG(down_proj, next input_layernorm). Qwen3-MoE: attn -> FFN is
GemmRSNormAG(o_proj, post_attention_layernorm), FFN -> next attn is
RsNormAgGemm(qkv_proj). Once a forward fuses, the residual is this rank's
[m_pad / tp, hidden] shard until the last layer's FFN input gathers it back.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary.contracts import FusedMlpInput
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.layers.layer_boundary.output import UnreducedOutput
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_lora,
    get_parallel,
    get_spec,
    max_prefill_buffer_tokens,
)
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_DENSE_ARCHITECTURES = frozenset({"Qwen3ForCausalLM", "LlamaForCausalLM"})
_MOE_ARCHITECTURES = frozenset({"Qwen3MoeForCausalLM"})
# The world sizes of TP group.
_SUPPORTED_TP_SIZES = (2, 4, 8)
# mega_ops kernel shape rules: comm n-tile is 256 columns, K-tile is 64, N-tile 128.
_HIDDEN_ALIGN = 256
_K_ALIGN = 64
_N_ALIGN = 128

_O_PROJ_TARGET: ContextVar[Optional[torch.Tensor]] = ContextVar(
    "sm90_gemm_norm_o_proj_target", default=None
)


def _architecture(config) -> Optional[str]:
    architectures = config.architectures
    return architectures[0] if architectures else None


def make_sm90_gemm_norm_fusion(
    config, *, layer_id: int
) -> Optional[Sm90GemmNormFusion]:
    """The boundary provider for one Qwen3 decoder layer, or None when off."""
    architecture = _architecture(config)
    if not envs.SGLANG_ENABLE_SM90_GEMM_NORM_FUSION.get() or architecture not in (
        _DENSE_ARCHITECTURES | _MOE_ARCHITECTURES
    ):
        return None
    return Sm90GemmNormFusion(
        sparse=architecture in _MOE_ARCHITECTURES,
        first=layer_id == 0,
        terminal=layer_id == config.num_hidden_layers - 1,
    )


class _RowOutput(msgspec.Struct, frozen=True):
    # GemmRSNormAG: full normed [m, hidden] and this rank's updated residual shard.
    normed: torch.Tensor
    residual: torch.Tensor


class _ColOutput(msgspec.Struct, frozen=True):
    # RsNormAgGemm: full normed [m, hidden] (its input) and the GEMM output [m, n].
    normed: torch.Tensor
    output: torch.Tensor


class _Partial(msgspec.Struct, frozen=True):
    # A row-parallel GEMM written into RsNormAgGemm's input buffer, unreduced.
    value: torch.Tensor
    buffer: torch.Tensor


class _Service:
    """The two shared mega_ops kernels and the handoffs between call sites."""

    def __init__(
        self, *, sparse: bool, hidden_size: int, row_k: int, col_n: int, eps: float
    ):
        import mega_ops

        group = get_parallel().tp_group
        self.group = group
        self.world_size = group.world_size
        self.max_m = ceil_align(max_prefill_buffer_tokens(), self.world_size)
        kwargs = dict(
            group_name=group.device_group.group_name,
            rank=group.rank_in_group,
            world_size=self.world_size,
            max_m=self.max_m,
            device=group.device,
            eps=eps,
        )
        # Both allocate symmetric memory collectively; keep the order on every rank.
        with torch.inference_mode(False), torch.cuda.device(group.device):
            self.row = mega_ops.GemmRSNormAG(n=hidden_size, **kwargs)
            self.col = mega_ops.RsNormAgGemm(hidden_size=hidden_size, n=col_n, **kwargs)
            torch.cuda.current_stream(group.device).synchronize()
        self._load_chunk_tables(hidden_size=hidden_size, row_k=row_k, col_n=col_n)
        self.sparse = sparse
        self.pending_partial: Optional[_Partial] = None
        self.pending_row: Optional[_RowOutput] = None
        self.pending_col: Optional[_ColOutput] = None

    def _load_chunk_tables(self, *, hidden_size: int, row_k: int, col_n: int) -> None:
        """Install the tuned per-m RS chunk tables of this TP size, shape and GPU.

        The tables are keyed by exact m_pad; any other m keeps the kernel's fixed grid.
        """
        import mega_ops
        from mega_ops.tile_order import build_physical_tile_m_order

        config_dir = envs.SGLANG_SM90_GEMM_NORM_CONFIG_DIR.get() or os.path.join(
            os.path.dirname(mega_ops.__file__), "configs"
        )
        device_name = torch.cuda.get_device_name(self.group.device).replace(" ", "_")
        world_size, rank = self.world_size, self.group.rank_in_group
        suffix = f"device_name={device_name}.json"
        for kernel, name, path in (
            (self.row, "gemm_rs_norm_ag", f"world={world_size},H={hidden_size},K={row_k},{suffix}"),
            (self.col, "rs_norm_ag_gemm", f"world={world_size},H={hidden_size},N={col_n},{suffix}"),
        ):
            path = os.path.join(config_dir, name, path)
            if not os.path.exists(path):
                logger.info("SM90 GEMM norm fusion: no chunk tables at %s", path)
                continue
            with open(path) as f:
                spec = json.load(f)
            match = spec["match"]
            loaded = []
            for m_text, entry in spec["configs"].items():
                m = int(m_text)
                if m > self.max_m or m % world_size:
                    continue
                if kernel is self.row:
                    tile_n, cluster_m = match["tile_shape"][1], match["cluster_m"]
                    chosen = kernel.config_for(m)
                    if chosen is None or chosen[1:] != (tile_n, cluster_m):
                        logger.warning(
                            "SM90 GEMM norm fusion: skip %s m=%d, tuned for "
                            "tile_n=%d cluster_m=%d but dispatches %s",
                            name, m, tile_n, cluster_m, chosen,
                        )
                        continue
                table = torch.tensor(entry["chunk_table"][str(rank)], dtype=torch.int32)
                kernel.set_chunk_table(m, table.view(-1, 4))
                loaded.append(m)
            if kernel is self.col and loaded and match.get("physical_tile_m_order") == "common":
                # These tables were tuned with the common M-tile order, not the default rotated one.
                native_order = kernel._physical_tile_m_order_for
                common = {}

                def order_for(m, tile_m, _native=native_order, _ms=frozenset(loaded)):
                    if m not in _ms:
                        return _native(m, tile_m)
                    if (m, tile_m) not in common:
                        common[(m, tile_m)] = torch.tensor(
                            build_physical_tile_m_order(m, tile_m, world_size),
                            dtype=torch.int32,
                            device=self.group.device,
                        )
                    return common[(m, tile_m)]

                kernel._physical_tile_m_order_for = order_for
            logger.info(
                "SM90 GEMM norm fusion: %s chunk tables for m=%s from %s", name, loaded, path
            )

    def active(self, forward_batch: ForwardBatch) -> Optional[int]:
        """m_pad when this forward fuses every boundary, else None.

        Reads only the batch's token count and mode, so every layer and rank agrees.
        """
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        if (
            forward_batch.forward_mode != ForwardMode.EXTEND
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        m = forward_batch.input_ids.shape[0]
        if m == 0:
            return None
        m_pad = ceil_align(m, self.world_size)
        if m_pad > self.max_m or self.row.config_for(m_pad) is None:
            return None
        return m_pad

    def shard(self, residual: torch.Tensor, m_pad: int) -> torch.Tensor:
        """This rank's rows of a full [m, hidden] residual, zero-padded to m_pad."""
        rows = m_pad // self.world_size
        begin = self.group.rank_in_group * rows
        shard = residual.new_zeros((rows, residual.shape[1]))
        valid = max(0, min(rows, residual.shape[0] - begin))
        if valid:
            shard[:valid] = residual[begin : begin + valid]
        return shard

    def gather(self, shard: torch.Tensor, m: int) -> torch.Tensor:
        full = shard.new_empty((shard.shape[0] * self.world_size, shard.shape[1]))
        self.group.all_gather_into_tensor(full, shard)
        return full[:m]

    def run_row(
        self, x: torch.Tensor, linear, norm, residual: torch.Tensor, m_pad: int
    ) -> _RowOutput:
        m = x.shape[0]
        if m != m_pad:
            padded = x.new_zeros((m_pad, x.shape[1]))
            padded[:m] = x
            x = padded
        normed = self.row.forward(
            input=x, weight=linear.weight, norm_weight=norm.weight, residual=residual
        )
        return _RowOutput(normed=normed[:m], residual=residual)

    def col_input(self, *, m: int, m_pad: int) -> torch.Tensor:
        """RsNormAgGemm's [m_pad, hidden] input; the caller writes rows [:m]."""
        buffer = self.col.input_buffer(m_pad)
        if m != m_pad:
            buffer[m:].zero_()
        return buffer

    def run_col(
        self, buffer: torch.Tensor, linear, norm, residual: torch.Tensor, m: int
    ) -> _ColOutput:
        output = self.col.forward(
            partial=buffer,
            residual=residual,
            norm_weight=norm.weight,
            weight=linear.weight,
        )
        return _ColOutput(normed=buffer[:m], output=output[:m])

    def put_partial(self, value: torch.Tensor, buffer: torch.Tensor) -> None:
        if self.pending_partial is not None:
            raise RuntimeError(
                "SM90 GEMM norm fusion: an o_proj output was not consumed"
            )
        self.pending_partial = _Partial(value=value, buffer=buffer)

    def take_partial(self, value: torch.Tensor) -> Optional[torch.Tensor]:
        pending = self.pending_partial
        if pending is None:
            return None
        self.pending_partial = None
        if pending.value is not value:
            raise RuntimeError(
                "SM90 GEMM norm fusion: FFN input is not the o_proj output"
            )
        return pending.buffer

    def put_row(self, row: _RowOutput) -> None:
        if self.pending_row is not None:
            raise RuntimeError("SM90 GEMM norm fusion: a fused norm was not consumed")
        self.pending_row = row

    def take_row(self, normed: torch.Tensor) -> Optional[_RowOutput]:
        pending = self.pending_row
        if pending is None:
            return None
        self.pending_row = None
        if pending.normed is not normed:
            raise RuntimeError(
                "SM90 GEMM norm fusion: boundary input is not the fused norm"
            )
        return pending

    def put_col(self, col: _ColOutput) -> None:
        # A stale output can only match its own normed tensor; overwriting is safe.
        self.pending_col = col

    def take_col(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        pending = self.pending_col
        if pending is None or pending.normed is not x:
            return None
        self.pending_col = None
        return pending.output


class Sm90GemmNormFusion:
    """One Qwen3 layer's boundary candidates; inert until installed."""

    def __init__(self, *, sparse: bool, first: bool, terminal: bool):
        self.sparse = sparse
        # Layer 0 receives the full embedding residual.
        self.first = first
        # The last layer gathers the residual back for the native final norm.
        self.terminal = terminal
        self.service: Optional[_Service] = None
        # Dense: this layer's gate_up_proj. MoE: this layer's qkv_proj.
        self.col_linear = None

    def install(self, service: _Service, *, col_linear) -> None:
        self.service = service
        self.col_linear = col_linear

    def attention_input(self, plan) -> tuple:
        return (partial(self._attention_input, plan.norm),)

    def ffn_input(self, plan) -> tuple:
        return (
            FusedMlpInput(
                completes=SumGroup.ATTN_TP, run=partial(self._ffn_input, plan.norm)
            ),
        )

    def can_defer_finalize(self, plan, forward_batch: ForwardBatch) -> bool:
        return False

    def can_defer_all_reduce(self, plan, forward_batch: ForwardBatch) -> bool:
        return False

    def _ffn_input(self, norm, hidden_states, residual, forward_batch):
        service = self.service
        if service is None:
            return None
        if self.sparse:
            row = service.take_row(hidden_states)
            if row is None:
                return None
            normed, residual = row.normed, row.residual
        else:
            buffer = service.take_partial(hidden_states)
            if buffer is None:
                return None
            m = hidden_states.shape[0]
            if self.first:
                residual = service.shard(residual, buffer.shape[0])
            col = service.run_col(buffer, self.col_linear, norm, residual, m)
            service.put_col(col)
            normed = col.normed
        if self.terminal:
            residual = service.gather(residual, normed.shape[0])
        return normed, residual

    def _attention_input(
        self, norm, owed, residual, forward_batch, post_residual_addition
    ):
        service = self.service
        if service is None or not isinstance(owed, UnreducedOutput):
            return None
        m_pad = service.active(forward_batch)
        if m_pad is None:
            return None
        if post_residual_addition is not None:
            raise RuntimeError(
                "SM90 GEMM norm fusion cannot add post_residual_addition"
            )
        if self.sparse:
            m = owed.partial.shape[0]
            buffer = service.col_input(m=m, m_pad=m_pad)
            buffer[:m].copy_(owed.partial)
            col = service.run_col(buffer, self.col_linear, norm, residual, m)
            service.put_col(col)
            return col.normed, residual
        row = service.take_row(owed.partial)
        if row is None:
            raise RuntimeError("SM90 GEMM norm fusion: down_proj did not fuse its norm")
        return row.normed, row.residual



def _install_col_forward(linear, service: _Service) -> None:
    """Return the fused GEMM output when the input is the normed tensor it consumed."""
    native_forward = linear.forward

    def forward(x, *args, **kwargs):
        if not args and not kwargs:
            output = service.take_col(x)
            if output is not None:
                return output, None
        return native_forward(x, *args, **kwargs)

    linear.forward = forward


def _o_proj_into(native_forward, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    output, _ = native_forward(x, skip_all_reduce=True, output_tensor=target)
    if output.data_ptr() != target.data_ptr():
        raise RuntimeError("SM90 GEMM norm fusion: o_proj did not write its buffer")
    return output


def _install_partial_o_proj(o_proj, service: _Service) -> None:
    """Qwen3 dense: write o_proj into RsNormAgGemm's input buffer, unreduced."""
    native_forward = o_proj.forward

    def forward(x, *args, forward_batch=None, **kwargs):
        m_pad = None if forward_batch is None else service.active(forward_batch)
        if m_pad is None or args or kwargs:
            return native_forward(x, *args, forward_batch=forward_batch, **kwargs)
        m = x.shape[0]
        buffer = service.col_input(m=m, m_pad=m_pad)
        output = _o_proj_into(native_forward, x, buffer[:m])
        service.put_partial(output, buffer)
        return output, None

    o_proj.forward = forward


def _install_targeted_o_proj(o_proj) -> None:
    """Llama: write o_proj into the buffer the fused decoder branch set, unreduced."""
    native_forward = o_proj.forward

    def forward(x, *args, **kwargs):
        target = _O_PROJ_TARGET.get()
        if target is None:
            return native_forward(x, *args, **kwargs)
        return _o_proj_into(native_forward, x, target), None

    o_proj.forward = forward


@contextmanager
def o_proj_output_into(target: torch.Tensor):
    token = _O_PROJ_TARGET.set(target)
    try:
        yield
    finally:
        _O_PROJ_TARGET.reset(token)


def _install_row_forward(
    linear, service: _Service, *, norm, shards_residual: bool, deferred_sum: bool
) -> None:
    """Run GemmRSNormAG in place of the linear and hand the norm to the next boundary."""
    from sglang.srt.layers.moe import should_skip_mlp_all_reduce

    native_forward = linear.forward

    def forward(x, *args, forward_batch=None, **kwargs):
        m_pad = None if forward_batch is None else service.active(forward_batch)
        if m_pad is None:
            return native_forward(x, *args, forward_batch=forward_batch, **kwargs)
        if args or kwargs or (deferred_sum and not should_skip_mlp_all_reduce()):
            raise RuntimeError("SM90 GEMM norm fusion: unexpected row-linear call")
        residual = residual_batch.current(forward_batch).residual
        if shards_residual:
            residual = service.shard(residual, m_pad)
        row = service.run_row(x, linear, norm, residual, m_pad)
        service.put_row(row)
        return row.normed, None

    linear.forward = forward


def install_sm90_gemm_norm_fusion(model) -> None:
    """Install on a Qwen3 / Qwen3-MoE / Llama ForCausalLM; a no-op otherwise.

    Runs before warmup and CUDA graph capture: it allocates symmetric memory.
    """
    architecture = _architecture(model.config)
    if not envs.SGLANG_ENABLE_SM90_GEMM_NORM_FUSION.get() or architecture not in (
        _DENSE_ARCHITECTURES | _MOE_ARCHITECTURES
    ):
        return
    sparse = architecture in _MOE_ARCHITECTURES
    layers = model.model.layers
    reason = _runtime_unsupported_reason() or _layer_unsupported_reason(
        layers, architecture=architecture, hidden_size=model.config.hidden_size
    )
    if reason is not None:
        logger.warning("SM90 GEMM norm fusion disabled; %s", reason)
        return
    col_linear = layers[0].self_attn.qkv_proj if sparse else layers[0].mlp.gate_up_proj
    row_linear = layers[0].self_attn.o_proj if sparse else layers[0].mlp.down_proj
    service = _Service(
        sparse=sparse,
        hidden_size=model.config.hidden_size,
        row_k=row_linear.weight.shape[1],
        col_n=col_linear.weight.shape[0],
        eps=layers[0].post_attention_layernorm.variance_epsilon,
    )
    if architecture == "LlamaForCausalLM":
        for layer in layers:
            _install_targeted_o_proj(layer.self_attn.o_proj)
        model.model.gemm_norm_fusion = service
    else:
        _install_qwen3(layers, service, sparse=sparse)
    logger.info(
        "SM90 GEMM norm fusion enabled (%s, TP%d, max_m=%d, %d layers)",
        architecture,
        service.world_size,
        service.max_m,
        len(layers),
    )


def _install_qwen3(layers, service: _Service, *, sparse: bool) -> None:
    for index, layer in enumerate(layers):
        fusion = layer.attn_boundary.fusions
        if sparse:
            fusion.install(service, col_linear=layer.self_attn.qkv_proj)
            _install_col_forward(layer.self_attn.qkv_proj, service)
            _install_row_forward(
                layer.self_attn.o_proj,
                service,
                norm=layer.post_attention_layernorm,
                shards_residual=index == 0,
                deferred_sum=False,
            )
            continue
        fusion.install(service, col_linear=layer.mlp.gate_up_proj)
        _install_col_forward(layer.mlp.gate_up_proj, service)
        _install_partial_o_proj(layer.self_attn.o_proj, service)
        if index + 1 < len(layers):
            # Captured by reference: never assign another layer's norm as a submodule.
            _install_row_forward(
                layer.mlp.down_proj,
                service,
                norm=layers[index + 1].input_layernorm,
                shards_residual=False,
                deferred_sum=True,
            )


def _runtime_unsupported_reason() -> Optional[str]:
    from sglang.srt.layers import layernorm_sp
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang.srt.layers.moe import (
        get_moe_a2a_backend,
        post_experts_sum_is_one_all_reduce,
    )
    from sglang.srt.utils.common import get_device_sm

    parallel = get_parallel()
    execution = get_exec()
    checks = (
        (get_device_sm() == 90, "requires SM90"),
        (parallel.tp_size in _SUPPORTED_TP_SIZES, "requires TP in {2, 4, 6, 8}"),
        (parallel.attn_tp_size == parallel.tp_size, "requires attn_tp == tp"),
        (parallel.pp_size == 1, "does not support PP"),
        (parallel.attn_cp_size == 1, "does not support attention CP"),
        (not is_dp_attention_enabled(), "does not support DP attention"),
        (parallel.moe_ep_size == 1, "does not support MoE EP"),
        (get_moe_a2a_backend().is_none(), "does not support a MoE a2a backend"),
        (
            not layernorm_sp.layernorm_sp_enabled(),
            "conflicts with --enable-layernorm-sp",
        ),
        (
            not execution.comm.enable_quant_communications,
            "does not support quant communications",
        ),
        (
            not (get_lora().enable_lora or get_lora().lora_paths),
            "does not support LoRA",
        ),
        (not execution.graph.enable_torch_compile, "does not support torch.compile"),
        (
            not execution.overlap.enable_two_batch_overlap,
            "does not support two-batch overlap",
        ),
        (
            not execution.overlap.enable_single_batch_overlap,
            "does not support single-batch overlap",
        ),
        (not get_disagg().enable_pdmux, "does not support PDMux"),
        (
            get_spec().speculative_algorithm is None,
            "does not support speculative decoding",
        ),
        (
            execution.deterministic.rl_on_policy_target is None,
            "does not support the RL on-policy target",
        ),
        # Non-terminal layers must leave their FFN sum for the next layer's input.
        (post_experts_sum_is_one_all_reduce(), "requires a deferrable FFN all-reduce"),
    )
    for ok, reason in checks:
        if not ok:
            return reason
    try:
        import mega_ops  # noqa: F401
    except ImportError as exc:
        return f"cannot import mega_ops: {exc}"
    missing = [
        name
        for name in ("gemm_rs_norm_ag", "rs_norm_ag_gemm")
        if not hasattr(torch.ops.mega_ops, name)
    ]
    if missing:
        return f"torch.ops.mega_ops is missing {', '.join(missing)}"
    return None


def _is_plain_bf16_linear(linear) -> bool:
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    return (
        type(linear.quant_method) is UnquantizedLinearMethod
        and linear.bias is None
        and linear.weight.dtype == torch.bfloat16
    )


def _layer_unsupported_reason(
    layers, *, architecture: str, hidden_size: int
) -> Optional[str]:
    from sglang.srt.layers.layernorm import RMSNorm

    if hidden_size % _HIDDEN_ALIGN:
        return f"hidden_size {hidden_size} is not a multiple of {_HIDDEN_ALIGN}"
    sparse = architecture in _MOE_ARCHITECTURES
    eps = layers[0].post_attention_layernorm.variance_epsilon
    col_shape = (
        layers[0].self_attn.qkv_proj if sparse else layers[0].mlp.gate_up_proj
    ).weight.shape
    for layer in layers:
        if architecture != "LlamaForCausalLM" and not isinstance(
            layer.attn_boundary.fusions, Sm90GemmNormFusion
        ):
            return "a decoder layer was built without the fusion provider"
        attn = layer.self_attn
        linears = (
            (attn.o_proj, attn.qkv_proj)
            if sparse
            else (attn.o_proj, layer.mlp.gate_up_proj, layer.mlp.down_proj)
        )
        for linear in linears:
            if not _is_plain_bf16_linear(linear):
                return "a fused linear is quantized, biased or not bf16"
            # weight is [N, K].
            n, k = linear.weight.shape
            if k % _K_ALIGN or n % _N_ALIGN:
                return f"linear shape {(n, k)} is not tile-aligned"
        col = attn.qkv_proj if sparse else layer.mlp.gate_up_proj
        if col.weight.shape != col_shape:
            return "the column-parallel linears differ across layers"
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            if (
                type(norm) is not RMSNorm
                or not norm.has_weight
                or norm.weight.dtype != torch.bfloat16
                or norm.variance_epsilon != eps
            ):
                return "every norm must be a bf16 RMSNorm with one epsilon"
    return None
