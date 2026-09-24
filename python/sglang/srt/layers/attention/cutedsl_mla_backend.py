"""
Attention backend for the flashinfer cute-dsl MLA decode kernels with decode
context parallelism (DCP).

Subclasses :class:`TRTLLMMLABackend` (``backend="cute-dsl"``) to reuse its MLA
data preparation, workspace, and prefill plumbing. The flashinfer cute-dsl
monolithic MLA decode kernel natively accepts cyclic DCP metadata
(``enable_dcp`` / ``cp_world`` / ``cp_rank`` / ``causal_seqlens_kv_global``) and
returns the rank-local ``(out, lse)`` needed by the cross-rank merge in
``deepseek_common/attention_forward_methods/forward_mla.py``.

Non-DCP (``dcp_size == 1``) decode falls through to the base cute-dsl path
unchanged. The DCP metadata helpers live on :class:`TRTLLMMLABackend`; this
module only supplies the cute-dsl kernel call and its decode forward.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows
from sglang.kernels.ops.attention.utils import (
    concat_mla_absorb_q_general,
    mla_quantize_and_rope_for_fp8,
    mla_quantize_without_rope_for_fp8,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.trtllm_mla_backend import (
    _ENABLE_PDL,
    TRTLLMMLABackend,
    TRTLLMMLAMultiStepDraftBackend,
)
from sglang.srt.layers.logits_processor import get_in_autotune_dummy_run
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import (
    get_cuda_graph_max_batch_size,
    get_eager_max_batch_size,
    is_flashinfer_available,
)

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class CuteDslMLABackend(TRTLLMMLABackend):
    """flashinfer cute-dsl MLA decode backend with decode context parallelism.

    SGLANG_CUTEDSL_MLA_NUM_KV_SPLITS optionally overrides the split planner
    for ordinary non-DCP decode; the default keeps the base cute-dsl path.
    """

    # This kernel does not support varlen queries.
    supports_varlen_absorbed_mla = False

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        # Resolve once: graph replay captures this split scalar and workspace.
        self._decode_with_splits = None
        num_splits = envs.SGLANG_CUTEDSL_MLA_NUM_KV_SPLITS.get()
        if num_splits != 0:
            from sglang.srt.layers.attention.cutedsl_mla_splitkv import (
                create_cutedsl_mla_decode_with_splits,
            )

            if get_parallel().dcp_enabled or not model_runner.spec_algorithm.is_none():
                raise ValueError(
                    "SGLANG_CUTEDSL_MLA_NUM_KV_SPLITS currently requires "
                    "DCP size 1 and speculative decoding disabled"
                )
            if envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get() is not None:
                raise ValueError("Fixed CuTeDSL MLA splits do not support skip-softmax")
            self._decode_with_splits = create_cutedsl_mla_decode_with_splits(num_splits)

        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
            backend="cute-dsl",
        )

        if self._decode_with_splits is not None:
            from sglang.srt.layers.attention.cutedsl_mla_splitkv import (
                plan_cutedsl_mla_splits,
            )

            # Same upper bound as the decode graph runner, including alignment
            # padding. Also covers eager batches up to the request-pool capacity.
            capacity = model_runner.req_to_token_pool.size
            max_bs = max(
                get_cuda_graph_max_batch_size(capacity),
                get_eager_max_batch_size(capacity),
            )
            _, required = plan_cutedsl_mla_splits(
                max_bs, 1, self.num_local_heads, self.kv_lora_rank, num_splits
            )
            if required > self.workspace_buffer.numel():
                # Keep the shared upstream buffer intact for other instances.
                # Allocate once before warmup/capture, never in the layer loop.
                self.workspace_buffer = torch.empty(
                    required, dtype=torch.int8, device=model_runner.device
                )
                self.workspace_size = required
            logger.info(
                "CuTeDSL MLA fixed KV splits=%d, max_batch=%d, workspace=%.2f MiB "
                "(short contexts may use fewer nonempty splits)",
                num_splits,
                max_bs,
                self.workspace_buffer.numel() / 1024**2,
            )

    # ------------------------------------------------------------------
    # Kernel + decode forward.
    # ------------------------------------------------------------------
    def _run_decode_kernel(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        layer: RadixAttention,
        *,
        causal_seqs: Optional[torch.Tensor] = None,
        cp_world: int = 1,
        cp_rank: int = 0,
        return_lse: bool = False,
    ):
        """Call the flashinfer cute-dsl MLA decode kernel.

        Without DCP (``cp_world <= 1``) this defers to the base cute-dsl path.
        With token DCP, ``seq_lens`` are this rank's cyclic-local KV lengths and
        ``causal_seqs`` the global per-request KV lengths; the kernel returns a
        rank-local ``(out, lse)``, the LSE in natural log.
        Page DCP uses local lengths for both inputs and passes ``cp_world=1``
        and ``cp_rank=0`` to the kernel.
        """
        if cp_world <= 1:
            if self._decode_with_splits is not None:
                if query.ndim != 4 or query.shape[1] != 1 or return_lse:
                    raise ValueError(
                        "Fixed CuTeDSL MLA splits currently support ordinary "
                        "decode only (Q length 1, no DCP/LSE merge)"
                    )
                return self._decode_with_splits(
                    query=query,
                    kv_cache=kv_cache,
                    workspace_buffer=self.workspace_buffer,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    max_seq_len=max_seq_len,
                    softmax_scale=self._compute_decode_bmm1_scale(layer),
                    output_scale=1.0,
                    out_dtype=torch.bfloat16,
                    is_var_seq=True,
                    enable_pdl=_ENABLE_PDL,
                )
            return super()._run_decode_kernel(
                query,
                kv_cache,
                block_tables,
                seq_lens,
                max_seq_len,
                layer,
                causal_seqs=causal_seqs,
                cp_world=cp_world,
                cp_rank=cp_rank,
                return_lse=return_lse,
            )
        if causal_seqs is None:
            raise ValueError(
                "causal_seqs (global per-request KV lengths) is required for DCP "
                "MLA decode."
            )
        if self.dcp_kv_layout == "page":
            # With q_len=1 every valid local KV is visible. For multiple queries,
            # cp_world=1 would incorrectly subtract the query suffix on every rank;
            # page shards need per-query local bounds, not a shared local tail.
            causal_seqs = seq_lens
            cp_world, cp_rank = 1, 0
        bmm1_scale = self._compute_decode_bmm1_scale(layer)
        raw_out, lse = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=(
                seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
            ),
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
            backend="cute-dsl",
            enable_dcp=True,
            cp_world=cp_world,
            cp_rank=cp_rank,
            causal_seqlens_kv_global=(
                causal_seqs
                if causal_seqs.dtype == torch.int32
                else causal_seqs.to(torch.int32)
            ),
            return_lse=True,  # DCP requires the rank-local LSE for the merge
        )
        return raw_out, lse

    def forward_decode(
        self,
        q: torch.Tensor,  # q_nope
        k: torch.Tensor,  # k_nope
        v: torch.Tensor,  # not used in this backend
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ):
        parallel = get_parallel()
        if parallel.dcp_enabled and get_in_autotune_dummy_run():
            return self._dummy_dcp_decode_for_autotune(q, layer)
        if not parallel.dcp_enabled:
            return super().forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
            )

        # Query / KV preparation mirrors the base cute-dsl decode (both FP16 and
        # FP8 KV), then swaps to the DCP kernel call + rank-local return.
        merge_query = q_rope is not None
        query = None
        if self.data_type == torch.float8_e4m3fn:
            assert q_rope is not None and k_rope is not None
            if cos_sin_cache is None:
                if save_kv_cache and self._fused_set_kv_concat_q_fp8:
                    loc = self._resolve_fused_write_loc(forward_batch)
                    if loc is not None:
                        query = self._set_kv_and_concat_q_fp8_fused(
                            layer=layer,
                            loc=loc,
                            q=q,
                            q_rope=q_rope,
                            k=k,
                            k_rope=k_rope,
                        )
                if query is None:
                    q, k, k_rope = mla_quantize_without_rope_for_fp8(
                        q, q_rope, k.squeeze(1), k_rope.squeeze(1)
                    )
            else:
                q, k, k_rope = mla_quantize_and_rope_for_fp8(
                    q,
                    q_rope,
                    k.squeeze(1),
                    k_rope.squeeze(1),
                    forward_batch.positions,
                    cos_sin_cache,
                    is_neox,
                    self.kv_lora_rank,
                    self.qk_rope_head_dim,
                )
            merge_query = False

        if query is None and save_kv_cache:
            assert k is not None and k_rope is not None
            self.token_to_kv_pool.set_mla_kv_buffer(
                layer, self._kv_write_loc(forward_batch), k, k_rope
            )

        if query is not None:
            pass  # fused fp8 path already built the query and wrote KV
        elif merge_query:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            query = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)
        else:
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        if llama_4_scaling is not None:
            query = (query.to(self.q_data_type) * llama_4_scaling).to(self.data_type)
        if query.dim() == 3:
            query = query.unsqueeze(1)

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_decode_metadata
        )
        metadata_batch_size = getattr(metadata, "batch_size", None)
        if (
            metadata_batch_size is not None
            and metadata_batch_size < forward_batch.batch_size
        ):
            self.init_forward_metadata(forward_batch)
            metadata = forward_batch.decode_trtllm_mla_metadata

        if metadata.seq_lens_k is not None and metadata.global_seq_lens_k is not None:
            # Hoisted path: int32 rank-local + global lens maintained once per
            # step by metadata init / graph replay-prep.
            local_seq_lens = metadata.seq_lens_k[: forward_batch.batch_size]
            global_seq_lens = metadata.global_seq_lens_k[: forward_batch.batch_size]
        else:
            global_seq_lens = forward_batch.seq_lens[: forward_batch.batch_size]
            local_seq_lens = self._get_dcp_local_seq_lens(global_seq_lens)
        raw_out, lse = self._run_decode_kernel(
            query=query,
            kv_cache=kv_cache,
            block_tables=metadata.block_kv_indices,
            seq_lens=local_seq_lens,
            max_seq_len=metadata.max_seq_len_k,
            layer=layer,
            causal_seqs=global_seq_lens,
            cp_world=parallel.dcp_size,
            cp_rank=parallel.dcp_rank,
            return_lse=True,
        )

        output = raw_out.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        lse = lse.view(-1, layer.tp_q_head_num)
        # Zero-KV rows (a request this rank owns no cyclic slice for) get a
        # neutral (out=0, lse=-inf) state so the cross-rank merge ignores them.
        fixup_zero_kv_rows(
            output,
            lse,
            local_seq_lens,
            self.q_indptr_decode[: forward_batch.batch_size + 1],
            1,
        )
        return output.flatten(1), lse


class CuteDslMLAMultiStepDraftBackend(TRTLLMMLAMultiStepDraftBackend):
    """Multi-step draft backend for cutedsl_mla used by EAGLE / DSPARK."""

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)
        # Parent populates self.attn_backends with TRT-LLM instances; replace
        # them with cute-dsl instances sharing the parent's index buffers.
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = CuteDslMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
            )
