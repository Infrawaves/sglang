"""Opt-in, Decode-only, fail-safe plugin interface for disaggregated DFLASH snapshots.

Key invariants:
- Explicitly gated by ``SGLANG_DFLASH_SNAPSHOT_OFFLOAD=1`` AND plugin ``register()``.
  When disabled, behavior and memory usage are 100% identical to base.
- Decode-only capture (Prefill has zero hooks): Decode already holds the full
  prompt + reply KV cache.
  * Verify hidden (attention-group hidden owner ``attn_dcp_rank == 0``, one per DP
    group under TP32/attn-DP4/DCP8): copied directly from the single shared graph
    buffer on ``copy_stream`` in parallel with accept and draft i+1, then aligned
    by position-overwrite on CPU (zero forward_stream gather kernels, zero H2D
    index_select sync, zero CommitGate).
  * KV: gathered once per sampled request right before release on a side stream
    via a compact 1/8 owned-only Triton gather kernel, protected by reuse_guard.
- Bounded resources accounted for before KV cache sizing (``gpu_budget_reservation_gb``).
- Fail-safe: any setup, recapture, or runtime error degrades cleanly to dropping
  samples without ever raising into serving.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.snapshot_transfer import (
    CompactKVPlan,
    HiddenD2HPool,
    KVSnapshotPool,
)
from sglang.srt.speculative.dflash_snapshot_rows import (
    MLALayout,
    OffloadAttempt,
    OffloadedRows,
    RequestHiddenBuffer,
    prompt_spans,
    visible_len,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OffloadConfig:
    layers: tuple[int, ...]
    select: Callable[[Req], OffloadAttempt | None]
    consume: Callable[[OffloadAttempt, OffloadedRows], None]
    on_complete: Callable[[OffloadAttempt], None] | None = None
    slots: int = 16
    kv_chunk_rows: int = 2048
    hidden_chunk_rows: int = 256
    hidden_slots: int = 8
    gpu_bytes: int = 64 << 20
    pinned_bytes: int = 192 << 20
    host_hidden_bytes: int = 512 << 20
    # Backward-compatible aliases (ignored in Decode-only v2; kept so existing plugins load).
    prompt_chunk_rows: int | None = None
    decode_batch_reqs: int = 32
    commit_timeout: float = 1.0

    @property
    def effective_kv_chunk_rows(self) -> int:
        return self.prompt_chunk_rows if self.prompt_chunk_rows is not None else self.kv_chunk_rows


config: OffloadConfig | None = None
runtime: OffloadRuntime | None = None


def is_enabled() -> bool:
    """Explicit engine switch + registered plugin."""
    return bool(config is not None and envs.SGLANG_DFLASH_SNAPSHOT_OFFLOAD.get())


def register(settings: OffloadConfig):
    """Call from an sglang.srt.plugins entry point before worker construction.

    select runs once on the scheduler and must use only local metadata.
    consume and on_complete run on a separate CPU thread. Budgets apply per rank.
    """
    global config
    if config is not None or runtime is not None:
        raise RuntimeError("DFLASH offload already registered")
    if not settings.layers or len(set(settings.layers)) != len(settings.layers):
        raise ValueError("Supply distinct MLA model-layer IDs")
    if any(type(i) is not int or i < 0 for i in settings.layers):
        raise ValueError("Invalid MLA model-layer ID")
    if (
        min(
            settings.slots,
            settings.effective_kv_chunk_rows,
            settings.hidden_chunk_rows,
            settings.hidden_slots,
            settings.gpu_bytes,
            settings.pinned_bytes,
            settings.host_hidden_bytes,
        )
        <= 0
    ):
        raise ValueError("Offload capacities must be positive")
    config = settings


def reset():
    """Close any active runtime and clear the registered plugin configuration."""
    global config
    try:
        close()
    finally:
        config = None


def _is_decode_mode() -> bool:
    from sglang.srt.runtime_context import get_disagg

    return get_disagg().disaggregation_mode == "decode"


def _parallel_coords():
    """Return (parallel, dp_size, dp_rank, dcp_size, dcp_rank) in attention-group coordinates."""
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    raw_attn_dp_size = getattr(p, "attn_dp_size", None)
    raw_attn_dp_rank = getattr(p, "attn_dp_rank", None)
    raw_dp_size = getattr(p, "dp_size", None)
    raw_dp_rank = getattr(p, "dp_rank", None)
    raw_dcp_size = getattr(p, "attn_dcp_size", None)
    raw_dcp_rank = getattr(p, "attn_dcp_rank", None)
    attn_dp_size = 1 if raw_attn_dp_size is None else int(raw_attn_dp_size)
    attn_dp_rank = 0 if raw_attn_dp_rank is None else int(raw_attn_dp_rank)
    dp_size = (1 if raw_dp_size is None else int(raw_dp_size)) * attn_dp_size
    dp_rank = (0 if raw_dp_rank is None else int(raw_dp_rank)) * attn_dp_size + attn_dp_rank
    dcp_size = 1 if raw_dcp_size is None else int(raw_dcp_size)
    dcp_rank = 0 if raw_dcp_rank is None else int(raw_dcp_rank)
    return p, dp_size, dp_rank, dcp_size, dcp_rank


def _is_hidden_owner() -> bool:
    """One teacher hidden owner per attention DP group (attn_dcp_rank == 0)."""
    _, _, _, _, dcp_rank = _parallel_coords()
    return dcp_rank == 0


def wants_teacher() -> bool:
    if not is_enabled():
        return False
    try:
        return _is_decode_mode() and _is_hidden_owner()
    except Exception:
        return False


def gpu_budget_reservation_gb() -> float:
    """GPU memory reservation deducted BEFORE KV cache sizing."""
    if not is_enabled():
        return 0.0
    try:
        if not _is_decode_mode():
            return 0.0
        assert config is not None
        return config.gpu_bytes / (1 << 30)
    except Exception:
        return 0.0


def _resolve_device(device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _target_logits_processor(model):
    if model is None:
        return None
    lp = getattr(model, "logits_processor", None)
    if lp is not None:
        return lp
    lm = getattr(model, "language_model", None)
    return getattr(lm, "logits_processor", None) if lm is not None else None


def _disable_target_hidden_output(runner) -> None:
    lp = _target_logits_processor(getattr(runner, "model", None))
    if lp is not None:
        lp.offload_verify_hidden = False
        lp.offload_verify_hidden_buffer = None
    runner.offload_shared_hidden_bytes = 0


def configure_target_model(runner) -> int:
    """Enable target verify hidden output and allocate one shared graph buffer.

    Called from cuda_graph_setup before graph capture. Never raises: any
    validation failure or mid-serving recapture degrades cleanly to disabled.
    All ranks check the shared-buffer budget identically so no rank diverges
    before a capture barrier, while only each attention DP group's hidden owner
    (attn_dcp_rank == 0) enables output and allocates the BF16 shared buffer.
    """
    if not is_enabled() or getattr(runner, "is_draft_worker", False):
        return 0
    try:
        if not _is_decode_mode():
            return 0
        assert config is not None
        if runtime is not None:
            logger.warning(
                "Recapturing target verify graphs while DFLASH offload is active; "
                "closing and disabling offload to keep serving uninterrupted."
            )
            close()
            _disable_target_hidden_output(runner)
            return 0
        if runner.model_config.dtype != torch.bfloat16:
            logger.warning(
                "Disabling DFLASH offload: requires BF16 target hidden state, got %s",
                runner.model_config.dtype,
            )
            _disable_target_hidden_output(runner)
            return 0
        lp = _target_logits_processor(getattr(runner, "model", None))
        if lp is None:
            logger.warning(
                "Disabling DFLASH offload: target model has no LogitsProcessor"
            )
            _disable_target_hidden_output(runner)
            return 0
        shared_out = getattr(runner, "graph_shared_output", None)
        max_rows = shared_out.max_rows if shared_out is not None else 0
        reserved = max_rows * runner.model_config.hidden_size * torch.bfloat16.itemsize
        if reserved >= config.gpu_bytes:
            logger.warning(
                "Disabling DFLASH offload: shared verify hidden buffer (%d B) exceeds "
                "gpu_bytes budget (%d B)",
                reserved,
                config.gpu_bytes,
            )
            _disable_target_hidden_output(runner)
            return 0
        if not _is_hidden_owner():
            _disable_target_hidden_output(runner)
            return 0
        lp.offload_verify_hidden = True
        if max_rows > 0:
            dev = _resolve_device(runner.device)
            existing = getattr(lp, "offload_verify_hidden_buffer", None)
            if (
                existing is None
                or existing.shape != (max_rows, runner.model_config.hidden_size)
                or existing.dtype != torch.bfloat16
                or existing.device != dev
            ):
                lp.offload_verify_hidden_buffer = torch.empty(
                    (max_rows, runner.model_config.hidden_size),
                    dtype=torch.bfloat16,
                    device=dev,
                )
        else:
            lp.offload_verify_hidden_buffer = None
        runner.offload_shared_hidden_bytes = reserved
        return reserved
    except Exception:
        logger.exception(
            "Failed to configure target model for DFLASH offload; degrading to disabled"
        )
        try:
            _disable_target_hidden_output(runner)
        except Exception:
            pass
        return 0


def initialize(scheduler):
    """Initialize Decode-only offload runtime. Never raises into serving."""
    global runtime
    if not is_enabled():
        return None
    try:
        from sglang.srt.layers.cp.base import is_cp_sharded_padded
        from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        role = scheduler.disaggregation_mode.value
        if role != "decode":
            return None
        if not isinstance(scheduler.draft_worker, DFlashWorkerV2):
            logger.warning(
                "Disabling DFLASH offload: requires DFlashWorkerV2, got %s",
                type(scheduler.draft_worker).__name__,
            )
            return None

        parallel, dp_size, dp_rank, dcp_size, dcp_rank = _parallel_coords()
        raw_attn_tp_size = getattr(parallel, "attn_tp_size", None)
        raw_attn_tp_rank = getattr(parallel, "attn_tp_rank", None)
        raw_attn_cp_size = getattr(parallel, "attn_cp_size", None)
        raw_pp_size = getattr(parallel, "pp_size", None)
        attn_tp_size = (
            int(parallel.tp_size) if raw_attn_tp_size is None else int(raw_attn_tp_size)
        )
        attn_tp_rank = (
            int(parallel.tp_rank) if raw_attn_tp_rank is None else int(raw_attn_tp_rank)
        )
        attn_cp_size = 1 if raw_attn_cp_size is None else int(raw_attn_cp_size)
        pp_size = 1 if raw_pp_size is None else int(raw_pp_size)
        if (
            pp_size != 1
            or attn_cp_size != 1
            or attn_tp_size != dcp_size
            or attn_tp_rank != dcp_rank
        ):
            logger.warning(
                "Disabling DFLASH offload: requires PP1, attn_cp=1, and one attn_tp rank "
                "per DCP owner (got pp=%d attn_cp=%d attn_tp=%d/%d dcp=%d/%d)",
                pp_size,
                attn_cp_size,
                attn_tp_rank,
                attn_tp_size,
                dcp_rank,
                dcp_size,
            )
            return None
        if getattr(scheduler, "kv_shard_widening", 1) != 1 or is_cp_sharded_padded():
            logger.warning(
                "Disabling DFLASH offload: requires token-interleaved DCP ownership"
            )
            return None

        runner = scheduler.tp_worker.model_runner
        assert config is not None
        rt = OffloadRuntime(
            runner,
            scheduler.draft_worker.block_size,
            scheduler.page_size,
            dp_size,
            dp_rank,
            dcp_size,
            dcp_rank,
            config,
            max_running_requests=int(
                getattr(scheduler, "max_running_requests", None) or 0
            ),
        )
        runtime = rt
        scheduler.draft_worker._dflash_offload_runtime = rt
        storage = kv_storage_of(
            runner.req_to_token_pool, runner.token_to_kv_pool_allocator
        )
        storage.add_pin_source(storage.sub_pool_names(), rt.kv_pool.reading)
        logger.info(
            "DFLASH Decode-only offload initialized: dp=%d/%d dcp=%d/%d hidden_owner=%s "
            "kv_slots=%d (max_owned=%d/chunk) hidden_slots=%d",
            dp_rank,
            dp_size,
            dcp_rank,
            dcp_size,
            bool(rt.hidden_width),
            rt.kv_pool.slot_count,
            rt.kv_plan.max_owned,
            0 if rt.hidden_pool is None else rt.hidden_pool.slot_count,
        )
        return rt
    except Exception:
        logger.exception(
            "Failed to initialize DFLASH offload runtime; degrading cleanly to disabled"
        )
        runtime = None
        return None


def close(target=None):
    global runtime
    rt = target if target is not None else runtime
    try:
        if rt is not None:
            rt.close()
    except Exception:
        logger.exception("Error while closing DFLASH offload runtime")
    finally:
        if target is None or target is runtime:
            runtime = None


def inspect_layout(
    router, layers, *, dcp_size: int = 1, page_size: int = 1, dp_size: int = 1
):
    from sglang.srt.mem_cache.kv.memory.views import MLAKVView

    layouts = []
    for layer_id in layers:
        layer = router.layer(layer_id)
        view = layer.view
        if not isinstance(view, MLAKVView) or not layer.owner or view.dcp_replicated:
            raise ValueError(f"Layer {layer_id} does not use owner-sharded MLA rows")
        row = layer.key()
        if row.ndim != 3 or row.shape[1] != 1 or row.stride(-1) != 1:
            raise ValueError(f"Unsupported MLA storage shape in layer {layer_id}")
        latent, rope = view.kv_lora_rank, view.qk_rope_head_dim
        packed = view.dsa_kv_cache_store_fp8
        if packed:
            expected = latent + (latent // 128) * 4 + rope * 2
            valid = latent % 128 == 0 and row.dtype in (
                torch.uint8,
                torch.float8_e4m3fn,
            )
        else:
            expected = (latent + rope) * row.element_size()
            valid = row.dtype == torch.bfloat16 or (
                row.dtype == torch.float8_e4m3fn
                and getattr(view, "fp8_cast_on_write", True)
            )
        if not valid or row.shape[-1] * row.element_size() != expected:
            raise ValueError(f"Unsupported MLA storage encoding in layer {layer_id}")
        layouts.append(
            MLALayout(
                layers,
                expected,
                row.dtype,
                latent,
                rope,
                packed,
                packed and view.kv_scale_ue8m0,
                dcp_size=dcp_size,
                page_size=page_size,
                scale_block_size=128 if packed else None,
                format_version=2,
                dp_size=dp_size,
            )
        )
    if any(layout != layouts[0] for layout in layouts):
        raise ValueError("Selected MLA layers have different physical encodings")
    return layouts[0]


def _max_owned_per_chunk(chunk_rows: int, dcp_size: int, page_size: int) -> int:
    if dcp_size <= 1:
        return chunk_rows
    # Under token-interleaved DCP, each page contributes page_size/dcp_size rows.
    # Add 2*page_size headroom for arbitrary partial-page/rejected-draft alignment.
    return min(chunk_rows, (chunk_rows + dcp_size - 1) // dcp_size + 2 * page_size)


class OffloadRuntime:
    def __init__(
        self,
        runner,
        verify_width: int,
        page_size: int,
        dp_size: int,
        dp_rank: int,
        dcp_size: int,
        dcp_rank: int,
        settings: OffloadConfig,
        *,
        max_running_requests: int = 0,
    ) -> None:
        self.settings = settings
        self.role = "decode"
        self.dp_size = int(dp_size)
        self.dp_rank = int(dp_rank)
        self.dcp_size = int(dcp_size)
        self.rank = int(dcp_rank)
        self.page_size = int(page_size)
        self.verify_width = int(verify_width)
        self.router = runner.token_to_kv_pool.layer_router
        self.ids = runner.runner_ids
        self.req_to_token_pool = runner.req_to_token_pool
        self.allocator = getattr(runner, "token_to_kv_pool_allocator", None)
        self.forward_stream = getattr(runner, "forward_stream", None)
        self.layout = inspect_layout(
            self.router,
            settings.layers,
            dcp_size=dcp_size,
            page_size=page_size,
            dp_size=dp_size,
        )
        self.hidden_width = (
            runner.model_config.hidden_size if self.rank == 0 else 0
        )
        if self.hidden_width:
            if runner.model_config.dtype != torch.bfloat16:
                raise ValueError(
                    "DFLASH teacher offload requires a BF16 target hidden state"
                )
            model = getattr(runner, "model", None)
            if model is not None:
                lp = _target_logits_processor(model)
                if lp is None or not getattr(lp, "offload_verify_hidden", False):
                    raise ValueError(
                        "Target model does not have verify hidden offload enabled"
                    )

        device = _resolve_device(runner.device)
        reserved = int(getattr(runner, "offload_shared_hidden_bytes", 0))
        kv_gpu_budget = settings.gpu_bytes - reserved
        if kv_gpu_budget <= 0:
            raise ValueError("Shared verify hidden buffer exhausts offload gpu_bytes")

        self.kv_chunk_rows = int(settings.effective_kv_chunk_rows)
        max_owned = _max_owned_per_chunk(
            self.kv_chunk_rows, self.dcp_size, self.page_size
        )
        self.kv_plan = CompactKVPlan.create(
            max_owned, len(self.layout.layers), self.layout.row_bytes
        )

        # Single FIFO consumer thread shared by Hidden D2H and KV D2H so
        # per-request hidden overwrites and release finalization are strictly ordered.
        self._consumer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dflash-offload"
        )
        if device.type == "cuda":
            self._consumer.submit(torch.cuda.set_device, device).result()

        # Split pinned budget between Hidden D2H (rank 0) and KV D2H.
        if self.hidden_width:
            kv_pinned_budget = max(
                self.kv_plan.nbytes * 2, settings.pinned_bytes // 4
            )
            hidden_pinned_budget = max(
                0, settings.pinned_bytes - kv_pinned_budget
            )
        else:
            kv_pinned_budget = settings.pinned_bytes
            hidden_pinned_budget = 0

        self.kv_pool = KVSnapshotPool(
            device,
            self.kv_plan.nbytes,
            settings.slots,
            kv_gpu_budget,
            kv_pinned_budget,
            consumer=self._consumer,
        )
        # Keep `pool` alias for backward compatibility with monitoring / inspection.
        self.pool = self.kv_pool

        self.hidden_pool: HiddenD2HPool | None = None
        if self.hidden_width:
            shared_out = getattr(runner, "graph_shared_output", None)
            shared_max = (
                int(shared_out.max_rows)
                if shared_out is not None and getattr(shared_out, "max_rows", 0) > 0
                else 0
            )
            running_max = int(max_running_requests or 0) * self.verify_width
            row_bytes = self.hidden_width * 2 + 16
            max_rows = max(shared_max, running_max, self.verify_width * 64, 256)
            # Ensure at least 2 pinned slots fit within hidden_pinned_budget.
            budget_rows = hidden_pinned_budget // max(1, row_bytes * 2)
            if budget_rows > 0:
                max_rows = min(max_rows, max(shared_max, budget_rows))
            self.hidden_pool = HiddenD2HPool(
                device,
                max_rows,
                self.hidden_width,
                settings.hidden_slots,
                hidden_pinned_budget,
                consumer=self._consumer,
            )

        # Pre-allocated layer pointer/stride buffers for zero-sync Triton compact KV gather.
        self._layer_ptrs_cache: tuple[int, ...] | None = None
        self._layer_strides_cache: tuple[int, ...] | None = None
        n_layers = len(self.layout.layers)
        if device.type == "cuda":
            self._layer_ptrs_cpu = torch.empty(
                (n_layers,), dtype=torch.int64, device="cpu", pin_memory=True
            )
            self._layer_strides_cpu = torch.empty(
                (n_layers,), dtype=torch.int64, device="cpu", pin_memory=True
            )
            self._layer_ptrs_gpu = torch.empty(
                (n_layers,), dtype=torch.int64, device=device
            )
            self._layer_strides_gpu = torch.empty(
                (n_layers,), dtype=torch.int64, device=device
            )
        else:
            self._layer_ptrs_cpu = None
            self._layer_strides_cpu = None
            self._layer_ptrs_gpu = None
            self._layer_strides_gpu = None

        self._state_lock = threading.Lock()
        self._active_attempts: dict[int, OffloadAttempt] = {}
        self._disabled = False

        # Accessed exclusively on the single FIFO consumer thread.
        self._hidden_buffers: dict[int, RequestHiddenBuffer] = {}
        self._host_hidden_bytes_used = 0

    def _track_attempt(self, attempt: OffloadAttempt) -> None:
        if self.settings.on_complete is None:
            return
        with self._state_lock:
            self._active_attempts[id(attempt)] = attempt

    def _invoke_on_complete(self, attempt: OffloadAttempt) -> None:
        callback = self.settings.on_complete
        if callback is None:
            return
        with self._state_lock:
            self._active_attempts.pop(id(attempt), None)
        try:
            callback(attempt)
        except Exception:
            logger.exception("Snapshot on_complete callback failed")

    def _finish_packet(self, attempt: OffloadAttempt, *, on_consumer: bool) -> None:
        if not attempt.release_packet():
            return
        if self.settings.on_complete is None:
            return
        if on_consumer:
            self._invoke_on_complete(attempt)
            return
        if not self.kv_pool.submit_callback(
            lambda a=attempt: self._invoke_on_complete(a)
        ):
            self._invoke_on_complete(attempt)

    def _mark_attempt_released(self, attempt: OffloadAttempt) -> None:
        if not attempt.mark_released():
            return
        if self.settings.on_complete is None:
            return
        if not self.kv_pool.submit_callback(
            lambda a=attempt: self._invoke_on_complete(a)
        ):
            self._invoke_on_complete(attempt)

    def _attempt(self, req) -> OffloadAttempt | None:
        if self._disabled:
            return None
        if not hasattr(req, "_dflash_offload_attempt"):
            attempt = None
            if (
                req.multimodal_inputs is None
                and getattr(req, "input_embeds", None) is None
                and req.origin_input_ids
            ):
                try:
                    attempt = self.settings.select(req)
                    if attempt is not None and (
                        not isinstance(attempt, OffloadAttempt)
                        or not attempt.capture_id
                        or not attempt.attempt_id
                    ):
                        raise ValueError(
                            "Offload selection requires capture and attempt IDs"
                        )
                except Exception:
                    attempt = None
                    logger.exception("DFLASH offload selection failed")
            req._dflash_offload_attempt = attempt
            if attempt is not None:
                self._track_attempt(attempt)
        attempt = req._dflash_offload_attempt
        if attempt is not None and (req.is_retracted or req.retraction_count):
            attempt.fail("request retracted")
        return attempt if attempt is not None and not attempt.error else None

    def before_step(self) -> None:
        """GPU-only wait before draft i+1 overwrites static positions / draft_tokens buffers."""
        if self._disabled or self.hidden_pool is None:
            return
        try:
            self.hidden_pool.wait_meta_ready(forward_stream=self.forward_stream)
        except Exception:
            logger.exception("DFLASH offload before_step failed; disabling offload")
            self._disabled = True

    def before_verify(self) -> None:
        """GPU-only stream wait before verify i+1 overwrites offload_verify_hidden_buffer."""
        if self._disabled or self.hidden_pool is None:
            return
        try:
            self.hidden_pool.wait_buffer_ready(forward_stream=self.forward_stream)
        except Exception:
            logger.exception("DFLASH offload before_verify failed; disabling offload")
            self._disabled = True

    def snapshot_verify_hidden(
        self, reqs, positions, tokens, hidden, width: int
    ) -> None:
        """Copy verify hidden + positions + input_ids directly on copy_stream.

        - Zero kernels on forward_stream.
        - Zero H2D index_select / host sync even for sparse non-contiguous sampling.
        - Non-hidden-owner ranks return immediately on the first branch.
        """
        if self._disabled or self.hidden_pool is None:
            return
        try:
            selected = []
            for i, req in enumerate(reqs):
                attempt = self._attempt(req)
                if attempt is not None:
                    selected.append((i, len(req.origin_input_ids), attempt))
            if not selected:
                return

            batch_size = len(reqs)
            expected_rows = batch_size * width
            if (
                positions is None
                or tokens is None
                or positions.numel() != expected_rows
                or tokens.numel() != expected_rows
                or positions.dtype not in (torch.int32, torch.int64)
                or tokens.dtype not in (torch.int32, torch.int64)
            ):
                for _, _, attempt in selected:
                    attempt.fail("invalid verify row metadata")
                return
            if (
                hidden is None
                or hidden.dtype != torch.bfloat16
                or hidden.shape != (expected_rows, self.hidden_width)
            ):
                for _, _, attempt in selected:
                    attempt.fail("missing final target hidden rows")
                return

            # Pure view slice covering [min_i, max_i]: zero H2D tensor construction!
            min_i = selected[0][0]
            max_i = selected[-1][0]
            row_lo = min_i * width
            row_hi = (max_i + 1) * width
            gpu_h = hidden[row_lo:row_hi]
            gpu_p = positions.reshape(-1)[row_lo:row_hi]
            gpu_t = tokens.reshape(-1)[row_lo:row_hi]

            entries = tuple(
                (i - min_i, prompt_len, attempt)
                for i, prompt_len, attempt in selected
            )
            for _, _, attempt in entries:
                attempt.acquire_packet()

            def deliver(
                pos_cpu: torch.Tensor,
                tok_cpu: torch.Tensor,
                hid_cpu: torch.Tensor,
                entries=entries,
                width=width,
            ):
                for rel_idx, prompt_len, attempt in entries:
                    try:
                        if attempt.error:
                            continue
                        s = rel_idx * width
                        e = s + width
                        key = id(attempt)
                        buf = self._hidden_buffers.get(key)
                        if buf is None:
                            buf = RequestHiddenBuffer(prompt_len, self.hidden_width)
                            self._hidden_buffers[key] = buf
                        budget_rem = (
                            self.settings.host_hidden_bytes
                            - self._host_hidden_bytes_used
                        )
                        delta = buf.write(
                            pos_cpu[s:e],
                            tok_cpu[s:e],
                            hid_cpu[s:e],
                            budget_remaining=budget_rem,
                        )
                        self._host_hidden_bytes_used += delta
                    except Exception as exc:
                        attempt.fail(str(exc) or type(exc).__name__)
                        logger.warning(
                            "DFLASH offload dropped hidden capture %s: %s",
                            attempt.capture_id,
                            attempt.error,
                        )
                    finally:
                        self._finish_packet(attempt, on_consumer=True)

            def fail(reason: str, entries=entries):
                for _, _, attempt in entries:
                    attempt.fail(reason)
                    self._finish_packet(attempt, on_consumer=False)

            self.hidden_pool.submit(
                gpu_h,
                gpu_p,
                gpu_t,
                deliver,
                fail,
                forward_stream=self.forward_stream,
            )
        except Exception:
            logger.exception("snapshot_verify_hidden failed; dropping batch samples")

    def _sync_layer_ptrs_on_stream(self, sources: tuple[torch.Tensor, ...]) -> None:
        if self._layer_ptrs_gpu is None:
            return
        ptrs = tuple(int(s.data_ptr()) for s in sources)
        strides = tuple(int(s.stride(0) * s.element_size()) for s in sources)
        if ptrs == self._layer_ptrs_cache and strides == self._layer_strides_cache:
            return
        for i, (ptr, st) in enumerate(zip(ptrs, strides, strict=True)):
            self._layer_ptrs_cpu[i] = ptr
            self._layer_strides_cpu[i] = st
        self._layer_ptrs_gpu.copy_(self._layer_ptrs_cpu, non_blocking=True)
        self._layer_strides_gpu.copy_(self._layer_strides_cpu, non_blocking=True)
        self._layer_ptrs_cache = ptrs
        self._layer_strides_cache = strides

    def _submit_kv_span(
        self,
        attempt: OffloadAttempt,
        ids_snapshot: torch.Tensor,
        start: int,
        seq_tokens: tuple[int, ...],
        wait_streams: tuple,
    ) -> bool:
        plan = self.kv_plan
        attempt.acquire_packet()

        def fill(storage: torch.Tensor, stream) -> None:
            sources = tuple(
                self.router.read(layer).key() for layer in self.layout.layers
            )
            self._sync_layer_ptrs_on_stream(sources)
            owned = (ids_snapshot > 0) & (
                ids_snapshot.remainder(self.dcp_size) == self.rank
            )
            local_ids = torch.where(owned, ids_snapshot, 0)
            physical = self.ids.translate_dcp_read_ids(local_ids).to(torch.int64)
            for src in (ids_snapshot, *sources):
                src.record_stream(stream)
            plan.gather(
                storage,
                sources=sources,
                owned=owned,
                physical=physical,
                layer_ptrs_gpu=self._layer_ptrs_gpu,
                layer_strides_gpu=self._layer_strides_gpu,
            )

        def deliver(
            host: torch.Tensor,
            start=start,
            seq_tokens=seq_tokens,
            attempt=attempt,
        ) -> None:
            try:
                if attempt.error:
                    return
                fields = plan.views(host)
                count = int(fields["meta"][0].item())
                overflow = int(fields["meta"][1].item())
                if overflow != 0 or count > plan.max_owned:
                    attempt.fail(
                        f"compact KV capacity exceeded ({count} > {plan.max_owned})"
                    )
                    return
                if count == 0:
                    return
                rel_idx = fields["orig_idx"][:count]
                phys = fields["physical"][:count]
                kv = fields["kv"][:count]
                # Atomic compaction order is arbitrary; sort by token offset on CPU.
                order = torch.argsort(rel_idx)
                rel_sorted = rel_idx[order]
                positions = rel_sorted + start
                input_ids = torch.tensor(
                    [seq_tokens[int(i)] for i in rel_sorted.tolist()],
                    dtype=torch.int64,
                    device="cpu",
                )
                self.settings.consume(
                    attempt,
                    OffloadedRows(
                        role=self.role,
                        owner=self.rank,
                        layout=self.layout,
                        positions=positions,
                        input_ids=input_ids,
                        owned=torch.ones((count,), dtype=torch.bool, device="cpu"),
                        physical_ids=phys[order],
                        raw_kv=kv[order],
                        next_token_ids=None,
                        final_hidden=None,
                        dp_rank=self.dp_rank,
                    ),
                )
            except Exception as exc:
                attempt.fail(str(exc) or type(exc).__name__)
                logger.exception("KV snapshot consumer failed")
            finally:
                self._finish_packet(attempt, on_consumer=True)

        def fail(reason: str, attempt=attempt) -> None:
            attempt.fail(reason)
            self._finish_packet(attempt, on_consumer=False)

        return self.kv_pool.submit(
            plan.nbytes, fill, deliver, fail, wait_streams=wait_streams
        )

    def _enqueue_hidden_finalize(
        self,
        attempt: OffloadAttempt,
        prompt_len: int,
        output_ids: tuple[int, ...],
    ) -> None:
        if not self.hidden_width:
            return
        attempt.acquire_packet()

        def finalize() -> None:
            key = id(attempt)
            buf = self._hidden_buffers.pop(key, None)
            if buf is not None:
                self._host_hidden_bytes_used = max(
                    0, self._host_hidden_bytes_used - buf.nbytes
                )
            try:
                if attempt.error:
                    return
                if len(output_ids) <= 1:
                    return
                if buf is None:
                    attempt.fail("missing teacher hidden buffer at release")
                    return
                extracted = buf.extract(output_ids)
                if extracted is None:
                    return
                positions, input_ids, next_token_ids, final_hidden = extracted
                n = int(positions.numel())
                chunk = int(self.settings.hidden_chunk_rows)
                empty_phys = torch.empty((0,), dtype=torch.int64, device="cpu")
                empty_kv = torch.empty(
                    (0, len(self.layout.layers), self.layout.row_bytes),
                    dtype=torch.uint8,
                    device="cpu",
                )
                for off in range(0, n, chunk):
                    end = min(off + chunk, n)
                    m = end - off
                    self.settings.consume(
                        attempt,
                        OffloadedRows(
                            role=self.role,
                            owner=self.rank,
                            layout=self.layout,
                            positions=positions[off:end],
                            input_ids=input_ids[off:end],
                            owned=torch.zeros((m,), dtype=torch.bool, device="cpu"),
                            physical_ids=empty_phys,
                            raw_kv=empty_kv,
                            next_token_ids=next_token_ids[off:end],
                            final_hidden=final_hidden[off:end],
                            dp_rank=self.dp_rank,
                        ),
                    )
            except Exception as exc:
                attempt.fail(str(exc) or type(exc).__name__)
                logger.warning(
                    "DFLASH offload dropped hidden finalize %s: %s",
                    attempt.capture_id,
                    attempt.error,
                )
            finally:
                self._finish_packet(attempt, on_consumer=True)

        if not self.kv_pool.submit_callback(finalize):
            attempt.fail("offload closed")
            self._finish_packet(attempt, on_consumer=False)

    def _drop_hidden_buffer_async(self, attempt: OffloadAttempt) -> None:
        if not self.hidden_width:
            return
        key = id(attempt)

        def drop() -> None:
            buf = self._hidden_buffers.pop(key, None)
            if buf is not None:
                self._host_hidden_bytes_used = max(
                    0, self._host_hidden_bytes_used - buf.nbytes
                )

        self.kv_pool.submit_callback(drop)

    def before_release(self, req) -> None:
        """One-shot compact 1/8 KV gather + hidden finalize before KV pages are freed."""
        if self._disabled:
            return
        try:
            attempt = self._attempt(req)
            if attempt is None:
                existing = getattr(req, "_dflash_offload_attempt", None)
                if existing is not None:
                    self._drop_hidden_buffer_async(existing)
                    self._mark_attempt_released(existing)
                return

            reason = req.finished_reason
            prompt_ids = tuple(req.origin_input_ids)
            out_len = visible_len(req)
            output_ids = tuple(req.output_ids[:out_len])
            full_tokens = prompt_ids + output_ids
            full_len = len(full_tokens)

            if (
                reason is None
                or reason.to_json()["type"] not in ("stop", "length")
                or req.is_retracted
                or req.retraction_count
                or req.kv.req_pool_idx is None
                or req.kv.kv_committed_len < len(prompt_ids)
            ):
                attempt.fail("incomplete or retracted decode request")
                self._drop_hidden_buffer_async(attempt)
                self._mark_attempt_released(attempt)
                return

            # Export all committed prompt + reply KV rows (optionally skipping verified prompt pages).
            kv_len = min(full_len, int(req.kv.kv_committed_len))
            try:
                # Validate verified_pages against prompt length (not reply pages).
                if attempt.verified_pages:
                    max_prompt_pages = len(prompt_ids) // self.page_size
                    if any(
                        type(i) is not int or not 0 <= i < max_prompt_pages
                        for i in attempt.verified_pages
                    ):
                        raise ValueError(
                            "A verified prompt page must be complete and within the prompt"
                        )
                spans = list(
                    prompt_spans(
                        kv_len,
                        attempt.verified_pages,
                        self.page_size,
                        self.kv_chunk_rows,
                    )
                )
            except ValueError as exc:
                attempt.fail(str(exc))
                self._drop_hidden_buffer_async(attempt)
                self._mark_attempt_released(attempt)
                return

            # Snapshot req_to_token IDs on caller stream before release_kv_cache frees req_pool_idx.
            submitted_any = False
            row_ids = self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :kv_len]
            if spans:
                ids_clone = row_ids.clone()
                caller_stream = (
                    torch.cuda.current_stream(ids_clone.device)
                    if ids_clone.is_cuda
                    else self.forward_stream
                )
                wait_streams = (caller_stream, self.forward_stream)
                for start, end in spans:
                    if not self._submit_kv_span(
                        attempt,
                        ids_clone[start:end],
                        start,
                        full_tokens[start:end],
                        wait_streams=wait_streams,
                    ):
                        break
                    submitted_any = True

            # Enqueue hidden finalize onto the same FIFO consumer thread (runs after all verify D2H).
            self._enqueue_hidden_finalize(attempt, len(prompt_ids), output_ids)
            self._mark_attempt_released(attempt)
            if submitted_any:
                self.kv_pool.guard_release(self.allocator)
        except Exception:
            logger.exception("DFLASH offload before_release failed; dropping sample")
            existing = getattr(req, "_dflash_offload_attempt", None)
            if existing is not None:
                existing.fail("before_release error")
                self._drop_hidden_buffer_async(existing)
                self._mark_attempt_released(existing)

    def close(self) -> None:
        self._disabled = True
        try:
            if self.hidden_pool is not None:
                self.hidden_pool.close()
            self.kv_pool.close()
            self._consumer.shutdown(wait=True)
        finally:
            self._hidden_buffers.clear()
            self._host_hidden_bytes_used = 0
            with self._state_lock:
                remaining = list(self._active_attempts.values())
                self._active_attempts.clear()
            for attempt in remaining:
                attempt.fail("offload closed")
                if attempt.mark_released():
                    self._invoke_on_complete(attempt)
