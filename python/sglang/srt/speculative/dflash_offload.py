"""Opt-in, process-local plugin interface for disaggregated DFLASH snapshots."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.snapshot_transfer import CommitGate, SnapshotPool, packed_rows
from sglang.srt.speculative.dflash_snapshot_rows import (
    MLALayout,
    OffloadAttempt,
    OffloadedRows,
    VerifyBatchCommit,
    VerifyCommit,
    committed_slice,
    prompt_spans,
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
    slots: int = 32
    prompt_chunk_rows: int = 2048
    decode_batch_reqs: int = 32
    gpu_bytes: int = 256 << 20
    pinned_bytes: int = 128 << 20
    commit_timeout: float = 1.0


config: OffloadConfig | None = None
runtime: OffloadRuntime | None = None


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
            settings.prompt_chunk_rows,
            settings.decode_batch_reqs,
            settings.gpu_bytes,
            settings.pinned_bytes,
            settings.commit_timeout,
        )
        <= 0
    ):
        raise ValueError("Offload capacities and timeout must be positive")
    config = settings


def reset():
    """Close any active runtime and clear the registered plugin configuration."""
    global config
    try:
        close()
    finally:
        config = None


def wants_teacher():
    if config is None:
        return False
    from sglang.srt.runtime_context import get_disagg, get_parallel

    return get_disagg().disaggregation_mode == "decode" and get_parallel().tp_rank == 0


def _target_logits_processor(model):
    if model is None:
        return None
    lp = getattr(model, "logits_processor", None)
    if lp is not None:
        return lp
    lm = getattr(model, "language_model", None)
    return getattr(lm, "logits_processor", None) if lm is not None else None


def configure_target_model(runner) -> int:
    """Enable target verify hidden output and allocate one shared graph buffer.

    Called from cuda_graph_setup before graph capture. All TP ranks validate
    the target dtype and buffer budget identically so no rank fails alone before
    a capture barrier, while only TP rank 0 enables the output and allocates
    the shared BF16 buffer.
    """
    if config is None or runner.is_draft_worker:
        return 0
    from sglang.srt.runtime_context import get_disagg, get_parallel

    if get_disagg().disaggregation_mode != "decode":
        return 0
    if runtime is not None:
        raise RuntimeError("Drain DFLASH offload before recapturing verify graphs")
    if runner.model_config.dtype != torch.bfloat16:
        raise ValueError("DFLASH teacher offload requires a BF16 target hidden state")
    lp = _target_logits_processor(getattr(runner, "model", None))
    if lp is None:
        raise ValueError(
            "DFLASH teacher offload requires a target model with LogitsProcessor"
        )
    shared_out = getattr(runner, "graph_shared_output", None)
    max_rows = shared_out.max_rows if shared_out is not None else 0
    reserved = max_rows * runner.model_config.hidden_size * torch.bfloat16.itemsize
    if reserved >= config.gpu_bytes:
        raise ValueError("Shared verify hidden buffer exceeds the offload GPU budget")
    if get_parallel().tp_rank != 0:
        runner.offload_shared_hidden_bytes = 0
        return 0
    lp.offload_verify_hidden = True
    if max_rows > 0:
        existing = getattr(lp, "offload_verify_hidden_buffer", None)
        if (
            existing is None
            or existing.shape != (max_rows, runner.model_config.hidden_size)
            or existing.dtype != torch.bfloat16
            or str(existing.device) != str(torch.device(runner.device))
        ):
            lp.offload_verify_hidden_buffer = torch.empty(
                (max_rows, runner.model_config.hidden_size),
                dtype=torch.bfloat16,
                device=runner.device,
            )
    else:
        lp.offload_verify_hidden_buffer = None
    runner.offload_shared_hidden_bytes = reserved
    return reserved


def initialize(scheduler):
    global runtime
    if config is None:
        return None
    from sglang.srt.layers.cp.base import is_cp_sharded_padded
    from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    parallel = get_parallel()
    role = scheduler.disaggregation_mode.value
    if role not in ("prefill", "decode") or not isinstance(
        scheduler.draft_worker, DFlashWorkerV2
    ):
        raise ValueError("Snapshot offload requires disaggregated DFLASH v2")
    if (
        parallel.pp_size != 1
        or parallel.dp_size != 1
        or parallel.tp_size != parallel.attn_dcp_size
        or parallel.tp_rank != parallel.attn_dcp_rank
    ):
        raise ValueError(
            "Snapshot offload requires DP1, PP1 and one TP rank per DCP owner"
        )
    if getattr(scheduler, "kv_shard_widening", 1) != 1 or is_cp_sharded_padded():
        raise ValueError(
            "Snapshot offload requires token-interleaved DCP ownership, not page-interleaved or CP-sharded KV"
        )
    runner = scheduler.tp_worker.model_runner
    rt = OffloadRuntime(
        runner,
        scheduler.draft_worker.block_size,
        scheduler.page_size,
        role,
        parallel.attn_dcp_size,
        parallel.attn_dcp_rank,
        config,
    )
    runtime = rt
    scheduler.draft_worker._dflash_offload_runtime = rt
    storage = kv_storage_of(runner.req_to_token_pool, runner.token_to_kv_pool_allocator)
    storage.add_pin_source(storage.sub_pool_names(), rt.pool.reading)
    return rt


def close(target=None):
    global runtime
    rt = target if target is not None else runtime
    try:
        if rt is not None:
            rt.close()
    finally:
        if target is None or target is runtime:
            runtime = None


def inspect_layout(router, layers, *, dcp_size: int = 1, page_size: int = 1):
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
                format_version=1,
            )
        )
    if any(layout != layouts[0] for layout in layouts):
        raise ValueError("Selected MLA layers have different physical encodings")
    return layouts[0]


class OffloadRuntime:
    def __init__(self, runner, verify_width, page_size, role, dcp_size, rank, settings):
        self.settings = settings
        self.role = role
        self.rank = rank
        self.dcp_size = dcp_size
        self.page_size = page_size
        self.router = runner.token_to_kv_pool.layer_router
        self.ids = runner.runner_ids
        self.req_to_token_pool = runner.req_to_token_pool
        self.allocator = getattr(runner, "token_to_kv_pool_allocator", None)
        self.forward_stream = getattr(runner, "forward_stream", None)
        self.layout = inspect_layout(
            self.router, settings.layers, dcp_size=dcp_size, page_size=page_size
        )
        self.hidden_width = (
            runner.model_config.hidden_size if role == "decode" and rank == 0 else 0
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
        reserved = getattr(runner, "offload_shared_hidden_bytes", 0)
        gpu_budget = settings.gpu_bytes - reserved
        if role == "decode":
            self.decode_batch_reqs = self._choose_decode_batch_reqs(
                verify_width, settings, gpu_budget
            )
            rows = verify_width * self.decode_batch_reqs
        else:
            self.decode_batch_reqs = 1
            rows = settings.prompt_chunk_rows
        plan = self._packed(rows)
        self.pool = SnapshotPool(
            runner.device,
            plan.nbytes,
            settings.slots,
            gpu_budget,
            settings.pinned_bytes,
        )
        self._state_lock = threading.Lock()
        self._pending_commits: set[VerifyBatchCommit] = set()
        self._active_attempts: dict[int, OffloadAttempt] = {}

    def _choose_decode_batch_reqs(self, verify_width, settings, gpu_budget):
        """Pack multiple selected decode requests per slot without shrinking the
        slot count below what single-request slots fit under the memory budget.
        """
        single_stride = (self._packed(verify_width).nbytes + 7) // 8 * 8
        if single_stride <= 0:
            return 1
        single_slots = min(
            settings.slots,
            max(0, gpu_budget) // (2 * single_stride),
            settings.pinned_bytes // single_stride,
        )
        target_slots = max(1, single_slots)
        req_cap = max(1, int(settings.decode_batch_reqs))
        while req_cap > 1:
            stride = (self._packed(verify_width * req_cap).nbytes + 7) // 8 * 8
            slots = min(
                settings.slots,
                max(0, gpu_budget) // (2 * stride),
                settings.pinned_bytes // stride,
            )
            if slots >= target_slots:
                return req_cap
            req_cap //= 2
        return 1

    def _packed(self, rows):
        return packed_rows(
            rows,
            len(self.layout.layers),
            self.layout.row_bytes,
            self.hidden_width,
            self.role == "decode",
        )

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
        submit_cb = getattr(self.pool, "submit_callback", None)
        if submit_cb is None or not submit_cb(
            lambda a=attempt: self._invoke_on_complete(a)
        ):
            self._invoke_on_complete(attempt)

    def _mark_attempt_released(self, attempt: OffloadAttempt) -> None:
        if not attempt.mark_released():
            return
        if self.settings.on_complete is None:
            return
        submit_cb = getattr(self.pool, "submit_callback", None)
        if submit_cb is None or not submit_cb(
            lambda a=attempt: self._invoke_on_complete(a)
        ):
            self._invoke_on_complete(attempt)

    def _attempt(self, req):
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

    def _make_fill(self, plan, ids, positions, tokens, hidden):
        def fill(storage, stream):
            sources = tuple(
                self.router.read(layer).key() for layer in self.layout.layers
            )
            # The allocator encodes the owner in widened IDs. Physical addressing
            # also includes the v2p translation on a migratable pool.
            owned = (ids > 0) & (ids.remainder(self.dcp_size) == self.rank)
            local_ids = torch.where(owned, ids, 0)
            physical = self.ids.translate_dcp_read_ids(local_ids).to(torch.int64)
            for src in (ids, positions, tokens, hidden, *sources):
                if src is not None:
                    src.record_stream(stream)
            plan.gather(
                storage,
                sources=sources,
                owned=owned,
                physical=physical,
                positions=positions,
                tokens=tokens,
                hidden=hidden,
            )

        return fill

    def _submit(
        self, attempt, ids, positions, tokens, hidden, deliver, *, stream=None
    ):
        plan = self._packed(ids.numel())
        fill = self._make_fill(plan, ids, positions, tokens, hidden)
        attempt.acquire_packet()

        def consume(storage):
            try:
                if not attempt.error:
                    deliver(plan.views(storage))
            except Exception as exc:
                attempt.fail(str(exc) or type(exc).__name__)
                logger.exception("Snapshot consumer failed")
            finally:
                self._finish_packet(attempt, on_consumer=True)

        def fail(reason):
            attempt.fail(reason)
            self._finish_packet(attempt, on_consumer=False)

        kwargs = {}
        if stream is not None:
            kwargs["stream"] = stream
        return self.pool.submit(plan.nbytes, fill, consume, fail, **kwargs)

    def _emit(self, attempt, fields, rows, positions, tokens, labels):
        if positions.numel() == 0 or attempt.error:
            return
        owned = fields["owned"][rows]
        hidden = fields.get("hidden")
        self.settings.consume(
            attempt,
            OffloadedRows(
                self.role,
                self.rank,
                self.layout,
                positions,
                tokens,
                owned,
                fields["physical"][rows][owned],
                fields["kv"][:, rows].transpose(0, 1)[owned],
                labels,
                None if hidden is None else hidden[rows],
            ),
        )

    def _select_verify_rows(self, tensor, batch_size, width, req_indices, idx_t=None):
        if len(req_indices) == batch_size:
            return tensor.reshape(-1)
        first, last = req_indices[0], req_indices[-1]
        if last - first + 1 == len(req_indices):
            return tensor.reshape(-1)[first * width : (last + 1) * width]
        if idx_t is None:
            idx_t = torch.tensor(req_indices, dtype=torch.int64, device=tensor.device)
        return tensor.reshape(batch_size, width).index_select(0, idx_t).reshape(-1)

    def _select_verify_hidden(self, hidden, batch_size, width, req_indices, idx_t=None):
        if hidden is None:
            return None
        if len(req_indices) == batch_size:
            return hidden
        first, last = req_indices[0], req_indices[-1]
        if last - first + 1 == len(req_indices):
            return hidden[first * width : (last + 1) * width]
        if idx_t is None:
            idx_t = torch.tensor(req_indices, dtype=torch.int64, device=hidden.device)
        return (
            hidden.reshape(batch_size, width, self.hidden_width)
            .index_select(0, idx_t)
            .reshape(-1, self.hidden_width)
        )

    def _unregister_commit(self, commit: VerifyBatchCommit) -> None:
        with self._state_lock:
            self._pending_commits.discard(commit)

    def snapshot_verify(self, reqs, positions, tokens, ids, hidden, width):
        selected = []
        for i, req in enumerate(reqs):
            attempt = self._attempt(req)
            if attempt is not None:
                selected.append((i, req, attempt))
        if not selected:
            return []

        batch_size = len(reqs)
        expected_rows = batch_size * width
        if any(
            t.numel() != expected_rows or t.dtype not in (torch.int32, torch.int64)
            for t in (positions, tokens, ids)
        ):
            for _, _, attempt in selected:
                attempt.fail("invalid verify row metadata")
            return []
        if self.hidden_width and (
            hidden is None
            or hidden.dtype != torch.bfloat16
            or hidden.shape != (expected_rows, self.hidden_width)
        ):
            for _, _, attempt in selected:
                attempt.fail("missing final target hidden rows")
            return []

        batch_commits = []
        chunk_reqs = max(1, self.decode_batch_reqs)
        for offset in range(0, len(selected), chunk_reqs):
            group = selected[offset : offset + chunk_reqs]
            req_indices = [i for i, _, _ in group]
            commits = [VerifyCommit(req, attempt) for _, req, attempt in group]
            lengths = tuple(len(req.origin_input_ids) for _, req, _ in group)
            attempts = tuple(attempt for _, _, attempt in group)

            is_contiguous = (
                len(req_indices) == batch_size
                or req_indices[-1] - req_indices[0] + 1 == len(req_indices)
            )
            idx_t = (
                None
                if is_contiguous
                else torch.tensor(req_indices, dtype=torch.int64, device=ids.device)
            )
            group_ids = self._select_verify_rows(
                ids, batch_size, width, req_indices, idx_t
            )
            group_pos = self._select_verify_rows(
                positions, batch_size, width, req_indices, idx_t
            )
            group_tok = self._select_verify_rows(
                tokens, batch_size, width, req_indices, idx_t
            )
            group_hidden = (
                self._select_verify_hidden(
                    hidden, batch_size, width, req_indices, idx_t
                )
                if self.hidden_width
                else None
            )
            plan = self._packed(len(group) * width)
            fill = self._make_fill(plan, group_ids, group_pos, group_tok, group_hidden)

            for attempt in attempts:
                attempt.acquire_packet()

            def deliver(
                storage,
                paths,
                plan=plan,
                lengths=lengths,
                attempts=attempts,
                width=width,
            ):
                fields = None if paths is None else plan.views(storage)
                for idx, (attempt, length) in enumerate(
                    zip(attempts, lengths, strict=True)
                ):
                    try:
                        if paths is None or attempt.error:
                            continue
                        path = paths[idx]
                        if path is None:
                            continue
                        start = idx * width
                        end = start + width
                        sub_pos = fields["positions"][start:end]
                        sub_tok = fields["tokens"][start:end]
                        sel, labels = committed_slice(path, length, sub_pos, sub_tok)
                        if not labels:
                            continue
                        rows = slice(start + sel.start, start + sel.stop)
                        self._emit(
                            attempt,
                            fields,
                            rows,
                            sub_pos[sel],
                            sub_tok[sel],
                            torch.tensor(labels, dtype=torch.int64, device="cpu"),
                        )
                    except Exception as exc:
                        attempt.fail(str(exc) or type(exc).__name__)
                        logger.exception("Snapshot verify consumer failed")
                    finally:
                        self._finish_packet(attempt, on_consumer=True)

            def fail(reason, attempts=attempts):
                for attempt in attempts:
                    attempt.fail(reason)
                    self._finish_packet(attempt, on_consumer=False)

            gate = CommitGate(deliver, fail)
            batch_commit = VerifyBatchCommit(
                commits, gate, on_done=self._unregister_commit
            )
            with self._state_lock:
                self._pending_commits.add(batch_commit)

            submitted = self.pool.submit(
                plan.nbytes,
                fill,
                deliver,
                fail,
                stream=self.forward_stream,
                gate=gate,
            )
            if submitted:
                batch_commits.append(batch_commit)
            else:
                self._unregister_commit(batch_commit)
        return batch_commits

    def _guard_release(self):
        guard = getattr(self.pool, "guard_release", None)
        if guard is not None:
            guard(self.allocator, forward_stream=self.forward_stream)
        else:
            self.pool.fence_reuse(stream=self.forward_stream)

    def before_release(self, req):
        if self.role != "prefill":
            self._guard_release()
            attempt = getattr(req, "_dflash_offload_attempt", None)
            if attempt is not None:
                reason = req.finished_reason
                if reason is None or reason.to_json()["type"] not in ("stop", "length"):
                    attempt.fail("decode released before completion")
                self._mark_attempt_released(attempt)
            return
        self.pool.fence_reuse(stream=self.forward_stream)
        attempt = self._attempt(req)
        if attempt is None:
            existing = getattr(req, "_dflash_offload_attempt", None)
            if existing is not None:
                self._mark_attempt_released(existing)
            return
        length = len(req.origin_input_ids)
        reason = req.finished_reason
        if (
            reason is None
            or reason.to_json()["type"] not in ("stop", "length")
            or req.kv.req_pool_idx is None
            or req.kv.kv_committed_len < length
        ):
            attempt.fail("incomplete prefill")
            self._mark_attempt_released(attempt)
            return
        try:
            spans = list(
                prompt_spans(
                    length,
                    attempt.verified_pages,
                    self.page_size,
                    self.settings.prompt_chunk_rows,
                )
            )
        except ValueError as exc:
            attempt.fail(str(exc))
            self._mark_attempt_released(attempt)
            return
        for start, end in spans:
            ids = self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, start:end]
            prompt = tuple(req.origin_input_ids[start:end])

            def deliver(fields, start=start, end=end, prompt=prompt, attempt=attempt):
                self._emit(
                    attempt,
                    fields,
                    slice(None),
                    torch.arange(start, end, device="cpu"),
                    torch.tensor(prompt, dtype=torch.int64, device="cpu"),
                    None,
                )

            if not self._submit(
                attempt,
                ids,
                None,
                None,
                None,
                deliver,
                stream=self.forward_stream,
            ):
                break
        self._mark_attempt_released(attempt)
        self._guard_release()

    def close(self):
        with self._state_lock:
            pending = list(self._pending_commits)
            self._pending_commits.clear()
        for commit in pending:
            commit.abort("offload closed")
        try:
            self.pool.close()
        finally:
            with self._state_lock:
                remaining = list(self._active_attempts.values())
                self._active_attempts.clear()
            for attempt in remaining:
                attempt.fail("offload closed")
                if attempt.mark_released():
                    self._invoke_on_complete(attempt)
