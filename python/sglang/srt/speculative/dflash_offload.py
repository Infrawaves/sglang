"""Opt-in, process-local plugin interface for disaggregated DFLASH snapshots."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.snapshot_transfer import SnapshotPool, packed_rows
from sglang.srt.speculative.dflash_snapshot_rows import (
    MLALayout,
    OffloadAttempt,
    OffloadedRows,
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
    slots: int = 32
    prompt_chunk_rows: int = 256
    gpu_bytes: int = 256 << 20
    pinned_bytes: int = 128 << 20
    commit_timeout: float = 1.0


config: OffloadConfig | None = None
runtime: OffloadRuntime | None = None


def register(settings: OffloadConfig):
    """Call from an sglang.srt.plugins entry point before worker construction.

    select runs once on the scheduler and must use only local metadata.
    consume runs on a separate CPU thread. Budgets apply per rank.
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
            settings.gpu_bytes,
            settings.pinned_bytes,
            settings.commit_timeout,
        )
        <= 0
    ):
        raise ValueError("Offload capacities and timeout must be positive")
    config = settings


def wants_teacher():
    if config is None:
        return False
    from sglang.srt.runtime_context import get_disagg, get_parallel

    return get_disagg().disaggregation_mode == "decode" and get_parallel().tp_rank == 0


def reserve_graph_bytes(graph):
    """Bound live final-hidden outputs across all target verify graph variants."""
    if (
        not wants_teacher()
        or graph.model_runner.is_draft_worker
        or not graph.capture_forward_mode.is_target_verify()
    ):
        return 0
    if runtime is not None:
        raise RuntimeError("Drain DFLASH offload before recapturing verify graphs")
    row_bytes = graph.model_runner.model_config.hidden_size * torch.bfloat16.itemsize
    sizes = []
    for shape in graph._capture_sizes():
        batch, rows = (
            shape
            if isinstance(shape, tuple)
            else (shape, shape * graph.captured_req_width)
        )
        variants = (
            len(graph.attention_graph_variants.capture_labels)
            if graph._attention_varies_at(batch)
            else 1
        )
        variants *= (2 if graph.record_nolora_graph else 1) * len(
            graph.logits_graph_variants
        )
        sizes.extend([rows * row_bytes] * variants)
    reserved = sum(sizes) + max(sizes, default=0)
    if reserved >= config.gpu_bytes:
        raise ValueError("Verify graph outputs exceed the offload GPU budget")
    return reserved


def initialize(scheduler):
    global runtime
    if config is None:
        return
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
    runner = scheduler.tp_worker.model_runner
    runtime = OffloadRuntime(
        runner,
        scheduler.draft_worker.block_size,
        scheduler.page_size,
        role,
        parallel.attn_dcp_size,
        parallel.attn_dcp_rank,
        config,
    )
    storage = kv_storage_of(runner.req_to_token_pool, runner.token_to_kv_pool_allocator)
    storage.add_pin_source(storage.sub_pool_names(), runtime.pool.reading)


def close():
    global runtime
    if runtime is not None:
        runtime.pool.close()
        runtime = None


def inspect_layout(router, layers):
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
        self.layout = inspect_layout(self.router, settings.layers)
        self.hidden_width = (
            runner.model_config.hidden_size if role == "decode" and rank == 0 else 0
        )
        if self.hidden_width and runner.model_config.dtype != torch.bfloat16:
            raise ValueError(
                "DFLASH teacher offload requires a BF16 target hidden state"
            )
        rows = verify_width if role == "decode" else settings.prompt_chunk_rows
        plan = self._packed(rows)
        reserved = getattr(runner.decode_cuda_graph_runner, "offload_graph_bytes", 0)
        self.pool = SnapshotPool(
            runner.device,
            plan.nbytes,
            settings.slots,
            settings.gpu_bytes - reserved,
            settings.pinned_bytes,
        )

    def _packed(self, rows):
        return packed_rows(
            rows,
            len(self.layout.layers),
            self.layout.row_bytes,
            self.hidden_width,
            self.role == "decode",
        )

    def _attempt(self, req):
        if not hasattr(req, "_dflash_offload_attempt"):
            attempt = None
            if req.multimodal_inputs is None and req.origin_input_ids:
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
        attempt = req._dflash_offload_attempt
        if attempt is not None and (req.is_retracted or req.retraction_count):
            attempt.fail("request retracted")
        return attempt if attempt is not None and not attempt.error else None

    def _submit(self, attempt, ids, positions, tokens, hidden, deliver):
        plan = self._packed(ids.numel())

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

        def consume(storage):
            if not attempt.error:
                deliver(plan.views(storage))

        return self.pool.submit(plan.nbytes, fill, consume, attempt.fail)

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

    def snapshot_verify(self, reqs, positions, tokens, ids, hidden, width):
        commits = []
        for i, req in enumerate(reqs):
            attempt = self._attempt(req)
            if attempt is None:
                continue
            if any(
                t.numel() != len(reqs) * width
                or t.dtype not in (torch.int32, torch.int64)
                for t in (positions, tokens, ids)
            ):
                attempt.fail("invalid verify row metadata")
                continue
            if self.hidden_width and (
                hidden is None
                or hidden.dtype != torch.bfloat16
                or hidden.shape != (len(reqs) * width, self.hidden_width)
            ):
                attempt.fail("missing final target hidden rows")
                continue
            part = slice(i * width, (i + 1) * width)
            commit = VerifyCommit(req, attempt)

            # Bind per-request values before moving work to the consumer thread.
            def deliver(
                fields,
                path=commit.path,
                length=len(req.origin_input_ids),
                attempt=attempt,
            ):
                selected, labels = committed_slice(
                    path.result(self.settings.commit_timeout),
                    length,
                    fields["positions"],
                    fields["tokens"],
                )
                self._emit(
                    attempt,
                    fields,
                    selected,
                    fields["positions"][selected],
                    fields["tokens"][selected],
                    torch.tensor(labels, dtype=torch.int64, device="cpu"),
                )

            teacher = hidden[part] if self.hidden_width else None
            if self._submit(
                attempt,
                ids.reshape(-1)[part],
                positions.reshape(-1)[part],
                tokens.reshape(-1)[part],
                teacher,
                deliver,
            ):
                commits.append(commit)
        return commits

    def before_release(self, req):
        self.pool.fence_reuse()
        if self.role != "prefill":
            attempt = getattr(req, "_dflash_offload_attempt", None)
            reason = req.finished_reason
            if attempt is not None and (
                reason is None or reason.to_json()["type"] not in ("stop", "length")
            ):
                attempt.fail("decode released before completion")
            return
        attempt = self._attempt(req)
        if attempt is None:
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
            return
        for start, end in spans:
            ids = self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, start:end]
            prompt = tuple(req.origin_input_ids[start:end])

            def deliver(fields, start=start, end=end, prompt=prompt):
                self._emit(
                    attempt,
                    fields,
                    slice(None),
                    torch.arange(start, end, device="cpu"),
                    torch.tensor(prompt, dtype=torch.int64, device="cpu"),
                    None,
                )

            if not self._submit(attempt, ids, None, None, None, deliver):
                break
        self.pool.fence_reuse()
