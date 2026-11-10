from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.context_bucket import (
    CONTEXT_BUCKET_BOUNDS,
    context_bucket_index,
)
from sglang.srt.managers.load_snapshot import (
    DisaggregationMetrics,
    LoadSnapshot,
    LoRAMetrics,
    MemoryMetrics,
    QueueMetrics,
    SpeculativeMetrics,
)
from sglang.srt.managers.scheduler_components.request_list import ListKind
from sglang.srt.runtime_context import get_lora, get_parallel

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler_components.pool_stats_observer import (
        SchedulerPoolStatsObserver,
    )
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerLoadInquirer:
    disaggregation_mode: DisaggregationMode
    server_args: ServerArgs
    max_total_num_tokens: int
    max_running_requests: int
    pool_stats_observer: SchedulerPoolStatsObserver
    tp_worker: BaseTpWorker
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    spec_algorithm: SpeculativeAlgorithm
    get_running_batch: Callable
    get_waiting_queue: Callable
    waiting_queue_prefix_matched: Callable
    get_recent_cache_hit_rate: Callable
    get_stats: Callable
    get_chunked_req: Callable
    get_request_lists: Callable
    get_disagg_decode_prealloc_queue: Callable
    get_spec_total_num_accept_tokens: Callable
    get_spec_total_num_forward_ct: Callable
    get_total_prefill_uncached_tokens: Callable
    get_total_prefill_busy_us: Callable
    get_decode_moment_totals: Callable
    get_waiting_storage_span: (
        Callable[[CacheRequestHandle], tuple[int, int] | None] | None
    ) = None
    get_round_robin_requests: Callable = lambda: ()
    # Maps a req to the prefix end of its queried L3 hit, or None.
    get_pending_storage_hit_end: Callable = lambda req: None
    get_total_prefill_input_tokens: Callable = lambda: 0

    def _cached_prefix_len(self, req, device_host_len: int) -> int:
        # storage_hit_end already counts the device+host prefix the L3 span
        # starts after, so take the max instead of adding the two.
        storage_hit_end = self.get_pending_storage_hit_end(req)
        if storage_hit_end is not None:
            device_host_len = max(device_host_len, storage_hit_end)
        return min(device_host_len, req.seqlen)

    def _get_num_pending_tokens(self, chunk_deduct: int = 0) -> int:
        """Get the total number of tokens pending prefill.

        This includes tokens from waiting queue requests plus remaining tokens
        from the currently chunked request.

        Args:
            chunk_deduct: extra tokens to subtract from the chunked request's
                remaining count. At batch-scheduling time the current chunk
                has been planned but ``prefix_len`` does not yet include it,
                so callers pass ``extend_input_len`` here. At load-reporting
                time ``prefix_len`` is already up-to-date, so the default
                0 is correct.
        """
        num_pending_tokens = sum(req.seqlen for req in self.get_waiting_queue())
        if self.get_chunked_req() is not None:
            req = self.get_chunked_req()
            num_pending_tokens += req.seqlen - req.prefix_len - chunk_deduct
        num_pending_tokens += sum(
            max(0, req.seqlen - req.kv.kv_allocated_len)
            for req in self.get_round_robin_requests()
            if req is not self.get_chunked_req()
        )
        return num_pending_tokens

    def get_num_waiting_uncached_tokens(self) -> int:
        """Estimate input tokens waiting for prefill compute."""
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            return 0
        waiting_queue_prefix_matched = self.waiting_queue_prefix_matched()
        cache_miss_rate = 1.0 - self.get_recent_cache_hit_rate()
        num_tokens = 0
        for req in self.get_waiting_queue():
            if waiting_queue_prefix_matched:
                matched = req.num_matched_prefix_tokens
                uncached = max(0, req.seqlen - matched)
                span = None
                if self.get_waiting_storage_span is not None:
                    span = self.get_waiting_storage_span(req.cache_request_handle)
                if span is not None:
                    storage_tokens = max(
                        0, min(span[1], req.seqlen) - max(span[0], matched)
                    )
                    uncached -= storage_tokens
                else:
                    # Cache mode reports no span; use the queried L3 hit end.
                    uncached = req.seqlen - self._cached_prefix_len(req, matched)
            else:
                # The historical hit rate already includes admitted storage hits.
                uncached = int(req.seqlen * cache_miss_rate)
            num_tokens += uncached
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Discount by the enqueue-time match plus the queried L3 hit, not the
            # windowed hit rate: that rate freezes when bootstrap stalls.
            num_tokens += sum(
                req.seqlen
                - self._cached_prefix_len(req, req.storage_prefetch_last_match_len or 0)
                for req in self.get_request_lists()[ListKind.PD_BOOTSTRAP]
            )
        cr = self.get_chunked_req()
        if cr is not None:
            num_tokens += max(0, cr.seqlen - cr.prefix_len)
        num_tokens += sum(
            max(0, req.seqlen - req.kv.kv_allocated_len)
            for req in self.get_round_robin_requests()
            if req is not cr
        )
        return num_tokens

    def get_loads(self) -> LoadSnapshot:
        """Build the per-DP-rank load snapshot for DP balancing and /v1/loads."""
        stats = self.get_stats()
        running_reqs = self.get_running_batch().reqs
        num_running_reqs = len(running_reqs)

        lists = self.get_request_lists()
        waiting_queues = [self.get_waiting_queue()]
        pending_token_queues = [self.get_waiting_queue()]
        awaiting_kv_tokens = 0
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            prefill_bootstrap_queue = lists[ListKind.PD_BOOTSTRAP]
            waiting_queues.append(prefill_bootstrap_queue)
            pending_token_queues.append(prefill_bootstrap_queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            decode_prealloc_queue = lists[ListKind.PD_PREALLOC]
            decode_transfer_queue = lists[ListKind.PD_RECEIVING]
            decode_retracted_queue = lists[ListKind.PD_RETRACTED]
            waiting_queues.append(decode_prealloc_queue)
            waiting_queues.append(decode_transfer_queue)
            waiting_queues.append(decode_retracted_queue)
            # In disaggregated decode, transfer-queue requests and transferred
            # waiting-queue requests have already pre-allocated decode-side KV
            # slots, so they are already included in num_used_tokens.
            pending_token_queues = [decode_prealloc_queue, decode_retracted_queue]
            # KV not yet arrived from the prefill side.
            awaiting_kv_tokens = sum(
                req.seqlen
                for queue in (decode_prealloc_queue, decode_transfer_queue)
                for req in queue
            )

        suspended = [
            r
            for r in self.get_round_robin_requests()
            if r is not self.get_chunked_req()
        ]
        num_waiting_reqs = sum(len(queue) for queue in waiting_queues) + len(suspended)
        num_used_tokens, kv_token_usage = (
            self.pool_stats_observer.get_pool_stats().get_kv_token_stats()
        )
        num_total_tokens = num_used_tokens + sum(
            req.seqlen for queue in pending_token_queues for req in queue
        )
        num_total_tokens += sum(
            max(0, r.seqlen - r.kv.kv_allocated_len) for r in suspended
        )
        num_active_tokens = max(0, num_total_tokens - awaiting_kv_tokens)

        context_length_histogram = None
        num_context_tokens = 0
        if get_parallel().load_balance_method == "context_bucket":
            context_length_histogram = [0] * (len(CONTEXT_BUCKET_BOUNDS) + 1)
            # The queues represent disjoint request lifecycle states. Count
            # transferred requests once, even though their KV is already in
            # num_used_tokens. Req.seqlen is CPU metadata (input + output).
            for queue in (running_reqs, *waiting_queues):
                for req in queue:
                    length = req.seqlen
                    context_length_histogram[context_bucket_index(length)] += 1
                    num_context_tokens += length

        memory = None
        try:
            memory = MemoryMetrics(
                weight_gb=round(self.tp_worker.model_runner.weight_load_mem_usage, 3),
                kv_cache_gb=round(
                    self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 3
                ),
                graph_gb=round(sum(self.tp_worker.graph_memory_usage.values()), 3),
                token_capacity=int(self.max_total_num_tokens),
            )
        except (AttributeError, TypeError) as e:
            logger.debug(f"Memory metrics not available: {e}")

        speculative = None
        if (
            not self.spec_algorithm.is_none()
            and self.get_spec_total_num_forward_ct() > 0
        ):
            speculative = SpeculativeMetrics(
                accept_length=(
                    self.get_spec_total_num_accept_tokens()
                    / self.get_spec_total_num_forward_ct()
                ),
                accept_rate=stats.spec_accept_rate,
            )

        lora = None
        if get_lora().enable_lora:
            lora = LoRAMetrics(
                slots_used=stats.lora_pool_slots_used,
                slots_total=stats.lora_pool_slots_total,
                utilization=stats.lora_pool_utilization,
            )

        mode_str = "null"
        prefill_bootstrap = prefill_inflight = 0
        decode_prealloc = decode_transfer = decode_retracted = 0
        decode_prealloc_ready = 0
        num_prealloc_ready_tokens = 0
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            mode_str = "prefill"
            prefill_bootstrap = len(lists[ListKind.PD_BOOTSTRAP])
            prefill_inflight = len(lists[ListKind.PD_SENDING])
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            mode_str = "decode"
            decode_prealloc = len(lists[ListKind.PD_PREALLOC])
            decode_transfer = len(lists[ListKind.PD_RECEIVING])
            decode_retracted = len(lists[ListKind.PD_RETRACTED])
            ready_reqs = [
                decode_req.req
                for decode_req in self.get_disagg_decode_prealloc_queue().queue.items()
                if decode_req.waiting_for_input
            ]
            decode_prealloc_ready = len(ready_reqs)
            num_prealloc_ready_tokens = sum(req.seqlen for req in ready_reqs)
        disaggregation = DisaggregationMetrics(
            mode=mode_str,
            prefill_bootstrap_queue_reqs=prefill_bootstrap,
            prefill_inflight_queue_reqs=prefill_inflight,
            decode_prealloc_queue_reqs=decode_prealloc,
            decode_transfer_queue_reqs=decode_transfer,
            decode_retracted_queue_reqs=decode_retracted,
            kv_transfer_speed_gb_s=stats.kv_transfer_speed_gb_s,
            kv_transfer_latency_ms=stats.kv_transfer_latency_ms,
        )

        queues = QueueMetrics(
            waiting=len(self.get_waiting_queue()),
            grammar=stats.num_grammar_queue_reqs,
            paused=stats.num_paused_reqs,
            retracted=stats.num_retracted_reqs,
            prealloc_ready=decode_prealloc_ready,
        )

        totals = self.get_decode_moment_totals()
        decode_moments = list(totals) if totals[0] > 0 else None

        return LoadSnapshot(
            dp_rank=int(get_parallel().dp_rank)
            if get_parallel().dp_rank is not None
            else 0,
            timestamp=time.time(),
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_waiting_uncached_tokens=self.get_num_waiting_uncached_tokens(),
            num_used_tokens=num_used_tokens,
            num_total_tokens=num_total_tokens,
            num_active_tokens=num_active_tokens,
            context_length_histogram=context_length_histogram,
            num_context_tokens=num_context_tokens,
            num_prealloc_ready_tokens=num_prealloc_ready_tokens,
            max_total_num_tokens=self.max_total_num_tokens,
            max_running_requests=self.max_running_requests,
            token_usage=round(kv_token_usage, 4),
            gen_throughput=round(stats.gen_throughput, 2),
            cache_hit_rate=round(stats.cache_hit_rate, 4),
            utilization=round(stats.utilization, 4),
            memory=memory,
            speculative=speculative,
            lora=lora,
            disaggregation=disaggregation,
            queues=queues,
            total_prefill_uncached_tokens=self.get_total_prefill_uncached_tokens(),
            total_prefill_busy_us=self.get_total_prefill_busy_us(),
            total_prefill_input_tokens=self.get_total_prefill_input_tokens(),
            decode_moments=decode_moments,
        )
