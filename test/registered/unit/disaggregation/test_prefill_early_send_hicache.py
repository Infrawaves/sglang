"""CPU coverage for skipping the cached-prefix early send during HiCache load-back."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ForwardLaunched(Exception):
    pass


class _EarlySendScheduler(SchedulerDisaggregationPrefillMixin):
    """Just enough Scheduler state for run_batch to reach the forward launch."""

    def __init__(self):
        self.scheduler_stage_metrics = None
        self.metrics_reporter = SimpleNamespace(record_scheduler_active=Mock())
        self.forward_ct = 0
        self._sched_idled = False
        self.processed_tokens_counter = 0
        self.scripted_scheduler_hook = None
        self.profiler_manager = SimpleNamespace(_profile_batch_predicate=Mock())
        self.forward_sleep_time = None
        self.disaggregation_mode = DisaggregationMode.PREFILL
        self.enable_staging = False
        self.enable_overlap = False
        self.token_to_kv_pool_allocator = SimpleNamespace(page_size=64)
        self.send_kv_chunk = Mock()

    @property
    def is_generation(self):
        raise _ForwardLaunched


class TestCachedPrefixEarlySend(CustomTestCase):
    def test_batch_that_starts_a_hicache_load_back_does_not_early_send(self):
        """A request can match pages whose HiCache load-back was submitted in the same
        build; early-sending them would read KV the H2D has not written yet (#44070)."""
        for consumer_index, expect_send in ((-1, True), (0, False)):
            scheduler = _EarlySendScheduler()
            req = SimpleNamespace(
                prefix_indices=list(range(128)),
                host_hit_length=0,
                start_send_idx=0,
                pending_bootstrap=False,
                early_send_prefix_end=None,
            )
            batch = SimpleNamespace(
                reqs=[req],
                hicache_consumer_index=consumer_index,
                extend_num_tokens=32,
                forward_mode=SimpleNamespace(
                    is_split_prefill=lambda: False, is_prebuilt=lambda: False
                ),
            )
            with self.subTest(consumer_index=consumer_index):
                with envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.override(True):
                    with self.assertRaises(_ForwardLaunched):
                        Scheduler.run_batch(scheduler, batch)
                if expect_send:
                    scheduler.send_kv_chunk.assert_called_once_with(
                        req, last_chunk=False, end_idx=128
                    )
                else:
                    scheduler.send_kv_chunk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
