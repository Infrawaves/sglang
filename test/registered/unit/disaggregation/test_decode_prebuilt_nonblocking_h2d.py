"""PD decode admission must not block the scheduler CPU on the in-flight forward.

Regression test for a decode stall: admitting a new request ran a synchronous
pageable H2D copy (``torch.tensor(..., device="cuda")``) right after
``schedule_stream.wait_stream(forward_stream)``. The copy's stream synchronize
parked the scheduler until the whole in-flight forward finished, so the next
batch's CPU prep (DP metadata all_gather, run_batch, graph launch) ran while
every GPU in the EP group sat idle.
"""

import time
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
    _h2d_nonblocking,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# Long enough that the fake forward is still running when admission returns.
_SLEEP_CYCLES = 1_000_000_000


class _FakeFutureMap:
    def __init__(self, size: int):
        self.output_tokens_buf = torch.full(
            (size,), -1, dtype=torch.int64, device="cuda"
        )

    def stash(self, indices, payload):
        self.output_tokens_buf[indices] = payload.bonus_tokens


def _make_prebuilt_batch(req_pool_indices, last_tokens):
    reqs = [
        SimpleNamespace(
            output_ids=[tok],
            swa_branching_seqlen=None,
            grammar=None,
            skip_radix_cache_insert=True,
        )
        for tok in last_tokens
    ]
    return SimpleNamespace(
        reqs=reqs,
        tree_cache=None,
        device="cuda",
        req_pool_indices=torch.tensor(req_pool_indices, device="cuda"),
        spec_algorithm=SimpleNamespace(build_disagg_draft_input=lambda *_: None),
        spec_info=None,
        input_ids=None,
    )


def _sync_error(fn):
    """The RuntimeError torch raises if ``fn`` synchronizes, or None."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn()
    except RuntimeError as exc:
        return exc
    finally:
        torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
    return None


class TestH2DNonblockingCpuFallback(unittest.TestCase):
    def test_cpu_device_keeps_plain_copy(self):
        out = _h2d_nonblocking([1, 2, 3], torch.int32, "cpu")
        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out.tolist(), [1, 2, 3])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDecodePrebuiltNonblockingH2D(unittest.TestCase):
    def setUp(self):
        # Warm the pinned host allocator so timing reflects steady state.
        _h2d_nonblocking([0, 0], torch.int64, "cuda")
        torch.cuda.synchronize()

    def test_h2d_values_and_no_sync(self):
        holder = {}

        def run():
            holder["out"] = _h2d_nonblocking([4, 5, 6], torch.int32, "cuda")

        self.assertIsNone(_sync_error(run))
        out = holder["out"]
        self.assertEqual(out.device.type, "cuda")
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out.cpu().tolist(), [4, 5, 6])

    def test_process_prebuilt_has_no_sync(self):
        future_map = _FakeFutureMap(8)
        batch = _make_prebuilt_batch([1, 2], [7, 8])
        err = _sync_error(
            lambda: ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(
                batch, future_map
            )
        )
        self.assertIsNone(err)
        self.assertEqual(future_map.output_tokens_buf.cpu().tolist()[1:3], [7, 8])

    def test_process_prebuilt_does_not_wait_for_inflight_forward(self):
        forward_stream = torch.cuda.Stream()
        schedule_stream = torch.cuda.Stream()
        future_map = _FakeFutureMap(8)
        batch = _make_prebuilt_batch([3, 5], [11, 22])
        torch.cuda.synchronize()

        with torch.cuda.stream(forward_stream):
            # The in-flight forward: runs long, then writes the req_pool row
            # that the newly admitted request reuses.
            torch.cuda._sleep(_SLEEP_CYCLES)
            future_map.output_tokens_buf[3] = -7
        forward_done = torch.cuda.Event()
        forward_done.record(forward_stream)

        with torch.cuda.stream(schedule_stream):
            # The admission fence from _get_new_prebuilt_batch.
            schedule_stream.wait_stream(forward_stream)
            self.assertFalse(forward_done.query(), "fake forward ended too early")
            t0 = time.perf_counter()
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(batch, future_map)
            cpu_s = time.perf_counter() - t0
            forward_still_running = not forward_done.query()

        torch.cuda.synchronize()
        self.assertLess(cpu_s, 0.1, f"admission blocked the CPU for {cpu_s:.3f}s")
        self.assertTrue(forward_still_running)
        # GPU ordering is unchanged: the stash lands after the forward's write.
        buf = future_map.output_tokens_buf.cpu().tolist()
        self.assertEqual(buf[3], 11)
        self.assertEqual(buf[5], 22)


if __name__ == "__main__":
    unittest.main()
