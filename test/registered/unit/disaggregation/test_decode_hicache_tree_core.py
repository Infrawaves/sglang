"""Unit tests for decode HiCache TreeCore interactions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation import utils as disagg_utils
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestDecodeHiCacheTreeCore(CustomTestCase):
    def test_stale_prefetch_anchor_degrades_to_l2(self):
        tree_cache = SimpleNamespace(
            tree_core=SimpleNamespace(is_eagle=False),
            hicache_storage_pass_prefix_keys=True,
            ongoing_prefetch={},
            get_last_hash_value=Mock(side_effect=KeyError(22)),
            get_prefix_hash_values=Mock(),
            prefetch_from_storage=Mock(),
        )
        harness = SimpleNamespace(tree_cache=tree_cache)
        req = SimpleNamespace(
            rid="req-0",
            cache_request_handle=CacheRequestHandle("req-0", 0),
            origin_input_ids=[0, 1, 2, 3, 4, 5],
            extra_key=None,
            cache_salt=None,
        )
        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.tensor([10, 11]),
            l2_host_hit_length=2,
            l3_storage_hit_length=2,
            last_device_node=11,
            last_host_node=22,
        )

        DecodeHiCachePreallocMixin._start_hicache_prefetch(harness, req, prefix_match)

        self.assertEqual(prefix_match.l3_storage_hit_length, 0)
        self.assertFalse(prefix_match.prefetch_registered)
        tree_cache.get_prefix_hash_values.assert_not_called()
        tree_cache.prefetch_from_storage.assert_not_called()

    def test_restored_indices_read_off_the_restored_path(self):
        restored_path = torch.tensor([10, 11, 12, 13, 14])
        rematch = SimpleNamespace(
            device_prefix_len=2,
            last_device_node="rematched",
            best_match_node="restored",
            host_hit_length=2,
        )
        last_node_during_load = []

        def init_load_back(params):
            last_node_during_load.append(req.last_node)
            return 2, "restored"

        tree_cache = SimpleNamespace(
            tree_core=SimpleNamespace(is_eagle=False),
            match_prefix=Mock(return_value=rematch),
            init_load_back=Mock(side_effect=init_load_back),
            path_device_indices=Mock(return_value=restored_path),
            lock=Mock(return_value="restore-lock"),
        )
        harness = SimpleNamespace(tree_cache=tree_cache)
        req = SimpleNamespace(
            rid="req-0",
            cache_request_handle=CacheRequestHandle("req-0", 0),
            origin_input_ids=[0, 1, 2, 3, 4, 5],
            last_node="prealloc",
        )
        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.tensor([10, 11]),
            l2_host_hit_length=2,
            l3_storage_hit_length=0,
            last_device_node="prealloc",
        )
        decode_req = SimpleNamespace(req=req, prefix_match=prefix_match)

        with patch(
            "sglang.srt.disaggregation.decode_hicache_mixin._req_radix_key",
            return_value="key",
        ):
            queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
                harness, decode_req
            )

        self.assertTrue(queued)
        # init_load_back reads the rematched device boundary off req.last_node;
        # the request keeps its own node afterwards.
        self.assertEqual(last_node_during_load, ["rematched"])
        self.assertEqual(req.last_node, "prealloc")
        tree_cache.path_device_indices.assert_called_once_with("restored")
        self.assertEqual(decode_req.hicache_restored_kv_indices.tolist(), [12, 13])
        tree_cache.lock.assert_called_once_with("restored")


class TestHiCacheRestoreConsensus(CustomTestCase):
    """Local restore and staging state reach the MIN-reduced poll (#43562)."""

    @staticmethod
    def _request(restore, transport=KVPoll.Success, *, require_staging=False):
        return SimpleNamespace(
            hicache_restore_status=restore,
            kv_receiver=SimpleNamespace(
                poll=lambda: transport, require_staging=require_staging
            ),
            metadata_buffer_index=0,
            req=SimpleNamespace(
                rid="req-123", bootstrap_host="127.0.0.1", bootstrap_room=123
            ),
        )

    @staticmethod
    def _poll(request, handler, metadata=None, *, hicache=True):
        pollers = (
            [HiCacheRestoreGatedKVReceiver(request)]
            if hicache
            else [request.kv_receiver]
        )
        with (
            envs.SGLANG_TEST_DISAGG_FAILURE_PROB.override(0),
            patch.object(
                disagg_utils, "_all_reduce_polls", side_effect=lambda polls, _: polls
            ) as reduce,
        ):
            result = disagg_utils.poll_and_all_reduce_with_staging(
                [request],
                handler,
                object(),
                metadata_buffers=metadata,
                pollers=pollers,
            )
        return result, reduce.call_args.args[0]

    def test_staging_restore_state_is_included_before_all_reduce(self):
        handler = SimpleNamespace(
            is_done=lambda _: True, is_failed=lambda _: False, advance_scatter=Mock()
        )
        for require_staging in (False, True):
            for restore, hicache, metadata_room, expected in (
                (HiCacheRestoreResult.FAILED, True, 123, KVPoll.Failed),
                (HiCacheRestoreResult.PENDING, True, 123, KVPoll.Transferring),
                (HiCacheRestoreResult.READY, True, 123, KVPoll.Success),
                (HiCacheRestoreResult.READY, True, 0, KVPoll.Transferring),
                # Without HiCache, DecodeRequest's default PENDING is inert.
                (HiCacheRestoreResult.PENDING, False, 123, KVPoll.Success),
            ):
                with self.subTest(
                    staging=require_staging,
                    restore=restore,
                    hicache=hicache,
                    metadata_room=metadata_room,
                ):
                    request = self._request(restore, require_staging=require_staging)
                    metadata = SimpleNamespace(
                        bootstrap_room=torch.tensor([[metadata_room]])
                    )
                    result, reduced = self._poll(
                        request, handler, metadata, hicache=hicache
                    )
                    self.assertEqual(result, [int(expected)])
                    self.assertEqual(reduced, [int(expected)])

    def test_staging_failure_is_not_hidden_by_pending_restore(self):
        request = self._request(HiCacheRestoreResult.PENDING, require_staging=True)
        handler = SimpleNamespace(
            is_done=lambda _: False, is_failed=lambda _: True, advance_scatter=Mock()
        )
        result, _ = self._poll(request, handler)
        self.assertEqual(result, [int(KVPoll.Failed)])


if __name__ == "__main__":
    unittest.main()
