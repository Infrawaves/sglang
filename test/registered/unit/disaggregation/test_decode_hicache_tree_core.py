"""Unit tests for decode HiCache TreeCore interactions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation import utils as disagg_utils
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodePrefixMatch,
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDecodeHiCacheTreeCore(CustomTestCase):
    def test_storage_probe_and_prefetch_use_node_handles(self):
        ongoing_prefetch = {}

        def register_prefetch(req_id, *_args, **_kwargs):
            ongoing_prefetch[req_id] = object()

        tree_cache = SimpleNamespace(
            hicache_storage_pass_prefix_keys=True,
            ongoing_prefetch=ongoing_prefetch,
            is_backuped=Mock(return_value=True),
            is_root=Mock(return_value=False),
            get_last_hash_value=Mock(return_value="h2"),
            get_prefix_hash_values=Mock(return_value=["h0", "h1"]),
            query_storage_hit_length=Mock(return_value=2),
            prefetch_from_storage=Mock(side_effect=register_prefetch),
        )
        harness = SimpleNamespace(
            scheduler=SimpleNamespace(enable_decode_hicache=True),
            tree_cache=tree_cache,
        )
        req = SimpleNamespace(
            rid="req-0",
            origin_input_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            extra_key="model",
            cache_salt="tenant-a",
        )
        result = SimpleNamespace(
            device_indices=torch.tensor([10, 11]),
            host_hit_length=2,
            last_device_node=11,
            last_host_node=22,
        )

        prefix_match = DecodeHiCachePreallocMixin._build_decode_prefix_match(
            harness, req, result
        )

        self.assertEqual(prefix_match.l3_storage_hit_length, 2)
        tree_cache.query_storage_hit_length.assert_called_once_with(
            22,
            [4, 5, 6, 7],
            "h2",
            ["h0", "h1"],
            extra_key="model",
            cache_salt="tenant-a",
        )

        DecodeHiCachePreallocMixin._start_hicache_prefetch(harness, req, prefix_match)

        self.assertTrue(prefix_match.prefetch_registered)
        tree_cache.prefetch_from_storage.assert_called_once_with(
            "req-0",
            22,
            [4, 5],
            "h2",
            ["h0", "h1"],
            extra_key="model",
            cache_salt="tenant-a",
        )

    def test_stale_prefetch_anchor_degrades_to_l2(self):
        tree_cache = SimpleNamespace(
            hicache_storage_pass_prefix_keys=True,
            ongoing_prefetch={},
            get_last_hash_value=Mock(side_effect=KeyError(22)),
            get_prefix_hash_values=Mock(),
            prefetch_from_storage=Mock(),
        )
        harness = SimpleNamespace(tree_cache=tree_cache)
        req = SimpleNamespace(
            rid="req-0",
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


class TestHiCacheRestoreConsensus(CustomTestCase):
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

    def test_restore_failure_precedes_every_transport_state(self):
        for transport in (
            KVPoll.Bootstrapping,
            KVPoll.WaitingForInput,
            KVPoll.Transferring,
            KVPoll.Success,
            KVPoll.Failed,
        ):
            with self.subTest(transport=transport):
                request = self._request(HiCacheRestoreResult.FAILED, transport)
                self.assertEqual(
                    HiCacheRestoreGatedKVReceiver(request).poll(), KVPoll.Failed
                )

    def test_pending_restore_preserves_transport_failure(self):
        for restore in (HiCacheRestoreResult.PENDING, HiCacheRestoreResult.READY):
            with self.subTest(restore=restore):
                request = self._request(restore, KVPoll.Failed)
                self.assertEqual(
                    HiCacheRestoreGatedKVReceiver(request).poll(), KVPoll.Failed
                )
                request = self._request(restore)
                self.assertEqual(
                    HiCacheRestoreGatedKVReceiver(request).poll(),
                    KVPoll.Transferring
                    if restore == HiCacheRestoreResult.PENDING
                    else KVPoll.Success,
                )

    def test_staging_restore_state_is_included_before_all_reduce(self):
        for require_staging in (False, True):
            for restore, enabled, metadata_room, expected in (
                (HiCacheRestoreResult.FAILED, True, 123, KVPoll.Failed),
                (HiCacheRestoreResult.FAILED, True, 0, KVPoll.Failed),
                (HiCacheRestoreResult.PENDING, True, 123, KVPoll.Transferring),
                (HiCacheRestoreResult.READY, True, 123, KVPoll.Success),
                (HiCacheRestoreResult.READY, True, 0, KVPoll.Transferring),
                # Without HiCache, DecodeRequest's default PENDING is inert.
                (HiCacheRestoreResult.PENDING, False, 123, KVPoll.Success),
            ):
                with self.subTest(
                    staging=require_staging,
                    restore=restore,
                    enabled=enabled,
                    metadata_room=metadata_room,
                ):
                    request = self._request(restore, require_staging=require_staging)
                    handler = SimpleNamespace(
                        is_done=lambda _: True,
                        is_failed=lambda _: False,
                        advance_scatter=Mock(),
                    )
                    metadata = SimpleNamespace(
                        bootstrap_room=torch.tensor([[metadata_room]])
                    )
                    with (
                        envs.SGLANG_TEST_DISAGG_FAILURE_PROB.override(0),
                        patch.object(
                            disagg_utils,
                            "_all_reduce_polls",
                            side_effect=lambda polls, _: polls,
                        ) as reduce,
                    ):
                        result = disagg_utils.poll_and_all_reduce_with_staging(
                            [request],
                            handler,
                            object(),
                            metadata_buffers=metadata,
                            enable_decode_hicache=enabled,
                        )
                    self.assertEqual(result, [int(expected)])
                    self.assertEqual(reduce.call_args.args[0], [int(expected)])

    def test_staging_failure_is_not_hidden_by_pending_restore(self):
        request = self._request(HiCacheRestoreResult.PENDING, require_staging=True)
        handler = SimpleNamespace(
            is_done=lambda _: False,
            is_failed=lambda _: True,
            advance_scatter=Mock(),
        )
        with (
            envs.SGLANG_TEST_DISAGG_FAILURE_PROB.override(0),
            patch.object(
                disagg_utils, "_all_reduce_polls", side_effect=lambda polls, _: polls
            ),
        ):
            result = disagg_utils.poll_and_all_reduce_with_staging(
                [request], handler, object(), enable_decode_hicache=True
            )
        self.assertEqual(result, [int(KVPoll.Failed)])


if __name__ == "__main__":
    unittest.main()
