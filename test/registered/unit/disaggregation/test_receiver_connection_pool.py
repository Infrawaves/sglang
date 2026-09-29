"""Unit tests for srt/disaggregation/common/conn — receiver connection_pool invalidation."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver
from sglang.test.test_utils import CustomTestCase


class _ConcreteReceiver(CommonKVReceiver):
    def poll(self) -> KVPoll:
        raise NotImplementedError

    def failure_exception(self):
        raise NotImplementedError


def _receiver(connection_pool, entries):
    receiver = object.__new__(_ConcreteReceiver)
    receiver.kv_mgr = SimpleNamespace(
        connection_pool=connection_pool,
        connection_lock=threading.Lock(),
        enable_deferred_decode_kv_release=False,
    )
    receiver._connection_pool_entries = entries
    receiver._connection_pool_entries_lock = threading.RLock()
    return receiver


class _FetchingReceiver(_ConcreteReceiver):
    def _get_bootstrap_info_from_server(
        self, prefill_dp_rank, prefill_cp_rank, target_tp_rank, target_pp_rank
    ):
        self.fetch_count += 1
        return {"rank_ip": "10.0.0.1", "rank_port": 2001, "pp_rank": target_pp_rank}

    def _register_kv_args(self):
        return True


def _fetching_receiver(connection_pool):
    receiver = object.__new__(_FetchingReceiver)
    receiver.kv_mgr = SimpleNamespace(
        connection_pool=connection_pool,
        connection_lock=threading.Lock(),
        is_mla_backend=False,
    )
    receiver.bootstrap_addr = "prefill:8998"
    receiver.bootstrap_room = 1
    receiver.prefill_dp_rank = 0
    receiver.prefill_info = SimpleNamespace(pp_size=1, attn_cp_size=1)
    receiver.target_cp_ranks = [0]
    receiver.target_tp_rank = 0
    receiver.target_tp_ranks = [0]
    receiver.target_pp_ranks = [0]
    receiver._connection_pool_entries = {}
    receiver._connection_pool_entries_lock = threading.RLock()
    receiver.fetch_count = 0
    return receiver


class TestReceiverConnectionPool(CustomTestCase):
    def test_invalidate_removes_matching_generation(self):
        stale = [
            {"rank_ip": "10.0.0.1", "rank_port": 1001},
            {"rank_ip": "10.0.0.1", "rank_port": 1002},
        ]
        retained = [{"rank_ip": "10.0.0.2", "rank_port": 2001}]
        receiver = _receiver(
            {"stale": stale, "retained": retained},
            {"stale": stale},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {"retained": retained})
        self.assertEqual(receiver._connection_pool_entries, {})

    def test_invalidate_preserves_concurrent_replacement_generation(self):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        replacement = [{"rank_ip": "10.0.0.1", "rank_port": 2001}]
        receiver = _receiver(
            {"key": replacement},
            {"key": stale},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {"key": replacement})

    def test_invalidate_removes_all_matching_cp_entries(self):
        stale_cp0 = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        stale_cp1 = [{"rank_ip": "10.0.0.1", "rank_port": 1002}]
        receiver = _receiver(
            {"cp0": stale_cp0, "cp1": stale_cp1},
            {"cp0": stale_cp0, "cp1": stale_cp1},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {})

    def test_next_receiver_refetches_after_invalidation(self):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        connection_pool = {"prefill:8998_0_0_0": stale}
        stale_receiver = _receiver(
            connection_pool,
            {"prefill:8998_0_0_0": stale},
        )
        stale_receiver.invalidate_cached_bootstrap_infos()

        receiver = _fetching_receiver(connection_pool)
        receiver._setup_bootstrap_infos()

        self.assertEqual(receiver.fetch_count, 1)
        self.assertEqual(receiver.bootstrap_infos[0]["rank_port"], 2001)
        self.assertIs(
            connection_pool["prefill:8998_0_0_0"],
            receiver._connection_pool_entries["prefill:8998_0_0_0"],
        )

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=3.0)
    def test_waiting_timeout_invalidates_cached_generation(self, _mock_time):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        receiver = _receiver({"key": stale}, {"key": stale})
        receiver.bootstrap_room = 1
        receiver.bootstrap_infos = stale
        receiver.init_time = 1.0
        receiver.abort_notified = True
        receiver.kv_mgr.waiting_timeout = 1.0
        receiver.kv_mgr.record_failure = Mock()
        receiver.kv_mgr.update_status = Mock()
        receiver._send_abort_notification = Mock()

        self.assertEqual(receiver._check_waiting_timeout(), KVPoll.Failed)
        self.assertEqual(receiver.kv_mgr.connection_pool, {})

    def test_abort_retry_drops_undrained_room_conflict(self):
        receiver = object.__new__(_ConcreteReceiver)
        receiver.bootstrap_room = 7
        receiver.bootstrap_infos = [{}]
        receiver._abort_token = "new-token"
        register = Mock(side_effect=RuntimeError("undrained room"))
        connection_lock = threading.Lock()
        receiver.kv_mgr = SimpleNamespace(
            connection_lock=connection_lock,
            enable_deferred_decode_kv_release=True,
            requires_transfer_drain=True,
            register_deferred_abort_room=register,
        )
        receiver._connect_to_bootstrap_server = Mock()

        receiver.retry_abort()

        register.assert_called_once_with(7, token="new-token")
        receiver._connect_to_bootstrap_server.assert_not_called()
        self.assertTrue(connection_lock.acquire(blocking=False))
        connection_lock.release()

    def test_stale_receiver_cannot_arm_or_notify_reused_room(self):
        receiver = object.__new__(_ConcreteReceiver)
        receiver.bootstrap_room = 7
        receiver.bootstrap_infos = [{}]
        receiver._bootstrap_setup_token = object()
        receiver.abort_notified = False
        receiver.kv_mgr = SimpleNamespace(
            _bootstrap_room_tokens={7: object()},
            register_deferred_abort_room=Mock(),
        )
        receiver._connect_to_bootstrap_server = Mock()

        receiver.ensure_abort_notified(force_arm=True)
        receiver.retry_abort()
        receiver._send_abort_notification(force_arm=True)

        self.assertFalse(receiver.abort_notified)
        receiver.kv_mgr.register_deferred_abort_room.assert_not_called()
        receiver._connect_to_bootstrap_server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
