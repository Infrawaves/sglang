"""Unit tests for srt/disaggregation/common/conn -- receiver connection_pool
invalidation and the decode-side bootstrap handshake lifecycle."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from disagg_test_utils import (
    complete_receiver_setup,
    make_decode_kv_manager,
    prefill_info_stub,
)

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver
from sglang.test.test_utils import CustomTestCase

ADDR = "prefill:8998"


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
    return receiver


class _FetchingReceiver(_ConcreteReceiver):
    fetch_count = 0
    fetch_gate = None

    def _get_bootstrap_info_from_server(
        self, prefill_dp_rank, prefill_cp_rank, target_tp_rank, target_pp_rank
    ):
        if self.fetch_gate is not None:
            self.fetch_gate.wait(5)
        type(self).fetch_count += 1
        return {"rank_ip": "10.0.0.1", "rank_port": 2001, "pp_rank": target_pp_rank}

    def _register_kv_args(self, bootstrap_infos):
        return True


class TestReceiverConnectionPool(CustomTestCase):
    def setUp(self):
        _FetchingReceiver.fetch_count = 0
        _FetchingReceiver.fetch_gate = None

    def _decode_manager(self, **overrides):
        mgr = make_decode_kv_manager(self, **overrides)
        mgr.prefill_info_table[ADDR] = prefill_info_stub()
        return mgr

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
        mgr = self._decode_manager()
        mgr.connection_pool[f"{ADDR}_0_0_0"] = stale
        stale_receiver = _receiver(mgr.connection_pool, {f"{ADDR}_0_0_0": stale})
        stale_receiver.kv_mgr = mgr
        stale_receiver.invalidate_cached_bootstrap_infos()

        receiver = _FetchingReceiver(mgr, ADDR, 1)
        receiver.init(0)
        complete_receiver_setup(receiver)

        self.assertEqual(_FetchingReceiver.fetch_count, 1)
        self.assertEqual(receiver.bootstrap_infos[0]["rank_port"], 2001)
        self.assertIs(
            mgr.connection_pool[f"{ADDR}_0_0_0"],
            receiver._connection_pool_entries[f"{ADDR}_0_0_0"],
        )
        self.assertEqual(mgr.request_status[1], KVPoll.WaitingForInput)

    def test_cached_connection_completes_handshake_inside_init(self):
        """A cache hit must reach WaitingForInput in init() itself: deferring it
        to an executor costs every request an extra scheduler cycle of TTFT."""
        mgr = self._decode_manager()
        cached = [{"rank_ip": "10.0.0.1", "rank_port": 2001}]
        mgr.connection_pool[f"{ADDR}_0_0_0"] = cached

        receiver = _FetchingReceiver(mgr, ADDR, 1)
        receiver.init(0)

        self.assertEqual(mgr.request_status[1], KVPoll.WaitingForInput)
        self.assertEqual(receiver.bootstrap_infos, cached)
        self.assertEqual(_FetchingReceiver.fetch_count, 0)

    def test_concurrent_cache_misses_share_one_fetch(self):
        """Receivers that miss the same connection key while its fetch is in
        flight wait for that one fetch instead of each querying the prefill."""
        mgr = self._decode_manager()
        _FetchingReceiver.fetch_gate = threading.Event()
        self.addCleanup(_FetchingReceiver.fetch_gate.set)

        receivers = [_FetchingReceiver(mgr, ADDR, room) for room in (1, 2, 3)]
        for receiver in receivers:
            receiver.init(0)
        self.assertEqual(mgr.request_status[1], KVPoll.Bootstrapping)
        self.assertIsNone(receivers[0].bootstrap_infos)

        _FetchingReceiver.fetch_gate.set()
        for receiver in receivers:
            complete_receiver_setup(receiver)

        self.assertEqual(_FetchingReceiver.fetch_count, 1)
        for room, receiver in zip((1, 2, 3), receivers):
            self.assertEqual(mgr.request_status[room], KVPoll.WaitingForInput)
            self.assertIs(receiver.bootstrap_infos[0], receivers[0].bootstrap_infos[0])

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=3.0)
    def test_waiting_timeout_invalidates_cached_generation(self, _mock_time):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        mgr = self._decode_manager(waiting_timeout=1.0)
        mgr.connection_pool["key"] = stale
        receiver = _ConcreteReceiver(mgr, ADDR, 1)
        receiver._connection_pool_entries = {"key": stale}
        receiver.bootstrap_infos = stale
        receiver.init_time = 1.0
        receiver._send_abort_notification = Mock()

        self.assertEqual(receiver._check_waiting_timeout(), KVPoll.Failed)
        self.assertEqual(mgr.connection_pool, {})
        self.assertEqual(mgr.request_status[1], KVPoll.Failed)

    def test_bootstrap_deadline_counts_from_request_arrival(self):
        """The decode-side handshake deadline runs from receiver creation, so time
        spent waiting for topology or the DP rank counts against it."""
        clock = "sglang.srt.disaggregation.common.conn.time.monotonic"
        with patch(clock, return_value=100.0):
            mgr = self._decode_manager()
            mgr.decode_bootstrap_timeout = 10
            receiver = _ConcreteReceiver(mgr, ADDR, 1)

        with patch(clock, return_value=109.9):
            self.assertEqual(receiver._poll_bootstrapping(), KVPoll.Bootstrapping)
        with patch(clock, return_value=110.0):
            self.assertEqual(receiver._poll_bootstrapping(), KVPoll.Failed)
        self.assertEqual(mgr.request_status[1], KVPoll.Failed)
        self.assertIn("Bootstrapping", mgr.failure_records[1])

    def test_decode_bootstrap_timeout_defaults_to_bootstrap_timeout(self):
        """Leaving the decode-side knob unset keeps the deadline deployments
        already tuned through SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT."""
        from sglang.srt.environ import envs

        with (
            envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.override(3600),
            envs.SGLANG_DISAGGREGATION_DECODE_BOOTSTRAP_TIMEOUT.override(None),
        ):
            self.assertEqual(self._decode_manager().decode_bootstrap_timeout, 3600)
        with (
            envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.override(3600),
            envs.SGLANG_DISAGGREGATION_DECODE_BOOTSTRAP_TIMEOUT.override(30),
        ):
            self.assertEqual(self._decode_manager().decode_bootstrap_timeout, 30)

    def test_setup_finishing_after_abort_does_not_publish(self):
        """A fetch that lands after the request was aborted must not move the
        room back to WaitingForInput or hand the receiver bootstrap infos."""
        mgr = self._decode_manager()
        _FetchingReceiver.fetch_gate = threading.Event()
        self.addCleanup(_FetchingReceiver.fetch_gate.set)
        receiver = _FetchingReceiver(mgr, ADDR, 1)
        receiver.init(0)
        pending = [part for _, part in receiver._setup_parts]

        receiver.abort()
        _FetchingReceiver.fetch_gate.set()
        for future in pending:
            future.result(timeout=5)
        receiver._advance_setup()

        self.assertEqual(mgr.request_status[1], KVPoll.Failed)
        self.assertIsNone(receiver.bootstrap_infos)

    def test_stale_receiver_clear_keeps_reused_room(self):
        """clear() of a receiver whose room was reused by a newer request must
        leave the newer request's status and room tracking alone."""
        mgr = self._decode_manager()
        old = _ConcreteReceiver(mgr, ADDR, 7)
        new = _ConcreteReceiver(mgr, ADDR, 7)

        old.clear()

        self.assertEqual(mgr.request_status[7], KVPoll.Bootstrapping)
        self.assertIn(7, mgr.addr_to_rooms_tracker[ADDR])
        new.clear()
        self.assertNotIn(7, mgr.request_status)

    def test_node_failure_does_not_match_address_prefix(self):
        """Evicting host:80 must not drop the cached connections of host:8000."""
        mgr = self._decode_manager()
        mgr.connection_pool["host:8000_0_0_0"] = [{"rank_ip": "10.0.0.1"}]
        mgr.connection_pool["host:80_0_0_0"] = [{"rank_ip": "10.0.0.2"}]

        with patch.object(CommonKVReceiver, "disconnect_endpoint"):
            mgr._handle_node_failure("host:80")

        self.assertEqual(list(mgr.connection_pool), ["host:8000_0_0_0"])

    def test_abort_retry_waits_for_undrained_room(self):
        """A room still held by an undrained receiver keeps that receiver's ACK
        state: the new receiver neither sends ABORT nor wipes it on clear()."""
        mgr = self._decode_manager(enable_deferred_decode_kv_release=True)
        mgr.requires_transfer_drain = True
        mgr.register_deferred_abort_room(7, token="old-token")
        mgr.note_abort_ack(7, 0, token="old-token")
        receiver = _ConcreteReceiver(mgr, ADDR, 7)
        receiver.bootstrap_infos = [{"abort_rank": 0}]
        receiver._connect_to_bootstrap_server = Mock()

        receiver.retry_abort()
        receiver.clear()

        receiver._connect_to_bootstrap_server.assert_not_called()
        self.assertEqual(mgr._deferred_abort_tokens[7], "old-token")
        self.assertEqual(mgr._deferred_abort_ack_tracker[7], {0})
        self.assertTrue(mgr.connection_lock.acquire(blocking=False))
        mgr.connection_lock.release()

    def test_stale_receiver_cannot_arm_or_notify_reused_room(self):
        mgr = self._decode_manager(enable_deferred_decode_kv_release=True)
        receiver = _ConcreteReceiver(mgr, ADDR, 7)
        receiver.bootstrap_infos = [{}]
        _ConcreteReceiver(mgr, ADDR, 7)
        mgr.register_deferred_abort_room = Mock()
        receiver._connect_to_bootstrap_server = Mock()

        receiver.ensure_abort_notified(force_arm=True)
        receiver.retry_abort()
        receiver._send_abort_notification(force_arm=True)

        self.assertFalse(receiver.abort_notified)
        receiver.kv_mgr.register_deferred_abort_room.assert_not_called()
        receiver._connect_to_bootstrap_server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
