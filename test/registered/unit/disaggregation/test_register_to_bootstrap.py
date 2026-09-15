"""Unit tests for srt/disaggregation/common/conn bootstrap registration and layout."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


import asyncio
import json
import struct
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    MooncakeKVReceiver,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase


def _run_control_message(manager, start_listener, message):
    """Feed one wire message through the real control loop, then stop at recv."""

    class Socket:
        def recv_multipart(self):
            nonlocal message
            if message is None:
                raise SystemExit
            current, message = message, None
            return current

    class Thread:
        def __init__(self, *, target):
            self.target = target

        def start(self):
            try:
                self.target()
            except SystemExit:
                pass

    manager.server_socket = Socket()
    with patch("sglang.srt.disaggregation.mooncake.conn.threading.Thread", Thread):
        start_listener()


class TestRegisterToBootstrap(CustomTestCase):
    """Tests for CommonKVManager.register_to_bootstrap retry/backoff behavior."""

    def setUp(self):
        # register_to_bootstrap reads get_parallel().load_balance_method /
        # .enable_dsa_cache_layer_split and get_serving().port from the
        # published config.
        override = get_context().override_server_args(
            load_balance_method="follow_bootstrap_room", port=30000
        )
        override.install()
        self.addCleanup(override.restore)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_succeeds_on_first_attempt(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_put.return_value = mock_response

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        mock_put.assert_called_once()
        mock_time.sleep.assert_not_called()

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_succeeds_after_retries(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.side_effect = [fail_resp, fail_resp, success_resp]

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 3)
        self.assertEqual(mock_time.sleep.call_count, 2)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_all_retries_exhausted(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        mock_put.return_value = fail_resp

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 5)
        # Sleep is only called between attempts, not after the final failure
        self.assertEqual(mock_time.sleep.call_count, 4)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_exception_with_nested_cause(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0

        root_exc = ConnectionRefusedError("connection refused")
        inner_exc = OSError("os error")
        inner_exc.__cause__ = root_exc
        outer_exc = Exception("wrapped")
        outer_exc.__cause__ = inner_exc

        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.side_effect = [outer_exc, success_resp]

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 2)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_exception_with_no_cause(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0

        exc = ConnectionError("plain connection error")
        exc.__cause__ = None

        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.side_effect = [exc, success_resp]

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 2)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_backoff_delay_exponential(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        mock_put.return_value = fail_resp

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        # With monotonic() = 0.0, jitter factor = 0.75 + 0.25 * (0.0 % 1) = 0.75
        # delay = min(1.0 * 2^attempt, 30.0) * 0.75
        # Sleep happens only between attempts (attempt 0..3), not after the final failure
        expected_calls = [call(0.75), call(1.5), call(3.0), call(6.0)]
        self.assertEqual(mock_time.sleep.call_args_list, expected_calls)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_jitter_never_exceeds_max_delay(self, mock_put, mock_time):
        """Guard against operator-precedence regressions in the jitter factor.

        The jitter factor must stay in [0.75, 1.0), so a delay capped at
        max_delay must never exceed max_delay after applying jitter.
        """
        # monotonic() returns a value whose fractional part is close to 1.
        # If the parentheses around `time.monotonic() % 1` were dropped, the
        # jitter factor could grow up to ~1.75 and blow past max_delay.
        mock_time.monotonic.return_value = 999.9999
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        mock_put.return_value = fail_resp

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        max_delay = 30.0
        for sleep_call in mock_time.sleep.call_args_list:
            actual_delay = sleep_call[0][0]
            self.assertLess(actual_delay, max_delay)
            self.assertGreaterEqual(actual_delay, 0.75)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_payload_contains_required_fields(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        call_kwargs = mock_put.call_args
        payload = call_kwargs[1]["json"]
        required_fields = [
            "attn_tp_size",
            "attn_tp_rank",
            "attn_cp_size",
            "attn_cp_rank",
            "attn_dp_size",
            "attn_dp_rank",
            "pp_size",
            "pp_rank",
            "system_dp_size",
            "system_dp_rank",
            "rank_ip",
            "rank_port",
            "page_size",
            "kv_cache_dtype",
            # Self-registered HTTP API port used to derive the PD retract
            # rebootstrap /generate URL on the decode side.
            "prefill_http_port",
        ]
        for field in required_fields:
            self.assertIn(field, payload)
        self.assertEqual(payload["prefill_http_port"], 30000)
        self.assertIs(payload["supports_dcp_page"], True)
        self.assertNotIn("dcp_kv_layout", payload)

    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_page_support_follows_prefill_configuration(self, mock_put):
        mock_put.return_value = MagicMock(status_code=200)
        for layout, backend, speculative, dcp_size, supported in (
            ("token", "mooncake", None, 1, True),
            ("page", "mooncake", None, 1, True),
            ("token", "nixl", None, 1, False),
            ("token", "mooncake", "EAGLE", 1, False),
            ("token", "mooncake", None, 2, False),
        ):
            with self.subTest(
                layout=layout,
                backend=backend,
                speculative=speculative,
                dcp_size=dcp_size,
            ):
                override = get_context().override_server_args(
                    dcp_kv_layout=layout,
                    disaggregation_transfer_backend=backend,
                    speculative_algorithm=speculative,
                )
                override.install()
                try:
                    manager = self._make_manager()
                    manager.dcp_size = dcp_size
                    manager.register_to_bootstrap()
                finally:
                    override.restore()
                self.assertIs(
                    mock_put.call_args.kwargs["json"]["supports_dcp_page"], supported
                )

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_url_with_dist_init_addr(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager(dist_init_addr="10.0.0.1:12345")
        mgr.register_to_bootstrap()

        url_used = mock_put.call_args[0][0]
        self.assertIn("10.0.0.1", url_used)

    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    @patch("sglang.srt.disaggregation.common.conn.get_world_group")
    def test_rust_attention_dp_replicates_complete_topology_across_hosts(
        self, mock_world_group, mock_put
    ):
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        schedulers = (
            (0, 0, "10.0.0.1", 17000, 8765),
            (0, 1, "10.0.0.1", 17001, None),
            (1, 0, "10.0.0.2", 17002, 8766),
            (1, 1, "10.0.0.2", 17003, None),
        )

        def gather_topology(payload):
            return [
                {
                    **payload,
                    "attn_dp_rank": dp_rank,
                    "attn_tp_rank": tp_rank,
                    "rank_ip": host,
                    "rank_port": rank_port,
                }
                for dp_rank, tp_rank, host, rank_port, _ in schedulers
            ]

        mock_world_group.return_value.all_gather_object.side_effect = gather_topology

        with envs.SGLANG_RUST_SERVER.override(True):
            for dp_rank, tp_rank, local_ip, _, rust_http_port in schedulers:
                manager = self._make_manager()
                manager.attn_dp_size = 2
                manager.attn_dp_rank = dp_rank
                manager.attn_tp_size = 2
                manager.attn_tp_rank = tp_rank
                manager.local_ip = local_ip
                manager.bootstrap_host = local_ip
                manager.kv_args.rust_http_port = rust_http_port
                manager.register_to_bootstrap()

        topology_by_registry = {}
        for put_call in mock_put.call_args_list:
            payload = put_call.kwargs["json"]
            topology_by_registry.setdefault(put_call.args[0], {})[
                (payload["attn_dp_rank"], payload["attn_tp_rank"])
            ] = (payload["rank_ip"], payload["rank_port"])
        complete_topology = {
            (dp, tp): (host, rank_port) for dp, tp, host, rank_port, _ in schedulers
        }
        self.assertEqual(
            topology_by_registry,
            {
                "http://10.0.0.1:8765/route": complete_topology,
                "http://10.0.0.2:8766/route": complete_topology,
            },
        )
        self.assertEqual(mock_put.call_count, 8)
        self.assertEqual(
            {
                (put_call.args[0], put_call.kwargs["json"]["prefill_http_port"])
                for put_call in mock_put.call_args_list
            },
            {
                ("http://10.0.0.1:8765/route", 8765),
                ("http://10.0.0.2:8766/route", 8766),
            },
        )
        self.assertEqual(
            [
                (
                    gather_call.args[0]["attn_dp_rank"],
                    gather_call.args[0]["attn_tp_rank"],
                )
                for gather_call in mock_world_group.return_value.all_gather_object.call_args_list
            ],
            [(dp, tp) for dp, tp, _, _, _ in schedulers],
        )

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_wildcard_host_0000_uses_ipv4_loopback(self, mock_put, mock_time):
        """When --host 0.0.0.0 is used, the PUT must target IPv4 loopback.

        Scenario: cross-node P/D disagg where each role runs on a single node
        (tp=1).  Each machine runs its own SGLang instance with --host 0.0.0.0
        to accept remote connections.  dist_init_addr is None because tp=1
        needs no multi-node rendezvous, so register_to_bootstrap takes the
        else-branch and would use bootstrap_host="0.0.0.0" as the PUT target.
        aiohttp >=3.9 rejects that with HTTP 403 because 0.0.0.0 is not a
        valid Host header value.

        Fix: substitute same-family loopback when bootstrap_host is a wildcard.
        """
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager()
        mgr.bootstrap_host = "0.0.0.0"
        mgr.local_ip = "192.168.1.10"
        mgr.register_to_bootstrap()

        url_used = mock_put.call_args[0][0]
        self.assertNotIn("0.0.0.0", url_used)
        self.assertIn("127.0.0.1", url_used)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_wildcard_host_ipv6_uses_ipv6_loopback(self, mock_put, mock_time):
        """Same fix for the IPv6 wildcard \"::\": must use IPv6 loopback."""
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager()
        mgr.bootstrap_host = "::"
        mgr.local_ip = "fd00::1"
        mgr.register_to_bootstrap()

        url_used = mock_put.call_args[0][0]
        # "::" bracketed as "[::]:port" should not appear; loopback should.
        self.assertNotIn("[::]", url_used)
        self.assertIn("[::1]", url_used)

    def _make_manager(self, dist_init_addr=None):
        """Create a lightweight mock manager that has the attributes needed
        by register_to_bootstrap, without going through CommonKVManager.__init__
        (which requires zmq, ServerArgs model resolution, etc.)."""
        mgr = MagicMock(spec=CommonKVManager)
        # Bind the real method to the mock
        mgr.register_to_bootstrap = CommonKVManager.register_to_bootstrap.__get__(
            mgr, CommonKVManager
        )
        mgr._register_topology_row = CommonKVManager._register_topology_row.__get__(
            mgr, CommonKVManager
        )

        # Set attributes that register_to_bootstrap reads
        mgr.dist_init_addr = dist_init_addr
        mgr.bootstrap_host = "127.0.0.1"
        mgr.bootstrap_port = 8765
        mgr.attn_tp_size = 1
        mgr.attn_tp_rank = 0
        mgr.attn_cp_size = 1
        mgr.attn_cp_rank = 0
        mgr.dcp_size = 1
        mgr.attn_dp_size = 1
        mgr.attn_dp_rank = 0
        mgr.pp_size = 1
        mgr.pp_rank = 0
        mgr.system_dp_size = 1
        mgr.system_dp_rank = 0
        mgr.local_ip = "127.0.0.1"
        mgr.rank_port = 12345

        mgr.kv_args = MagicMock()
        mgr.kv_args.page_size = 16
        mgr.kv_args.rust_http_port = None
        # Resolved per-runner value threaded through KVArgs (the payload field).
        mgr.kv_cache_dtype_str = "auto"

        return mgr


class _RouteRequest:
    def __init__(self, *, data=None, query=None):
        self._data = data
        self.query = query or {}

    async def json(self):
        return self._data


class TestBootstrapDcpPageSupport(CustomTestCase):
    def _make_server(self, *, dp_size=1, tp_size=1):
        server = object.__new__(CommonKVBootstrapServer)
        server.attn_tp_size = tp_size
        server.attn_cp_size = 1
        server.dp_size = dp_size
        server.pp_size = 1
        server.page_size = 16
        server.kv_cache_dtype = "auto"
        server.supports_dcp_page = None
        server.follow_bootstrap_room = True
        server.enable_dsa_cache_layer_split = False
        server.prefill_http_port = None
        server.prefill_port_table = {}
        server._registered_count = 0
        server.lock = asyncio.Lock()
        return server

    @staticmethod
    def _registration(*, dp_rank=0, tp_rank=0, supports_dcp_page=None):
        data = {
            "attn_tp_size": 2,
            "attn_tp_rank": tp_rank,
            "attn_cp_size": 1,
            "attn_cp_rank": 0,
            "attn_dp_size": 1,
            "attn_dp_rank": dp_rank,
            "pp_size": 1,
            "pp_rank": 0,
            "system_dp_size": 1,
            "system_dp_rank": 0,
            "rank_ip": "127.0.0.1",
            "rank_port": 12345 + tp_rank,
            "page_size": 16,
            "kv_cache_dtype": "auto",
        }
        if supports_dcp_page is not None:
            data["supports_dcp_page"] = supports_dcp_page
        return data

    @staticmethod
    def _static_query(*, want_page_support=False):
        query = {
            "prefill_dp_rank": "-1",
            "prefill_cp_rank": "-1",
            "target_tp_rank": "-1",
            "target_pp_rank": "-1",
        }
        if want_page_support:
            query["want_dcp_page_support"] = "1"
        return query

    def test_route_page_support_is_opt_in(self):
        for supported in (None, False, True):
            with self.subTest(supported=supported):
                server = self._make_server(tp_size=2)
                for rank in range(2):
                    asyncio.run(
                        server._handle_route_put(
                            _RouteRequest(
                                data=self._registration(
                                    tp_rank=rank, supports_dcp_page=supported
                                )
                            )
                        )
                    )

                legacy = asyncio.run(
                    server._handle_route_get(_RouteRequest(query=self._static_query()))
                )
                self.assertEqual(legacy.status, 200)
                self.assertNotIn("supports_dcp_page", json.loads(legacy.text))

                response = asyncio.run(
                    server._handle_route_get(
                        _RouteRequest(query=self._static_query(want_page_support=True))
                    )
                )
                self.assertEqual(response.status, 200)
                self.assertIs(json.loads(response.text)["supports_dcp_page"], supported)

    @patch("sglang.srt.disaggregation.common.conn.requests.get")
    @patch("sglang.srt.disaggregation.common.conn.get_parallel")
    def test_decode_checks_page_support_before_caching_prefill_info(
        self, mock_parallel, mock_get
    ):
        bootstrap_addr = "127.0.0.1:30000"
        prefill_info = {
            "attn_tp_size": 1,
            "attn_cp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "page_size": 16,
            "kv_cache_dtype": "auto",
            "follow_bootstrap_room": True,
        }
        for decode_layout, support_fields, accepted in (
            ("page", {}, False),
            ("page", {"supports_dcp_page": None}, False),
            ("page", {"supports_dcp_page": False}, False),
            ("page", {"supports_dcp_page": True}, True),
            ("token", {}, True),
        ):
            with self.subTest(
                decode_layout=decode_layout, support_fields=support_fields
            ):
                manager = object.__new__(CommonKVManager)
                manager.prefill_info_table = {}
                manager.kv_args = SimpleNamespace(page_size=16)
                manager.kv_cache_dtype_str = "auto"
                manager.dcp_size = 1
                manager._resolve_rank_mapping = MagicMock()
                mock_parallel.return_value = SimpleNamespace(
                    dcp_kv_layout=decode_layout
                )
                mock_get.reset_mock()
                response = MagicMock(status_code=200)
                response.json.return_value = {**prefill_info, **support_fields}
                mock_get.return_value = response

                if accepted:
                    self.assertTrue(manager.try_ensure_parallel_info(bootstrap_addr))
                    cached_info = manager.prefill_info_table[bootstrap_addr]
                    self.assertEqual(
                        cached_info.supports_dcp_page,
                        support_fields.get("supports_dcp_page"),
                    )
                    self.assertTrue(manager.try_ensure_parallel_info(bootstrap_addr))
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "DCP page transfer support"
                    ):
                        manager.try_ensure_parallel_info(bootstrap_addr)
                    self.assertEqual(manager.prefill_info_table, {})
                    manager._resolve_rank_mapping.assert_not_called()

                mock_get.assert_called_once()
                self.assertEqual(
                    "want_dcp_page_support=1" in mock_get.call_args.args[0],
                    decode_layout == "page",
                )

    @patch("sglang.srt.disaggregation.mooncake.conn.get_parallel")
    def test_token_and_page_share_cached_registration_and_room_metadata(self, parallel):
        bootstrap_addr = "127.0.0.1:30000"
        for layout, registration_frames in (("token", 19), ("page", 20)):
            with self.subTest(layout=layout):
                parallel.return_value = SimpleNamespace(dcp_kv_layout=layout)
                manager = SimpleNamespace(
                    kv_args=SimpleNamespace(
                        kv_data_ptrs=[4096],
                        aux_data_ptrs=[],
                        state_data_ptrs=[],
                        state_item_lens=[],
                        state_dim_per_tensor=[],
                        state_layer_ids=[],
                        kv_layer_ids=[0],
                        engine_rank=0,
                        kv_item_lens=[64],
                        page_size=16,
                    ),
                    attn_tp_size=2,
                    dcp_size=2,
                    dcp_rank=0,
                    is_mla_backend=True,
                    enable_staging=False,
                    local_ip="127.0.0.1",
                    rank_port=30001,
                    get_session_id=lambda: "session",
                    addr_to_rooms_tracker=defaultdict(set),
                    update_status=MagicMock(),
                    connection_pool={},
                    connection_lock=threading.Lock(),
                    required_prefill_response_num_table={},
                    prefill_info_table={
                        bootstrap_addr: SimpleNamespace(
                            target_tp_rank=0,
                            target_tp_ranks=[0],
                            target_cp_ranks=[0],
                            target_pp_ranks=[0],
                            required_dst_info_num=1,
                            required_prefill_response_num=1,
                        )
                    },
                )
                socket = MagicMock()
                with (
                    patch.object(
                        MooncakeKVReceiver,
                        "_get_bootstrap_info_from_server",
                        return_value={"rank_ip": "127.0.0.1", "rank_port": 30000},
                    ) as lookup,
                    patch.object(
                        MooncakeKVReceiver,
                        "_connect_to_bootstrap_server",
                        return_value=(socket, threading.Lock()),
                    ),
                ):
                    for room in (9, 10):
                        receiver = MooncakeKVReceiver(manager, bootstrap_addr, room)
                        receiver.init(0)
                        manager.update_status.assert_any_call(
                            room, KVPoll.WaitingForInput
                        )
                        receiver.send_metadata(
                            np.array([1], dtype=np.int32), aux_index=0
                        )

                lookup.assert_called_once()
                messages = [
                    call.args[0] for call in socket.send_multipart.call_args_list
                ]
                self.assertEqual(
                    [len(message) for message in messages],
                    [registration_frames, 10, 10],
                )
                registration = KVArgsRegisterInfo.from_zmq(messages[0])
                self.assertEqual(registration.dcp_kv_layout, layout)
                if layout == "page":
                    self.assertEqual(messages[0][19:], [b"page"])

    @patch("sglang.srt.disaggregation.mooncake.conn.get_parallel")
    def test_prefill_registers_decode_layout_and_validates_geometry(self, parallel):
        parallel.return_value = SimpleNamespace(dcp_kv_layout="token")
        manager = object.__new__(MooncakeKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.server_args = object()
        manager.kv_args = SimpleNamespace(
            page_size=4, num_draft_entries=0, kv_item_lens=[8]
        )
        manager.dcp_size = 1
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.decode_kv_args_table = {}
        manager.transfer_infos = {}
        manager.local_ip = "127.0.0.1"
        manager.rank_port = 30000
        manager._send_multipart_locked = MagicMock()
        manager.session_lock = threading.Lock()
        manager.failed_sessions = {"session"}
        manager.session_failures = {"session": 1}
        manager._init_dcp_pack_buffers_once = MagicMock()
        message = [
            b"None",
            b"127.0.0.1",
            b"30001",
            b"session",
            struct.pack("Q", 4096),
            b"",
            b"",
            b"0",
            b"2",
            b"12",
            b"",
            b"",
            struct.pack("I", 0),
            b"",
            b"",
            b"",
            b"2",
            b"0",
            b"",
            b"page",
        ]

        with self.assertRaisesRegex(RuntimeError, "KV geometry differs"):
            _run_control_message(manager, manager.start_prefill_thread, message)

        self.assertEqual(manager.decode_kv_args_table, {})
        manager._send_multipart_locked.assert_not_called()

        message[9] = b"8"
        _run_control_message(manager, manager.start_prefill_thread, message)
        self.assertEqual(
            manager.decode_kv_args_table["session"].dcp_kv_layout,
            "page",
        )
        manager._init_dcp_pack_buffers_once.assert_not_called()
        self.assertNotIn("session", manager.failed_sessions)
        self.assertNotIn("session", manager.session_failures)

        manager.dcp_size = 2
        with self.assertRaisesRegex(RuntimeError, "prefill=token, decode=page"):
            _run_control_message(manager, manager.start_prefill_thread, message)
        manager.dcp_size = 1
        token_message = message[:19]
        token_message[3] = b"token-session"
        _run_control_message(manager, manager.start_prefill_thread, token_message)
        manager._init_dcp_pack_buffers_once.assert_called_once_with(2)
        self.assertEqual(manager.decode_kv_args_table["session"].dcp_kv_layout, "page")
        self.assertEqual(
            manager.decode_kv_args_table["token-session"].dcp_kv_layout, "token"
        )


if __name__ == "__main__":
    unittest.main()
