"""CPU-only tests for the NVLink-vs-NIC transport decision on the
remote-instance weight loading path: the fabric identity comparison, and the
parsing of what a seed publishes -- including a seed that predates the
protocol/fabric fields, whose absence must not crash the reader.
"""

import ctypes
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import msgspec

from sglang.srt.environ import envs
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    SeedTransferEngineInfo,
    _parse_seed_transfer_engine_info,
    deregister_memory_region,
)
from sglang.srt.utils import nvlink_fabric_utils
from sglang.srt.utils.nvlink_fabric_utils import (
    NvlinkFabricIdentity,
    _format_cluster_uuid,
    _query_fabric_info,
    get_nvlink_fabric_identity,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_WEIGHTS = {"model.layers.0.weight": [140737488355328, 4096, 2]}
_UUID_BYTES = bytes.fromhex("dde92ffd754e4ed1bf5e404aaebc4781")


def _identity(cluster_uuid="dde92ffd", clique_id=32766) -> NvlinkFabricIdentity:
    return NvlinkFabricIdentity(cluster_uuid=cluster_uuid, clique_id=clique_id)


class TestNvlinkFabricIdentity(CustomTestCase):
    def test_reachable_only_when_both_fields_agree(self):
        # Both halves are load-bearing: same rack different partition, and
        # same partition number on a different rack, are both unreachable.
        self.assertEqual(_identity(), _identity())
        self.assertNotEqual(_identity(), _identity(clique_id=1))
        self.assertNotEqual(_identity(), _identity(cluster_uuid="other"))

    def test_survives_a_json_round_trip(self):
        identity = _identity()
        restored = msgspec.convert(
            msgspec.structs.asdict(identity), NvlinkFabricIdentity
        )
        self.assertEqual(identity, restored)

    def test_cluster_uuid_formatting_is_deterministic(self):
        # NVML hands back raw bytes on some pynvml versions and a str on
        # others; both ends of a comparison only have to agree with themselves.
        self.assertEqual(_format_cluster_uuid(b"\x01\x02"), "0102")
        self.assertEqual(_format_cluster_uuid("dde92ffd"), "dde92ffd")

    def test_probe_reports_none_off_fabric(self):
        # No pynvml, no GPU, or no MNNVL all have to answer the same way, since
        # the caller's only question is whether NVLink is an option.
        self.assertIsNone(get_nvlink_fabric_identity(gpu_id=0))


class _FabricInfo(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("clusterUuid", ctypes.c_char * 16),
        ("status", ctypes.c_uint),
        ("cliqueId", ctypes.c_uint),
        ("state", ctypes.c_uint),
    ]


class TestQueryFabricInfo(CustomTestCase):
    """nvmlDeviceGetGpuFabricInfoV takes the struct as an out-param; calling it
    as though it returned one raised TypeError that the probe swallowed into
    "off fabric", so a GB300 in a healthy NVL72 reported no NVLink.
    """

    def _fake_pynvml(self, **overrides):
        def get_info_v(handle, ref):
            info = ctypes.cast(ref, ctypes.POINTER(_FabricInfo)).contents
            # Only a caller that set version first gets a filled struct, which
            # is what NVML itself does.
            if info.version == 0:
                raise ValueError("version not set")
            info.state, info.status, info.cliqueId = 3, 0, 32766
            info.clusterUuid = _UUID_BYTES

        base = dict(
            c_nvmlGpuFabricInfoV_t=_FabricInfo,
            nvmlGpuFabricInfo_v3=0x3000038,
            byref=ctypes.byref,
            nvmlDeviceGetGpuFabricInfoV=get_info_v,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_fills_the_struct_through_the_out_param(self):
        info = _query_fabric_info(self._fake_pynvml(), handle="h", gpu_id=0)
        self.assertIsNotNone(info)
        self.assertEqual(info.cliqueId, 32766)
        self.assertEqual(_format_cluster_uuid(info.clusterUuid), _UUID_BYTES.hex())

    def test_falls_back_to_the_unversioned_entry_point(self):
        # Older pynvml has no versioned call and returns the struct directly.
        legacy = SimpleNamespace(
            nvmlDeviceGetGpuFabricInfo=lambda handle: SimpleNamespace(
                state=3, status=0, cliqueId=7, clusterUuid=_UUID_BYTES
            )
        )
        info = _query_fabric_info(legacy, handle="h", gpu_id=0)
        self.assertEqual(info.cliqueId, 7)

    def test_no_entry_point_at_all_is_not_an_error(self):
        self.assertIsNone(_query_fabric_info(SimpleNamespace(), handle="h", gpu_id=0))

    def test_a_standalone_node_reports_no_fabric(self):
        # A host outside any MNNVL domain still answers COMPLETED/Success, with
        # a zeroed UUID and clique 0. Taking that as an identity would make every
        # such host match every other one, so two unrelated nodes would agree
        # they share a clique and try a fabric handle that cannot resolve.
        standalone = SimpleNamespace(
            nvmlDeviceGetGpuFabricInfo=lambda handle: SimpleNamespace(
                state=3, status=0, cliqueId=0, clusterUuid=bytes(16)
            )
        )
        with mock.patch.object(
            nvlink_fabric_utils, "_open_nvml_handle", return_value="h"
        ), mock.patch.dict(sys.modules, {"pynvml": standalone}):
            self.assertIsNone(get_nvlink_fabric_identity(gpu_id=0))


class TestParseSeedTransferEngineInfo(CustomTestCase):
    def test_reads_a_seed_that_publishes_the_transport(self):
        info = _parse_seed_transfer_engine_info(
            ["1.2.3.4:9000", _WEIGHTS, "nvlink", msgspec.structs.asdict(_identity())]
        )
        self.assertEqual(
            info,
            SeedTransferEngineInfo(
                session_id="1.2.3.4:9000",
                weights_info_dict=_WEIGHTS,
                protocol="nvlink",
                fabric_identity=_identity(),
            ),
        )

    def test_seed_without_the_new_fields_still_loads(self):
        # A seed older than the protocol/fabric fields sends two entries. It
        # must keep working, reported as "did not say" rather than as failure.
        info = _parse_seed_transfer_engine_info(["1.2.3.4:9000", _WEIGHTS])
        self.assertIsNotNone(info)
        self.assertEqual(info.session_id, "1.2.3.4:9000")
        self.assertIsNone(info.protocol)
        self.assertIsNone(info.fabric_identity)

    def test_nic_seed_carries_no_fabric(self):
        info = _parse_seed_transfer_engine_info(
            ["1.2.3.4:9000", _WEIGHTS, "rdma", None]
        )
        self.assertEqual(info.protocol, "rdma")
        self.assertIsNone(info.fabric_identity)

    def test_missing_session_or_weights_is_a_failure(self):
        self.assertIsNone(_parse_seed_transfer_engine_info([None, None]))
        self.assertIsNone(_parse_seed_transfer_engine_info(["1.2.3.4:9000", None]))


_SEED_HOST = "10.1.1.4"
_OTHER_HOST = "10.1.1.5"


def _dual_transport_wire():
    # The NIC endpoint is primary because every peer can reach it; NVLink rides
    # along as an alternate for peers inside the same fabric clique.
    return [
        f"{_SEED_HOST}:16000",
        _WEIGHTS,
        "rdma",
        None,
        [
            [
                f"{_SEED_HOST}:16002",
                _WEIGHTS,
                "nvlink",
                msgspec.structs.asdict(_identity()),
            ]
        ],
    ]


def _all_transport_wire():
    """What a seed with a usable fabric publishes: every transport it can serve.

    A seed cannot know where its clients will land, so it offers all three and
    each client picks the fastest it can reach.
    """
    return [
        f"{_SEED_HOST}:16000",
        _WEIGHTS,
        "rdma",
        None,
        [
            [f"{_SEED_HOST}:16001", _WEIGHTS, "nvlink_intra", None],
            [
                f"{_SEED_HOST}:16002",
                _WEIGHTS,
                "nvlink",
                msgspec.structs.asdict(_identity()),
            ],
        ],
    ]


class TestSeedEndpointSelection(CustomTestCase):
    def test_peer_in_the_same_clique_gets_nvlink(self):
        info = _parse_seed_transfer_engine_info(_dual_transport_wire())
        endpoint = info.endpoint_for(protocol="nvlink", fabric=_identity())
        self.assertIsNotNone(endpoint)
        self.assertEqual(endpoint.session_id, f"{_SEED_HOST}:16002")

    def test_peer_outside_the_clique_cannot_use_nvlink(self):
        # A fabric handle only resolves within one clique, so a same-rack peer
        # in another partition is as unreachable as one on another rack. Serving
        # it the NVLink endpoint would transfer into unmapped memory.
        info = _parse_seed_transfer_engine_info(_dual_transport_wire())
        for fabric in (_identity(clique_id=5), _identity(cluster_uuid="other"), None):
            with self.subTest(fabric=fabric):
                self.assertIsNone(info.endpoint_for(protocol="nvlink", fabric=fabric))

    def test_that_peer_still_reaches_the_seed_over_the_nic(self):
        info = _parse_seed_transfer_engine_info(_dual_transport_wire())
        endpoint = info.endpoint_for(protocol="rdma", fabric=None)
        self.assertIsNotNone(endpoint)
        self.assertEqual(endpoint.session_id, f"{_SEED_HOST}:16000")

    def test_the_nic_endpoint_ignores_fabric(self):
        # Only NVLink is clique-gated; gating the NIC too would strand every
        # peer whose fabric differs from the seed's.
        info = _parse_seed_transfer_engine_info(_dual_transport_wire())
        self.assertIsNotNone(
            info.endpoint_for(protocol="rdma", fabric=_identity(clique_id=5))
        )

    def test_a_client_too_old_to_read_alternates_finds_the_nic(self):
        # Fields are only appended, so an old client truncates at fabric and
        # must still land on a reachable endpoint -- hence the NIC is primary.
        info = _parse_seed_transfer_engine_info(_dual_transport_wire()[:4])
        self.assertEqual(info.offered_protocols(), ("rdma",))
        self.assertEqual(info.session_id, f"{_SEED_HOST}:16000")

    def test_alternates_do_not_nest(self):
        # A tree would let a malformed payload recurse; alternates are leaves.
        info = _parse_seed_transfer_engine_info(_dual_transport_wire())
        self.assertEqual(info.alternates[0].alternates, ())

    def test_unreachable_protocol_reports_none_rather_than_guessing(self):
        info = _parse_seed_transfer_engine_info(_dual_transport_wire())
        self.assertIsNone(info.endpoint_for(protocol="tcp", fabric=None))


class TestIntraNodeRouting(CustomTestCase):
    """nvlink_intra is CUDA IPC, so it reaches only peers on the seed's host.

    A seed offers it unasked -- it needs no fabric registration, and only the
    client knows which host it landed on. Getting that gate wrong is silent:
    the transfer reads into memory the peer never mapped.
    """

    def test_a_client_elsewhere_cannot(self):
        info = _parse_seed_transfer_engine_info(_all_transport_wire())
        for host in (_OTHER_HOST, None):
            with self.subTest(local_host=host):
                self.assertIsNone(
                    info.endpoint_for(protocol="nvlink_intra", local_host=host)
                )

    def test_each_transport_is_gated_on_its_own_reachability(self):
        # One seed, three peers, three answers -- the case a single global
        # protocol could not express.
        info = _parse_seed_transfer_engine_info(_all_transport_wire())
        same_host = info.endpoint_for(
            protocol="nvlink_intra", fabric=_identity(), local_host=_SEED_HOST
        )
        same_clique = info.endpoint_for(
            protocol="nvlink", fabric=_identity(), local_host=_OTHER_HOST
        )
        far_away = info.endpoint_for(
            protocol="rdma", fabric=_identity(clique_id=5), local_host=_OTHER_HOST
        )
        self.assertEqual(
            [e.session_id for e in (same_host, same_clique, far_away)],
            [f"{_SEED_HOST}:16001", f"{_SEED_HOST}:16002", f"{_SEED_HOST}:16000"],
        )

    def test_an_unparseable_session_id_is_not_treated_as_local(self):
        # Guessing "local" here would hand a remote peer a cudaIpc handle.
        info = _parse_seed_transfer_engine_info(
            ["not-a-host-port", _WEIGHTS, "nvlink_intra", None]
        )
        self.assertIsNone(
            info.endpoint_for(protocol="nvlink_intra", local_host=_SEED_HOST)
        )


class TestMooncakeTransportEnv(CustomTestCase):
    """mooncake picks a TransferEngine's transport from MC_INTRANODE_NVLINK /
    MC_FORCE_MNNVL, read at each initialize() rather than once per process, and
    ignores the protocol argument. Setting them around the call is what lets one
    seed hold engines on different transports.
    """

    def _env_snapshot(self):
        return (
            os.environ.get("MC_INTRANODE_NVLINK"),
            os.environ.get("MC_FORCE_MNNVL"),
        )

    def _transport_env(self, protocol):
        from sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter import (
            RemoteInstanceWeightTransporter,
        )

        return RemoteInstanceWeightTransporter._mooncake_transport_env(protocol)

    def test_each_protocol_sets_only_its_own_variable(self):
        for protocol, expected in (
            ("nvlink_intra", ("True", None)),
            ("nvlink", (None, "True")),
        ):
            with self.subTest(protocol=protocol):
                with self._transport_env(protocol):
                    self.assertEqual(self._env_snapshot(), expected)

    def test_the_nic_clears_rather_than_setting_false(self):
        # mooncake reads these with getenv, to which the string "False" is as
        # true as "1", so a NIC engine has to see them absent.
        with envs.MC_INTRANODE_NVLINK.override(True):
            with self._transport_env("rdma"):
                self.assertEqual(self._env_snapshot(), (None, None))

    def test_the_operators_own_values_come_back(self):
        with envs.MC_INTRANODE_NVLINK.override(True):
            before = self._env_snapshot()
            with self._transport_env("rdma"):
                pass
            self.assertEqual(self._env_snapshot(), before)

    def test_they_come_back_even_when_initialize_raises(self):
        # A transport mooncake was not built with raises here, and the next
        # engine in this process must not inherit the leftover variables.
        with envs.MC_INTRANODE_NVLINK.override(True):
            before = self._env_snapshot()
            with self.assertRaises(RuntimeError):
                with self._transport_env("rdma"):
                    raise RuntimeError("initialize failed")
            self.assertEqual(self._env_snapshot(), before)


class TestDeregisterAfterPartialBatch(CustomTestCase):
    """mooncake's batch unregister returns its first error even after freeing
    most of the list, and an address it already freed answers a retry with
    ERR_ADDRESS_NOT_REGISTERED. Reading that as "still pinned" reported ~85% of
    a Kimi-K3 registration as leaked when it had in fact been released.
    """

    def _blocks(self, count):
        return [(0x1000 + i * 0x100, 0x100) for i in range(count)]

    def test_addresses_the_batch_already_freed_count_as_done(self):
        class PartialBatch:
            def __init__(self):
                self.freed = set()
                self.per_address_calls = 0

            def batch_unregister_memory(self, addresses):
                self.freed.update(addresses[: int(len(addresses) * 0.85)])
                return -3  # its first error, despite having freed most

            def unregister_memory(self, address):
                self.per_address_calls += 1
                if address in self.freed:
                    return -3  # ERR_ADDRESS_NOT_REGISTERED
                self.freed.add(address)
                return 0

        engine = PartialBatch()
        blocks = self._blocks(200)
        self.assertTrue(deregister_memory_region(engine, blocks, protocol="rdma"))
        self.assertEqual(len(engine.freed), 200)
        # One sweep for the remainder, not a redo of the whole list.
        self.assertEqual(engine.per_address_calls, 200)

    def test_a_real_failure_is_still_reported(self):
        class Stuck:
            def batch_unregister_memory(self, addresses):
                return -1

            def unregister_memory(self, address):
                return -202  # ERR_CONTEXT

        self.assertFalse(
            deregister_memory_region(Stuck(), self._blocks(20), protocol="rdma")
        )


class TestDeregisterRetry(CustomTestCase):
    """The retry loop existed for EAGAIN from a driver busy with the other ranks,
    which clears. An address mooncake refuses as a registration base never does,
    and on Kimi-K3 that is ~85% of them -- five backoffs over those cost 5s of
    startup for nothing.
    """

    def _blocks(self, count):
        return [(0x1000 + i * 0x100, 0x100) for i in range(count)]

    def test_stops_once_an_attempt_frees_nothing(self):
        class Stalling:
            def __init__(self):
                self.calls = 0
                self.freed = 0

            def batch_unregister_memory(self, addresses):
                return -1

            def unregister_memory(self, address):
                self.calls += 1
                if self.freed < 10:
                    self.freed += 1
                    return 0
                return -1

        engine = Stalling()
        released = deregister_memory_region(engine, self._blocks(100), protocol="rdma")
        self.assertFalse(released)
        # One pass over all 100, one over the 90 that stalled, then stop.
        self.assertEqual(engine.calls, 100 + 90)

    def test_keeps_retrying_while_it_makes_progress(self):
        # A driver that frees 40 per pass needs three passes; stopping at the
        # first partial failure would leave pages pinned that do come free.
        class Draining:
            def __init__(self):
                self.calls = 0
                self.freed_this_pass = 0
                self.pinned = set(range(100))

            def batch_unregister_memory(self, addresses):
                return -1

            def unregister_memory(self, address):
                self.calls += 1
                if self.freed_this_pass >= 40:
                    return -1
                self.freed_this_pass += 1
                self.pinned.discard((address - 0x1000) // 0x100)
                if not self.pinned:
                    return 0
                if self.freed_this_pass == 40:
                    # Next call starts the following pass.
                    self.freed_this_pass = -1_000_000
                return 0

        engine = Draining()
        released = deregister_memory_region(engine, self._blocks(100), protocol="rdma")
        self.assertTrue(released)
        self.assertEqual(engine.pinned, set())

    def test_nvlink_never_reaches_the_retry_loop(self):
        class Exploding:
            def batch_unregister_memory(self, addresses):
                return -1

            def unregister_memory(self, address):
                raise AssertionError("NVLink should not retry block by block")

        self.assertTrue(
            deregister_memory_region(
                Exploding(), self._blocks(10), protocol="nvlink_intra"
            )
        )


class TestReaderRegistrationRelease(CustomTestCase):
    """What a registration costs is a transport property, not a preference:
    ibv_reg_mr pins pages the KV pool sizes itself around, an NVLink one frees
    nothing. A single global boolean silently harms one of the two.
    """

    def _release_decision(self, protocol) -> bool:
        from sglang.srt.model_loader.loader import (
            _should_release_reader_registration,
        )

        return _should_release_reader_registration(protocol)

    def test_nic_transports_release(self):
        self.assertTrue(self._release_decision("rdma"))

    def test_nvlink_transports_do_not(self):
        for protocol in ("nvlink", "nvlink_intra"):
            with self.subTest(protocol=protocol):
                self.assertFalse(self._release_decision(protocol))

    def test_the_engines_transport_decides_not_the_environment(self):
        # A client resolves its transport against what the seed offers, so with
        # the env unset the engine can be on nvlink_intra while the env still
        # reads "rdma" -- deriving the decision from the env would spend seconds
        # failing to release a registration that holds nothing.
        self.assertFalse(self._release_decision("nvlink_intra"))

    def test_an_explicit_setting_overrides_the_transport(self):
        # The env stays an escape hatch for a transport whose cost we got wrong.
        with envs.SGLANG_ENABLE_REMOTE_INSTANCE_MR_RELEASE.override(True):
            self.assertTrue(self._release_decision("nvlink"))
        with envs.SGLANG_ENABLE_REMOTE_INSTANCE_MR_RELEASE.override(False):
            self.assertFalse(self._release_decision("rdma"))

    def test_malformed_payloads_do_not_raise(self):
        # A str has a length and is indexable, so it passes a naive shape check
        # and would yield a nonsense session id.
        for payload in ("garbage", [], None, {}, ["only-one"]):
            with self.subTest(payload=payload):
                self.assertIsNone(_parse_seed_transfer_engine_info(payload))


class TestFabricExportability(CustomTestCase):
    """MNNVL serves a buffer by exporting the allocation behind it as a fabric
    handle, which exists only for cuMemCreate memory -- and the torch caching
    allocator's default path is cudaMalloc. Mooncake answers a registration it
    cannot export with *success* and an empty segment, so the seed looks healthy
    and the reader fails at transfer time with "Requested address ... not
    found!". Observed on GB300: all 8 client ranks fell through to disk.
    """

    def _fake_cuda(self, *, retain, export):
        calls = []

        class FakeCuda:
            def cuMemRetainAllocationHandle(self, handle, pointer):
                calls.append("retain")
                return retain

            def cuMemExportToShareableHandle(self, out, handle, kind, flags):
                calls.append(("export", kind.value))
                return export

            def cuMemRelease(self, handle):
                calls.append("release")
                return 0

        return FakeCuda(), calls

    def _probe(self, *, retain, export):
        cuda, calls = self._fake_cuda(retain=retain, export=export)
        with mock.patch.object(ctypes, "CDLL", return_value=cuda):
            result = nvlink_fabric_utils._pointer_is_fabric_exportable(0x7F0000000000)
        return result, calls

    def test_cuda_malloc_memory_cannot_be_exported(self):
        # The GB300 case: retain fails, which is the whole bug.
        result, calls = self._probe(retain=1, export=0)
        self.assertFalse(result)
        self.assertEqual(calls, ["retain"])

    def test_retaining_is_not_enough_on_its_own(self):
        # An allocation carries the handle types it was created with, so
        # cuMemCreate memory that never asked for FABRIC retains fine and still
        # cannot export. Dropping the second step would call
        # expandable_segments:True a fix when it is not.
        result, calls = self._probe(retain=0, export=1)
        self.assertFalse(result)
        self.assertEqual(calls, ["retain", ("export", 0x8), "release"])

    def test_fabric_exportable_memory_passes(self):
        result, calls = self._probe(retain=0, export=0)
        self.assertTrue(result)
        self.assertEqual(calls, ["retain", ("export", 0x8), "release"])

    def test_the_retained_handle_is_released_even_when_export_fails(self):
        _, calls = self._probe(retain=0, export=1)
        self.assertIn("release", calls)

    def test_no_libcuda_is_not_exportable(self):
        with mock.patch.object(ctypes, "CDLL", side_effect=OSError("no libcuda")):
            self.assertFalse(
                nvlink_fabric_utils._pointer_is_fabric_exportable(0x7F0000000000)
            )


def _transporter_module():
    from sglang.srt.model_executor.model_runner_components import (
        remote_instance_weight_transporter,
    )

    return remote_instance_weight_transporter


def _transporter(*, fabric_identity):
    return _transporter_module().RemoteInstanceWeightTransporter(
        get_model=lambda: None,
        tp_rank=0,
        gpu_id=0,
        fabric_identity=fabric_identity,
    )


def _patch_exportable(value):
    """Patch where the transporter looks the probe up, not where it is defined.

    The transporter imports it by name at module load, so patching
    nvlink_fabric_utils would leave that binding untouched and the test would
    exercise the real CUDA probe.
    """
    return mock.patch.object(
        _transporter_module(),
        "caching_allocator_memory_is_fabric_exportable",
        **value,
    )


class TestSeedWithdrawsAnUnservableMnnvlOffer(CustomTestCase):
    """A seed in a clique can still be unable to serve MNNVL. Offering on clique
    membership alone is what made every GB300 client pick the one transport that
    could not work, and lose the NIC endpoint that would have.
    """

    def _offered(self, *, exportable):
        transporter = _transporter(fabric_identity=_identity())
        with _patch_exportable({"return_value": exportable}):
            return transporter._servable_nvlink_protocols()

    def test_an_unexportable_seed_offers_only_intra_node(self):
        self.assertEqual(self._offered(exportable=False), ("nvlink_intra",))

    def test_an_exportable_seed_offers_both(self):
        self.assertEqual(self._offered(exportable=True), ("nvlink_intra", "nvlink"))

    def test_intra_node_survives_the_withdrawal(self):
        # nvlink_intra registers through cudaIpcGetMemHandle, which is fine on
        # cudaMalloc memory, so the fabric verdict must not gate it. It is the
        # only NVLink transport that works for a same-host client.
        self.assertIn("nvlink_intra", self._offered(exportable=False))

    def test_a_seed_off_the_fabric_never_reaches_the_probe(self):
        transporter = _transporter(fabric_identity=None)
        with _patch_exportable({"side_effect": AssertionError("should not be probed")}):
            self.assertEqual(
                transporter._servable_nvlink_protocols(), ("nvlink_intra",)
            )


class TestClientSkipsAStaleMnnvlOffer(CustomTestCase):
    """A seed predating the check above still offers MNNVL it cannot serve. The
    client has to decline, because the cost of accepting is not the NIC -- it is
    a failed transfer, and the fallback from there is disk.
    """

    def _credible(self, *, exportable):
        transporter = _transporter(fabric_identity=_identity())
        with _patch_exportable({"return_value": exportable}):
            return transporter._mnnvl_offer_is_credible()

    def test_declines_when_this_ranks_memory_cannot_be_exported(self):
        self.assertFalse(self._credible(exportable=False))

    def test_accepts_when_it_can(self):
        self.assertTrue(self._credible(exportable=True))


def _patch_arena_capable(value):
    return mock.patch.object(
        _transporter_module(), "device_can_back_fabric_arena", **value
    )


class TestArenaChangesWhatASeedOffers(CustomTestCase):
    """With the arena the weights are on FABRIC memory, so the pre-load gate has
    to stop asking about the default allocator -- and nvlink_intra has to go,
    because cudaIpcGetMemHandle fails on a VMM pointer.
    """

    def _offered(self, *, requested, arena_capable):
        transporter = _transporter(fabric_identity=_identity())
        with mock.patch.object(
            type(transporter), "_fabric_weights_requested", lambda self: requested
        ):
            with _patch_arena_capable({"return_value": arena_capable}):
                # An arena seed must not consult the default allocator: its answer
                # is False by construction and would withdraw a working offer.
                with _patch_exportable(
                    {"side_effect": AssertionError("should not be probed")}
                    if requested
                    else {"return_value": False}
                ):
                    return transporter._servable_nvlink_protocols()

    def test_an_arena_seed_offers_mnnvl_alone(self):
        # nvlink_intra cannot register VMM memory at all, and MNNVL reaches
        # same-host peers too, so dropping it costs them nothing.
        self.assertEqual(self._offered(requested=True, arena_capable=True), ("nvlink",))

    def test_a_device_that_cannot_back_an_arena_still_offers_intra_node(self):
        # No MNNVL endpoint means the migration is skipped, so the weights stay on
        # the default allocator -- which is exactly the memory nvlink_intra can
        # serve. Withdrawing it here would cost same-host peers for nothing.
        self.assertEqual(
            self._offered(requested=True, arena_capable=False), ("nvlink_intra",)
        )

    def test_without_the_arena_the_old_behaviour_stands(self):
        self.assertEqual(
            self._offered(requested=False, arena_capable=True), ("nvlink_intra",)
        )


class TestFabricWeightsAreOptIn(CustomTestCase):
    """The migration rebinds every weight's storage, which a module that cached
    ``.data`` rather than the Parameter would not survive. So it stays off unless
    asked for, and the env check has to come first -- _is_seed() reads global
    runtime context that a plain transporter has no business touching.
    """

    def test_off_by_default_without_consulting_anything_else(self):
        transporter = _transporter(fabric_identity=_identity())
        with mock.patch.object(
            type(transporter), "_is_seed", lambda self: 1 / 0  # would raise
        ):
            self.assertFalse(transporter._fabric_weights_requested())

    def test_a_draft_worker_never_requests_it(self):
        # The draft neither serves weights nor owns the manifest key.
        transporter = _transporter_module().RemoteInstanceWeightTransporter(
            get_model=lambda: None, tp_rank=0, gpu_id=0, is_draft_worker=True
        )
        with envs.SGLANG_ENABLE_REMOTE_INSTANCE_FABRIC_WEIGHTS.override(True):
            with mock.patch.object(type(transporter), "_is_seed", lambda self: True):
                self.assertFalse(transporter._fabric_weights_requested())

    def test_a_client_never_requests_it(self):
        # A client reads into ordinary buffers; only a seed has to export.
        transporter = _transporter(fabric_identity=_identity())
        with envs.SGLANG_ENABLE_REMOTE_INSTANCE_FABRIC_WEIGHTS.override(True):
            with mock.patch.object(type(transporter), "_is_seed", lambda self: False):
                self.assertFalse(transporter._fabric_weights_requested())

    def test_a_seed_that_asked_for_it_gets_it(self):
        transporter = _transporter(fabric_identity=_identity())
        with envs.SGLANG_ENABLE_REMOTE_INSTANCE_FABRIC_WEIGHTS.override(True):
            with mock.patch.object(type(transporter), "_is_seed", lambda self: True):
                self.assertTrue(transporter._fabric_weights_requested())


class TestWithdrawMnnvlWhenWeightsEscape(CustomTestCase):
    """_can_serve_mnnvl answers on intent, before the weights exist. This is where
    intent meets the real pointers, and it is the only thing standing between an
    escaped weight and the client-side failure the arena was built to remove.
    """

    def _transporter_with(self, *, mappings, manifest):
        transporter = _transporter(fabric_identity=_identity())
        transporter.weight_arena = SimpleNamespace(mappings=mappings)
        transporter.endpoints["nvlink"] = _transporter_module()._TransportEndpoint(
            protocol="nvlink",
            engine=object(),
            session_id="10.1.1.4:16002",
            weight_info=manifest,
        )
        return transporter

    def test_an_escaped_weight_withdraws_the_endpoint(self):
        transporter = self._transporter_with(
            mappings=[(0x1000, 0x1000)],
            manifest={"inside": (0x1000, 8, 1), "escaped": (0x99000, 8, 1)},
        )
        transporter._withdraw_mnnvl_if_weights_escaped_the_arena()
        self.assertNotIn("nvlink", transporter.endpoints)

    def test_a_fully_covered_manifest_keeps_it(self):
        transporter = self._transporter_with(
            mappings=[(0x1000, 0x1000)], manifest={"a": (0x1000, 8, 1)}
        )
        transporter._withdraw_mnnvl_if_weights_escaped_the_arena()
        self.assertIn("nvlink", transporter.endpoints)

    def test_no_arena_leaves_the_endpoint_alone(self):
        # The non-arena path publishes per-tensor pointers that were never meant
        # to fall inside a mapping, so this check must not run against it.
        transporter = self._transporter_with(
            mappings=[(0x1000, 0x1000)], manifest={"a": (0x99000, 8, 1)}
        )
        transporter.weight_arena = None
        transporter._withdraw_mnnvl_if_weights_escaped_the_arena()
        self.assertIn("nvlink", transporter.endpoints)


if __name__ == "__main__":
    unittest.main()
