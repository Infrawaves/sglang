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


class _FakeWeight:
    def __init__(self, ptr, numel=8, element_size=2):
        self._ptr, self._numel, self._element_size = ptr, numel, element_size

    def data_ptr(self):
        return self._ptr

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size


class _FakeModel:
    """Stands in for nn.Module far enough for the registration walk."""

    def __init__(self, weights):
        self._weights = list(weights)

    def named_parameters(self, remove_duplicate=True):
        return iter(self._weights)


class _FakeDriver:
    """cuMemGetAddressRange over a fixed allocation map.

    ``allocations`` is a list of ``(base, size)``; a pointer resolves to the one
    whose range contains it, and to an error when none does.
    """

    class CUresult:
        CUDA_SUCCESS = 0
        CUDA_ERROR_INVALID_VALUE = 1

    def __init__(self, allocations):
        self.allocations = allocations

    def cuMemGetAddressRange(self, ptr):
        for base, size in self.allocations:
            if base <= ptr < base + size:
                return (self.CUresult.CUDA_SUCCESS, base, size)
        return (self.CUresult.CUDA_ERROR_INVALID_VALUE, 0, 0)


class _RecordingEngine:
    def __init__(self, fail_on_call=None):
        self.registered = []
        self.deregistered = []
        self._fail_on_call = fail_on_call

    def register_memory(self, address, size):
        if (
            self._fail_on_call is not None
            and len(self.registered) == self._fail_on_call
        ):
            return -1
        self.registered.append((address, size))
        return 0

    def batch_unregister_memory(self, addresses):
        self.deregistered.extend(addresses)
        return 0


class TestVmmRegistration(CustomTestCase):
    """Pooled weights are invisible to the snapshot walk, which reads the caching
    allocator's segments, so every transport reaches them through the allocations
    instead. MNNVL additionally needs the deduplication: it registers the
    allocation enclosing a weight and appends a descriptor per call, so
    registering per weight would publish the same allocation many times over and
    serialize every copy into the segment other ranks read.
    """

    def _register(self, model, engine, allocations, vmm_bases=None, protocol="nvlink"):
        """``vmm_bases`` limits which allocation bases came from cuMemCreate;
        None means all of them did."""
        from sglang.srt.model_loader import remote_instance_weight_loader_utils as u

        def is_vmm(ptr):
            return vmm_bases is None or ptr in vmm_bases

        with (
            mock.patch(
                "sglang.srt.utils.cuda_vmm_utils.is_vmm_pointer", side_effect=is_vmm
            ),
            mock.patch(
                "sglang.srt.utils.cuda_vmm_utils._get_cuda_driver",
                return_value=_FakeDriver(allocations),
            ),
        ):
            return u.register_memory_region_vmm(model, engine, protocol=protocol)

    def test_one_registration_per_allocation_not_per_weight(self):
        # Three weights, two allocations. Per-weight registration would push
        # three descriptors, two of them duplicates of the same base.
        allocations = [(0x1000, 0x1000), (0x4000, 0x1000)]
        model = _FakeModel(
            [
                ("a", _FakeWeight(0x1000)),
                ("b", _FakeWeight(0x1200)),
                ("c", _FakeWeight(0x4000)),
            ]
        )
        engine = _RecordingEngine()
        weight_info, blocks = self._register(model, engine, allocations)

        self.assertEqual(engine.registered, [(0x1000, 0x1000), (0x4000, 0x1000)])
        self.assertEqual(blocks, [(0x1000, 0x1000), (0x4000, 0x1000)])
        # The manifest stays keyed per weight at the weight's own pointer: that
        # is what the client looks its own parameters up by.
        self.assertEqual(weight_info["b"], (0x1200, 8, 2))
        self.assertEqual(set(weight_info), {"a", "b", "c"})

    def test_a_weight_straddling_two_allocations_registers_both(self):
        # The pool grew mid-tensor, so covering only the first allocation would
        # leave the tail unreachable -- and the client reads the whole span.
        allocations = [(0x1000, 0x1000), (0x2000, 0x1000)]
        model = _FakeModel([("big", _FakeWeight(0x1FF0, numel=0x20, element_size=1))])
        engine = _RecordingEngine()
        _, blocks = self._register(model, engine, allocations)
        self.assertEqual(blocks, [(0x1000, 0x1000), (0x2000, 0x1000)])

    def test_weights_outside_a_vmm_allocation_register_nothing(self):
        # Half a model behind a handle nobody can complete is worse than none:
        # the client would fail mid-transfer instead of picking another endpoint.
        allocations = [(0x1000, 0x1000)]
        model = _FakeModel(
            [("a", _FakeWeight(0x1000)), ("stray", _FakeWeight(0x99000))]
        )
        engine = _RecordingEngine()
        self.assertIsNone(self._register(model, engine, allocations))
        self.assertEqual(engine.registered, [])

    def test_cudamalloc_weights_fall_through_rather_than_registering(self):
        engine = _RecordingEngine()
        self.assertIsNone(
            self._register(
                _FakeModel([("a", _FakeWeight(0x1000))]),
                engine,
                [(0x1000, 0x1000)],
                vmm_bases=(),
            )
        )
        self.assertEqual(engine.registered, [])

    def test_one_un_exportable_allocation_withdraws_all_of_them(self):
        # cuMemGetAddressRange resolves cudaMalloc allocations too, so it cannot
        # tell the allocators apart. Sampling one base would register the pooled
        # ones and publish a manifest covering the rest, which no client can read.
        allocations = [(0x1000, 0x1000), (0x4000, 0x1000)]
        model = _FakeModel(
            [("pooled", _FakeWeight(0x1000)), ("heap", _FakeWeight(0x4000))]
        )
        engine = _RecordingEngine()
        self.assertIsNone(
            self._register(model, engine, allocations, vmm_bases=(0x1000,))
        )
        self.assertEqual(engine.registered, [])

    def test_a_failure_partway_releases_what_it_already_took(self):
        allocations = [(0x1000, 0x1000), (0x4000, 0x1000)]
        model = _FakeModel([("a", _FakeWeight(0x1000)), ("c", _FakeWeight(0x4000))])
        engine = _RecordingEngine(fail_on_call=1)
        with self.assertRaises(RuntimeError):
            self._register(model, engine, allocations)
        # The caller never learns about the first one, so leaving it registered
        # would strand it with nobody holding a handle.
        self.assertEqual(engine.deregistered, [0x1000])


class TestRegistrationDispatch(CustomTestCase):
    """Which walk to use is a property of where the weights live, not of the
    transport. Keying it on the transport strands the NIC on pooled weights: the
    snapshot walk reads the caching allocator's segments, finds none of them, and
    registers nothing -- so the one transport a peer outside the NVLink domain can
    use would serve a manifest whose addresses were never registered.
    """

    def _dispatch_for(self, protocol, weights_are_pooled):
        from sglang.srt.model_loader import remote_instance_weight_loader_utils as u

        vmm = ("vmm", [(0x1000, 0x1000)]) if weights_are_pooled else None
        with (
            mock.patch.object(
                u, "register_memory_region_vmm", return_value=vmm
            ) as vmm_walk,
            mock.patch.object(
                u, "register_memory_region_v2", return_value=("v2", [])
            ) as snapshot_walk,
        ):
            result, _ = u.register_memory_region(
                _FakeModel([]), _RecordingEngine(), protocol=protocol
            )
        return result, vmm_walk.called, snapshot_walk.called

    def test_pooled_weights_take_the_allocation_walk_on_every_transport(self):
        # Including rdma: a peer in another NVL72 has only the NIC, and the
        # snapshot walk cannot see pooled weights to register for it.
        for protocol in ("nvlink", "nvlink_intra", "rdma", None):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self._dispatch_for(protocol, weights_are_pooled=True),
                    ("vmm", True, False),
                )

    def test_caching_allocator_weights_keep_the_snapshot_walk(self):
        for protocol in ("nvlink", "nvlink_intra", "rdma", None):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self._dispatch_for(protocol, weights_are_pooled=False),
                    ("v2", True, True),
                )


def _transporter(model, endpoints):
    from sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter import (
        RemoteInstanceWeightTransporter,
        _TransportEndpoint,
    )

    transporter = RemoteInstanceWeightTransporter(
        get_model=lambda: model, tp_rank=0, gpu_id=0
    )
    transporter.endpoints = {
        protocol: _TransportEndpoint(
            protocol=protocol, engine=_RecordingEngine(), session_id="10.1.1.4:16000"
        )
        for protocol in endpoints
    }
    return transporter


class TestServableProtocolsUnderTheFabricPool(CustomTestCase):
    """CUDA IPC cannot export cuMemCreate memory, so a seed whose weights come
    from the fabric pool can never serve nvlink_intra. Building it anyway costs a
    registration walk over every weight on every rank before it is dropped, and
    the knob is readable now while the weights are not: endpoints are built
    before load_model.
    """

    def _servable(self, pool_value):
        transporter = _transporter(_FakeModel([]), ())
        transporter.fabric_identity = _identity()
        with envs.SGLANG_REMOTE_INSTANCE_WEIGHT_MEM_POOL.override(pool_value):
            return transporter._servable_nvlink_protocols()

    def test_the_pool_withdraws_cuda_ipc_before_it_is_built(self):
        self.assertEqual(self._servable("NVLINK"), ("nvlink",))

    def test_without_the_pool_cuda_ipc_is_still_offered(self):
        self.assertEqual(self._servable(None), ("nvlink_intra", "nvlink"))

    def test_a_seed_off_the_fabric_serves_only_cuda_ipc(self):
        # No clique to publish a fabric handle into, so MNNVL is not servable --
        # and without the pool CUDA IPC still is.
        transporter = _transporter(_FakeModel([]), ())
        transporter.fabric_identity = None
        with envs.SGLANG_REMOTE_INSTANCE_WEIGHT_MEM_POOL.override(None):
            self.assertEqual(
                transporter._servable_nvlink_protocols(), ("nvlink_intra",)
            )


class TestSeedWithdrawsUnservableMnnvl(CustomTestCase):
    """mooncake answers a fabric-handle refusal with success after registering
    nothing, so `ret == 0` does not mean the seed can serve. Advertising anyway
    costs every client a full handshake, registration and transfer before it
    reads `Requested address ... not found!` and disk-loads -- on every rank.
    """

    def _drop(self, transporter, is_vmm):
        with mock.patch(
            "sglang.srt.utils.cuda_vmm_utils.is_vmm_pointer", side_effect=is_vmm
        ):
            transporter._drop_endpoints_that_registered_nothing()
        return tuple(transporter.endpoints)

    def test_cudamalloc_weights_withdraw_mnnvl_and_keep_the_rest(self):
        transporter = _transporter(
            _FakeModel([("a", _FakeWeight(0x1000))]), ("nvlink", "rdma")
        )
        self.assertEqual(self._drop(transporter, lambda ptr: False), ("rdma",))

    def test_pooled_weights_keep_mnnvl(self):
        transporter = _transporter(
            _FakeModel([("a", _FakeWeight(0x1000))]), ("nvlink", "rdma")
        )
        self.assertEqual(self._drop(transporter, lambda ptr: True), ("nvlink", "rdma"))

    def test_an_unprobeable_allocator_keeps_mnnvl(self):
        # The probe failing says nothing about what mooncake's own registration
        # managed to do; dropping on it would disable a transport that works.
        transporter = _transporter(
            _FakeModel([("a", _FakeWeight(0x1000))]), ("nvlink", "rdma")
        )
        self.assertEqual(
            self._drop(transporter, RuntimeError("no driver")), ("nvlink", "rdma")
        )

    def test_cuda_ipc_is_not_gated_on_the_allocator(self):
        # nvlink_intra reaches cudaMalloc weights fine, so the withdrawal must
        # not widen to it.
        transporter = _transporter(
            _FakeModel([("a", _FakeWeight(0x1000))]), ("nvlink_intra", "rdma")
        )
        self.assertEqual(
            self._drop(transporter, lambda ptr: False), ("nvlink_intra", "rdma")
        )


class TestPerEndpointRegistrationIsolation(CustomTestCase):
    """A seed holds one engine per transport. CUDA IPC cannot export VMM-backed
    memory, so a seed allocating weights from the fabric pool has nvlink_intra
    refuse the same buffers MNNVL accepts. Letting that out would leave the
    instance serving nothing over the transports that do work.
    """

    def _register_with(self, transporter, register):
        from sglang.srt.model_executor.model_runner_components import (
            remote_instance_weight_transporter as t,
        )

        with mock.patch.object(t, "register_memory_region", side_effect=register):
            transporter._register_each_endpoint()
        return tuple(transporter.endpoints)

    def test_one_transports_refusal_does_not_take_the_others(self):
        transporter = _transporter(_FakeModel([]), ("nvlink_intra", "nvlink", "rdma"))

        def register(model, engine, protocol=None):
            if protocol == "nvlink_intra":
                raise RuntimeError("cudaIpcGetMemHandle failed")
            return ({"a": (0x1000, 8, 2)}, [(0x1000, 0x1000)])

        self.assertEqual(self._register_with(transporter, register), ("nvlink", "rdma"))
        self.assertIsNotNone(transporter.endpoints["nvlink"].weight_info)

    def test_registering_zero_regions_withdraws_the_endpoint(self):
        # Registering nothing reports success, so without this check the seed
        # publishes a manifest whose addresses no client can read.
        transporter = _transporter(_FakeModel([]), ("nvlink_intra", "rdma"))

        def register(model, engine, protocol=None):
            if protocol == "nvlink_intra":
                return ({"a": (0x1000, 8, 2)}, [])
            return ({"a": (0x1000, 8, 2)}, [(0x1000, 0x1000)])

        self.assertEqual(self._register_with(transporter, register), ("rdma",))


class TestWeightMemPool(CustomTestCase):
    """The pool exists so MNNVL can export a handle for the weights. Asking for
    it and silently getting the default allocator reproduces exactly the failure
    it prevents, so an unavailable pool is a startup error.
    """

    def setUp(self):
        super().setUp()
        from sglang.srt.model_loader import weight_mem_pool

        weight_mem_pool._pool = None

    def test_unset_returns_no_pool_without_touching_mooncake(self):
        # The default path must not import mooncake at all: most deployments do
        # not have it, and importing it here would make weight loading depend on
        # a package the NIC path never needs.
        from sglang.srt.model_loader.weight_mem_pool import maybe_init_weight_mem_pool

        with (
            envs.SGLANG_REMOTE_INSTANCE_WEIGHT_MEM_POOL.override(None),
            mock.patch.dict(sys.modules, {"mooncake.allocator": None}),
        ):
            self.assertIsNone(maybe_init_weight_mem_pool())

    def test_a_misspelled_pool_is_not_silently_ignored(self):
        # Accepting it would leave the operator believing MNNVL is served from a
        # fabric pool while the weights came from the caching allocator.
        from sglang.srt.model_loader.weight_mem_pool import maybe_init_weight_mem_pool

        with envs.SGLANG_REMOTE_INSTANCE_WEIGHT_MEM_POOL.override("NVLNIK"):
            with self.assertRaises(ValueError):
                maybe_init_weight_mem_pool()

    def test_a_gpu_in_a_clique_is_told_to_check_the_imex_channel(self):
        # The two causes need different fixes and the handle type alone cannot
        # tell them apart: in a clique but not FABRIC is the IMEX channel not
        # reaching the process, which is what a container misses.
        from sglang.srt.model_loader.weight_mem_pool import unusable_fabric_reason

        reason = unusable_fabric_reason(
            device="cuda:0", handle_type_name="POSIX_FD", fabric_ready=True
        )
        self.assertIn("imex-channels", reason)
        self.assertNotIn("x86 HGX", reason)

    def test_a_gpu_outside_any_clique_is_told_there_is_no_domain(self):
        from sglang.srt.model_loader.weight_mem_pool import unusable_fabric_reason

        reason = unusable_fabric_reason(
            device="cuda:0", handle_type_name="POSIX_FD", fabric_ready=False
        )
        self.assertIn("x86 HGX", reason)
        self.assertNotIn("imex-channels", reason)


if __name__ == "__main__":
    unittest.main()
