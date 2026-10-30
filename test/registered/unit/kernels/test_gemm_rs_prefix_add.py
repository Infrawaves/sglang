"""CPU tests for the external-library GEMM + RS + prefix wrapper.

Only allocation/collective boundaries are mocked. These tests do not claim
CUDA Graph execution, numeric kernel validation, or model smoke coverage.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

_ROOT = Path(__file__).resolve().parents[4]
_PACKAGE_PATH = _ROOT / "python/sglang/kernels/ops/communication/gemm_rs_prefix_add"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# File imports keep the tests independent of optional server/CUDA packages.
_CI = _load_module("gemm_rs_test_ci", _ROOT / "python/sglang/test/ci/ci_register.py")
register_cpu_ci = _CI.register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")
_PACKAGE = types.ModuleType("gemm_rs_policy_test_package")
_PACKAGE.__path__ = [str(_PACKAGE_PATH)]
sys.modules[_PACKAGE.__name__] = _PACKAGE
_WORKSPACE = _load_module(
    _PACKAGE.__name__ + ".workspace", _PACKAGE_PATH / "workspace.py"
)
_MODULE = _load_module(_PACKAGE.__name__ + ".dispatch", _PACKAGE_PATH / "dispatch.py")
_TILE_ORDER = _load_module(
    _PACKAGE.__name__ + ".tile_order", _PACKAGE_PATH / "tile_order.py"
)
KernelConfig = _WORKSPACE.KernelConfig
WorkspaceLayout = _WORKSPACE.WorkspaceLayout
SUPPORTED_CONFIGS = _WORKSPACE.SUPPORTED_CONFIGS
DispatchConfig = _MODULE.DispatchConfig
load_dispatch_table = _MODULE.load_dispatch_table
build_tile_m_order = _TILE_ORDER.build_tile_m_order


class WorkspaceLayoutTest(unittest.TestCase):
    def test_partial_n_tile_is_in_ready_capacity(self):
        layout = WorkspaceLayout(1000, 136, 4, configs=(KernelConfig(),))
        self.assertEqual(layout.tile_n_capacity, 2)
        self.assertEqual(layout.ready_numel, 16)
        self.assertEqual(layout.partial_numel, 136000)

    def test_push_plane_stride_is_capacity_based_and_16_byte_aligned(self):
        layout = WorkspaceLayout(1000, 136, 4)
        self.assertEqual(layout.plane_stride_elements, 250 * 136)
        self.assertEqual(layout.plane_stride_elements * 2 % 16, 0)
        for m in (4, 8, 500, 1000, 4):
            self.assertTrue(layout.can_fuse(m, 1536))
            for source in range(4):
                plane_begin = source * layout.plane_stride_elements
                write_end = plane_begin + m // 4 * 136
                self.assertLessEqual(
                    write_end, (source + 1) * layout.plane_stride_elements
                )
        self.assertEqual(layout.recv_data_numel, 4 * 250 * 136)

    def test_shared_arena_has_disjoint_aligned_partial_and_receive_regions(self):
        # Include tiny and non-tile-aligned shapes. Both paths keep the same
        # capacity-based addresses when active M and tile configuration change.
        for world_size in (2, 4, 8):
            for max_m, n in ((world_size, 8), (1000, 136), (4096, 7168)):
                with self.subTest(world_size=world_size, max_m=max_m, n=n):
                    layout = WorkspaceLayout(max_m, n, world_size)
                    # The GEMM partial starts right after the receive planes.
                    partial_offset = layout.recv_data_numel
                    self.assertEqual(partial_offset * 2 % 16, 0)
                    self.assertEqual(
                        layout.arena_numel, partial_offset + layout.partial_numel
                    )
                    for m in (world_size, max_m):
                        for source in range(world_size):
                            begin = source * layout.plane_stride_elements
                            end = begin + (m // world_size) * n
                            self.assertLessEqual(end, partial_offset)
                        partial_end = partial_offset + m * n
                        self.assertLessEqual(partial_end, layout.arena_numel)

    def test_all_compiled_configs_fit_one_fixed_counter_capacity(self):
        for p in (2, 4, 8):
            layout = WorkspaceLayout(4096, 7168, p)
            self.assertEqual(layout.tile_n_capacity, 112)
            self.assertEqual(layout.ready_numel, 256 * 112)
            for config in SUPPORTED_CONFIGS:
                for m in (p, p * 33, 4096):
                    self.assertTrue(layout.can_fuse(m, 1536, config))
                    m_tiles = (m + config.tile_m - 1) // config.tile_m
                    n_tiles = (layout.n + config.tile_n - 1) // config.tile_n
                    final_slot = (m_tiles - 1) * layout.tile_n_capacity + n_tiles - 1
                    self.assertLess(final_slot, layout.ready_numel)
                    self.assertEqual(
                        layout.cluster_work(m, config),
                        m_tiles * (n_tiles // config.cluster_hidden),
                    )

    def test_config_set_controls_capacity_without_changing_payload_strides(self):
        configs = (KernelConfig(64, 128, 1), KernelConfig(128, 64, 2))
        layout = WorkspaceLayout(1000, 256, 4, configs=configs)
        default = WorkspaceLayout(1000, 256, 4)
        self.assertEqual(layout.tile_n_capacity, 4)
        self.assertEqual(layout.ready_numel, 16 * 4)
        self.assertEqual(layout.partial_numel, default.partial_numel)
        self.assertEqual(layout.plane_stride_elements, default.plane_stride_elements)
        self.assertEqual(layout.recv_data_numel, default.recv_data_numel)
        self.assertEqual(layout.arena_numel, default.arena_numel)
        for config in configs:
            self.assertTrue(layout.can_fuse(1000, 128, config))
        with self.assertRaisesRegex(ValueError, "not declared"):
            layout.can_fuse(1000, 128, KernelConfig())

    def test_hidden_cluster_eligibility_depends_on_selected_tile_n(self):
        layout = WorkspaceLayout(1024, 128, 4)
        narrow = KernelConfig(64, 64, 2)
        wide = KernelConfig(64, 128, 2)
        self.assertTrue(layout.can_fuse(1024, 128, narrow))
        self.assertFalse(layout.can_fuse(1024, 128, wide))
        self.assertEqual(layout.cluster_work(4, narrow), 1)
        self.assertEqual(layout.cluster_work(1024, narrow), 16)
        # A partial hidden tile is valid for the 1-CTA kernel, but the 2-CTA
        # path cannot leave one member of a cluster without a physical tile.
        tail = WorkspaceLayout(1024, 136, 4)
        self.assertTrue(tail.can_fuse(1024, 128, KernelConfig(64, 64, 1)))
        self.assertFalse(tail.can_fuse(1024, 128, narrow))

    def test_invalid_capacities_calls_and_configs(self):
        for args in ((0, 128, 4), (999, 128, 4), (1000, 129, 4), (1000, 128, 3)):
            with self.assertRaises(ValueError):
                WorkspaceLayout(*args)
        for args in (
            (0, 128, 1),
            (48, 128, 1),
            (128, 32, 1),
            (128, 128, 3),
            (128, 256, 1),
        ):
            with self.assertRaises(ValueError):
                KernelConfig(*args)
        with self.assertRaises(ValueError):
            WorkspaceLayout(1000, 128, 4, configs=())
        layout = WorkspaceLayout(1000, 128, 4)
        for m, k in ((0, 128), (999, 128), (4, 0)):
            with self.assertRaises(ValueError):
                layout.can_fuse(m, k)
        for m, k in ((1004, 128), (4, 127)):
            self.assertFalse(layout.can_fuse(m, k))


class Sm100TileOrderTest(unittest.TestCase):
    def test_pull_uses_identical_round_robin_order_on_four_ranks(self):
        expected = (0, 2, 4, 6, 1, 3, 5, 7)
        for _rank in range(4):
            self.assertEqual(build_tile_m_order(256, 4, 128, None), expected)

    def test_push_starts_every_round_at_its_source_rank(self):
        expected = (
            (0, 2, 4, 6, 1, 3, 5, 7),
            (2, 4, 6, 0, 3, 5, 7, 1),
            (4, 6, 0, 2, 5, 7, 1, 3),
            (6, 0, 2, 4, 7, 1, 3, 5),
        )
        for rank, order in enumerate(expected):
            self.assertEqual(build_tile_m_order(256, 4, 128, rank), order)

    def test_cross_shard_tile_occurs_once_but_intersects_both_shards(self):
        num_shard_tokens, world_size, tile_m = 250, 4, 128
        num_tile_m = (num_shard_tokens * world_size + tile_m - 1) // tile_m
        push_orders = [
            build_tile_m_order(num_shard_tokens, world_size, tile_m, rank)
            for rank in range(world_size)
        ]
        for order in [
            build_tile_m_order(num_shard_tokens, world_size, tile_m, None),
            *push_orders,
        ]:
            self.assertEqual(sorted(order), list(range(num_tile_m)))
            self.assertEqual(order.count(1), 1)
        self.assertLess(tile_m, num_shard_tokens)
        self.assertGreater(2 * tile_m, num_shard_tokens)
        for rank, order in enumerate(push_orders):
            first_tile_start = order[0] * tile_m
            first_tile_end = min(
                first_tile_start + tile_m, world_size * num_shard_tokens
            )
            shard_start = rank * num_shard_tokens
            shard_end = shard_start + num_shard_tokens
            self.assertLess(first_tile_start, shard_end)
            self.assertGreater(first_tile_end, shard_start)

    def test_non_divisible_last_tile_and_small_shards_are_permutations(self):
        for num_shard_tokens, world_size, tile_m in (
            (250, 4, 128),
            (33, 3, 128),
            (1, 8, 128),
            (257, 4, 128),
        ):
            num_tile_m = (num_shard_tokens * world_size + tile_m - 1) // tile_m
            all_orders = [
                build_tile_m_order(num_shard_tokens, world_size, tile_m, None)
            ]
            all_orders.extend(
                build_tile_m_order(num_shard_tokens, world_size, tile_m, rank)
                for rank in range(world_size)
            )
            for order in all_orders:
                self.assertEqual(sorted(order), list(range(num_tile_m)))

    def test_invalid_arguments(self):
        for args in ((0, 4, 128), (128, 0, 128), (128, 4, 0)):
            with self.assertRaises(ValueError):
                build_tile_m_order(*args, None)
        for rank in (-1, 4):
            with self.assertRaises(ValueError):
                build_tile_m_order(128, 4, 128, rank)


class DispatchTableTest(unittest.TestCase):
    def test_checked_in_table_marks_baseline_and_disabled_shapes(self):
        table = load_dispatch_table(4, 7168, "NVIDIA_GB300")
        pull = DispatchConfig("pull", KernelConfig(256, 128, 2), "common")
        baseline = DispatchConfig("baseline")
        self.assertEqual(table.k, 3072)
        # Baseline 512-2048 and the disabled historical 4K push stay native.
        self.assertEqual(
            table.configs,
            {
                512: baseline,
                1024: baseline,
                2048: baseline,
                4096: baseline,
                8192: pull,
                12288: pull,
                16384: pull,
            },
        )

    def test_gb300_tp8_table_mirrors_b300_tp8(self):
        self.assertEqual(
            load_dispatch_table(8, 7168, "NVIDIA_GB300"),
            load_dispatch_table(8, 7168, "NVIDIA_B300_SXM6_AC"),
        )

    def test_selection_takes_the_largest_bucket_not_exceeding_m(self):
        table = load_dispatch_table(8, 7168, "NVIDIA_B300_SXM6_AC")
        pull = table.configs[8192]
        for m, expected in (
            (8, None),
            (511, None),
            (4096, None),
            (8184, None),
            (8192, pull),
            (13280, table.configs[12288]),
            (16384, table.configs[16384]),
            (32768, table.configs[16384]),
        ):
            with self.subTest(m=m):
                self.assertEqual(table.select(m), expected)

    def test_native_bucket_bounds_the_fused_bucket_below_it(self):
        fused = DispatchConfig("pull", KernelConfig(256, 128, 2), "common")
        table = _MODULE.DispatchTable(
            1536, {2048: fused, 4096: DispatchConfig("baseline")}
        )
        self.assertEqual(table.select(3000), fused)
        self.assertIsNone(table.select(4096))
        self.assertIsNone(table.select(9000))
        # The disabled GB300 4K push keeps 4096-8191 native.
        gb300 = load_dispatch_table(4, 7168, "NVIDIA_GB300")
        self.assertIsNone(gb300.select(6000))

    def test_missing_table_selects_native(self):
        self.assertIsNone(load_dispatch_table(2, 7168, "NVIDIA_GB300"))
        self.assertIsNone(load_dispatch_table(4, 4096, "NVIDIA_GB300"))
        self.assertIsNone(load_dispatch_table(4, 7168, "NVIDIA_B300"))


PULL = DispatchConfig("pull", KernelConfig(256, 128, 2), "common")
PUSH = DispatchConfig("push", KernelConfig(128, 128, 2), "rotated")


class FakeTensor:
    """Track slice offsets and allocation identity without importing Torch."""

    def __init__(self, numel, pointer):
        self.shape = (numel,)
        self._pointer = pointer
        self.zero_ = Mock()

    def data_ptr(self):
        return self._pointer

    def element_size(self):
        return 2

    def __getitem__(self, key):
        begin, end, _ = key.indices(self.shape[0])
        return FakeTensor(end - begin, self._pointer + begin * 2)


class ContextTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        torch = types.ModuleType("torch")
        dist = types.ModuleType("torch.distributed")
        symm = types.ModuleType("torch.distributed._symmetric_memory")
        torch.distributed = dist
        dist._symmetric_memory = symm

        @contextlib.contextmanager
        def inference_mode(enabled):
            yield

        torch.inference_mode = inference_mode
        stream = types.SimpleNamespace(synchronize=lambda: self.events.append("sync"))
        torch.cuda = types.SimpleNamespace(
            device=lambda _: contextlib.nullcontext(),
            current_stream=lambda _: stream,
        )
        torch.bfloat16, torch.uint32, torch.int32 = "bf16", "uint32", "int32"
        torch.zeros = Mock(side_effect=lambda *args, **kwargs: object())
        torch.tensor = Mock(side_effect=lambda data, **kwargs: ("lut", tuple(data)))
        self.capacity = Mock(return_value=8)
        torch.ops = types.SimpleNamespace(
            mega_ops=types.SimpleNamespace(
                sm100_device_uuid=Mock(return_value="uuid-0"),
                sm100_check_peer_access=Mock(),
                sm100_resident_cluster_capacity=self.capacity,
                sm100_gemm_rs_prefix_add_pull=Mock(),
                sm100_gemm_rs_prefix_add_push=Mock(),
            )
        )
        self.allocations = []

        def allocate(numel, **kwargs):
            tensor = FakeTensor(numel, 0x1000000 * (len(self.allocations) + 1))
            self.allocations.append(tensor)
            return tensor

        self.multicast_ptr = 256
        symm.empty = Mock(side_effect=allocate)
        symm.rendezvous = Mock(
            side_effect=lambda *args: types.SimpleNamespace(
                multicast_ptr=self.multicast_ptr, buffer_ptrs_dev=512
            )
        )

        def all_gather_object(output, value, group):
            output[:] = [value, "uuid-1"]

        dist.all_gather_object = Mock(side_effect=all_gather_object)
        dist.get_world_size = Mock(return_value=2)
        dist.barrier = Mock(side_effect=lambda group: self.events.append("barrier"))
        self.torch, self.dist, self.symm = torch, dist, symm
        jit_utils = types.ModuleType("sglang.kernels.jit.utils")
        jit_utils.cache_once = lambda function: function
        jit_utils.empty_sentinel = Mock()
        jit_utils.load_jit = Mock()
        jit_utils.make_cpp_args = Mock()
        nvtx_utils = types.ModuleType("sglang.srt.utils.nvtx_utils")
        nvtx_utils.operations_nvtx_range = lambda _: contextlib.nullcontext()
        self.same_node = [True, True]
        self.one_nvlink_clique = True
        ar_utils = types.ModuleType(
            "sglang.srt.distributed.device_communicators.custom_all_reduce_utils"
        )
        ar_utils.is_one_nvlink_clique = Mock(
            side_effect=lambda group, device: self.one_nvlink_clique
        )
        parallel_state = types.ModuleType("sglang.srt.distributed.parallel_state")
        parallel_state.in_the_same_node_as = Mock(
            side_effect=lambda group, source_rank: self.same_node
        )
        self.ar_utils, self.parallel_state = ar_utils, parallel_state
        patcher = patch.dict(
            sys.modules,
            {
                "torch": torch,
                "torch.distributed": dist,
                "torch.distributed._symmetric_memory": symm,
                "sglang.kernels.jit.utils": jit_utils,
                "sglang.srt.utils.nvtx_utils": nvtx_utils,
                ar_utils.__name__: ar_utils,
                parallel_state.__name__: parallel_state,
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.inkling = _load_module(
            "sglang.kernels.ops.communication.inkling_all_reduce",
            _ROOT / "python/sglang/kernels/ops/communication/inkling_all_reduce.py",
        )
        _load_module(
            "sglang.kernels.ops.communication.sm100_peer_access",
            _ROOT / "python/sglang/kernels/ops/communication/sm100_peer_access.py",
        )
        self.module = _load_module(
            _PACKAGE.__name__ + ".context", _PACKAGE_PATH / "context.py"
        )
        self.group = types.SimpleNamespace(group_name="attn_tp")

    def context(self):
        return self.module.GemmRSPrefixAdd(
            self.group,
            0,
            2,
            1024,
            256,
            types.SimpleNamespace(index=0),
            [PULL.kernel, PUSH.kernel],
        )

    def test_context_reuses_existing_barrier_sizes_without_compiling(self):
        self.assertIs(self.module.flags_numel, self.inkling.flags_numel)
        self.assertEqual(self.module.STATE_SIZE, 264)
        for world_size in (2, 4, 8):
            self.assertEqual(self.module.flags_numel(world_size), world_size * 257)
        self.inkling.load_jit.assert_not_called()

    def test_constructor_allocates_one_shared_arena_and_publishes_flags(self):
        ctx = self.context()
        layout = ctx.layout
        self.assertEqual(
            [call.args for call in self.symm.empty.call_args_list],
            [
                (layout.arena_numel,),
                (layout.ready_numel * 2,),
                (self.inkling.flags_numel(2),),
            ],
        )
        workspace = ctx._symmetric_workspace
        self.assertIs(workspace, self.allocations[0])
        self.assertEqual(ctx._push_receive_buffer.data_ptr(), workspace.data_ptr())
        self.assertEqual(
            ctx._gemm_partial_output.data_ptr(),
            workspace.data_ptr() + layout.recv_data_numel * 2,
        )
        self.assertEqual(ctx._partial_multicast_ptr, 256 + layout.recv_data_numel * 2)
        # Push receive slots and both flag arrays start zeroed; partials do not.
        ctx._push_receive_buffer.zero_.assert_called_once_with()
        ctx._gemm_partial_output.zero_.assert_not_called()
        for flags in self.allocations[1:]:
            flags.zero_.assert_called_once_with()
        self.assertTrue(ctx.supports_pull)
        self.torch.ops.mega_ops.sm100_check_peer_access.assert_called_once_with(
            0, ["uuid-0", "uuid-1"]
        )
        self.assertIs(self.dist.all_gather_object.call_args.kwargs["group"], self.group)
        # Initialization is published before any peer can launch.
        self.assertEqual(self.events, ["sync", "barrier"])

    def test_single_host_group_skips_the_fabric_clique_query(self):
        self.context()
        self.parallel_state.in_the_same_node_as.assert_called_once_with(
            self.group, source_rank=0
        )
        self.ar_utils.is_one_nvlink_clique.assert_not_called()

    def test_multi_host_group_in_one_nvlink_clique_skips_the_native_check(self):
        self.same_node = [True, False]
        self.context()
        self.ar_utils.is_one_nvlink_clique.assert_called_once()
        self.torch.ops.mega_ops.sm100_check_peer_access.assert_not_called()

    def test_multi_host_group_outside_one_nvlink_clique_raises(self):
        self.same_node = [True, False]
        self.one_nvlink_clique = False
        with self.assertRaisesRegex(RuntimeError, "NVLink fabric clique"):
            self.context()
        self.torch.ops.mega_ops.sm100_check_peer_access.assert_not_called()

    def test_without_multicast_only_push_is_available(self):
        self.multicast_ptr = 0
        ctx = self.context()
        self.assertFalse(ctx.supports_pull)
        with self.assertRaisesRegex(RuntimeError, "NVLS"):
            ctx.prepare(256, 128, PULL)
        ctx.prepare(256, 128, PUSH)
        self.assertIn((256, 128, PUSH), ctx._prepared_kernel_grids)

    def test_prepare_builds_order_and_grid_once(self):
        ctx = self.context()
        self.capacity.return_value = 3
        ctx.prepare(1024, 128, PUSH)
        ctx.prepare(1024, 128, PUSH)
        self.capacity.assert_called_once_with(
            ctx._gemm_partial_output, 2, True, 128, 128
        )
        self.assertEqual(
            ctx._prepared_kernel_grids[1024, 128, PUSH],
            (("lut", build_tile_m_order(512, 2, 128, 0)), 3),
        )
        # The grid never exceeds the shape's cluster work.
        self.capacity.return_value = 1000
        ctx.prepare(1024, 128, PULL)
        self.assertEqual(
            ctx._prepared_kernel_grids[1024, 128, PULL],
            (
                ("lut", build_tile_m_order(512, 2, 256, None)),
                ctx.layout.cluster_work(1024, PULL.kernel),
            ),
        )

    def test_unfusable_or_undeclared_shape_raises(self):
        ctx = self.context()
        for m, k in ((2048, 128), (256, 127)):
            with self.assertRaisesRegex(ValueError, "cannot fuse"):
                ctx.prepare(m, k, PULL)
        with self.assertRaisesRegex(ValueError, "not declared"):
            ctx.prepare(
                256, 128, DispatchConfig("pull", KernelConfig(64, 64, 1), "common")
            )
        self.assertEqual(ctx._prepared_kernel_grids, {})

    def test_forward_launches_the_prepared_grid_for_each_mode(self):
        ctx = self.context()
        ctx.prepare(1024, 128, PULL)
        ctx.prepare(1024, 128, PUSH)
        input = types.SimpleNamespace(shape=(1024, 128))
        weight, prefix, output = object(), object(), object()
        for dispatch in (PULL, PUSH):
            with self.subTest(mode=dispatch.mode):
                kernel = getattr(
                    self.torch.ops.mega_ops,
                    "sm100_gemm_rs_prefix_add_" + dispatch.mode,
                )
                self.assertIs(ctx(input, weight, prefix, output, dispatch), output)
                args = kernel.call_args.args
                self.assertEqual(args[:4], (input, weight, prefix, output))
                self.assertIs(args[4], ctx._gemm_partial_output)
                tile_m_order, num_clusters = ctx._prepared_kernel_grids[
                    1024, 128, dispatch
                ]
                self.assertEqual(
                    args[-13:],
                    (
                        ctx._tile_ready_flags,
                        ctx._ready_peer_ptrs,
                        ctx._barrier_state,
                        ctx._barrier_flags,
                        ctx._barrier_peer_ptrs,
                        tile_m_order,
                        0,
                        2,
                        ctx.layout.tile_n_capacity,
                        num_clusters,
                        2,
                        dispatch.kernel.tile_m,
                        128,
                    ),
                )
                if dispatch.mode == "pull":
                    self.assertEqual(
                        args[5:7],
                        (ctx._partial_multicast_ptr, ctx._ready_multicast_ptr),
                    )
                else:
                    self.assertEqual(
                        args[5:8],
                        (
                            ctx._push_receive_buffer,
                            ctx._payload_peer_ptrs,
                            ctx.layout.plane_stride_elements,
                        ),
                    )

    def test_unprepared_shape_is_rejected(self):
        ctx = self.context()
        with self.assertRaises(KeyError):
            ctx(types.SimpleNamespace(shape=(1024, 128)), None, None, None, PULL)


if __name__ == "__main__":
    unittest.main()
