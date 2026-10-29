"""CPU tests for the external-library all-gather + GEMM wrapper.

Only the native op boundary is mocked. These tests do not claim CUDA Graph
execution, numeric kernel validation, or model smoke coverage.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.communication import ag_gemm
from sglang.kernels.ops.communication.ag_gemm import context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

KernelConfig = ag_gemm.KernelConfig
WorkspaceLayout = ag_gemm.WorkspaceLayout
DispatchConfig = ag_gemm.DispatchConfig


class WorkspaceLayoutTest(unittest.TestCase):
    def test_only_compiled_configurations_are_accepted(self):
        for tile_m, cluster_hidden in ((128, 1), (128, 2), (256, 2)):
            KernelConfig(tile_m, cluster_hidden)
        # A 256-token tile needs the 2-SM MMA; 64 was never compiled.
        for args in ((256, 1), (64, 1), (128, 4)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                KernelConfig(*args)

    def test_layout_rejects_shapes_the_kernel_cannot_address(self):
        for args in ((1001, 7168, 8), (1024, 7000, 8), (1024, 7168, 3)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                WorkspaceLayout(*args)
        with self.assertRaises(ValueError):
            WorkspaceLayout(1024, 7168, 8, configs=())

    def test_configs_are_deduplicated_and_sorted(self):
        layout = WorkspaceLayout(
            1024, 7168, 8, configs=(KernelConfig(256, 2), (128, 1), (256, 2))
        )
        self.assertEqual(layout.configs, (KernelConfig(128, 1), KernelConfig(256, 2)))
        with self.assertRaises(ValueError):
            layout.can_fuse(1024, 2112, KernelConfig(128, 2))

    def test_can_fuse_bounds(self):
        layout = WorkspaceLayout(16384, 7168, 8)
        self.assertTrue(layout.can_fuse(16384, 6144))
        self.assertFalse(layout.can_fuse(16392, 6144))
        self.assertFalse(layout.can_fuse(8192, 2116))
        with self.assertRaises(ValueError):
            layout.can_fuse(8196, 2112)

    def test_cluster_work_counts_every_source_shard(self):
        layout = WorkspaceLayout(16384, 7168, 8)
        config = KernelConfig(256, 2)
        # 1024-row shards -> 4 token tiles; 2112 -> 17 hidden tiles -> 9 clusters.
        self.assertEqual(layout.cluster_work(8192, 2112, config), 8 * 4 * 9)
        # A 1250-row shard rounds up to 5 tiles; 6144 -> 48 tiles -> 24 clusters.
        self.assertEqual(layout.cluster_work(10000, 6144, config), 8 * 5 * 24)


class DispatchTableTest(unittest.TestCase):
    PULL = DispatchConfig("pull", KernelConfig(256, 2))

    def test_floor_bucket_and_baseline_selection(self):
        table = ag_gemm.DispatchTable(
            7168,
            2112,
            {4096: DispatchConfig("baseline"), 8192: self.PULL},
        )
        self.assertIsNone(table.select(2048))
        self.assertIsNone(table.select(8188))
        self.assertEqual(table.select(8192), self.PULL)
        self.assertEqual(table.select(13000), self.PULL)

    def test_checked_in_tp8_tables(self):
        # GB300 TP8 mirrors the B300 TP8 dispatch.
        for device in ("NVIDIA_B300_SXM6_AC", "NVIDIA_GB300"):
            with self.subTest(device=device):
                kda = ag_gemm.load_dispatch_table(8, 7168, 6144, device)
                mla = ag_gemm.load_dispatch_table(8, 7168, 2112, device)
                self.assertEqual((kda.k, kda.n, mla.k, mla.n), (7168, 6144, 7168, 2112))
                self.assertIsNone(kda.select(4095))
                self.assertEqual(kda.select(4096), self.PULL)
                self.assertIsNone(mla.select(8191))
                self.assertEqual(mla.select(8192), self.PULL)
                self.assertIsNone(ag_gemm.load_dispatch_table(4, 7168, 6144, device))


def _operator(rank=3, world_size=8, max_m=1024, k=128):
    # Bypass the collective constructor; only workspace views are exercised.
    operator = object.__new__(ag_gemm.AgGemm)
    operator.rank, operator.world_size, operator.k = rank, world_size, k
    operator.layout = WorkspaceLayout(max_m, k, world_size)
    operator._symmetric_workspace = torch.arange(
        operator.layout.gathered_input_numel, dtype=torch.float32
    )
    return operator


class GatheredInputTest(unittest.TestCase):
    def test_live_shards_pack_into_one_contiguous_input(self):
        operator = _operator()
        for m in (64, 1000 - 1000 % 8, 1024):
            with self.subTest(m=m):
                shards = operator.input_shards(m)
                gathered = operator.gathered_input(m)
                self.assertEqual(tuple(shards.shape), (8, m // 8, 128))
                # The kernel strides source shards by the live, not max, size.
                self.assertEqual(shards.stride(0), m // 8 * 128)
                self.assertEqual(gathered.data_ptr(), shards.data_ptr())
                self.assertTrue(gathered.is_contiguous())
                local = operator.local_input_shard(m)
                rows = slice(3 * m // 8, 4 * m // 8)
                self.assertEqual(local.data_ptr(), gathered[rows].data_ptr())
                self.assertTrue(torch.equal(local, gathered[rows]))

    def test_forward_matches_the_native_schema(self):
        operator = _operator()
        operator._payload_peer_ptrs = tuple(range(8))
        names = (
            "_shard_ready_flags",
            "_barrier_state",
            "_barrier_flags",
            "_barrier_peer_ptrs",
        )
        for name in names:
            setattr(operator, name, Mock(name=name))
        operator._copy_handles = (11, 12, 13)
        dispatch = DispatchConfig("pull", KernelConfig(256, 2))
        operator._prepared_kernel_grids = {(512, 6144, dispatch): 77}
        pull = Mock()
        local_input = torch.empty(64, 128)
        weight = torch.empty(6144, 128)
        output = torch.empty(512, 6144)
        with patch.object(
            context.torch,
            "ops",
            SimpleNamespace(mega_ops=SimpleNamespace(sm100_ag_gemm_pull=pull)),
        ):
            self.assertIs(operator(local_input, weight, output, dispatch), output)
        args = pull.call_args.args
        # Order of sm100_ag_gemm_pull in MegaAttention-v2 torch_bindings.cpp.
        self.assertIs(args[0], local_input)
        self.assertIs(args[1], weight)
        self.assertIs(args[2], output)
        self.assertEqual(tuple(args[3].shape), (8, 64, 128))
        self.assertEqual(args[4], tuple(range(8)))
        self.assertEqual(args[5:9], tuple(getattr(operator, n) for n in names))
        self.assertEqual(args[9:], (3, 8, 256, 2, 77, 11, 12, 13))


if __name__ == "__main__":
    unittest.main()
