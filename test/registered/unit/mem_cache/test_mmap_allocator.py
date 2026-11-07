from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

import sys

sys.modules["libtpu"] = None
import ctypes
import ctypes.util
import mmap
import os
import signal
import tempfile
import unittest
import unittest.mock

import torch

from sglang.srt.mem_cache.pool_host.common import ShmHostTensorAllocator
from sglang.srt.mem_cache.storage.mmap import alloc_mmap, alloc_shm
from sglang.srt.mem_cache.storage.mmap.mmap_allocator import (
    _mmap_prefaulted,
    _mmap_private_parallel_prefaulted,
    _prefault_shm_shard,
    create_shm,
    map_shm,
)
from sglang.test.test_utils import CustomTestCase


class TestMmapAllocator(unittest.TestCase):
    def _assert_resident(self, mm, alloc_bytes):
        # mincore() reports per-page residency, so the invariant is checked
        # rather than inferred from the flags that were passed.
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        npages = alloc_bytes // mmap.PAGESIZE
        vec = (ctypes.c_ubyte * npages)()
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(alloc_bytes), vec) != 0:
            self.skipTest("mincore unavailable")
        self.assertTrue(all(v & 1 for v in vec), "mapping was not fully pre-faulted")

    def test_mmap_prefaulted_leaves_no_lazy_page(self):
        """Both populate paths must return a mapping with every page resident.

        cudaHostRegister pins these buffers, so a page still unfaulted at
        registration lets the device read memory that is not backed yet.
        """
        alloc_bytes = 64 * mmap.PAGESIZE
        flags = mmap.MAP_SHARED | mmap.MAP_ANONYMOUS

        with self.subTest(path="madvise"):
            mm = _mmap_prefaulted(-1, alloc_bytes, flags)
            try:
                self._assert_resident(mm, alloc_bytes)
            finally:
                mm.close()

        # MAP_POPULATE is unreachable on a 5.14+ kernel, so CI never runs it;
        # force the branch or it ships untested.
        with (
            self.subTest(path="map_populate"),
            unittest.mock.patch(
                "sglang.srt.mem_cache.storage.mmap.mmap_allocator._has_madv_populate_write",
                return_value=False,
            ),
        ):
            mm = _mmap_prefaulted(-1, alloc_bytes, flags)
            try:
                self._assert_resident(mm, alloc_bytes)
            finally:
                mm.close()

    def test_private_parallel_prefaulted_is_resident_and_zeroed(self):
        """Fully resident and zeroed, including a ragged non-THP-aligned tail."""
        alloc_bytes = 9 * 2 * 1024 * 1024 + 3 * mmap.PAGESIZE
        for threads in (1, 4, 64):
            with self.subTest(threads=threads):
                mm = _mmap_private_parallel_prefaulted(alloc_bytes, threads)
                try:
                    self.assertEqual(len(mm), alloc_bytes)
                    self._assert_resident(mm, alloc_bytes)
                    self.assertEqual(mm[:].count(0), alloc_bytes)
                finally:
                    mm.close()

    def test_private_parallel_mapping_is_not_inherited_by_fork(self):
        """MADV_DONTFORK: a forked child touching the buffer must SIGSEGV."""
        alloc_bytes = 4 * mmap.PAGESIZE
        mm = _mmap_private_parallel_prefaulted(alloc_bytes, 2)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        try:
            pid = os.fork()
            if pid == 0:  # child: the mapping must not exist here
                ctypes.c_char.from_address(addr).value  # noqa: B018
                os._exit(0)
            _, status = os.waitpid(pid, 0)
            self.assertTrue(os.WIFSIGNALED(status), f"child exited with {status}")
            self.assertEqual(os.WTERMSIG(status), signal.SIGSEGV)
        finally:
            mm.close()

    def test_alloc_mmap_parallel_flag(self):
        from sglang.srt.environ import envs

        dims = (3, 1024 * 1024)
        with envs.SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS.override(4):
            with unittest.mock.patch(
                "sglang.srt.mem_cache.storage.mmap.mmap_allocator._mmap_prefaulted",
                side_effect=AssertionError("shared path must not run"),
            ):
                tensor = alloc_mmap(dims, torch.uint8)
            self.assertEqual(tensor.shape, dims)
            self.assertTrue(bool(tensor.eq(0).all()))
            tensor[2, -1] = 7
            self.assertEqual(int(tensor[2, -1]), 7)

            # A failure on the new path falls back to the shared mapping.
            with unittest.mock.patch(
                "sglang.srt.mem_cache.storage.mmap.mmap_allocator._mmap_private_parallel_prefaulted",
                side_effect=OSError("boom"),
            ):
                tensor = alloc_mmap(dims, torch.uint8)
            self.assertEqual(tensor.shape, dims)

    def test_alloc_mmap_parallel_flag_off_by_default(self):
        """Default off: the shared prefaulted mapping is used unchanged."""
        with unittest.mock.patch(
            "sglang.srt.mem_cache.storage.mmap.mmap_allocator._mmap_private_parallel_prefaulted",
            side_effect=AssertionError("parallel path must not run by default"),
        ):
            tensor = alloc_mmap((2, 1024), torch.uint8)
        self.assertEqual(tensor.shape, (2, 1024))

    def test_alloc_shm_hugepage_warning(self):
        from sglang.srt.environ import envs

        envs.SGLANG_HUGEPAGE_SIZE.override("2MB")
        try:
            # Should succeed by falling back to plain page size mapping
            dims = (2, 2)
            tensor, fd, mm = alloc_shm(dims, torch.float32)
            self.assertEqual(tensor.shape, dims)
            mm.close()
            os.close(fd)
        finally:
            envs.SGLANG_HUGEPAGE_SIZE.override(None)

    def test_shm_host_tensor_allocator_invalid_device(self):
        allocator = ShmHostTensorAllocator()
        with self.assertRaises(AssertionError) as ctx:
            allocator.allocate((2, 2), torch.float32, device="cuda")
        self.assertIn("only supports CPU allocations", str(ctx.exception))


class TestShardedShmPopulation(CustomTestCase):
    def test_failed_shared_mapping_is_closed(self):
        """Failure during prefault or tensor creation must release the open mapping."""
        module = "sglang.srt.mem_cache.storage.mmap.mmap_allocator"
        for stage in ("prefault", "tensor"):
            with self.subTest(stage=stage), tempfile.TemporaryFile() as file:
                file.truncate(mmap.PAGESIZE)
                mappings = []

                def populate(mm, *, index, count):
                    mappings.append(mm)
                    if stage == "prefault":
                        raise OSError("prefault allocation failed")

                try:
                    with (
                        unittest.mock.patch(
                            f"{module}._prefault_shm_shard", side_effect=populate
                        ),
                        unittest.mock.patch(
                            "torch.frombuffer",
                            side_effect=MemoryError("tensor allocation failed"),
                        ),
                    ):
                        error = OSError if stage == "prefault" else MemoryError
                        with self.assertRaisesRegex(
                            error, f"{stage} allocation failed"
                        ):
                            map_shm(
                                file.fileno(),
                                (mmap.PAGESIZE,),
                                torch.uint8,
                                populate_shard=(0, 1),
                            )
                    self.assertTrue(mappings[0].closed)
                finally:
                    for mm in mappings:
                        mm.close()

    @unittest.skipUnless(sys.platform == "linux", "memfd residency needs Linux")
    def test_create_does_not_claim_pages_before_workers_touch_them(self):
        """The creator must leave placement to the workers' first touches."""
        fd = create_shm((16 * mmap.PAGESIZE,), torch.uint8)
        try:
            self.assertEqual(os.fstat(fd).st_size, 16 * mmap.PAGESIZE)
            self.assertEqual(os.fstat(fd).st_blocks, 0)
        finally:
            os.close(fd)

    @unittest.skipUnless(sys.platform == "linux", "Shared prefault uses Linux madvise")
    def test_shards_cover_uneven_and_smaller_than_world_mappings(self):
        """Shards must not overlap or omit pages, including empty shares."""
        for pages, workers in ((13, 4), (2, 4)):
            with self.subTest(pages=pages, workers=workers):
                fd = create_shm((pages * mmap.PAGESIZE,), torch.uint8)
                mm = mmap.mmap(fd, pages * mmap.PAGESIZE, flags=mmap.MAP_SHARED)
                os.close(fd)
                try:
                    # Keep mincore residency page-granular;
                    # a THP fault would mark a whole huge page.
                    mm.madvise(mmap.MADV_NOHUGEPAGE)
                    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
                    addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                    previous = [0] * pages
                    sizes = []
                    for rank in range(workers):
                        _prefault_shm_shard(mm, index=rank, count=workers)
                        resident = (ctypes.c_ubyte * pages)()
                        self.assertEqual(
                            libc.mincore(
                                ctypes.c_void_p(addr),
                                ctypes.c_size_t(len(mm)),
                                resident,
                            ),
                            0,
                        )
                        current = [value & 1 for value in resident]
                        new_pages = [
                            i for i in range(pages) if current[i] and not previous[i]
                        ]
                        if new_pages:
                            self.assertEqual(
                                new_pages,
                                list(range(new_pages[0], new_pages[-1] + 1)),
                            )
                        sizes.append(len(new_pages))
                        previous = current
                    self.assertEqual(previous, [1] * pages)
                    self.assertEqual(sum(sizes), pages)
                    self.assertLessEqual(max(sizes) - min(sizes), 1)
                finally:
                    mm.close()

    def test_shard_population_failure_propagates(self):
        """A failed shard madvise must raise instead of leaving pages lazily faulted."""
        mm = unittest.mock.create_autospec(mmap.mmap, instance=True)
        mm.__len__.return_value = 4 * mmap.PAGESIZE
        mm.madvise.side_effect = OSError("unsupported population")
        with self.assertRaisesRegex(OSError, "unsupported population"):
            _prefault_shm_shard(mm, index=0, count=2)


if __name__ == "__main__":
    unittest.main()
