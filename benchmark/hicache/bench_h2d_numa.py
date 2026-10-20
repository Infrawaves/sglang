"""HiCache MLA load-back bandwidth from host memory placed on chosen NUMA nodes.

Decides whether one shared MLA HiCache host copy per GB300 tray (interleaved
across both Graces) loads as fast as one private copy per GPU. All GPUs load
at once, one process per GPU as in production, from
  private: one buffer per GPU on that GPU's local node (baseline),
  interleaved: one buffer shared by all GPUs, interleaved across CPU nodes,
  node0: one buffer shared by all GPUs, all on the first CPU node.
The shared cases run twice: "same" pages, where all ranks load the same host
indices in each rep (as TP ranks loading one request do), and "different"
pages, where each rank draws its own.

Buffers come from the production allocator (alloc_shm inside
numa_utils.interleave_memory, as the MLA leader; one node means bind). The parent allocates the shared buffers before
forking; each GPU process maps the inherited memfd, pins it itself with
cudaHostRegister and times only its own stream. Page placement is checked
with move_pages: every expected node must hold an even share of the pages.

Each row: the production MLA load kernel (transfer_hicache_one_layer_mla, as
MLATokenToKVPoolHost.load_to_device_per_layer on a page_first pool) gathering
--load-tokens random tokens, in whole --page-size pages, for every layer.
--row-bytes defaults to 656 B/token/layer, DeepSeek-style fp8 MLA (512 B fp8
latent + 16 B scales + 128 B bf16 RoPE); assumed for K3, override if needed.

Validity notes:
  - Device L2: host loads do not go through device L2, and each rep scatters
    into a random set of device rows.
  - Host caches: every rep draws new random pages from a buffer
    several times Grace's L3; only "same" rows share them across ranks.
  - Aggregate = sum of per-GPU medians; reps start together at a barrier.

Usage (on one tray; check GPU-to-node mapping with `nvidia-smi topo -m`):
  python benchmark/hicache/bench_h2d_numa.py --gpus 0,1,2,3 --gpu-numa 0,0,1,1
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import mmap
import os
import platform
import queue
import random
import statistics
from collections import Counter

import torch
import torch.multiprocessing as mp

from sglang.srt.mem_cache.storage.mmap.mmap_allocator import alloc_shm, map_shm
from sglang.srt.utils.numa_utils import interleave_memory

_SYS_MOVE_PAGES = {"x86_64": 279, "aarch64": 239}
_PLACEMENT_SAMPLES = 4096
# Arbitrary; far wider than sampling noise (about 1.5% at 4096 samples).
_PLACEMENT_TOLERANCE = 0.1


def _check_placement(tensor: torch.Tensor, nodes: tuple[int, ...]) -> dict:
    n_pages = tensor.numel() // mmap.PAGESIZE
    base = tensor.data_ptr()
    sampled = random.Random(0).sample(range(n_pages), _PLACEMENT_SAMPLES)
    pages = (ctypes.c_void_p * _PLACEMENT_SAMPLES)(
        *[base + i * mmap.PAGESIZE for i in sampled]
    )
    status = (ctypes.c_int * _PLACEMENT_SAMPLES)()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    rc = libc.syscall(
        ctypes.c_long(_SYS_MOVE_PAGES[platform.machine()]),
        ctypes.c_int(0),
        ctypes.c_ulong(_PLACEMENT_SAMPLES),
        pages,
        None,
        status,
        ctypes.c_int(0),
    )
    if rc != 0:
        raise OSError(ctypes.get_errno(), "move_pages placement query failed")
    placement = Counter(status)
    share = 1 / len(nodes)
    if set(placement) != set(nodes) or any(
        abs(placement[n] / _PLACEMENT_SAMPLES - share) > _PLACEMENT_TOLERANCE
        for n in nodes
    ):
        raise RuntimeError(
            f"Sampled pages per node {dict(placement)}, expected {nodes}."
        )
    return dict(placement)


def _worker(rank, args, gpu, local_node, shared, barrier, results):
    from sglang.kernels.ops.kvcache.hicache import (
        TMA_BLOCK_QUOTA,
        can_use_hicache_jit_kernel,
        transfer_hicache_one_layer_mla,
        use_hicache_tma_kernel,
    )
    from sglang.srt.mem_cache.pool_host.common import (
        _cuda_host_register,
        _cuda_host_unregister,
    )

    torch.cuda.set_device(gpu)
    stream = torch.cuda.Stream()
    rng = torch.Generator().manual_seed(rank)
    page, row = args.page_size, args.row_bytes
    if not can_use_hicache_jit_kernel(page_size=page, element_size=row):
        raise RuntimeError(f"No JIT HiCache kernel for {row} B rows on GPU{gpu}.")
    kernel = (
        "tma"
        if use_hicache_tma_kernel(
            element_size=row, block_quota=TMA_BLOCK_QUOTA, page_size=page
        )
        else "register"
    )
    load_pages = args.load_tokens // page
    dev_pages = 4 * load_pages
    dst = torch.empty(dev_pages * page, row, dtype=torch.uint8, device=gpu)
    tokens_in_page = torch.arange(page)

    def token_indices(n_pages, pages_total, gen):
        pages = torch.randperm(pages_total, generator=gen)[:n_pages]
        return (pages[:, None] * page + tokens_in_page).flatten().to(gpu)

    def timed(run, inputs):
        """Median GiB/s; each rep uses its own inputs, all ranks start together."""
        for x in inputs[: args.warmup]:
            run(*x)
        torch.cuda.synchronize()
        gib_s = []
        for x in inputs[args.warmup :]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            barrier.wait()
            with torch.cuda.stream(stream):
                start.record()
                n_bytes = run(*x)
                end.record()
            end.synchronize()
            gib_s.append(n_bytes / (start.elapsed_time(end) * 1e-3) / 1024**3)
        return statistics.median(gib_s)

    out = {"kernel": kernel}
    for case, fd in [("private", None), *shared.items()]:
        if fd is None:
            with interleave_memory((local_node,)):
                src, private_fd, mm = alloc_shm(
                    (args.host_tokens, args.layers, row), torch.uint8
                )
            os.close(private_fd)
            _check_placement(src, (local_node,))
        else:
            src = map_shm(fd, (args.host_tokens, args.layers, row), torch.uint8)
            mm = None
        _cuda_host_register(src)
        # Per-layer [tokens, row] views with layer stride, as data_refs on page_first.
        layers = src.transpose(0, 1)

        def gather(src_idx, dst_idx):
            for layer in range(args.layers):
                transfer_hicache_one_layer_mla(
                    cache_dst=dst,
                    indices_dst=dst_idx,
                    cache_src=layers[layer],
                    indices_src=src_idx,
                    element_dim=row,
                    page_size=page,
                )
            return src_idx.numel() * row * args.layers

        reps = args.warmup + args.reps
        for pages in ("different",) if fd is None else ("same", "different"):

            def host_gen(rep):
                # Same pages: seeded by rep, so every rank draws the same.
                return torch.Generator().manual_seed(rep) if pages == "same" else rng

            gather_inputs = [
                (
                    token_indices(load_pages, args.host_tokens // page, host_gen(r)),
                    token_indices(load_pages, dev_pages, rng),
                )
                for r in range(reps)
            ]
            out[case, pages] = timed(gather, gather_inputs)
        _cuda_host_unregister(src)
        # Unmap before the next case allocates.
        src = layers = mm = None
    results.put((rank, out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--gpu-numa",
        default=None,
        help="Local CPU NUMA node per GPU, e.g. 0,0,1,1. Default: split GPUs "
        "evenly over the CPU nodes in order.",
    )
    parser.add_argument("--size-gb", type=float, default=16.0, help="Per buffer.")
    parser.add_argument("--layers", type=int, default=61)
    parser.add_argument("--row-bytes", type=int, default=656)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--load-tokens", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    # NUMA nodes with CPUs; Grace exposes GPU HBM as CPU-less nodes.
    nodes = sorted(
        int(path.split("/node")[-1].split("/")[0])
        for path in glob.glob("/sys/devices/system/node/node*/cpulist")
        if open(path).read().strip()
    )
    if args.gpu_numa is None:
        gpu_numa = [nodes[i * len(nodes) // len(gpus)] for i in range(len(gpus))]
    else:
        gpu_numa = [int(n) for n in args.gpu_numa.split(",")]
    token_bytes = args.layers * args.row_bytes
    args.host_tokens = (
        int(args.size_gb * 1024**3) // token_bytes // args.page_size * args.page_size
    )
    print(
        f"CPU NUMA nodes={nodes} GPU->node={dict(zip(gpus, gpu_numa))} "
        f"buffer={args.host_tokens * token_bytes / 1024**3:.2f} GiB "
        f"layers={args.layers} row={args.row_bytes} B page={args.page_size} "
        f"load={args.load_tokens} tokens"
    )

    # Allocated before fork; the children inherit and map the memfds.
    shared = {}
    for case, case_nodes in (("interleaved", tuple(nodes)), ("node0", (nodes[0],))):
        with interleave_memory(case_nodes):
            tensor, fd, _ = alloc_shm(
                (args.host_tokens, args.layers, args.row_bytes), torch.uint8
            )
        print(
            f"  {case}: sampled pages per node={_check_placement(tensor, case_nodes)}"
        )
        shared[case] = fd

    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(len(gpus), timeout=600)
    results = ctx.Queue()
    procs = [
        ctx.Process(
            target=_worker,
            args=(rank, args, gpu, gpu_numa[rank], shared, barrier, results),
        )
        for rank, gpu in enumerate(gpus)
    ]
    for p in procs:
        p.start()
    per_rank = {}
    while len(per_rank) < len(procs) and all(p.exitcode in (None, 0) for p in procs):
        try:
            rank, out = results.get(timeout=5)
            per_rank[rank] = out
        except queue.Empty:
            pass
    for p in procs:
        p.join()
    if len(per_rank) < len(procs):
        raise RuntimeError("A GPU process failed; see its traceback above.")
    for fd in shared.values():
        os.close(fd)

    print(f"\nkernel={per_rank[0]['kernel']}; GiB/s per GPU, aggregate = sum")
    summary = {}
    rows = [("private", "different")] + [
        (case, pages)
        for case in ("interleaved", "node0")
        for pages in ("same", "different")
    ]
    for case, pages in rows:
        per_gpu = [per_rank[r][case, pages] for r in range(len(gpus))]
        summary[case, pages] = (sum(per_gpu), min(per_gpu))
        per = " ".join(f"GPU{g}={v:6.1f}" for g, v in zip(gpus, per_gpu))
        print(
            f"  {case:11s} {pages:9s} aggregate={sum(per_gpu):7.1f}  "
            f"{per}  min={min(per_gpu):6.1f}"
        )
    base_agg, base_min = summary["private", "different"]
    for pages in ("same", "different"):
        shm_agg, shm_min = summary["interleaved", pages]
        print(
            f"  gather interleaved ({pages} pages)/private: aggregate "
            f"{shm_agg / base_agg:.3f}, slowest GPU {shm_min / base_min:.3f}"
        )


if __name__ == "__main__":
    main()
