#!/usr/bin/env python3
"""Compare byte-flat, row-tiled, and fused page-index DCP gather.

Run on the target GPU with no serving workload for an isolated measurement.
Both paths reuse the same preuploaded metadata and DCP rank-major indices.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.kvcache import pd_dcp_gather


@triton.jit
def _baseline_byte_gather(metadata, indices, pack, n, BLOCK: tl.constexpr):
    # Frozen pre-rewrite implementation; keep this independent of production.
    layer = tl.program_id(0)
    src = tl.load(metadata + layer * 3).to(pack.dtype)
    width = tl.load(metadata + layer * 3 + 1)
    base = tl.load(metadata + layer * 3 + 2)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n * width
    rows = tl.load(indices + offsets // width, mask=mask, other=0)
    values = tl.load(src + rows * width + offsets % width, mask=mask)
    tl.store(pack + base + offsets, values, mask=mask)


def bench_case(args, items, pattern):
    width = args.row_bytes
    # Match the production DCP=8, physical page=64 window alignment.
    n = args.rows or (args.window_mb * 1024**2 // (items * width * 512)) * 512
    if n == 0:
        raise ValueError("Window cannot hold one virtual DCP page")
    page_count = triton.cdiv(n, 64)
    pages = torch.arange(page_count, device="cuda")
    if pattern == "fragmented":
        pages = torch.randperm(page_count, device="cuda")
    positions = torch.arange(n, device="cuda")
    logical = torch.cat([positions[r::8] for r in range(8)])
    indices = pages[logical // 64] * 64 + logical % 64
    sources = [
        torch.randint(
            0, 256, (page_count * 64, width), dtype=torch.uint8, device="cuda"
        )
        for _ in range(items)
    ]
    ptrs = [src.data_ptr() for src in sources]
    widths = [width] * items
    payload = n * items * width
    old_pack = torch.empty(payload, dtype=torch.uint8, device="cuda")
    new_pack = torch.empty_like(old_pack)
    fused_pack = torch.empty_like(old_pack)
    metadata = torch.tensor(
        [[ptr, width, i * n * width] for i, ptr in enumerate(ptrs)],
        dtype=torch.int64,
        device="cuda",
    )

    def old():
        return _baseline_byte_gather[(items, triton.cdiv(n * width, 1024))](
            metadata, indices, old_pack, n, BLOCK=1024, num_warps=4
        )

    def new():
        pd_dcp_gather.copy_mla_rows_into_pack(
            ptrs, indices, new_pack, widths, src_metadata=metadata
        )

    page_gather = pd_dcp_gather.PagedMLAGather(ptrs, widths, "cuda", 64)
    gpu_pages = pages.to(torch.int32)

    def fused():
        page_gather(gpu_pages, n, fused_pack)

    baseline_kernel = old()
    new()
    fused()
    for i, src in enumerate(sources):
        expected = src[indices].flatten()
        for packed in (old_pack, new_pack, fused_pack):
            torch.testing.assert_close(
                packed[i * n * width : (i + 1) * n * width],
                expected,
                rtol=0,
                atol=0,
            )
    torch.cuda.synchronize()
    result = dict(
        items=items, row_bytes=width, rows=n, pattern=pattern, payload_bytes=payload
    )
    result.update(token_index_bytes=n * 8, fused_page_index_bytes=page_count * 4)
    if args.include_prepare:
        result["cpu_preparation"] = measure_preparation(
            pages, n, args.prepare_iterations
        )
    if args.profile:
        # JIT and correctness checks are outside the capture range. These
        # captures explain execution; they are excluded from speedup results.
        torch.cuda.cudart().cudaProfilerStart()
        try:
            for _ in range(10):
                with torch.cuda.nvtx.range(
                    f"DCP_GATHER_BASELINE items={items} pattern={pattern}"
                ):
                    old()
                with torch.cuda.nvtx.range(
                    f"DCP_GATHER_ROW_TILED items={items} pattern={pattern}"
                ):
                    new()
                with torch.cuda.nvtx.range(
                    f"DCP_GATHER_PAGED_FUSED items={items} pattern={pattern}"
                ):
                    fused()
            torch.cuda.synchronize()
        finally:
            torch.cuda.cudart().cudaProfilerStop()
        result["profiled"] = True
    else:
        times = {"baseline": [], "row_tiled": [], "paged_fused": []}
        for repeat in range(args.repeats):
            cases = [("baseline", old), ("row_tiled", new), ("paged_fused", fused)]
            shift = repeat % len(cases)
            cases = cases[shift:] + cases[:shift]
            for name, fn in cases:
                ms = triton.testing.do_bench(
                    fn, warmup=args.warmup_ms, rep=args.rep_ms, return_mode="median"
                )
                times[name].append(ms)
        old_ms = statistics.median(times["baseline"])
        new_ms = statistics.median(times["row_tiled"])
        fused_ms = statistics.median(times["paged_fused"])
        result.update(
            profiled=False,
            samples_ms=times,
            baseline_ms=old_ms,
            row_tiled_ms=new_ms,
            speedup=old_ms / new_ms,
            paged_fused_ms=fused_ms,
            fused_speedup=old_ms / fused_ms,
            payload_GBps=payload / fused_ms / 1e6,
            logical_read_write_GBps=2 * payload / fused_ms / 1e6,
        )
    if args.dump_ptx:
        # Compile the same specialization used by the production wrapper.
        copy_words = width % 4 == 0
        max_cols = width // (4 if copy_words else 1)
        cols = min(256, triton.next_power_of_2(max_cols))
        compiled = pd_dcp_gather._copy_mla_rows_into_pack_kernel.warmup(
            metadata,
            indices,
            new_pack,
            n,
            ROW_BYTES=width,
            COPY_WORDS=copy_words,
            ALIGN_BYTES=16 if width % 16 == 0 else (4 if copy_words else 1),
            BLOCK_ROWS=16,
            BLOCK_COLS=cols,
            num_warps=4,
            grid=(items, triton.cdiv(n, 16), triton.cdiv(max_cols, cols)),
        )
        args.dump_ptx.mkdir(parents=True, exist_ok=True)
        fused_compiled = pd_dcp_gather._copy_mla_rows_into_pack_kernel.warmup(
            page_gather.metadata,
            gpu_pages,
            fused_pack,
            n,
            ROW_BYTES=width,
            COPY_WORDS=copy_words,
            ALIGN_BYTES=16 if width % 16 == 0 else (4 if copy_words else 1),
            BLOCK_ROWS=16,
            BLOCK_COLS=cols,
            num_warps=4,
            start_token=0,
            PAGE_SIZE=64,
            DCP_SIZE=8,
            grid=(
                items,
                triton.cdiv(triton.cdiv(n, 8), 16),
                8 * triton.cdiv(max_cols, cols),
            ),
        )
        for name, kernel in (
            ("baseline", baseline_kernel),
            ("row_tiled", compiled),
            ("paged_fused", fused_compiled),
        ):
            path = args.dump_ptx / f"{name}-{items}items-{n}rows-{pattern}.ptx"
            with path.open("x") as f:
                f.write(kernel.asm["ptx"])
    return result


def measure_preparation(pages, n, iterations):
    """CPU index/run construction only; not a scheduler or RDMA benchmark."""
    import numpy as np

    from sglang.srt.disaggregation.common.dcp_window_plan import (
        DCPDestinationPlan,
        dcp_rank_layout,
    )
    from sglang.srt.disaggregation.common.utils import build_dcp_token_transfer_plan

    source = pages.cpu().numpy().astype(np.int32)
    destinations = [np.arange(triton.cdiv(n, 512), dtype=np.int32) for _ in range(8)]
    plans = [DCPDestinationPlan(dst, 64) for dst in destinations]
    host_rows = np.empty(n, dtype=np.int64)
    host_pages = np.empty_like(source)

    def old_prepare():
        offset = 0
        result = []
        for rank, dst in enumerate(destinations):
            plan = build_dcp_token_transfer_plan(
                source,
                dst,
                physical_page_size=64,
                dcp_size=8,
                dcp_rank=rank,
                num_kv_tokens=n,
            )
            count = len(plan.src_token_indices)
            host_rows[offset : offset + count] = plan.src_token_indices
            offset += count
            rows = plan.dst_token_indices
            if not len(rows):
                result.append([])
                continue
            starts = np.r_[0, np.flatnonzero(np.diff(rows) != 1) + 1]
            ends = np.r_[starts[1:], len(rows)]
            result.append(
                [(int(a), int(rows[a]), int(b - a)) for a, b in zip(starts, ends)]
            )
        return result

    def fused_prepare():
        host_pages[:] = source
        return [
            plan.runs(first // 8, count)
            for plan, (first, count, _) in zip(plans, dcp_rank_layout(n, 0))
        ]

    assert old_prepare() == fused_prepare()
    samples = {"legacy_us": [], "paged_us": []}
    for iteration in range(iterations + 10):
        cases = [("legacy_us", old_prepare), ("paged_us", fused_prepare)]
        if iteration % 2:
            cases.reverse()
        for name, fn in cases:
            started = time.perf_counter_ns()
            fn()
            elapsed = (time.perf_counter_ns() - started) / 1000
            if iteration >= 10:
                samples[name].append(elapsed)
    return {name: statistics.median(values) for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--row-bytes", type=int, default=576)
    parser.add_argument("--rows", type=int, help="Override production full-window size")
    parser.add_argument("--window-mb", type=int, default=128)
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["dcp", "fragmented"],
        default=["dcp", "fragmented"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dump-ptx", type=Path)
    parser.add_argument(
        "--include-prepare",
        action="store_true",
        help="Also time CPU index/run construction (excludes one-time caching and GPU/RDMA)",
    )
    parser.add_argument("--prepare-iterations", type=int, default=100)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture one case, skip timing comparison",
    )
    args = parser.parse_args()
    if min(
        *args.items,
        args.row_bytes,
        args.window_mb,
        args.repeats,
        args.warmup_ms,
        args.rep_ms,
        args.prepare_iterations,
    ) <= 0 or (args.rows is not None and args.rows <= 0):
        parser.error("Sizes and timing parameters must be positive")
    if args.profile and (len(args.items) != 1 or len(args.patterns) != 1):
        parser.error("--profile requires exactly one --items and one --patterns value")
    if args.output and args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    torch.cuda.set_device(args.device)
    torch.manual_seed(42)
    info = dict(
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        torch=torch.__version__,
        triton=triton.__version__,
        kernel_file=pd_dcp_gather.__file__,
        note="Logical read+write bytes, not measured HBM traffic. Isolated gather, not PD TTFT.",
    )
    print(json.dumps(info, ensure_ascii=False), flush=True)
    results = []
    for items in args.items:
        for pattern in args.patterns:
            result = bench_case(args, items, pattern)
            results.append(result)
            print(json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(dict(environment=info, results=results), f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
