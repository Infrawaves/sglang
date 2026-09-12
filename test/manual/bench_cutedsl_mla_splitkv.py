"""Manual Blackwell MLA split-KV experiment; not a CI or model-accuracy test.

Run from an environment with this SGLang checkout and FlashInfer 0.6.17:
  python test/manual/bench_cutedsl_mla_splitkv.py --batch-size 128
  python test/manual/bench_cutedsl_mla_splitkv.py --max-seq-len 1048576

Every candidate sees identical Q, KV, lengths, and disjoint randomized pages.
Timing covers CUDA-graph replay of the full MLA call, including split reduction.
Repeated replays do not flush L2. Output parity with stock is an experimental
check, not independent reference accuracy or end-to-end serving validation.
"""

import argparse
import importlib.metadata
import json
import math
import random
import statistics
from pathlib import Path


def make_lengths(name, count, scale, q_len, seed):
    if name == "uniform64k":
        lengths = [65536] * count
    elif name == "mixedmean64k":
        long_count = max(1, count // 4)
        lengths = [8192] * (count - long_count)
        total_long = count * 65536 - sum(lengths)
        base, extra = divmod(total_long, long_count)
        lengths += [base + (i < extra) for i in range(long_count)]
    else:  # At B128: 112 x 8K + 16 x 256K, mean 39K.
        long_count = max(1, count // 8)
        lengths = [8192] * (count - long_count) + [262144] * long_count
    lengths = [max(q_len, round(n * scale)) for n in lengths]
    random.Random(seed).shuffle(lengths)
    return lengths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--active-batch-size",
        type=int,
        help="Pad remaining rows with one query-width of KV (SGLang Q1 sentinel is 1)",
    )
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp8", "bf16"], default="fp8")
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=["uniform64k", "mixedmean64k", "shortermeanlongtail"],
        default=["uniform64k", "mixedmean64k", "shortermeanlongtail"],
    )
    parser.add_argument(
        "--lengths-json",
        type=Path,
        help="JSON list of active KV lengths; overrides distributions and length scale",
    )
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        help="Static graph capacity, must cover all actual lengths",
    )
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-3,
        help="Experimental output parity absolute tolerance",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Experimental output parity relative tolerance",
    )
    parser.add_argument("--enable-pdl", action="store_true")
    args = parser.parse_args()
    if args.active_batch_size is None:
        args.active_batch_size = args.batch_size
    if not 0 < args.active_batch_size <= args.batch_size or not 0 < args.heads <= 128:
        parser.error("require 0 < active batch <= batch and 0 < heads <= 128")
    if (
        min(args.q_len, args.page_size, args.iterations, args.rounds) <= 0
        or args.warmup < 1
        or args.length_scale <= 0
    ):
        parser.error("shape, scale, and timing counts must be positive")
    if any(not 1 <= n <= 32 for n in args.splits):
        parser.error("splits must be in [1, 32]")
    if (
        args.atol < 0
        or args.rtol < 0
        or (args.max_seq_len is not None and args.max_seq_len <= 0)
    ):
        parser.error("tolerances must be nonnegative and max-seq-len positive")
    return args


def run_case(args, name, lengths, torch, stock, create_decode, plan_splits):
    batch, page, heads, q_len = args.batch_size, args.page_size, args.heads, args.q_len
    if len(lengths) != args.active_batch_size or any(
        type(n) is not int or n < q_len for n in lengths
    ):
        raise ValueError("lengths must contain active-batch-size integers >= q-len")
    # Non-DCP FlashInfer 0.6.17's reducer does not support K=0. SGLang ordinary
    # decode pads seq_lens with 1, so keep these rows as real minimal KV work.
    lengths = lengths + [q_len] * (batch - len(lengths))
    bound = args.max_seq_len or max(lengths)
    if bound < max(lengths):
        raise ValueError("max-seq-len is below an actual KV length")
    dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    counts = [(n + page - 1) // page for n in lengths]
    total_pages = sum(counts)
    # Allocate only real pages; rectangular page-table capacity remains graph-safe.
    cpu_gen = torch.Generator().manual_seed(args.seed)
    ids = torch.randperm(total_pages, generator=cpu_gen, dtype=torch.int32)
    # Match the K128 tile's page-table lookahead, including very short custom
    # contexts. Unused entries point to a valid page and are masked by seq_lens.
    page_alignment = max(1, 128 // page)
    table_cols = math.ceil(math.ceil(bound / page) / page_alignment) * page_alignment
    tables = torch.zeros((batch, table_cols), dtype=torch.int32)
    offset = 0
    for i, count in enumerate(counts):
        tables[i, :count] = ids[offset : offset + count]
        offset += count
    gpu_gen = torch.Generator(device="cuda").manual_seed(args.seed)
    query = torch.randn(
        (batch, q_len, heads, 576),
        device="cuda",
        dtype=torch.bfloat16,
        generator=gpu_gen,
    ).to(dtype)
    cache = torch.empty((total_pages, page, 576), device="cuda", dtype=dtype)
    for start in range(0, total_pages, 1024):
        chunk = cache[start : start + 1024]
        chunk.copy_(
            torch.randn(
                chunk.shape, device="cuda", dtype=torch.bfloat16, generator=gpu_gen
            ).to(dtype)
        )
    table_gpu, lens_gpu = tables.cuda(), torch.tensor(
        lengths, dtype=torch.int32, device="cuda"
    )
    call = dict(
        query=query,
        kv_cache=cache,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=table_gpu,
        seq_lens=lens_gpu,
        max_seq_len=bound,
        softmax_scale=576**-0.5,
        is_var_seq=True,
        enable_pdl=args.enable_pdl,
    )
    sms = torch.cuda.get_device_properties(query.device).multi_processor_count
    stock_split, stock_ws = stock._get_split_kv_and_workspace_size(
        batch, q_len, heads, 512, sms, bound
    )
    candidates = [("stock", stock.cute_dsl_mla_decode, stock_split, stock_ws)]
    for split in dict.fromkeys(args.splits):
        effective, size = plan_splits(batch, q_len, heads, 512, split, bound)
        candidates.append((str(split), create_decode(split), effective, size))
    runs = []
    for label, fn, effective, size in candidates:
        workspace = torch.empty(max(1, size), dtype=torch.int8, device="cuda")
        out = torch.empty(
            (batch, q_len, heads, 512), dtype=torch.bfloat16, device="cuda"
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(args.warmup):
                fn(**call, workspace_buffer=workspace, out=out)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(**call, workspace_buffer=workspace, out=out)
        runs.append(
            dict(
                label=label,
                graph=graph,
                out=out,
                workspace=workspace,
                effective=effective,
                size=size,
                times=[],
            )
        )

    def check(actual, expected):
        if not torch.isfinite(actual).all().item():
            raise AssertionError(
                "MLA returned non-finite output, including padded rows"
            )
        torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)

    runs[0]["graph"].replay()
    reference = runs[0]["out"].clone()
    for run in runs:
        run["graph"].replay()
        check(run["out"], reference)
        run["max_abs_error"] = (
            (run["out"].float() - reference.float()).abs().max().item()
        )
    # Exercise live metadata at the captured addresses, then restore it.
    changed_tables = tables.clone()
    for i, count in enumerate(counts):
        changed_tables[i, :count] = tables[i, :count].flip(0)
    changed_lengths = [max(q_len, n // 2) if n else 0 for n in lengths]
    table_gpu.copy_(changed_tables)
    lens_gpu.copy_(torch.tensor(changed_lengths, dtype=torch.int32))
    eager_ref = torch.empty_like(reference)
    stock.cute_dsl_mla_decode(
        **call, workspace_buffer=runs[0]["workspace"], out=eager_ref
    )
    for run in runs:
        run["graph"].replay()
        check(run["out"], eager_ref)
    table_gpu.copy_(tables)
    lens_gpu.copy_(torch.tensor(lengths, dtype=torch.int32))
    for run in runs:
        run["graph"].replay()
        check(run["out"], reference)
    torch.cuda.synchronize()
    rng = random.Random(args.seed)
    for _ in range(args.rounds):
        order = list(runs)
        rng.shuffle(order)
        for run in order:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            begin.record()
            for _ in range(args.iterations):
                run["graph"].replay()
            end.record()
            end.synchronize()
            run["times"].append(begin.elapsed_time(end) / args.iterations)
    baseline = statistics.median(runs[0]["times"])
    active = lengths[: args.active_batch_size]
    for run in runs:
        ms = statistics.median(run["times"])
        print(
            json.dumps(
                dict(
                    distribution=name,
                    batch=batch,
                    active_batch=len(active),
                    heads=heads,
                    q_len=q_len,
                    dtype=args.dtype,
                    page_size=page,
                    min=min(active),
                    mean=statistics.mean(active),
                    max=max(active),
                    sum=sum(active),
                    graph_bound=bound,
                    requested_split=run["label"],
                    effective_split=run["effective"],
                    workspace_bytes=run["size"],
                    kv_bytes=cache.numel() * cache.element_size(),
                    median_ms=ms,
                    speedup_vs_stock=baseline / ms,
                    round_ms=run["times"],
                    max_abs_error=run["max_abs_error"],
                    metadata_replay_checked=True,
                    atol=args.atol,
                    rtol=args.rtol,
                    enable_pdl=args.enable_pdl,
                    is_var_seq=True,
                )
            ),
            flush=True,
        )


def main():
    args = parse_args()
    import torch
    from flashinfer.cute_dsl.attention.monolithic import mla_decode as stock

    from sglang.srt.layers.attention.cutedsl_mla_splitkv import (
        create_cutedsl_mla_decode_with_splits,
        plan_cutedsl_mla_splits,
    )

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError(
            "This experiment requires Blackwell SM100/SM103 CUDA hardware"
        )
    version = importlib.metadata.version("flashinfer-python")
    if version != "0.6.17":
        raise RuntimeError(f"This experiment pins FlashInfer 0.6.17; found {version}")
    print(
        json.dumps(
            dict(
                device=torch.cuda.get_device_name(),
                capability=torch.cuda.get_device_capability(),
                torch=torch.__version__,
                flashinfer=version,
                timing="CUDA graph MLA plus reducer; repeated data, no L2 flush",
                accuracy="experimental stock parity, not independent reference",
            )
        ),
        flush=True,
    )
    cases = (
        [("custom", json.loads(args.lengths_json.read_text()))]
        if args.lengths_json
        else [
            (
                name,
                make_lengths(
                    name,
                    args.active_batch_size,
                    args.length_scale,
                    args.q_len,
                    args.seed,
                ),
            )
            for name in args.distributions
        ]
    )
    with torch.inference_mode():
        for name, lengths in cases:
            run_case(
                args,
                name,
                lengths,
                torch,
                stock,
                create_cutedsl_mla_decode_with_splits,
                plan_cutedsl_mla_splits,
            )


if __name__ == "__main__":
    main()
