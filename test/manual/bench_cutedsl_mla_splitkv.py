"""Manual Blackwell MLA split-KV experiment; not a CI or model-accuracy test.

Run from an environment with this SGLang checkout and FlashInfer 0.6.17/0.6.18:
  python test/manual/bench_cutedsl_mla_splitkv.py --batch-size 128
  python test/manual/bench_cutedsl_mla_splitkv.py --batch-sizes 1 8 32 128 --check-only

Every candidate sees identical Q, KV, lengths, and disjoint randomized pages.
Timing covers the full MLA call including split reduction, using CUDA-graph
replay by default (or eager launches with --execution-mode eager).
Repeated replays do not flush L2. Output parity with stock is an experimental
check, not independent reference accuracy or end-to-end serving validation.
"""

import argparse
import functools
import importlib.metadata
import json
import math
import random
import statistics
from contextlib import nullcontext
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument("--batch-size", type=int, default=128)
    batch_group.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument(
        "--active-batch-size",
        type=int,
        help="Pad remaining rows with one query-width of KV (SGLang Q1 sentinel is 1)",
    )
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp8", "bf16"], default="fp8")
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 4, 8, 16, 32])
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
    parser.add_argument("--execution-mode", choices=["graph", "eager"], default="graph")
    parser.add_argument("--check-only", action="store_true", help="Skip timing")
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop at the first failed comparison"
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Also write flushed JSONL records to this file",
    )
    args = parser.parse_args(argv)
    args.batch_sizes = list(dict.fromkeys(args.batch_sizes or [args.batch_size]))
    if (
        min(args.batch_sizes) <= 0
        or not 0 < args.heads <= 128
        or (
            args.active_batch_size is not None
            and not 0 < args.active_batch_size <= min(args.batch_sizes)
        )
    ):
        parser.error("require 0 < active batch <= batch and 0 < heads <= 128")
    if (
        min(args.q_len, args.page_size, args.iterations, args.rounds) <= 0
        or args.warmup < 1
        or not math.isfinite(args.length_scale)
        or args.length_scale <= 0
    ):
        parser.error("shape, scale, and timing counts must be positive")
    if any(not 1 <= n <= 32 for n in args.splits):
        parser.error("splits must be in [1, 32]")
    if (
        not math.isfinite(args.atol)
        or not math.isfinite(args.rtol)
        or args.atol < 0
        or args.rtol < 0
        or (args.max_seq_len is not None and args.max_seq_len <= 0)
    ):
        parser.error("tolerances must be nonnegative and max-seq-len positive")
    return args


def build_cases(args):
    """Resolve each graph batch and its actual lengths before allocating on GPU."""
    custom = json.loads(args.lengths_json.read_text()) if args.lengths_json else None
    cases = []
    for batch in args.batch_sizes:
        case_args = argparse.Namespace(**vars(args))
        case_args.batch_size = batch
        case_args.active_batch_size = args.active_batch_size or batch
        distributions = (
            [("custom", custom)]
            if args.lengths_json
            else [
                (
                    name,
                    make_lengths(
                        name,
                        case_args.active_batch_size,
                        args.length_scale,
                        args.q_len,
                        args.seed,
                    ),
                )
                for name in args.distributions
            ]
        )
        for name, lengths in distributions:
            if (
                not isinstance(lengths, list)
                or len(lengths) != case_args.active_batch_size
                or any(type(n) is not int or n < args.q_len for n in lengths)
            ):
                raise ValueError(
                    f"B{batch} {name}: lengths must contain "
                    f"{case_args.active_batch_size} integers >= q-len"
                )
            if args.max_seq_len is not None and max(lengths) > args.max_seq_len:
                raise ValueError(
                    f"B{batch} {name}: max-seq-len is below an actual KV length"
                )
            cases.append((case_args, name, lengths))
    return cases


def compare_outputs(actual, expected, atol, rtol):
    """Stock parity statistics; non-finite values always fail, even if equal.

    Absolute/relative statistics describe finite pairs only. Relative error
    excludes exact-zero references, which have a separate mismatch count.
    None denotes an undefined or non-finite scalar so every record is valid JSON.
    """
    import torch

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Comparison requires identical output shapes and dtypes")
    actual, expected = actual.float(), expected.float()
    finite_actual, finite_expected = torch.isfinite(actual), torch.isfinite(expected)
    finite = finite_actual & finite_expected
    error = (actual - expected).abs()
    mismatch = ~finite | (error > atol + rtol * expected.abs())
    count = int(mismatch.sum().item())

    def number(value):
        value = float(value.item())
        return value if math.isfinite(value) else None

    stats = dict(
        passed=count == 0,
        numel=actual.numel(),
        mismatch_count=count,
        mismatch_fraction=count / actual.numel(),
        nonfinite_actual=int((~finite_actual).sum().item()),
        nonfinite_reference=int((~finite_expected).sum().item()),
        finite_pairs=int(finite.sum().item()),
        max_abs_error=None,
        mean_abs_error=None,
        rmse=None,
        max_rel_error=None,
        worst_abs_index=None,
        worst_abs_actual=None,
        worst_abs_reference=None,
        zero_ref_mismatch_count=int((mismatch & finite & (expected == 0)).sum().item()),
    )
    if stats["finite_pairs"]:
        finite_errors = error[finite]
        flat_index = int(error.masked_fill(~finite, -1).reshape(-1).argmax().item())
        remainder, index = flat_index, []
        for size in reversed(actual.shape):
            index.append(remainder % size)
            remainder //= size
        stats.update(
            max_abs_error=number(finite_errors.max()),
            mean_abs_error=number(finite_errors.double().mean()),
            rmse=number(finite_errors.double().square().mean().sqrt()),
            worst_abs_index=list(reversed(index)),
            worst_abs_actual=number(actual.reshape(-1)[flat_index]),
            worst_abs_reference=number(expected.reshape(-1)[flat_index]),
        )
        relative_mask = finite & (expected != 0)
        if relative_mask.any().item():
            stats["max_rel_error"] = number(
                (error[relative_mask] / expected[relative_mask].abs()).max()
            )
    return stats


def result_status(checks, reference_valid):
    if not reference_valid:
        return "invalid_reference"
    return "pass" if all(check["passed"] for check in checks.values()) else "fail"


def run_case(args, name, lengths, torch, stock, create_decode, plan_splits, emit):
    batch, page, heads, q_len = args.batch_size, args.page_size, args.heads, args.q_len
    if len(lengths) != args.active_batch_size or any(
        type(n) is not int or n < q_len for n in lengths
    ):
        raise ValueError("lengths must contain active-batch-size integers >= q-len")
    # The supported non-DCP reducers do not support K=0. SGLang ordinary
    # decode pads seq_lens with 1, so keep these rows as real minimal KV work.
    lengths = lengths + [q_len] * (batch - len(lengths))
    bound = args.max_seq_len or max(lengths)
    if bound < max(lengths):
        raise ValueError("max-seq-len is below an actual KV length")
    dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    counts = [(n + page - 1) // page for n in lengths]
    total_pages = sum(counts)
    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    stock_split, stock_ws = stock._get_split_kv_and_workspace_size(
        batch, q_len, heads, 512, sms, bound
    )
    active = lengths[: args.active_batch_size]
    context = dict(
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
        reference_split=stock_split,
        num_sms=sms,
        kv_bytes=total_pages * page * 576 * (1 if args.dtype == "fp8" else 2),
        atol=args.atol,
        rtol=args.rtol,
        enable_pdl=args.enable_pdl,
        execution_mode=args.execution_mode,
        seed=args.seed,
        is_var_seq=True,
    )
    emit(dict(event="case_start", **context))
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
    table_gpu, lens_gpu = (
        tables.cuda(),
        torch.tensor(lengths, dtype=torch.int32, device="cuda"),
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
    candidates = [("stock", stock.cute_dsl_mla_decode, stock_split, stock_ws)]
    for split in dict.fromkeys(args.splits):
        effective, size = plan_splits(batch, q_len, heads, 512, split, bound)
        candidates.append((str(split), create_decode(split), effective, size))
    runs = []
    for label, fn, effective, size in candidates:
        emit(
            dict(
                event="candidate_start",
                **context,
                requested_split=label,
                effective_split=effective,
                workspace_bytes=size,
            )
        )
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
        eager = functools.partial(fn, **call, workspace_buffer=workspace, out=out)
        if args.execution_mode == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                eager()
            launch = graph.replay
        else:
            launch = eager
        runs.append(
            dict(
                label=label,
                launch=launch,
                eager=eager,
                out=out,
                workspace=workspace,
                effective=effective,
                size=size,
                times=[],
                checks={},
            )
        )

    def check(run, expected, phase, reference_mode):
        stats = compare_outputs(run["out"], expected, args.atol, args.rtol)
        run["checks"][phase] = stats
        checked_lengths = changed_lengths if phase == "metadata_changed" else lengths
        emit(
            dict(
                event="check",
                **context,
                requested_split=run["label"],
                effective_split=run["effective"],
                phase=phase,
                checked_min=min(checked_lengths),
                checked_max=max(checked_lengths),
                reference_mode=reference_mode,
                **stats,
            )
        )
        if args.fail_fast and not stats["passed"]:
            raise SystemExit(1)

    def run_and_check(run, expected, phase, reference_mode):
        emit(
            dict(
                event="check_start",
                **context,
                requested_split=run["label"],
                effective_split=run["effective"],
                phase=phase,
            )
        )
        run["launch"]()
        check(run, expected, phase, reference_mode)

    emit(dict(event="reference_start", **context))
    runs[0]["launch"]()
    reference = runs[0]["out"].clone()
    for run in runs:
        phase = "stock_repeat" if run["label"] == "stock" else "initial"
        run_and_check(run, reference, phase, args.execution_mode)
    # A graph result must also agree with an eager call of the stock planner.
    # This does not establish an independent numerical ground truth.
    eager_ref = torch.empty_like(reference)
    stock.cute_dsl_mla_decode(
        **call, workspace_buffer=runs[0]["workspace"], out=eager_ref
    )
    run_and_check(
        runs[0],
        eager_ref,
        "eager_vs_graph" if args.execution_mode == "graph" else "eager_repeat",
        "eager",
    )
    # Exercise live metadata at the captured addresses, then restore it.
    changed_tables = tables.clone()
    for i, count in enumerate(counts):
        changed_tables[i, :count] = tables[i, :count].flip(0)
    changed_lengths = [max(q_len, n // 2) if n else 0 for n in lengths]
    table_gpu.copy_(changed_tables)
    lens_gpu.copy_(torch.tensor(changed_lengths, dtype=torch.int32))
    stock.cute_dsl_mla_decode(
        **call, workspace_buffer=runs[0]["workspace"], out=eager_ref
    )
    for run in runs:
        run_and_check(run, eager_ref, "metadata_changed", "eager")
    table_gpu.copy_(tables)
    lens_gpu.copy_(torch.tensor(lengths, dtype=torch.int32))
    for run in runs:
        run_and_check(run, reference, "metadata_restored", args.execution_mode)
    torch.cuda.synchronize()
    reference_valid = all(c["passed"] for c in runs[0]["checks"].values())
    for run in runs:
        run["status"] = result_status(run["checks"], reference_valid)
    timed = [] if args.check_only else [r for r in runs if r["status"] == "pass"]
    rng = random.Random(args.seed)
    for _ in range(args.rounds):
        order = list(timed)
        rng.shuffle(order)
        for run in order:
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record()
            for _ in range(args.iterations):
                run["launch"]()
            end.record()
            end.synchronize()
            run["times"].append(begin.elapsed_time(end) / args.iterations)
    baseline = statistics.median(runs[0]["times"]) if runs[0]["times"] else None
    results = []
    for run in runs:
        ms = statistics.median(run["times"]) if run["times"] else None
        initial = run["checks"][
            "stock_repeat" if run["label"] == "stock" else "initial"
        ]
        result = dict(
            event="result",
            **context,
            requested_split=run["label"],
            effective_split=run["effective"],
            workspace_bytes=run["size"],
            status=run["status"],
            passed=run["status"] == "pass",
            reference_valid=reference_valid,
            median_ms=ms,
            speedup_vs_stock=baseline / ms if baseline is not None and ms else None,
            round_ms=run["times"],
            max_abs_error=initial["max_abs_error"],
            mismatch_count=initial["mismatch_count"],
            mismatch_fraction=initial["mismatch_fraction"],
            failed_phases=[p for p, c in run["checks"].items() if not c["passed"]],
            phase_checks=run["checks"],
            metadata_checked=True,
            metadata_replay_checked=args.execution_mode == "graph",
        )
        emit(result)
        results.append(result)
    return results


def main():
    args = parse_args()
    cases = build_cases(args)
    import torch
    from flashinfer.cute_dsl.attention.monolithic import mla_decode as stock

    from sglang.srt.layers.attention.cutedsl_mla_splitkv import (
        SUPPORTED_FLASHINFER_VERSIONS,
        create_cutedsl_mla_decode_with_splits,
        plan_cutedsl_mla_splits,
    )

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError(
            "This experiment requires Blackwell SM100/SM103 CUDA hardware"
        )
    version = importlib.metadata.version("flashinfer-python")
    if version not in SUPPORTED_FLASHINFER_VERSIONS:
        raise RuntimeError(
            f"This experiment requires FlashInfer in {SUPPORTED_FLASHINFER_VERSIONS}; "
            f"found {version}"
        )
    output = args.output_jsonl.open("w") if args.output_jsonl else nullcontext(None)
    with output as report:

        def emit(record):
            line = json.dumps(record, allow_nan=False)
            print(line, flush=True)
            if report is not None:
                print(line, file=report, flush=True)

        emit(
            dict(
                event="environment",
                device=torch.cuda.get_device_name(),
                capability=torch.cuda.get_device_capability(),
                torch=torch.__version__,
                flashinfer=version,
                flashinfer_path=stock.__file__,
                benchmark_path=str(Path(__file__).resolve()),
                split_adapter_path=create_cutedsl_mla_decode_with_splits.__code__.co_filename,
                batch_sizes=args.batch_sizes,
                splits=args.splits,
                timing=f"{args.execution_mode} MLA plus reducer; repeated data, no L2 flush",
                accuracy="experimental stock parity, not independent reference",
            )
        )
        results = []
        with torch.inference_mode():
            for case_args, name, lengths in cases:
                try:
                    results.extend(
                        run_case(
                            case_args,
                            name,
                            lengths,
                            torch,
                            stock,
                            create_cutedsl_mla_decode_with_splits,
                            plan_cutedsl_mla_splits,
                            emit,
                        )
                    )
                except Exception as exc:
                    # An illegal access can poison CUDA. Never continue after a
                    # runtime failure; only numerical mismatches are recoverable.
                    emit(
                        dict(
                            event="runtime_error",
                            batch=case_args.batch_size,
                            distribution=name,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                    )
                    raise
                torch.cuda.empty_cache()
        failed = sum(not result["passed"] for result in results)
        emit(
            dict(
                event="summary",
                cases=len(cases),
                candidates=len(results),
                passed=len(results) - failed,
                failed=failed,
                invalid_reference=sum(
                    r["status"] == "invalid_reference" for r in results
                ),
                exit_code=int(failed > 0),
            )
        )
        return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
