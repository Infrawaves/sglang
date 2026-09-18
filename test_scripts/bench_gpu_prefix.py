#!/usr/bin/env python3
"""Small, serial GPU-prefix A/B benchmark through the PD router's /generate."""

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import time
from pathlib import Path


def make_workload(pool, prefix_len, suffix_len, count):
    if len(pool) < count or suffix_len < 1:
        raise ValueError(
            "Need a positive suffix and enough distinct first suffix tokens"
        )
    rng = random.Random(42)
    prefix = rng.choices(pool, k=prefix_len)
    # Seeding and JIT-warmup suffixes are distinct from every measured suffix.
    return [
        prefix + [pool[i]] + rng.choices(pool, k=suffix_len - 1) for i in range(count)
    ]


def consume_stream(lines, started, clock=time.perf_counter):
    ttft = None
    meta = {}
    text = ""
    done = False
    for line in lines:
        if not line.startswith(b"data:"):
            continue
        now = clock()
        payload = line[5:].strip()
        if payload == b"[DONE]":
            done = True
            break
        data = json.loads(payload)
        if "error" in data:
            raise ValueError(f"Server error: {data['error']}")
        meta.update(data.get("meta_info") or {})
        if meta.get("completion_tokens", 0) > 0 and ttft is None:
            ttft = (now - started) * 1000
        text = data.get("text", text)
    if not done or ttft is None or not meta.get("finish_reason"):
        raise ValueError(
            "incomplete SSE response: missing token, finish_reason or [DONE]"
        )
    if meta["finish_reason"].get("type") == "abort":
        raise ValueError(f"Request aborted: {meta['finish_reason']}")
    return {
        "ttft_ms": ttft,
        "e2e_ms": (now - started) * 1000,
        "meta": meta,
        "text": text,
    }


def validate_meta(meta, prefix, suffix, output):
    details = meta.get("cached_tokens_details")
    if (
        not isinstance(details, dict)
        or "device" not in details
        or "host" not in details
    ):
        raise ValueError("Missing cache breakdown; cannot certify GPU-prefix hit")
    if any(
        type(details.get(k, 0)) is not int or details.get(k, 0) < 0
        for k in ("device", "host", "storage")
    ):
        raise ValueError(f"Invalid cache breakdown: {details}")
    if details["host"] or details.get("storage", 0):
        raise ValueError(f"Not a pure GPU hit (host/storage loadback): {details}")
    if not prefix - 512 <= details["device"] <= prefix:
        raise ValueError(
            f"Unexpected GPU hit length: {details}; expected {prefix} +/- page tail"
        )
    if meta.get("prompt_tokens") != prefix + suffix:
        raise ValueError(f"Unexpected prompt length: {meta.get('prompt_tokens')}")
    if meta.get("completion_tokens") != output:
        raise ValueError(f"Unexpected output length: {meta.get('completion_tokens')}")
    if meta.get("cached_tokens") != details["device"]:
        raise ValueError("Total cached_tokens differs from GPU hit count")
    if meta.get("finish_reason", {}).get("type") != "length":
        raise ValueError(f"Unexpected finish reason: {meta.get('finish_reason')}")


def transfer_metrics(log_text):
    """Aggregate calls on one host; use interval union, never sum parallel times."""
    pattern = (
        r"PD_TRANSFER pid=(\d+) pp=(\d+) peer=(\S+) bytes=(\d+) descriptors=(\d+) "
        r"start_ns=(\d+) end_ns=(\d+) send_ms=\S+ effective_GBps=\S+ ret=(-?\d+)"
    )
    calls = []
    for pid, pp, peer, size, descriptors, start, end, ret in re.findall(
        pattern, log_text
    ):
        calls.append(
            dict(
                pid=int(pid),
                pp=int(pp),
                peer=peer,
                bytes=int(size),
                descriptors=int(descriptors),
                start_ns=int(start),
                end_ns=int(end),
                ret=int(ret),
            )
        )
    good = [c for c in calls if c["ret"] == 0 and c["end_ns"] > c["start_ns"]]
    result = {
        "calls": len(calls),
        "successful_calls": len(good),
        "failed_calls": sum(c["ret"] != 0 for c in calls),
        "invalid_timing_calls": sum(c["end_ns"] <= c["start_ns"] for c in calls),
    }
    if not good:
        return result
    intervals = sorted((c["start_ns"], c["end_ns"]) for c in good)
    lo, hi = intervals[0]
    union_ns = 0
    for start, end in intervals[1:]:
        if start > hi:
            union_ns += hi - lo
            lo, hi = start, end
        else:
            hi = max(hi, end)
    union_ns += hi - lo
    span_ns = max(end for _, end in intervals) - intervals[0][0]
    total_bytes = sum(c["bytes"] for c in good)
    durations = [(c["end_ns"] - c["start_ns"]) / 1e6 for c in good]
    rates = [c["bytes"] / (c["end_ns"] - c["start_ns"]) for c in good]
    result.update(
        successful_bytes=total_bytes,
        descriptors=sum(c["descriptors"] for c in good),
        call_send_ms_median=statistics.median(durations),
        call_send_ms_max=max(durations),
        call_effective_GBps_median=statistics.median(rates),
        active_union_ms=union_ns / 1e6,
        active_union_GBps=total_bytes / union_ns,
        span_ms=span_ns / 1e6,
        span_GBps=total_bytes / span_ns,
    )
    return result


def log_evidence(path, offset):
    if path.stat().st_size < offset:
        raise ValueError("Prefill log was truncated during the benchmark")
    with path.open("rb") as f:
        startup = f.read(offset).decode(errors="replace")
        measured = f.read().decode(errors="replace")
    windows = re.findall(r"DCP window pack PP\d+:.*window_tokens=\d+", startup)
    # New servers log one completed chunk instead of every gather/shard/window.
    # Prefer summaries when details are also enabled, avoiding double counting.
    summaries = re.findall(
        r"DCP window chunk room=\S+ windows=(\d+) gathers=(\d+) sends=(\d+) "
        r"tokens=\d+ bytes=\d+ ret=0",
        measured,
    )
    return {
        "startup": list(dict.fromkeys(windows)),
        "gathers": sum(int(row[1]) for row in summaries)
        if summaries
        else len(re.findall(r"DCP window room=.*gather_ms=", measured)),
        "sends": sum(int(row[2]) for row in summaries)
        if summaries
        else len(re.findall(r"DCP window room=.*descriptors=.*ret=0", measured)),
        "windows": sum(int(row[0]) for row in summaries)
        if summaries
        else len(re.findall(r"DCP window room=.*tokens=.*bytes=", measured)),
        "fallbacks": measured.count("falling back to per-token RDMA"),
        "failed_sends": len(
            re.findall(r"DCP window room=.*ret=(?!0(?:\s|$))-?\d+", measured)
        ),
        "transfer_metrics": transfer_metrics(measured),
    }


def validate_evidence(mode, evidence):
    metrics = evidence.get("transfer_metrics", {})
    if metrics.get("failed_calls", 0) or metrics.get("invalid_timing_calls", 0):
        raise ValueError("Transfer metrics contain failed calls or invalid timestamps")
    if mode == "A":
        if evidence["gathers"] or evidence["sends"] or not evidence["fallbacks"]:
            raise ValueError("A must show legacy fallback and no window execution")
    elif (
        not evidence["startup"]
        or not evidence["gathers"]
        or not evidence["sends"]
        or not evidence["windows"]
        or evidence["fallbacks"]
        or evidence["failed_sends"]
    ):
        raise ValueError(
            "B must show window allocation/Gather/send/completion and no fallback/failure"
        )


def summarize(samples):
    return {
        "completed": len(samples),
        "median_ttft_ms": statistics.median(s["ttft_ms"] for s in samples),
        "mean_ttft_ms": statistics.mean(s["ttft_ms"] for s in samples),
        "median_e2e_latency_ms": statistics.median(s["e2e_ms"] for s in samples),
    }


def compare(a, b):
    if a["mode"] != "A" or b["mode"] != "B":
        raise ValueError("Comparison requires A then B")
    if not a.get("valid") or not b.get("valid"):
        raise ValueError("Invalid run; refusing to report speedup")
    if a.get("profiled") or b.get("profiled"):
        raise ValueError("Profiled runs are diagnostic only; do not compare timing")
    if a["workload"] != b["workload"]:
        raise ValueError("Workload/token IDs differ between A and B")
    workload = a["workload"]
    for run in (a, b):
        if len(run["samples"]) != workload["requests"] or not run["samples"]:
            raise ValueError("Incomplete sample count; refusing comparison")
        validate_evidence(run["mode"], run["log_evidence"])
        for sample in run["samples"]:
            validate_meta(
                sample["meta"],
                workload["prefix"],
                workload["suffix"],
                workload["output"],
            )
            if not math.isfinite(sample["ttft_ms"]) or sample["ttft_ms"] <= 0:
                raise ValueError("Invalid TTFT")
    for i, (x, y) in enumerate(zip(a["samples"], b["samples"]), 1):
        if x["meta"]["cached_tokens_details"] != y["meta"]["cached_tokens_details"]:
            raise ValueError(f"Request {i}: A/B cache hits differ; refusing comparison")
    print("\nrequest    A TTFT(ms)    B TTFT(ms)    reduction")
    reductions = []
    for i, (x, y) in enumerate(zip(a["samples"], b["samples"]), 1):
        gain = 100 * (1 - y["ttft_ms"] / x["ttft_ms"])
        reductions.append(gain)
        print(f"{i:7d} {x['ttft_ms']:13.2f} {y['ttft_ms']:13.2f} {gain:10.1f}%")
    am = statistics.median(x["ttft_ms"] for x in a["samples"])
    bm = statistics.median(x["ttft_ms"] for x in b["samples"])
    print(f"median  {am:13.2f} {bm:13.2f} {100 * (1 - bm / am):10.1f}%")
    print(
        f"B faster on {sum(g > 0 for g in reductions)}/{len(reductions)} requests; "
        f"median speedup={am / bm:.3f}x (small-sample result, not P95)."
    )
    same = sum(x["text"] == y["text"] for x, y in zip(a["samples"], b["samples"]))
    print(f"Output text identical: {same}/{len(reductions)} (smoke check only).")
    transport = {}
    for label, run in (("A", a), ("B", b)):
        metrics = run["log_evidence"].get("transfer_metrics", {})
        transport[label] = metrics
        if metrics.get("successful_calls"):
            print(
                f"{label} transport: {metrics['successful_bytes']} bytes; "
                f"call median={metrics['call_effective_GBps_median']:.3f} GB/s; "
                f"active-union={metrics['active_union_GBps']:.3f} GB/s; "
                f"span={metrics['span_GBps']:.3f} GB/s"
            )
        else:
            print(
                f"{label}: no PD_TRANSFER metrics; restart Prefill with the updated code"
            )
    if all(transport[k].get("successful_calls") for k in ("A", "B")):
        if transport["A"]["successful_bytes"] != transport["B"]["successful_bytes"]:
            print(
                "WARNING: observed transfer bytes differ; do not attribute bandwidth changes to packing alone."
            )
    return {
        "a_median_ttft_ms": am,
        "b_median_ttft_ms": bm,
        "ttft_reduction_pct": 100 * (1 - bm / am),
        "speedup": am / bm,
        "b_faster_requests": sum(g > 0 for g in reductions),
        "requests": len(reductions),
        "identical_texts": same,
        "transfer_metrics": transport,
    }


def run(args):
    import requests
    from transformers import AutoTokenizer

    log = Path(args.prefill_log or f"log/prefill-{args.mode}-latest.log").resolve()
    if not log.is_file():
        raise ValueError(
            f"Prefill log not found: {log}; run this client on the Prefill host"
        )
    dest = Path(args.results) / f"{args.mode}-gpu-prefix.json"
    if dest.exists():
        raise ValueError(
            f"Result already exists: {dest}; use a fresh --results directory"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ordinary = (
        "The river passes through the city. A scientist records temperature, "
        "pressure, time, distance and the colors of trees. "
        "Numbers: " + " ".join(str(i) for i in range(100))
    )
    pool = sorted(
        set(tokenizer.encode(ordinary, add_special_tokens=False))
        - set(tokenizer.all_special_ids)
    )
    prefix, suffix = args.prefix_k * 1024, args.suffix_k * 1024
    rows = make_workload(pool, prefix, suffix, args.requests + 2)
    workload = {
        "sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
        "prefix": prefix,
        "suffix": suffix,
        "output": args.output_tokens,
        "requests": args.requests,
        "model": args.model,
        "concurrency": 1,
    }
    result = {
        "mode": args.mode,
        "workload": workload,
        "samples": [],
        "valid": False,
        "profiled": bool(args.profile_prefill),
        "base": args.base,
        "prefill_log": str(log),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    session = requests.Session()

    def generate(ids, tag):
        # Serialize before starting the clock, keeping payload construction out of TTFT.
        payload = json.dumps(
            {
                "input_ids": ids,
                "stream": True,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": args.output_tokens,
                    "ignore_eos": True,
                },
            }
        )
        start = time.perf_counter()
        with session.post(
            args.base.rstrip("/") + "/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=1800,
        ) as response:
            response.raise_for_status()
            sample = consume_stream(response.iter_lines(chunk_size=1), start)
        sample["tag"] = tag
        return sample

    profiling = False
    offset = None
    log_identity = (log.stat().st_dev, log.stat().st_ino)
    try:
        response = session.post(
            args.base.rstrip("/") + "/flush_cache?timeout=60", timeout=90
        )
        response.raise_for_status()
        print("Cache flushed. Seeding prefix (cold prefill, excluded)...", flush=True)
        result["seed_meta"] = generate(rows[0], "seed")["meta"]
        print("Warming cached-prefix path / Gather JIT (excluded)...", flush=True)
        warm = generate(rows[1], "warm")
        validate_meta(warm["meta"], prefix, suffix, args.output_tokens)
        result["warmup_meta"] = warm["meta"]
        time.sleep(0.2)
        offset = log.stat().st_size
        if args.profile_prefill:
            response = session.post(
                args.profile_prefill.rstrip("/") + "/start_profile",
                json={"activities": ["CUDA_PROFILER"]},
                timeout=120,
            )
            response.raise_for_status()
            profiling = True
        for i, ids in enumerate(rows[2:], 1):
            sample = generate(ids, f"measure-{i}")
            result["samples"].append(sample)
            validate_meta(sample["meta"], prefix, suffix, args.output_tokens)
            print(
                f"{i}/{args.requests}: TTFT={sample['ttft_ms']:.2f} ms "
                f"E2E={sample['e2e_ms']:.2f} ms "
                f"cache={sample['meta']['cached_tokens_details']}",
                flush=True,
            )
        # The final completion notification precedes the final log write.
        # This wait is outside the measured request interval.
        time.sleep(0.2)
        if (log.stat().st_dev, log.stat().st_ino) != log_identity:
            raise ValueError("Prefill log was replaced during the benchmark")
        evidence = result["log_evidence"] = log_evidence(log, offset)
        print(
            "Measured log evidence:", json.dumps(evidence, ensure_ascii=False, indent=2)
        )
        validate_evidence(args.mode, evidence)
        result["summary"] = summarize(result["samples"])
        print("Summary:", json.dumps(result["summary"], indent=2), flush=True)
        result["valid"] = True
    except Exception as exc:
        result["error"] = str(exc)
        raise
    finally:
        if profiling:
            try:
                response = session.post(
                    args.profile_prefill.rstrip("/") + "/stop_profile", timeout=120
                )
                response.raise_for_status()
            except Exception as exc:
                result["valid"] = False
                result["error"] = f"stop_profile failed: {exc}"
                print(f"WARNING: stop_profile failed: {exc}", flush=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if offset is not None:
            with log.open("rb") as f:
                f.seek(offset)
                dest.with_suffix(".prefill.log").write_bytes(f.read())
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        session.close()
        print(f"Saved {dest}", flush=True)
    if not result["valid"]:
        raise ValueError(result.get("error", "Invalid run"))
    if not args.profile_prefill:
        a_path = dest.with_name("A-gpu-prefix.json")
        b_path = dest.with_name("B-gpu-prefix.json")
        if a_path.exists() and b_path.exists():
            comparison = compare(
                json.loads(a_path.read_text()), json.loads(b_path.read_text())
            )
            dest.with_name("comparison.json").write_text(
                json.dumps(comparison, indent=2) + "\n"
            )
        else:
            print(
                "Run the other mode in the same --results directory for automatic comparison."
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--mode", choices=["A", "B"], required=True)
    run_parser.add_argument("--model", default="/model/Kimi-K3")
    run_parser.add_argument("--base", required=True)
    run_parser.add_argument("--prefix-k", type=int, default=512)
    run_parser.add_argument("--suffix-k", type=int, default=16)
    run_parser.add_argument("--requests", type=int, default=6)
    run_parser.add_argument("--output-tokens", type=int, default=8)
    run_parser.add_argument("--results", default="ab-results/gpu-prefix")
    run_parser.add_argument(
        "--prefill-log", help="Defaults to log/prefill-{mode}-latest.log"
    )
    run_parser.add_argument("--profile-prefill", help="Direct Prefill URL")
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("a", type=Path)
    compare_parser.add_argument("b", type=Path)
    args = parser.parse_args()
    if args.command == "compare":
        compare(json.loads(args.a.read_text()), json.loads(args.b.read_text()))
    else:
        if min(args.prefix_k, args.suffix_k, args.requests, args.output_tokens) <= 0:
            parser.error("Lengths and requests must be positive")
        run(args)


if __name__ == "__main__":
    main()
