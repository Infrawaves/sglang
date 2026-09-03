"""Lightweight process-local CSV writer for PD profiling."""
import csv
import logging
import os
import threading
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()
_HEADER = ("rid", "phase", "duration_us")
_CACHE_HEADER = (
    "rid",
    "prompt_tokens",
    "device_hit_tokens",
    "host_hit_tokens",
    "storage_hit_tokens",
)
_KV_TRANSFER_HEADER = ("rid", "latency_ms", "total_mb", "speed_gb_s")


def enabled() -> bool:
    return envs.SGLANG_ENABLE_PD_PROFILING.get()


def observe(rid: str, phase: str, duration_us: int) -> None:
    if not enabled():
        return
    path = envs.SGLANG_PD_PROFILING_OUT.get()
    try:
        parent = os.path.dirname(path)
        with _LOCK:
            if parent:
                os.makedirs(parent, exist_ok=True)
            needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_HEADER)
                if needs_header:
                    writer.writeheader()
                writer.writerow(
                    {"rid": rid, "phase": phase, "duration_us": duration_us}
                )
    except (OSError, csv.Error) as e:
        logger.warning("Failed to write PD profiling record to %s: %s", path, e)


def observe_cache(
    rid: str,
    prompt_tokens: int,
    device_hit_tokens: int,
    host_hit_tokens: int,
    storage_hit_tokens: int,
) -> None:
    if not enabled():
        return
    base_path = envs.SGLANG_PD_PROFILING_OUT.get()
    root, ext = os.path.splitext(base_path)
    path = f"{root}.cache{ext or '.csv'}"
    row = {
        "rid": rid,
        "prompt_tokens": prompt_tokens,
        "device_hit_tokens": device_hit_tokens,
        "host_hit_tokens": host_hit_tokens,
        "storage_hit_tokens": storage_hit_tokens,
    }
    try:
        parent = os.path.dirname(path)
        with _LOCK:
            if parent:
                os.makedirs(parent, exist_ok=True)
            needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_CACHE_HEADER)
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
    except (OSError, csv.Error) as e:
        logger.warning("Failed to write PD cache profiling record to %s: %s", path, e)


def observe_kv_transfer(rid: str, metrics: dict) -> None:
    if not enabled():
        return
    base_path = envs.SGLANG_PD_PROFILING_OUT.get()
    root, ext = os.path.splitext(base_path)
    path = f"{root}.kv_transfer{ext or '.csv'}"
    row = {"rid": rid, **{key: metrics.get(key) for key in _KV_TRANSFER_HEADER[1:]}}
    try:
        parent = os.path.dirname(path)
        with _LOCK:
            if parent:
                os.makedirs(parent, exist_ok=True)
            needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_KV_TRANSFER_HEADER)
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
    except (OSError, csv.Error) as e:
        logger.warning("Failed to write KV transfer profiling record to %s: %s", path, e)
