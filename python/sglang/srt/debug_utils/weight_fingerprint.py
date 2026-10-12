"""Fingerprint every tensor on a loaded model, to diff two load paths.

Set SGLANG_DEBUG_WEIGHT_FINGERPRINT_DIR=<dir> and every rank writes one JSONL
right after load: parameters, buffers, and the plain tensors / float scalars
that post-load hooks stash on modules (w_kc, merged GEMM buffers, *_scale_float,
...). Run once loading from disk and once from a remote instance, then:

    python -m sglang.srt.debug_utils.weight_fingerprint <a.jsonl> <b.jsonl>

Checksums are over the raw bytes in logical order, so a layout change with the
same bytes shows up in stride, and a permutation shows up in the weighted sum.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from collections.abc import Iterator
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Bytes per chunk; the int64 copy of a chunk is 8x this.
_CHUNK_BYTES = 16 << 20
# Keeps position weights small enough that per-chunk sums cannot wrap.
_WEIGHT_MODULUS = 65521
_FLOAT_STAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
# nn.Module internals already covered by named_parameters / named_buffers.
_SKIP_ATTRS = {"_parameters", "_buffers", "_modules"}


def _iter_model_entries(model: torch.nn.Module) -> Iterator[tuple[str, str, Any]]:
    for name, param in model.named_parameters(remove_duplicate=False):
        yield "param", name, param
    for name, buf in model.named_buffers(remove_duplicate=False):
        yield "buffer", name, buf
    for mod_name, module in model.named_modules(remove_duplicate=False):
        prefix = f"{mod_name}." if mod_name else ""
        for attr, value in vars(module).items():
            if attr in _SKIP_ATTRS:
                continue
            name = prefix + attr
            if isinstance(value, torch.Tensor):
                yield "attr", name, value
            elif isinstance(value, float):
                yield "float_attr", name, value
            elif isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, torch.Tensor):
                        yield "attr", f"{name}[{key}]", item
            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    if isinstance(item, torch.Tensor):
                        yield "attr", f"{name}[{idx}]", item


def _tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach()
    record: dict[str, Any] = {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "storage_offset": t.storage_offset(),
        "dtype": str(t.dtype),
        "device": str(t.device),
    }
    if t.numel() == 0:
        record.update(sum=0, wsum=0)
        return record
    if t.dim() == 0:
        t = t.reshape(1)

    row_bytes = max(1, (t.numel() // t.shape[0]) * t.element_size())
    rows_per_chunk = max(1, _CHUNK_BYTES // row_bytes)
    want_float_stats = t.dtype in _FLOAT_STAT_DTYPES

    byte_sum = torch.zeros((), dtype=torch.int64, device=t.device)
    weighted_sum = torch.zeros((), dtype=torch.int64, device=t.device)
    nonfinite = torch.zeros((), dtype=torch.int64, device=t.device)
    absmax = torch.zeros((), dtype=torch.float32, device=t.device)
    byte_offset = 0
    for start in range(0, t.shape[0], rows_per_chunk):
        chunk = t[start : start + rows_per_chunk].contiguous().reshape(-1)
        raw = chunk.view(torch.uint8).to(torch.int64)
        positions = torch.arange(
            byte_offset, byte_offset + raw.numel(), device=t.device, dtype=torch.int64
        )
        byte_sum += raw.sum()
        weighted_sum += (raw * (positions % _WEIGHT_MODULUS + 1)).sum()
        byte_offset += raw.numel()
        if want_float_stats:
            as_float = chunk.float()
            finite = torch.isfinite(as_float)
            nonfinite += (~finite).sum()
            absmax = torch.maximum(
                absmax, torch.where(finite, as_float.abs(), 0.0).max()
            )
        del chunk, raw, positions

    record["sum"] = int(byte_sum.item())
    record["wsum"] = int(weighted_sum.item())
    if want_float_stats:
        record["nonfinite"] = int(nonfinite.item())
        record["absmax"] = float(absmax.item())
    return record


def dump_model_fingerprint(model: torch.nn.Module, path: str) -> int:
    """Write one JSON line per entry on ``model`` to ``path``; returns the count."""
    # Aliases (merged views, tied weights) share one computation.
    cache: dict[tuple, dict[str, Any]] = {}
    seen_names: dict[str, int] = {}
    count = 0
    with open(path, "w") as f:
        for kind, name, value in _iter_model_entries(model):
            key_name = name
            if key_name in seen_names:
                seen_names[key_name] += 1
                key_name = f"{name}#{seen_names[name]}"
            else:
                seen_names[key_name] = 0

            if kind == "float_attr":
                record = {"value": value}
            else:
                cache_key = (
                    value.data_ptr(),
                    tuple(value.shape),
                    tuple(value.stride()),
                    value.dtype,
                    value.device,
                )
                if cache_key not in cache:
                    cache[cache_key] = _tensor_fingerprint(value)
                record = dict(cache[cache_key])
            record.update(kind=kind, name=key_name)
            f.write(json.dumps(record) + "\n")
            count += 1
    return count


def maybe_dump_loaded_model_fingerprint(
    model: torch.nn.Module, *, load_format: Any, is_draft_worker: bool
) -> None:
    from sglang.srt.environ import envs

    out_dir = envs.SGLANG_DEBUG_WEIGHT_FINGERPRINT_DIR.get()
    if not out_dir:
        return
    try:
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        fmt = getattr(load_format, "value", load_format)
        role = "draft" if is_draft_worker else "target"
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"rank{rank:05d}_{fmt}_{role}.jsonl")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        count = dump_model_fingerprint(model, path)
        logger.info("Wrote %d weight fingerprint entries to %s", count, path)
    except Exception:
        # Debug instrumentation must never take the load down with it.
        logger.exception("Failed to dump the loaded-model weight fingerprint.")


def _load(path: str) -> dict[tuple[str, str], dict[str, Any]]:
    entries = {}
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            entries[(record.pop("kind"), record.pop("name"))] = record
    return entries


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def compare(path_a: str, path_b: str, *, ignore: set | None = None) -> int:
    """Print every entry that differs between two fingerprint files."""
    ignore = ignore or {"device"}
    a, b = _load(path_a), _load(path_b)
    num_diff = 0
    for key in sorted(a.keys() | b.keys()):
        kind, name = key
        if key not in b:
            print(f"only in A  [{kind}] {name}")
            num_diff += 1
            continue
        if key not in a:
            print(f"only in B  [{kind}] {name}")
            num_diff += 1
            continue
        fields = (a[key].keys() | b[key].keys()) - ignore
        diffs = [
            f"{field}: {a[key].get(field)} != {b[key].get(field)}"
            for field in sorted(fields)
            if not _same(a[key].get(field), b[key].get(field))
        ]
        if diffs:
            print(f"differs    [{kind}] {name}\n    " + "\n    ".join(diffs))
            num_diff += 1
    print(f"{num_diff} differing entries out of {len(a.keys() | b.keys())}")
    return num_diff


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <a.jsonl> <b.jsonl>")
    sys.exit(1 if compare(sys.argv[1], sys.argv[2]) else 0)
