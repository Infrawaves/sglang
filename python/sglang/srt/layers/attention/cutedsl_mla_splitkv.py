"""Opt-in fixed split-KV planning for FlashInfer 0.6.17 CuTeDSL MLA.

FlashInfer's public MLA API does not expose a split override. Give this backend
its own copy of the Python launch wrapper, replacing only its planner. The
upstream module, compiled kernels, and other backend instances stay untouched.
Remove this adapter once FlashInfer provides a public split-KV argument.
"""

import functools
from importlib import import_module
from importlib.metadata import version
from types import FunctionType


@functools.lru_cache(maxsize=1024, typed=True)
def plan_cutedsl_mla_splits(
    batch_size: int,
    q_len: int,
    num_heads: int,
    kv_lora_rank: int,
    num_splits: int,
    max_seq_len: int | None = None,
) -> tuple[int, int]:
    """Return effective splits and FP32 partial-output/LSE workspace bytes.

    These are the M128/K128 layout and nonempty-partition rules from the pinned
    monolithic wrapper. Planning uses host shape bounds, never GPU seq_lens.
    Omitting max_seq_len reserves capacity for the requested split count.
    """
    if type(num_splits) is not int or not 1 <= num_splits <= 32:
        raise ValueError("CuTeDSL MLA num_splits must be an integer in [1, 32]")
    if min(batch_size, q_len, num_heads, kv_lora_rank) <= 0:
        raise ValueError("CuTeDSL MLA batch and query/KV dimensions must be positive")
    if max_seq_len is not None:
        if max_seq_len <= 0:
            raise ValueError("CuTeDSL MLA max_seq_len must be positive")
        k_tiles = (max_seq_len + 127) // 128
        tiles_per_split = (k_tiles + num_splits - 1) // num_splits
        num_splits = (k_tiles + tiles_per_split - 1) // tiles_per_split

    q_tiles = (q_len * num_heads + 127) // 128
    workspace_bytes = (
        0
        if num_splits == 1
        else batch_size * 128 * q_tiles * num_splits * (kv_lora_rank + 1) * 4
    )
    return num_splits, workspace_bytes


def create_cutedsl_mla_decode_with_splits(num_splits: int):
    """Build an isolated wrapper once, before warmup/CUDA graph capture."""
    plan_cutedsl_mla_splits(1, 1, 1, 512, num_splits)
    installed = version("flashinfer-python")
    if installed != "0.6.17":
        raise RuntimeError(
            "SGLANG_CUTEDSL_MLA_NUM_KV_SPLITS requires flashinfer-python==0.6.17 "
            f"(found {installed}); unset the override for the upstream path"
        )
    module = import_module("flashinfer.cute_dsl.attention.monolithic.mla_decode")
    source = module.cute_dsl_mla_decode
    planner_name = "_get_split_kv_and_workspace_size"
    if (
        not isinstance(source, FunctionType)
        or planner_name not in source.__code__.co_names
        or planner_name not in source.__globals__
    ):
        raise RuntimeError("Unsupported FlashInfer CuTeDSL MLA wrapper/planner ABI")

    def plan(
        B,
        q_len,
        H,
        kv_lora_rank,
        max_active_blocks,
        max_seq_len=None,
        occupancy_q_tiles=None,
    ):
        # Occupancy is deliberately overridden: mixed lengths need enough
        # partitions to keep the GPU busy after the short requests finish.
        return plan_cutedsl_mla_splits(
            B, q_len, H, kv_lora_rank, num_splits, max_seq_len
        )

    globals_copy = dict(source.__globals__)
    globals_copy[planner_name] = plan
    decode = FunctionType(
        source.__code__,
        globals_copy,
        source.__name__,
        source.__defaults__,
        source.__closure__,
    )
    decode.__kwdefaults__ = (
        None if source.__kwdefaults__ is None else dict(source.__kwdefaults__)
    )
    return decode
