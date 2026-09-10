from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.arg_groups.model_override_base import mamba_extra_buffer_of


def _cfg(*, mode="decode", backup="ssd", strategy="extra_buffer"):
    return SimpleNamespace(
        disable_radix_cache=True,
        disaggregation_mode=mode,
        disaggregation_decode_retraction_backup=backup,
        mamba_radix_cache_strategy=strategy,
    )


def test_pd_ssd_retraction_allows_explicit_mamba_extra_buffer():
    assert mamba_extra_buffer_of(_cfg())


def test_pd_ssd_retraction_does_not_allow_lazy_extra_buffer():
    assert not mamba_extra_buffer_of(_cfg(strategy="extra_buffer_lazy"))


def test_extra_buffer_gate_stays_closed_for_non_retraction_chunk_cache():
    assert not mamba_extra_buffer_of(_cfg(backup="cpu_tensor"))


def test_kimi_k3_without_explicit_backup_does_not_auto_select_host_pool():
    """Recurrent KVs must use cpu_tensor unless host/SSD backup is explicit."""
    pytest.importorskip("xgrammar")
    from sglang.srt.mem_cache.kv_cache_builder import resolve_decode_retraction_backup

    class FakeHybridLinearKVPool:
        pass

    class FakeContext:
        def __init__(self):
            self.overrides = []

        def override(self, source, **fields):
            self.overrides.append((source, fields))

    context = FakeContext()
    disagg = SimpleNamespace(
        disaggregation_mode="decode",
        disaggregation_decode_retraction_backup=None,
        disaggregation_decode_enable_radix_cache=False,
        disaggregation_decode_enable_offload_kvcache=False,
    )
    memory = SimpleNamespace(hicache_ratio=1.0, enable_unified_memory=False)
    parallel = SimpleNamespace(dcp_enabled=False)
    allocator = SimpleNamespace(get_kvcache=lambda: FakeHybridLinearKVPool())
    worker = SimpleNamespace(
        get_memory_pool=lambda: (None, allocator),
        is_hybrid_swa=False,
        model_runner=SimpleNamespace(model_config=object()),
    )

    with (
        patch("sglang.srt.mem_cache.kv_cache_builder.get_context", return_value=context),
        patch("sglang.srt.mem_cache.kv_cache_builder.get_disagg", return_value=disagg),
        patch("sglang.srt.mem_cache.kv_cache_builder.get_memory", return_value=memory),
        patch("sglang.srt.mem_cache.kv_cache_builder.get_parallel", return_value=parallel),
        patch("sglang.srt.mem_cache.kv_cache_builder.is_hip", return_value=False),
        patch("sglang.srt.mem_cache.kv_cache_builder.uses_ssm_state", return_value=True),
        patch(
            "sglang.srt.mem_cache.kv_cache_builder.HybridLinearKVPool",
            FakeHybridLinearKVPool,
            create=True,
        ),
    ):
        assert resolve_decode_retraction_backup(tp_worker=worker) == "cpu_tensor"

    assert context.overrides == [
        (
            "kv_cache_builder.decode_retraction",
            {"disaggregation_decode_retraction_backup": "cpu_tensor"},
        )
    ]
