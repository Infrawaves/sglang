from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.attention import trtllm_mla_backend as trt
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("prepacked", [False, True])
def test_quantization_applies_scales_once(prepacked):
    q = torch.ones(2, 3, 4, dtype=torch.bfloat16)
    k = torch.full_like(q, 4)
    v = torch.full_like(q, 8)
    layer = SimpleNamespace(
        k_scale_float=2.0,
        v_scale_float=4.0,
        k_scale=torch.tensor([2.0]),
        v_scale=torch.tensor([4.0]),
    )
    if prepacked:
        k = (k.float() / 2).to(torch.float8_e4m3fn)
        v = (v.float() / 4).to(torch.float8_e4m3fn)

    def quantize(x, scale):
        return (x.float() / scale).to(torch.float8_e4m3fn), scale

    with patch.object(trt, "scaled_fp8_quant", side_effect=quantize) as quant:
        q_out, k_out, v_out, ks, vs = trt._quantize_fp8_qkv(q, k, v, layer)
    assert quant.call_count == (0 if prepacked else 2)
    assert q_out.dtype == torch.float8_e4m3fn
    assert (ks, vs) == (2.0, 4.0)
    torch.testing.assert_close(k_out.float(), torch.full_like(q.float(), 2))
    torch.testing.assert_close(v_out.float(), torch.full_like(q.float(), 2))
    if prepacked:
        assert k_out is k and v_out is v


@pytest.mark.parametrize("subclass,fp8", [(False, True), (True, True), (False, False)])
def test_backend_pack_dispatch(subclass, fp8):
    backend_type = (
        type("OtherBackend", (trt.TRTLLMMLABackend,), {})
        if subclass
        else trt.TRTLLMMLABackend
    )
    backend = backend_type.__new__(backend_type)
    backend.data_type = torch.float8_e4m3fn if fp8 else torch.bfloat16
    source = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
    layer = SimpleNamespace(
        k_scale_float=2.0, v_scale_float=1.0, k_scale=torch.tensor([2.0])
    )
    with (
        patch.object(torch.version, "cuda", "13.0"),
        patch(
            "sglang.kernels.ops.attention.mla_kv_pack.pack_mla_kv_fp8",
            return_value=("K", "V"),
        ) as pack,
    ):
        result = backend.pack_mha_kv(source, None, source, layer, None)
    if subclass or not fp8:
        assert result is None
        pack.assert_not_called()
    else:
        assert result == ("K", "V")
        pack.assert_called_once_with(
            source, None, source, k_scale=layer.k_scale, v_scale=None
        )


@pytest.mark.parametrize("packed", [False, True])
def test_current_chunk_uses_backend_pack_or_original_concat(packed):
    attn = forward_mha.DeepseekMHAForwardMixin()
    attn.q_lora_rank = None
    attn.num_local_heads = 2
    attn.qk_nope_head_dim = 4
    attn.qk_rope_head_dim = 2
    attn.qk_head_dim = 6
    attn.v_head_dim = 4
    attn.kv_lora_rank = 3
    attn.rotary_emb = None
    attn.q_proj = Mock(return_value=(torch.randn(3, 12), None))
    attn.kv_a_proj_with_mqa = Mock(return_value=(torch.randn(3, 5), None))
    attn.kv_a_layernorm = lambda x: x
    attn.kv_b_proj = Mock(return_value=(torch.randn(3, 16), None))
    attn.attn_mha = object()
    attn._set_mla_kv_buffer = Mock()
    attn._concat_and_cast_mha_k = Mock(return_value="original K")
    batch = SimpleNamespace(mha_one_shot=False)
    backend = SimpleNamespace(
        pack_mha_kv=Mock(return_value=("packed K", "packed V") if packed else None)
    )

    with patch.object(forward_mha, "resolve_attn_backend", return_value=backend):
        _, k, v, result_batch = attn.forward_normal_prepare(
            None, torch.randn(3, 8), batch, None
        )
    assert result_batch is batch
    backend.pack_mha_kv.assert_called_once()
    attn._set_mla_kv_buffer.assert_called_once()
    if packed:
        assert (k, v) == ("packed K", "packed V")
        attn._concat_and_cast_mha_k.assert_not_called()
    else:
        assert k == "original K"
        attn._concat_and_cast_mha_k.assert_called_once()


def test_prefix_uses_packed_kv_and_keeps_projection_dtype():
    attn = forward_mha.DeepseekMHAForwardMixin()
    attn.num_local_heads, attn.qk_nope_head_dim, attn.v_head_dim = 2, 4, 4
    attn.qk_rope_head_dim = 2
    attn._get_mla_kv_buffer = Mock(
        return_value=(torch.randn(3, 3), torch.randn(3, 1, 2))
    )
    attn.kv_b_proj = Mock(return_value=(torch.randn(3, 16), None))
    q = torch.randn(2, 2, 6, dtype=torch.bfloat16)
    output, lse = torch.zeros(2, 2, 4), torch.zeros(2, 2)
    k_pack, v_pack = (
        torch.empty(3, 2, 6, dtype=torch.float8_e4m3fn),
        torch.empty(3, 2, 4, dtype=torch.float8_e4m3fn),
    )
    attn.attn_mha = Mock(return_value=(output, lse))
    batch = SimpleNamespace(
        num_prefix_chunks=1,
        extend_prefix_lens_cpu=[3],
        set_prefix_chunk_idx=Mock(),
        prefix_chunk_kv_indices=[torch.arange(3)],
        prefix_chunk_seq_lens_cpu=[[3]],
        prefix_chunk_starts_cpu=[[0]],
    )
    backend = SimpleNamespace(pack_mha_kv=Mock(return_value=(k_pack, v_pack)))
    with (
        patch.object(forward_mha, "resolve_attn_backend", return_value=backend),
        patch.object(
            forward_mha,
            "all_gather_kv_cache_for_mha_chunk_extend",
            side_effect=lambda a, b, *_: (a, b),
        ),
        patch.dict(
            "sys.modules",
            {
                "sglang.srt.layers.attention.merge_state": SimpleNamespace(
                    merge_state=Mock()
                )
            },
        ),
    ):
        attn._chunked_prefix_attn_mha(q, output, lse, batch)
    backend.pack_mha_kv.assert_called_once()
    assert attn._get_mla_kv_buffer.call_args.args[1] == torch.bfloat16
    args = attn.attn_mha.call_args.args
    assert args[1] is k_pack and args[2] is v_pack
