"""DFLASH aux graph padding must preserve the eager MLA KV writes."""

import unittest

import torch

from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_triton
from sglang.srt.mem_cache.memory_pool import _prefix_valid_mla_write_loc
from sglang.srt.speculative.dflash_aux_graph import DFlashAuxGraphs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

_BLOCK = 3
_NOPE = 512
_WIDTH = _NOPE + 64
_SLOTS = 128


def _append_into(kv):
    def append(hidden, cache_loc, cache_loc_2d, positions, commit_lens):
        values = hidden + positions[:, None].to(hidden.dtype)
        nope, rope = values[:, :_NOPE], values[:, _NOPE:]
        loc = _prefix_valid_mla_write_loc(cache_loc_2d, commit_lens, nope).loc
        set_mla_kv_buffer_triton(kv, loc, nope, rope)

    return append


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA graphs")
class TestDFlashAuxGraphs(CustomTestCase):
    def test_replay_writes_what_the_eager_append_writes(self):
        """Capture must leave KV untouched; padding after a larger batch must
        not rewrite that batch's slots, even with stale inputs in the shared tail."""
        device = "cuda"
        kv = torch.full((_SLOTS, _WIDTH), -1.0, dtype=torch.bfloat16, device=device)
        graphs = DFlashAuxGraphs(
            buckets=(1, 2, 4, 8),
            block_size=_BLOCK,
            hidden_width=_WIDTH,
            hidden_dtype=kv.dtype,
            device=device,
        )
        graphs.capture(_append_into(kv), small_copy=True)
        torch.cuda.synchronize()
        torch.testing.assert_close(kv, torch.full_like(kv, -1.0))

        for small_copy in (False, True):
            with self.subTest(small_copy=small_copy):
                generator = torch.Generator().manual_seed(0)
                next_slot = 1
                # 7 and 3 replay buckets 8 and 4 just after full batches filled them.
                for bs in (8, 7, 4, 3, 1):
                    rows = bs * _BLOCK
                    hidden = torch.randn(rows, _WIDTH, generator=generator).to(
                        device=device, dtype=kv.dtype
                    )
                    # Fresh slots make stale writes visible outside the active batch.
                    cache_loc = (
                        torch.randperm(rows, generator=generator) + next_slot
                    ).to(device)
                    next_slot += rows
                    positions = torch.randint(0, 100, (rows,), generator=generator).to(
                        device
                    )
                    commit_lens = torch.randint(
                        1, _BLOCK + 1, (bs,), generator=generator, dtype=torch.int32
                    ).to(device)

                    expected = torch.full_like(kv, -1.0)
                    _append_into(expected)(
                        hidden,
                        cache_loc,
                        cache_loc.view(bs, _BLOCK),
                        positions,
                        commit_lens,
                    )
                    kv.fill_(-1.0)
                    graph = graphs.load(
                        target_hidden=hidden,
                        cache_loc=cache_loc,
                        positions=positions,
                        commit_lens=commit_lens,
                        small_copy=small_copy,
                    )
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(kv, expected, msg=f"bs={bs}")

        for bs, dtype, width in (
            (9, kv.dtype, _WIDTH),
            (1, torch.float32, _WIDTH),
            (1, kv.dtype, _WIDTH + 1),
        ):
            with self.subTest(bs=bs, dtype=dtype, width=width):
                self.assertIsNone(
                    graphs.load(
                        target_hidden=torch.zeros(
                            bs * _BLOCK, width, dtype=dtype, device=device
                        ),
                        cache_loc=torch.ones(
                            bs * _BLOCK, dtype=torch.int64, device=device
                        ),
                        positions=torch.zeros(
                            bs * _BLOCK, dtype=torch.int64, device=device
                        ),
                        commit_lens=torch.ones(bs, dtype=torch.int32, device=device),
                        small_copy=True,
                    )
                )


if __name__ == "__main__":
    unittest.main()
