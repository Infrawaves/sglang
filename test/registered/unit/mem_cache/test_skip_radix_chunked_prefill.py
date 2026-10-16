"""CPU 回归：跳过 radix 插入仍须保留当前请求已完成的 chunk。

通过 AST 执行真实调度、索引写入和释放函数，避免导入 CUDA/Torch。
轻量替身只保存 token 索引和物理页归属，不验证 GPU 内核。
"""

import ast
import unittest
from array import array
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"


def _definitions(path, names, namespace, class_name=None):
    body = ast.parse(path.read_text(), filename=str(path)).body
    if class_name is not None:
        body = next(
            node.body for node in body if getattr(node, "name", None) == class_name
        )
    selected = [node for node in body if getattr(node, "name", None) in names]
    assert {node.name for node in selected} == set(names)
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + selected,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


_ci = {}
_definitions(ROOT / "python/sglang/test/ci/ci_register.py", ["register_cpu_ci"], _ci)
register_cpu_ci = _ci["register_cpu_ci"]
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Scalar(int):
    def item(self):
        return int(self)


class _Indices(list):
    @property
    def shape(self):
        return (len(self),)

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return _Indices(result) if isinstance(key, slice) else _Scalar(result)

    def to(self, *args, **kwargs):
        return _Indices(self) if kwargs.get("copy") else self


class _Rows:
    def __init__(self):
        self.row = []

    def __getitem__(self, key):
        slot, positions = key
        assert slot == 0
        return _Indices(self.row[positions])

    def write(self, key, values):
        slot, positions = key
        assert slot == 0
        self.row[positions] = values


class _Pages:
    page_size = 64

    def __init__(self):
        self.next_page = 1
        self.live = set()

    def alloc(self, length):
        count = (length + self.page_size - 1) // self.page_size
        start = self.next_page * self.page_size
        self.live.update(range(self.next_page, self.next_page + count))
        self.next_page += count
        return _Indices(range(start, start + length))

    def free(self, indices):
        pages = {index // self.page_size for index in indices}
        assert pages <= self.live, "重复释放了物理页"
        self.live.difference_update(pages)


class _Range(namedtuple("_Range", "start end")):
    @property
    def length(self):
        return self.end - self.start


def _runtime():
    namespace = {
        "array": array,
        "Range": _Range,
        "torch": SimpleNamespace(int64="int64"),
        "get_serving": lambda: SimpleNamespace(strip_thinking_cache=False),
        "get_spec": lambda: SimpleNamespace(speculative_algorithm=None),
        "HybridReqToTokenPool": type("HybridReqToTokenPool", (), {}),
        "ceil_align": lambda value, alignment: (
            (value + alignment - 1) // alignment * alignment
        ),
        "attention_backends": lambda: ("torch_native", "torch_native"),
        "support_triton": lambda backend: False,
        "CLIP_MAX_NEW_TOKENS": 4096,
    }
    _definitions(
        SRT / "mem_cache/common.py",
        [
            "maybe_cache_unfinished_req",
            "release_kv_cache",
            "_release_overallocated_kv_indices",
        ],
        namespace,
    )
    _definitions(SRT / "mem_cache/allocation.py", ["write_cache_indices"], namespace)
    _definitions(
        SRT / "mem_cache/chunk_cache.py",
        ["cache_unfinished_req"],
        namespace,
        "ChunkCache",
    )
    req_methods = [
        "init_next_round_input",
        "_refresh_fill_ids",
        "_compute_max_prefix_len",
        "set_extend_range",
        "owned_kv_len",
    ]
    _definitions(SRT / "managers/schedule_batch.py", req_methods, namespace, "Req")
    req_type = type("Req", (), {name: namespace[name] for name in req_methods})
    _definitions(
        SRT / "managers/schedule_policy.py",
        ["add_chunked_req"],
        namespace,
        "PrefillAdder",
    )
    return SimpleNamespace(**namespace), req_type


class TestSkipRadixChunkedPrefill(unittest.TestCase):
    def setUp(self):
        self.prod, req_type = _runtime()
        self.pages, self.rows = _Pages(), _Rows()
        self.req = req_type()
        self.req.__dict__.update(
            origin_input_ids=array("q", [1]) * 13573,
            output_ids=array("q"),
            prefix_indices=_Indices(),
            skip_radix_cache_insert=True,
            session=None,
            return_logprob=False,
            dllm_config=None,
            positional_embed_overrides=None,
            is_retracted=False,
            retracted_stain=False,
            sampling_params=SimpleNamespace(max_new_tokens=1),
            is_dllm=lambda: False,
        )
        self.req.full_untruncated_fill_ids = self.req.origin_input_ids
        kv = SimpleNamespace(
            req_pool_idx=0,
            holds_kv=True,
            is_kv_released=False,
            retraction_l3_keys=None,
            cache_protected_len=0,
            kv_allocated_len=0,
            kv_committed_len=0,
        )
        kv.mark_kv_released = lambda: setattr(kv, "kv_allocated_len", 0)
        self.req.kv = kv
        pool = SimpleNamespace(
            req_to_token=self.rows,
            write=self.rows.write,
            free=lambda req: setattr(req.kv, "req_pool_idx", None),
        )
        self.cache = SimpleNamespace(
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=self.pages,
            claim_kv_row=lambda req: False,
            insert_req=Mock(),
            unpin=Mock(),
            on_release=Mock(),
            free_kv_row=lambda kv, ranges: [
                self.pages.free(self.rows[(kv.req_pool_idx, slice(start, end))])
                for start, end in ranges
            ],
        )
        self.cache.cache_unfinished_req = Mock(
            side_effect=lambda req, **kwargs: self.prod.cache_unfinished_req(
                self.cache, req, **kwargs
            )
        )

    def _allocate(self, start, end):
        self.req.set_extend_range(start, end)
        prefix, seq, extend, slot = map(_Indices, ([start], [end], [end - start], [0]))
        self.prod.write_cache_indices(
            self.pages.alloc(end - start),
            slot,
            slot,
            prefix,
            prefix,
            seq,
            seq,
            extend,
            extend,
            [self.req.prefix_indices],
            self.cache.req_to_token_pool,
        )
        self.req.kv.kv_allocated_len = self.req.kv.kv_committed_len = end

    def test_fake_chunk_continues_from_2752_and_releases_every_page(self):
        self._allocate(0, 2752)
        first_chunk = list(self.rows.row)
        self.prod.maybe_cache_unfinished_req(self.req, self.cache, chunked=True)
        self.req.init_next_round_input()
        adder = SimpleNamespace(
            dllm_config=None,
            rem_chunk_tokens=16384,
            kv_shard_granule=0,
            prefill_delayer_single_pass=None,
            chunked_req_limit=None,
            memory_budget=SimpleNamespace(
                available_chunk_tokens=lambda tokens: tokens,
                fit_chunk=lambda **kwargs: min(
                    kwargs["extend_input_len"], kwargs["chunk_limit"]
                ),
            ),
            _swa_new_tokens=lambda req: 1,
            _kv_shard_reserve_scratch=lambda **kwargs: True,
            _mamba_gap_budget_for_req=lambda req: 0,
            _update_prefill_budget=Mock(),
            can_run_list=[],
            exact_chunk_fill=False,
        )
        self.assertIsNone(self.prod.add_chunked_req(adder, self.req))
        continuation = self.req.extend_range
        self._allocate(continuation.start, continuation.end)
        prefix_retained = self.rows.row[:2752] == first_chunk
        self.prod.release_kv_cache(self.req, self.cache)
        self.cache.cache_unfinished_req.assert_not_called()
        self.cache.insert_req.assert_not_called()
        self.assertEqual(
            (
                continuation.start,
                continuation.length,
                prefix_retained,
                len(self.pages.live),
            ),
            (2752, 10821, True, 0),
        )

    def test_skip_without_chunk_keeps_insertion_disabled_and_releases(self):
        self._allocate(0, 13573)
        self.prod.maybe_cache_unfinished_req(self.req, self.cache)
        self.assertEqual(self.req.prefix_indices, self.rows.row)
        self.assertIsNot(self.req.prefix_indices, self.rows.row)
        self.prod.release_kv_cache(self.req, self.cache)
        self.cache.cache_unfinished_req.assert_not_called()
        self.cache.insert_req.assert_not_called()
        self.assertFalse(self.pages.live)

    def test_private_snapshot_preserves_tree_and_mamba_ownership(self):
        self._allocate(0, 2752)
        self.req.kv.cache_protected_len = 128
        self.req.kv.mamba_last_track_idx = 1
        self.req.kv.mamba_last_track_seqlen = 2048
        self.prod.maybe_cache_unfinished_req(self.req, self.cache, chunked=True)
        self.assertEqual(self.req.prefix_indices, self.rows.row)
        self.assertEqual(self.req.kv.cache_protected_len, 128)
        self.assertEqual(self.req.kv.mamba_last_track_idx, 1)
        self.assertEqual(self.req.kv.mamba_last_track_seqlen, 2048)
        self.cache.cache_unfinished_req.assert_not_called()
        self.cache.insert_req.assert_not_called()

    def test_skip_chunk_without_kv_is_a_noop(self):
        self.req.kv.req_pool_idx = None
        self.req.kv.holds_kv = False
        self.req.extend_range = _Range(0, 2752)
        self.prod.maybe_cache_unfinished_req(self.req, self.cache, chunked=True)
        self.assertEqual(self.req.prefix_indices, [])
        self.cache.cache_unfinished_req.assert_not_called()

    def test_regular_cache_still_receives_chunked_call(self):
        self.req.skip_radix_cache_insert = False
        self._allocate(0, 2752)
        self.prod.maybe_cache_unfinished_req(self.req, self.cache, chunked=True)
        self.cache.cache_unfinished_req.assert_called_once_with(self.req, chunked=True)
        self.assertEqual(self.req.prefix_indices, self.rows.row)


if __name__ == "__main__":
    unittest.main()
