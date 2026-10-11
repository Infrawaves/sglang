"""CPU protocol tests for RID routing and tokenizer-scoped decode state."""

import unittest
import zlib
from array import array
from collections import Counter, defaultdict
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import multi_tokenizer_mixin as routing
from sglang.srt.managers.detokenizer_manager import (
    DetokenizerManager,
    LimitedCapacityDict,
)
from sglang.srt.managers.io_struct import (
    BaseReq,
    BatchEmbeddingOutput,
    BatchTokenIDOutput,
    FreezeGCReq,
    msgpack_decode,
    msgpack_encode,
    unwrap_from_pickle,
    wrap_as_pickle,
)

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def token_batch(rids, ipcs, chunks, finished=None, **overrides):
    n = len(rids)
    fields = dict.fromkeys(
        (
            "input_token_logprobs_val",
            "input_token_logprobs_idx",
            "output_token_logprobs_val",
            "output_token_logprobs_idx",
            "input_top_logprobs_val",
            "input_top_logprobs_idx",
            "output_top_logprobs_val",
            "output_top_logprobs_idx",
            "input_token_ids_logprobs_val",
            "input_token_ids_logprobs_idx",
            "output_token_ids_logprobs_val",
            "output_token_ids_logprobs_idx",
            "output_token_entropy_val",
            "output_token_sampling_mask",
            "output_token_sampling_logprobs",
            "output_hidden_states",
            "routed_experts",
            "indexer_topk",
            "placeholder_tokens_idx",
            "placeholder_tokens_val",
        )
    )
    fields.update(
        rids=rids,
        http_worker_ipcs=ipcs,
        finished_reasons=finished if finished is not None else [None] * n,
        decoded_texts=[""] * n,
        decode_ids=[array("i", list(chunk)) for chunk in chunks],
        read_offsets=[0] * n,
        output_ids=None,
        skip_special_tokens=[True] * n,
        spaces_between_special_tokens=[False] * n,
        no_stop_trim=[False] * n,
        prompt_tokens=[10] * n,
        completion_tokens=[len(chunk) for chunk in chunks],
        reasoning_tokens=[0] * n,
        cached_tokens=[0] * n,
    )
    fields.update(overrides)
    return BatchTokenIDOutput(**fields)


class ByteTokenizer:
    """A deterministic tokenizer boundary, including incomplete UTF-8 bytes."""

    is_fast = True

    def decode(self, ids, **kwargs):
        return bytes(ids).decode("utf-8", errors="replace")

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row, **kwargs) for row in rows]


def make_detokenizer():
    manager = DetokenizerManager.__new__(DetokenizerManager)
    manager.decode_status = LimitedCapacityDict(capacity=100)
    manager.tokenizer = ByteTokenizer()
    manager.vocab_size = 256
    manager.disable_tokenizer_batch_decode = False
    manager.is_tool_call_parser_gpt_oss = False
    manager.recv_from_scheduler = "scheduler-output"
    manager.soft_watchdog = SimpleNamespace(disable=nullcontext, feed=lambda: None)
    manager.init_request_dispatcher()
    return manager


class TestDetokenizerRouting(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.sent = []
        # Preserve SocketMapping's registration/cache/send logic; replace only ZMQ.
        for name, kwargs in (
            ("zmq.Context", {}),
            ("get_zmq_socket", {"side_effect": lambda ctx, kind, ipc, bind: ipc}),
            (
                "sock_send",
                {"side_effect": lambda ipc, obj: self.sent.append((ipc, obj))},
            ),
        ):
            patcher = patch(
                "sglang.srt.managers.multi_tokenizer_mixin." + name, **kwargs
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_router(self, n=4):
        return routing.MultiDetokenizerRouter(
            [f"detokenizer-{i}" for i in range(n)],
            SimpleNamespace(detokenizer_ipc_name="scheduler-output"),
        )

    def route(self, router, messages):
        start = len(self.sent)
        with (
            patch.object(routing, "sock_recv", side_effect=iter(messages)),
            self.assertRaises(StopIteration),
        ):
            router.event_loop()
        return self.sent[start:]

    def test_request_balance_with_eight_skewed_tokenizers(self):
        # Synthetic protocol workload, not a measured latency/capacity claim.
        # Six T addresses collide in slot 0, and two in slots 1/2, for both 4/8 D.
        ipcs = []
        candidate = 0
        for slot in [0] * 6 + [1, 2]:
            while True:
                ipc = f"tokenizer-{candidate}"
                candidate += 1
                if zlib.crc32(ipc.encode()) % 8 == slot:
                    ipcs.append(ipc)
                    break
        messages = []
        expected_owners = {}
        for t, count in enumerate([512, 128, 64, 64, 64, 64, 64, 64]):
            rids = [f"request-{len(expected_owners) + i:06d}" for i in range(count)]
            expected_owners.update(dict.fromkeys(rids, ipcs[t]))
            messages.append(token_batch(rids, [ipcs[t]] * count, [b"x"] * count))
        for n in (4, 8):
            with self.subTest(workers=n):
                router = self.make_router(n)
                deliveries = self.route(router, messages)
                counts = Counter(ipc for ipc, _ in deliveries)
                self.assertEqual(len(deliveries), 1024)
                self.assertEqual(set(counts), set(router.ipc_name_list))
                self.assertLess(max(counts.values()), 1.15 * 1024 / n)
                self.assertGreater(min(counts.values()), 0.85 * 1024 / n)
                for _, item in deliveries:
                    self.assertEqual(
                        item.http_worker_ipcs, [expected_owners[item.rids[0]]]
                    )
                old_counts = Counter(
                    router._pick(None, expected_owners[rid]) for rid in expected_owners
                )
                self.assertEqual(sorted(old_counts.values()), [64, 64, 896])

    def test_chunks_keep_worker_when_batch_order_or_return_address_changes(self):
        router = self.make_router(8)
        deliveries = self.route(
            router,
            [
                token_batch(["a", "b"], ["T0", "T1"], [b"a", b"b"]),
                token_batch(["b", "a"], ["T1", "T0"], [b"c", b"d"]),
                BaseReq(rid="a", http_worker_ipc="T2"),
            ],
        )
        destinations = defaultdict(set)
        for ipc, item in deliveries:
            rid = item.rid if isinstance(item, BaseReq) else item.rids[0]
            destinations[rid].add(ipc)
        self.assertEqual(destinations["a"], {router._pick("a", "ignored")})
        self.assertEqual(destinations["b"], {router._pick("b", "ignored")})
        self.assertEqual(deliveries[-1][1].http_worker_ipc, "T2")

    def test_no_rid_fallback_and_broadcasts(self):
        router = self.make_router()
        missing = BaseReq(http_worker_ipc="T0")
        empty_rid = BaseReq(rid="", http_worker_ipc="T1")
        freeze = FreezeGCReq()
        idle = token_batch([], [], [])
        deliveries = self.route(router, [missing, empty_rid, freeze, idle])
        self.assertEqual(
            deliveries[:2],
            [
                (router._pick(None, "T0"), missing),
                (router._pick(None, "T1"), empty_rid),
            ],
        )
        for obj in (freeze, idle):
            self.assertEqual(
                [ipc for ipc, item in deliveries if item is obj], router.ipc_name_list
            )

    def test_token_and_embedding_batches_preserve_split_metadata(self):
        router = self.make_router()
        token = token_batch(
            ["a", "b"],
            ["T0", "T1"],
            [b"a", b"b"],
            finished=[None, {"type": "length"}],
            prompt_tokens=[11, 22],
            dp_ranks=[3, 7],
            output_token_logprobs_val=[[-0.1], [-0.2]],
            customized_info=wrap_as_pickle({"trace": ["trace-a", "trace-b"]}),
            time_stats=wrap_as_pickle([{"created": 1}, {"created": 2}]),
        )
        embedding = BatchEmbeddingOutput(
            rids=["c", "d"],
            http_worker_ipcs=["T2", "T3"],
            finished_reasons=[{"type": "length"}] * 2,
            embeddings=[[1.0, 2.0], [3.0, 4.0]],
            prompt_tokens=[33, 44],
            cached_tokens=[3, 4],
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
        )
        deliveries = self.route(router, [token, embedding])
        self.assertEqual(len(deliveries), 4)
        for i, (ipc, item) in enumerate(deliveries):
            self.assertEqual(item.rids, ["abcd"[i]])
            self.assertEqual(item.http_worker_ipcs, [f"T{i}"])
            self.assertEqual(item.prompt_tokens, [11 * (i + 1)])
            self.assertEqual(ipc, router._pick("abcd"[i], f"T{i}"))
            if i < 2:
                self.assertEqual(item.dp_ranks, [[3], [7]][i])
                self.assertEqual(
                    item.output_token_logprobs_val, [[[-0.1]], [[-0.2]]][i]
                )
                self.assertEqual(
                    unwrap_from_pickle(item.customized_info),
                    {"trace": [f"trace-{'ab'[i]}"]},
                )
                self.assertEqual(
                    unwrap_from_pickle(item.time_stats), [{"created": i + 1}]
                )
            else:
                self.assertEqual(item.embeddings, [embedding.embeddings[i - 2]])
                self.assertEqual(item.cached_tokens, [i + 1])

    def test_restarted_tokenizer_same_rid_ignores_old_ipc_finish(self):
        manager = make_detokenizer()
        # A new tokenizer IPC can reuse a RID while the old IPC still has output.
        first = token_batch(["same", "same"], ["T-old", "T-new"], [b"a", b"x"])
        self.assertEqual(
            manager.handle_batch_token_id_out(first).output_strs, ["a", "x"]
        )
        second = token_batch(
            ["same", "same"],
            ["T-new", "T-old"],
            [b"y", b"b"],
            finished=[None, {"type": "length"}],
        )
        self.assertEqual(
            manager.handle_batch_token_id_out(second).output_strs, ["y", "b"]
        )
        self.assertEqual(set(manager.decode_status), {("T-new", "same")})
        last = token_batch(["same"], ["T-new"], [b"c"], finished=[{"type": "length"}])
        self.assertEqual(manager.handle_batch_token_id_out(last).output_strs, ["c"])
        self.assertFalse(manager.decode_status)
        reused = token_batch(
            ["same"], ["T-new"], [b"fresh"], finished=[{"type": "length"}]
        )
        self.assertEqual(
            manager.handle_batch_token_id_out(reused).output_strs, ["fresh"]
        )
        self.assertFalse(manager.decode_status)

    def test_utf8_and_stop_trimming(self):
        for disable_batch in (False, True):
            for no_stop_trim in (False, True):
                with self.subTest(
                    disable_batch=disable_batch, no_stop_trim=no_stop_trim
                ):
                    manager = make_detokenizer()
                    manager.disable_tokenizer_batch_decode = disable_batch
                    first = token_batch(["utf8"], ["T0"], [[0xE2]])
                    self.assertEqual(
                        manager.handle_batch_token_id_out(first).output_strs, [""]
                    )
                    final = token_batch(
                        ["utf8"],
                        ["T0"],
                        [[0x82, 0xAC] + list(b" hi<STOP>tail")],
                        finished=[{"type": "stop", "matched": "<STOP>"}],
                        no_stop_trim=[no_stop_trim],
                    )
                    self.assertEqual(
                        manager.handle_batch_token_id_out(final).output_strs,
                        ["€ hi<STOP>" if no_stop_trim else "€ hi"],
                    )
                    token_stop = token_batch(
                        ["stop-token"],
                        ["T0"],
                        [b"ok!"],
                        finished=[{"type": "stop", "matched": ord("!")}],
                        no_stop_trim=[no_stop_trim],
                    )
                    self.assertEqual(
                        manager.handle_batch_token_id_out(token_stop).output_strs,
                        ["ok!" if no_stop_trim else "ok"],
                    )
                    self.assertFalse(manager.decode_status)

    def test_single_tokenizer_legacy_and_none_ipc_list(self):
        for ipcs, key in ((None, "legacy"), ([None], (None, "legacy"))):
            with self.subTest(ipcs=ipcs):
                manager = make_detokenizer()
                first = token_batch(["legacy"], ipcs, [b"a"])
                self.assertEqual(
                    manager.handle_batch_token_id_out(first).output_strs, ["a"]
                )
                self.assertEqual(set(manager.decode_status), {key})
                last = token_batch(
                    ["legacy"], ipcs, [b"b"], finished=[{"type": "length"}]
                )
                self.assertEqual(
                    manager.handle_batch_token_id_out(last).output_strs, ["b"]
                )
                self.assertFalse(manager.decode_status)

    def test_router_to_detokenizer_to_original_tokenizer(self):
        router = self.make_router()
        batches = [
            token_batch(["same", "same"], ["T0", "T1"], [[0xE2], b"b"]),
            token_batch(
                ["same", "same"],
                ["T1", "T0"],
                [b"!", [0x82]],
                finished=[{"type": "length"}, None],
            ),
            token_batch(
                ["same"], ["T0"], [[0xAC, ord("?")]], finished=[{"type": "length"}]
            ),
        ]
        # One T owns additional streams spanning every D, independent of its IPC.
        fanout = {}
        for candidate in range(64):
            rid = f"fanout-{candidate}"
            fanout.setdefault(router._pick(rid, "T0"), rid)
        self.assertEqual(len(fanout), router.num_workers)
        rids = list(fanout.values())
        batches.extend(
            [
                token_batch(rids, ["T0"] * len(rids), [b"start"] * len(rids)),
                token_batch(
                    rids,
                    ["T0"] * len(rids),
                    [b"-end"] * len(rids),
                    finished=[{"type": "length"}] * len(rids),
                ),
            ]
        )
        # Use the production codec on each hop; BaseReq is deliberately not used
        # here because it is not in io_struct's concrete wire-type union.
        deliveries = self.route(
            router, [msgpack_decode(msgpack_encode(x)) for x in batches]
        )
        self.assertEqual(len({ipc for ipc, _ in deliveries[:5]}), 1)
        self.assertEqual({ipc for ipc, _ in deliveries}, set(router.ipc_name_list))
        returned_start = len(self.sent)
        for ipc in router.ipc_name_list:
            queue = [
                msgpack_decode(msgpack_encode(obj))
                for destination, obj in deliveries
                if destination == ipc
            ]
            manager = make_detokenizer()
            with (
                patch.object(routing, "sock_recv", side_effect=iter(queue)),
                self.assertRaises(StopIteration),
            ):
                manager.multi_http_worker_event_loop()
            self.assertFalse(manager.decode_status)
        outputs = defaultdict(list)
        for ipc, item in self.sent[returned_start:]:
            item = msgpack_decode(msgpack_encode(item))
            outputs[(ipc, item.rids[0])].extend(item.output_strs)
        expected = {("T0", "same"): ["", "", "€?"], ("T1", "same"): ["b", "!"]}
        expected.update({("T0", rid): ["start", "-end"] for rid in rids})
        self.assertEqual(dict(outputs), expected)

    def test_router_rejects_invalid_batch_return_addresses(self):
        router = self.make_router()
        for ipcs in (None, [], [None], ["T0", "T1"]):
            with self.subTest(ipcs=ipcs):
                message = token_batch(["request"], ipcs, [b"x"])
                start = len(self.sent)
                with (
                    patch.object(routing, "sock_recv", side_effect=iter([message])),
                    self.assertRaisesRegex(AssertionError, "invalid http_worker_ipcs"),
                ):
                    router.event_loop()
                self.assertEqual(len(self.sent), start)


if __name__ == "__main__":
    unittest.main()
