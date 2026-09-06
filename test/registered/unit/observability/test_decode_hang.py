"""CPU-only regression tests for diagnostic hooks; no server/model/GPU needed."""

import ast
import enum
import fnmatch
import functools
import inspect
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.observability import decode_hang as trace
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_group_facts_impl = trace.group_facts


def req(rid):
    return SimpleNamespace(
        rid=rid,
        output_ids=[17],
        finished_reason=None,
        is_retracted=False,
        is_demoted=False,
    )


def batch(*rids):
    return SimpleNamespace(
        reqs=[req(rid) for rid in rids],
        forward_iter=None,
        forward_mode=SimpleNamespace(name="DECODE"),
    )


class MetadataOnlyTensor:
    shape = (63,)
    dtype = "torch.int32"
    device = "cuda:0"

    def numel(self):
        return self.shape[0]

    def cpu(self):
        raise AssertionError("diagnostics must not transfer GPU data")

    def tolist(self):
        raise AssertionError("diagnostics must not synchronize")

    def item(self):
        raise AssertionError("diagnostics must not synchronize")


class TestDecodeHang(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.enterContext(envs.SGLANG_DEBUG_DECODE_HANG.override(True))
        self.enterContext(
            envs.SGLANG_DEBUG_DECODE_HANG_DIR.override(self.directory.name)
        )
        self.enterContext(patch.object(trace, "_writer", None))
        self.enterContext(patch.object(trace, "_writer_pid", None))
        self.enterContext(patch.object(trace, "_failed", False))
        self.enterContext(patch.object(trace, "_event_seq", itertools.count(1)))
        self.enterContext(patch.object(trace, "_sample_seq", itertools.count(1)))
        self.enterContext(patch.object(trace, "_min_seq", itertools.count(1)))
        self.enterContext(patch.object(trace, "_identity", return_value={"tp_rank": 0}))
        self.enterContext(
            patch.object(
                trace,
                "group_facts",
                return_value={
                    "pg_name": "tp",
                    "pg_ranks": list(range(8)),
                    "pg_seq": 120,
                },
            )
        )
        trace.enabled.cache_clear()
        self.addCleanup(trace.enabled.cache_clear)
        self.addCleanup(self.close_writer)

    def close_writer(self):
        if trace._writer is not None:
            trace._writer.close()

    def records(self):
        lines = []
        for path in Path(self.directory.name).glob("*.jsonl*"):
            lines.extend(json.loads(line) for line in path.read_text().splitlines())
        return sorted(lines, key=lambda record: record["event_seq"])

    def test_group_metadata_reads_optional_host_sequence(self):
        torch_module = ModuleType("torch")
        dist_module = ModuleType("torch.distributed")
        dist_module.get_backend = lambda group: "nccl"
        dist_module.get_process_group_ranks = lambda group: list(range(8))
        torch_module.distributed = dist_module
        group = SimpleNamespace(
            group_name="tp", _get_sequence_number_for_group=lambda: 777
        )
        with patch.dict(
            sys.modules, {"torch": torch_module, "torch.distributed": dist_module}
        ):
            facts = _group_facts_impl(group)
            self.assertEqual(facts["pg_seq"], 777)
            self.assertEqual(facts["pg_ranks"], list(range(8)))
            del group._get_sequence_number_for_group
            self.assertIsNone(_group_facts_impl(group)["pg_seq"])

    def test_disabled_does_not_inspect_tensor_or_batch(self):
        with envs.SGLANG_DEBUG_DECODE_HANG.override(False):
            trace.enabled.cache_clear()
            self.assertIsNone(
                trace.token_sync_enter(
                    object(),
                    object(),
                    sync_enabled=True,
                    env_enabled=True,
                    has_grammar=False,
                )
            )
            original = Mock(return_value=17)
            self.assertEqual(trace.trace_run_batch(original)(object(), object()), 17)
            self.assertEqual(self.records(), [])

    def test_sync_skip_and_enter_have_distinct_sequences(self):
        tensor = MetadataOnlyTensor()
        trace.token_sync_enter(
            tensor, None, sync_enabled=False, env_enabled=False, has_grammar=False
        )
        ticket = trace.token_sync_enter(
            tensor, None, sync_enabled=True, env_enabled=False, has_grammar=True
        )
        trace.token_sync_return(ticket)
        records = self.records()[1:]
        self.assertEqual(
            [r["event"] for r in records],
            ["token_sync_skip", "token_sync_enter", "token_sync_host_return"],
        )
        self.assertIsNone(records[0]["min_seq"])
        self.assertEqual((records[1]["sample_seq"], records[1]["min_seq"]), (2, 1))
        self.assertEqual((records[1]["numel"], records[1]["pg_seq"]), (63, 120))
        self.assertNotIn("gpu_complete", records[2])

    def test_delayed_sampling_keeps_original_context_after_next_batch(self):
        owner = SimpleNamespace(disaggregation_mode="decode", forward_ct=0)
        observed = []

        @trace.trace_run_batch
        def run(self, current):
            self.forward_ct += 1
            current.forward_iter = self.forward_ct

            def sample():
                observed.append(dict(trace._context.get()))
                return "sampled"

            return SimpleNamespace(delay_sample_func=sample, can_run_cuda_graph=True)

        first = batch("rid-a", "rid-b")
        first_result = run(owner, first)
        first_hash = self.records()[1]["rids_hash"]
        first.reqs.reverse()  # mutable batch must not affect a delayed closure
        run(owner, batch("rid-c"))
        self.assertEqual(first_result.delay_sample_func(), "sampled")
        self.assertEqual(observed[0]["forward_iter"], 1)
        self.assertEqual(observed[0]["rids_hash"], first_hash)
        self.assertEqual(trace._context.get(), {})

    def test_original_exception_propagates_and_context_is_restored(self):
        def fail():
            raise ValueError("original failure")

        with self.assertRaisesRegex(ValueError, "original failure"):
            trace.bind_delayed_sample(fail, {"forward_iter": 9})()
        self.assertEqual(trace._context.get(), {})

    def test_result_records_cpu_finish_transition(self):
        owner = SimpleNamespace(disaggregation_mode="decode")
        current = batch("a")
        current.forward_iter = 8

        @trace.trace_process_result
        def process(self, b, result):
            b.reqs[0].output_ids.append(23)
            b.reqs[0].finished_reason = SimpleNamespace()
            return result

        result = object()
        self.assertIs(process(owner, current, result), result)
        before, after = self.records()[1:]
        self.assertEqual(before["rows"][0][1:4], [1, 17, None])
        self.assertEqual(after["rows"][0][1:4], [2, 23, "SimpleNamespace"])
        self.assertEqual(after["forward_iter"], 8)

    def test_rotation_keeps_bounded_files_with_parseable_records(self):
        trace.emit("first")
        trace._writer.maxBytes = 1000
        trace._writer.backupCount = 2
        for i in range(20):
            trace.emit("rotate", value="x" * 200, index=i)
        paths = list(Path(self.directory.name).glob("*.jsonl*"))
        self.assertEqual(len(paths), 3)
        records = self.records()
        self.assertEqual(records[-1]["index"], 19)
        self.assertTrue(all(r["schema"] == 1 for r in records))

    def test_disk_error_does_not_fail_inference(self):
        with patch.object(trace, "_TraceHandler", side_effect=OSError("disk full")):
            with self.assertLogs(trace.logger, level="WARNING"):
                trace.emit("event")
            trace.emit("another_event")
        self.assertTrue(trace._failed)

    def test_request_order_changes_digest(self):
        self.assertNotEqual(
            trace._batch_context(batch("a", "b"))["rids_hash"],
            trace._batch_context(batch("b", "a"))["rids_hash"],
        )

    def test_nested_prebuilt_does_not_rebind_inner_sampling(self):
        owner = SimpleNamespace(disaggregation_mode="decode", forward_ct=0)
        observed = []

        @trace.trace_run_batch
        def run(self, current):
            self.forward_ct += 1
            current.forward_iter = self.forward_ct
            if current.forward_mode.name == "PREBUILT":
                return run(self, batch("inner"))
            return SimpleNamespace(
                delay_sample_func=lambda: observed.append(
                    trace._context.get()["forward_iter"]
                )
            )

        outer = batch("outer")
        outer.forward_mode.name = "PREBUILT"
        result = run(owner, outer)
        result.delay_sample_func()
        self.assertEqual(observed, [2])
        self.assertEqual(
            sum(r["event"] == "delayed_sample_enter" for r in self.records()), 1
        )

    def test_schedule_records_filtered_request_and_preserves_plan(self):
        owner = SimpleNamespace(forward_ct=4, waiting_queue=[])
        current = batch("done", "running")

        @trace.trace_schedule
        def schedule(self, running_batch):
            running_batch.reqs.pop(0)
            return SimpleNamespace(batch_to_run=running_batch)

        plan = schedule(owner, running_batch=current)
        self.assertIs(plan.batch_to_run, current)
        before, after = self.records()[1:]
        self.assertEqual([r[0] for r in before["running"]], ["done", "running"])
        self.assertEqual([r[0] for r in after["selected"]], ["running"])

    def test_kernel_metadata_does_not_repr_or_probe_opaque_gpu_objects(self):
        source = Path(trace.__file__).parents[2] / "kernels" / "kernel_api_logging.py"
        tree = ast.parse(source.read_text())
        method = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_serialize_value"
        )
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))

        class Opaque:
            @property
            def shape(self):
                raise AssertionError("do not probe FFI GPU storage")

            def __repr__(self):
                raise AssertionError("repr can read GPU values")

        namespace = {
            "Any": object,
            "torch": SimpleNamespace(Tensor=MetadataOnlyTensor),
            "enum": enum,
            "_KERNEL_API_LOG_LEVEL": 3,
        }
        exec(compile(module, str(source), "exec"), namespace)
        serialize = namespace[method.name]
        self.assertEqual(serialize(Opaque()), ["Opaque(metadata omitted)"])
        algorithm = enum.Enum("Algo", "PUSH PULL")
        self.assertEqual(serialize(algorithm.PULL), ["Algo.PULL"])

    def test_kernel_filter_preserves_excluded_callable_and_selected_result(self):
        source = Path(trace.__file__).parents[2] / "kernels" / "kernel_api_logging.py"
        tree = ast.parse(source.read_text())
        method = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "debug_kernel_api"
        ][-1]
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        sections = Mock()
        namespace = {
            "Callable": Callable,
            "envs": envs,
            "fnmatch": fnmatch,
            "functools": functools,
            "inspect": inspect,
            "Any": object,
            "_KERNEL_API_LOG_LEVEL": 3,
            "_logger": Mock(),
            "Path": Path,
            "_is_compiling": lambda: False,
            "_timestamp": lambda: "time",
            "_infer_func_name": lambda f: f.__name__,
            "_log_section": sections,
        }
        exec(compile(module, str(source), "exec"), namespace)

        def original(x):
            return x + 1

        with envs.SGLANG_KERNEL_API_LOG_INCLUDE.override("k3.*"):
            decorate = namespace["debug_kernel_api"]
            self.assertIs(decorate(original, op_name="other.op"), original)
            selected = decorate(original, op_name="k3.all_reduce_pull_res")
            self.assertEqual(selected(8), 9)
            self.assertEqual(sections.call_count, 2)  # input and output metadata

    def test_sampler_preserves_all_reduce_condition_and_call_count(self):
        # Execute the actual leaf method without importing GPU-only sampler dependencies.
        source = Path(trace.__file__).parents[1] / "layers" / "sampler.py"
        tree = ast.parse(source.read_text())
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_sync_token_ids_across_tp"
        )
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        for env_enabled, grammars, expected in (
            (False, None, 0),
            (False, [], 0),
            (False, [object()], 1),
            (True, None, 1),
        ):
            with self.subTest(env=env_enabled, grammars=bool(grammars)):
                all_reduce = Mock()
                namespace = {
                    "SYNC_TOKEN_IDS_ACROSS_TP": env_enabled,
                    "torch": SimpleNamespace(
                        distributed=SimpleNamespace(all_reduce=all_reduce)
                    ),
                    "dist": SimpleNamespace(ReduceOp=SimpleNamespace(MIN="MIN")),
                    "token_sync_enter": trace.token_sync_enter,
                    "token_sync_return": trace.token_sync_return,
                }
                exec(compile(module, str(source), "exec"), namespace)
                tensor, group = MetadataOnlyTensor(), object()
                namespace[method.name](
                    SimpleNamespace(tp_sync_group=group),
                    tensor,
                    SimpleNamespace(grammars=grammars),
                )
                self.assertEqual(all_reduce.call_count, expected)
                if expected:
                    all_reduce.assert_called_once_with(tensor, op="MIN", group=group)


if __name__ == "__main__":
    unittest.main()
