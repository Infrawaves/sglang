# SPDX-License-Identifier: Apache-2.0
"""CPU regression of actual scheduler/IO methods without GPU imports.

Run directly with a Python environment containing msgspec:
    python test/registered/unit/managers/test_cache_input_metadata.py -v
"""

import ast
import logging
import pickle
import struct
import sys
import types
import unittest
from array import array
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec

# CI registration is parsed statically; direct CPU execution avoids sglang imports.
if "sglang.test.ci.ci_register" in sys.modules:
    from sglang.test.ci.ci_register import register_cpu_ci
else:

    def register_cpu_ci(**kwargs):
        """No-op when the serving package has not been imported."""


register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4]
_MANAGERS = _ROOT / "python/sglang/srt/managers"


def load_nodes(module_name, path, names, namespace, owner=None):
    """Execute selected production definitions with explicit dependency fixtures."""
    tree = ast.parse(path.read_text())
    body = tree.body
    if owner is not None:
        body = next(
            node
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == owner
        ).body
    selected = [
        node
        for node in body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    if {node.name for node in selected} != set(names):
        raise AssertionError(f"Missing production definitions in {path}: {names}")
    module = types.ModuleType(module_name)
    module.__dict__.update(namespace)
    sys.modules[module_name] = module
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), module.__dict__)
    return module


IO = load_nodes(
    "_cache_input_metadata_io",
    _MANAGERS / "io_struct.py",
    {
        "BaseReq",
        "BaseBatchReq",
        "TokenizedGenerateReqInput",
        "BatchTokenIDOutput",
        "BatchStrOutput",
        "BatchEmbeddingOutput",
    },
    {"msgspec": msgspec},
)


class Mode(Enum):
    NULL = "null"
    DECODE = "decode"
    PREFILL = "prefill"


OUTPUT = load_nodes(
    "_cache_input_metadata_output",
    _MANAGERS / "scheduler_components/output_streamer.py",
    {"_GenerationStreamAccumulator"},
    {
        "dataclass": dataclass,
        "field": field,
        "BatchTokenIDOutput": IO.BatchTokenIDOutput,
        "DisaggregationMode": Mode,
        "wrap_as_pickle": lambda value: value,
        "compute_weight_version_spans": lambda *args, **kwargs: None,
    },
)
ROUTING = load_nodes(
    "_cache_input_metadata_routing",
    _MANAGERS / "multi_tokenizer_mixin.py",
    {"_extract_field_by_indices", "_handle_output_by_indices"},
    {
        "BatchTokenIDOutput": IO.BatchTokenIDOutput,
        "BatchStrOutput": IO.BatchStrOutput,
        "BatchEmbeddingOutput": IO.BatchEmbeddingOutput,
        "unwrap_from_pickle": lambda value: value,
        "wrap_as_pickle": lambda value: value,
    },
)
DETOK = load_nodes(
    "_cache_input_metadata_detok",
    _MANAGERS / "detokenizer_manager.py",
    {"handle_batch_token_id_out"},
    {"BatchStrOutput": IO.BatchStrOutput, "is_beam_search_batch": lambda value: False},
    owner="DetokenizerManager",
)
_MSGPACK_SOURCE = _ROOT / "python/sglang/srt/utils/msgpack_utils.py"
_WIRE_CONSTANTS = {
    node.targets[0].id: ast.literal_eval(node.value)
    for node in ast.parse(_MSGPACK_SOURCE.read_text()).body
    if isinstance(node, ast.Assign)
    and isinstance(node.targets[0], ast.Name)
    and node.targets[0].id.startswith("_MSGPACK_EXT_")
}
WIRE = load_nodes(
    "_cache_input_metadata_wire",
    _MSGPACK_SOURCE,
    {"enc_hook", "ext_hook", "_pack_buffer_ext", "_unpack_buffer_ext"},
    {
        "msgspec": msgspec,
        "array": array,
        "_MSGPACK_BUFFER_METADATA_SIZE": struct.Struct(">I"),
        **_WIRE_CONSTANTS,
    },
)


def metadata(value=1):
    return {"input_ids": array("q", [value] * 257), "media_spans": [(0, 255, "a" * 64)]}


class FakeReq:
    def __init__(
        self,
        rid="mm",
        output_ids=(10,),
        *,
        cache=None,
        finished=False,
        reason="stop",
        **kwargs,
    ):
        self.rid = rid
        self.http_worker_ipc = "http-worker"
        self.origin_input_ids = array("q", [1] * 257)
        self.output_ids = array("q", output_ids)
        self._finished = finished
        self.finished_reason = (
            SimpleNamespace(to_json=lambda: {"type": reason}) if finished else None
        )
        self.finished_output = False
        self.finished_len = None
        self.beam_group = None
        self.stream = True
        self.sampling_params = SimpleNamespace(
            stream_interval=None,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            no_stop_trim=False,
        )
        self.send_token_offset = 0
        self.send_output_token_logprobs_offset = 0
        self.send_decode_id_offset = 0
        self.decoded_text = ""
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.retraction_count = 0
        self.time_stats = None
        self.mm_image_tokens = self.mm_audio_tokens = self.mm_video_tokens = 0
        self.multimodal_inputs = None
        self.customized_info = None
        self.weight_version_events = []
        self.cache_input_metadata = cache
        self.cache_input_metadata_sent = False
        self.stop_prefix = False

    @property
    def output_ids_through_stop(self):
        return self.output_ids

    def finished(self):
        return self._finished

    def init_incremental_detokenize(self):
        return self.output_ids, 0

    def check_match_stop_str_prefix(self):
        return self.stop_prefix


def accumulator(**changes):
    args = dict(
        return_logprob=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        disaggregation_mode=Mode.NULL,
        default_stream_interval=1,
        default_force_stream_interval=4,
        get_cached_tokens_details=lambda req: None,
        current_weight_version=None,
    )
    args.update(changes)
    return OUTPUT._GenerationStreamAccumulator(**args)


def emit(reqs, **changes):
    acc = accumulator(**changes)
    for req in reqs:
        acc.accept(req=req)
    return acc.to_payload(dp_rank=0, is_idle_batch=False)


def required_struct(cls, **overrides):
    values = {
        name: None
        for name, default in zip(cls.__struct_fields__, cls.__struct_defaults__)
        if default is msgspec.NODEFAULT
    }
    values.update(overrides)
    return cls(**values)


class TestCacheInputOutput(unittest.TestCase):
    def test_tokenized_request_defaults_to_no_snapshot(self):
        request = required_struct(IO.TokenizedGenerateReqInput)
        self.assertFalse(request.return_cache_input)
        self.assertEqual(
            IO.TokenizedGenerateReqInput.__struct_fields__[-1], "return_cache_input"
        )

    def test_first_success_sends_once_and_releases_req_snapshot(self):
        snapshot = metadata()
        req = FakeReq(cache=snapshot)
        first = emit([req])
        self.assertIs(first.cache_input_metadata[0], snapshot)
        self.assertTrue(req.cache_input_metadata_sent)
        self.assertIsNone(req.cache_input_metadata)
        req.output_ids.append(11)
        second = emit([req])
        self.assertIsNone(second.cache_input_metadata)
        self.assertEqual(first.prompt_tokens, second.prompt_tokens)

    def test_stream_interval_does_not_consume_unsent_snapshot(self):
        req = FakeReq(cache=metadata(), output_ids=(10, 11))
        req.sampling_params.stream_interval = 2
        self.assertIsNone(emit([req]))
        self.assertFalse(req.cache_input_metadata_sent)
        req.output_ids.append(12)
        self.assertIsNotNone(emit([req]).cache_input_metadata)

    def test_stop_prefix_does_not_consume_unsent_snapshot(self):
        req = FakeReq(cache=metadata())
        req.stop_prefix = True
        self.assertIsNone(emit([req]))
        self.assertFalse(req.cache_input_metadata_sent)
        req.stop_prefix = False
        self.assertIsNotNone(emit([req]).cache_input_metadata)

    def test_nonstream_force_interval_sends_once(self):
        req = FakeReq(cache=metadata())
        req.stream = False
        self.assertIsNone(emit([req]))
        req.output_ids.extend((11, 12, 13))
        self.assertIsNotNone(emit([req]).cache_input_metadata)
        req.output_ids.extend((14, 15, 16, 17))
        self.assertIsNone(emit([req]).cache_input_metadata)

    def test_finished_eos_with_no_visible_text_still_sends(self):
        req = FakeReq(cache=metadata(), finished=True)
        req.multimodal_inputs = None  # final cleanup cannot discard the frozen copy
        self.assertIsNotNone(emit([req]).cache_input_metadata)
        self.assertTrue(req.finished_output)

    def test_zero_tokens_does_not_send_or_consume_metadata(self):
        for finished in (False, True):
            with self.subTest(finished=finished):
                req = FakeReq(cache=metadata(), output_ids=(), finished=finished)
                payload = emit([req])
                self.assertIsNone(payload.cache_input_metadata)
                self.assertFalse(req.cache_input_metadata_sent)

    def test_abort_and_error_never_send(self):
        for reason in ("abort", "error"):
            with self.subTest(reason=reason):
                req = FakeReq(cache=metadata(), finished=True, reason=reason)
                payload = emit([req])
                self.assertIsNone(payload.cache_input_metadata)
                self.assertFalse(req.cache_input_metadata_sent)

    def test_normal_batch_keeps_field_none(self):
        payload = emit([FakeReq("text-a"), FakeReq("text-b")])
        self.assertIsNone(payload.cache_input_metadata)
        self.assertEqual(payload.rids, ["text-a", "text-b"])

    def test_mixed_batch_and_metadata_error_remain_index_aligned(self):
        error = {"error": "invalid_multimodal_metadata"}
        payload = emit(
            [
                FakeReq("text"),
                FakeReq("mm", cache=metadata()),
                FakeReq("bad-mm", cache=error),
            ]
        )
        self.assertIsNone(payload.cache_input_metadata[0])
        self.assertIs(payload.cache_input_metadata[2], error)
        self.assertEqual(payload.rids, ["text", "mm", "bad-mm"])

    def test_skip_tokenizer_path_preserves_snapshot_in_token_output(self):
        snapshot = metadata()
        payload = emit([FakeReq(cache=snapshot)])
        # skip_tokenizer_init consumes this exact BatchTokenIDOutput directly.
        transferred = pickle.loads(pickle.dumps(payload))
        self.assertIsInstance(transferred, IO.BatchTokenIDOutput)
        self.assertEqual(transferred.cache_input_metadata, [snapshot])
        self.assertEqual(transferred.cache_input_metadata[0]["input_ids"].typecode, "q")
        self.assertNotIn("api_key", transferred.cache_input_metadata[0])

    def test_msgpack_hooks_keep_array_and_span_values(self):
        snapshot = metadata()
        encoded = msgspec.msgpack.encode(snapshot, enc_hook=WIRE.enc_hook)
        decoded = msgspec.msgpack.decode(encoded, ext_hook=WIRE.ext_hook)
        self.assertEqual(decoded["input_ids"], snapshot["input_ids"])
        self.assertEqual(decoded["input_ids"].typecode, "q")
        # Untyped dict entries use ordinary MessagePack arrays for tuples.
        self.assertEqual(decoded["media_spans"], [[0, 255, "a" * 64]])

    def test_detokenizer_and_both_output_slicers_preserve_metadata(self):
        snapshots = [None, metadata(2), metadata(3)]
        payload = emit(
            [FakeReq(str(i), cache=snapshot) for i, snapshot in enumerate(snapshots)]
        )
        detok = SimpleNamespace(
            _decode_batch_token_id_output=lambda recv: ["decoded"] * len(recv.rids),
            _b64_encode_per_request=lambda values: values,
        )
        decoded = DETOK.handle_batch_token_id_out(detok, payload)
        self.assertIs(decoded.cache_input_metadata, payload.cache_input_metadata)
        for output in (payload, decoded):
            with self.subTest(output=type(output).__name__):
                sliced = ROUTING._handle_output_by_indices(output, [2, 0])
                self.assertEqual(sliced.rids, ["2", "0"])
                self.assertEqual(sliced.cache_input_metadata, [snapshots[2], None])
                self.assertEqual(sliced.prompt_tokens, [257, 257])

    def test_detokenizer_keeps_none_on_ordinary_batch(self):
        payload = emit([FakeReq()])
        detok = SimpleNamespace(
            _decode_batch_token_id_output=lambda recv: ["decoded"],
            _b64_encode_per_request=lambda values: values,
        )
        decoded = DETOK.handle_batch_token_id_out(detok, payload)
        self.assertIsNone(decoded.cache_input_metadata)
        self.assertIsNone(
            ROUTING._handle_output_by_indices(decoded, [0]).cache_input_metadata
        )


class SchedulerReq(FakeReq):
    def __init__(self, rid, text, ids, sampling_params, **kwargs):
        super().__init__(rid=rid, output_ids=())
        self.origin_input_ids = array("q", ids)
        self.sampling_params = sampling_params
        self.return_logprob = kwargs.get("return_logprob", False)
        self.return_sampling_mask = kwargs.get("return_sampling_mask", False)
        self.logprob_start_len = -1
        self.is_prefill_only = False
        self.return_routed_experts = kwargs.get("return_routed_experts", False)
        self.routed_experts_start_len = kwargs.get("routed_experts_start_len", 0)

    def set_finish_with_abort(self, message, **kwargs):
        self.abort_message = message

    def extend_image_inputs(self, mm_inputs):
        self.multimodal_inputs = mm_inputs


class TestCacheInputScheduler(unittest.TestCase):
    def setUp(self):
        self.freeze = Mock(
            side_effect=lambda ids, mm: {
                "input_ids": array("q", ids),
                "media_spans": [],
            }
        )
        self.validate = Mock(return_value=None)
        module = load_nodes(
            "_cache_input_metadata_scheduler",
            _MANAGERS / "scheduler.py",
            {"handle_generate_request"},
            {
                "Req": SchedulerReq,
                "BeamCoordinator": SimpleNamespace(request_beam_width=lambda req: 1),
                "DisaggregationMode": Mode,
                "get_disagg": lambda: SimpleNamespace(
                    disaggregation_bootstrap_port=8998
                ),
                "get_parallel": lambda: SimpleNamespace(pp_rank=0),
                "get_serving": lambda: SimpleNamespace(allow_auto_truncate=True),
                "get_device": lambda: SimpleNamespace(mlx_enable_sampling=False),
                "freeze_cache_input_metadata": self.freeze,
                "validate_input_length": self.validate,
                "duplicate_pad_mm_input_ids": lambda ids, mm, fn: fn(ids, mm),
                "logger": logging.getLogger(__name__),
                "HTTPStatus": HTTPStatus,
            },
            owner="Scheduler",
        )
        self.handle = module.handle_generate_request
        self.queued = []
        self.mm = SimpleNamespace(mm_items=[])
        self.scheduler = SimpleNamespace(
            model_config=SimpleNamespace(vocab_size=256, hf_eos_token_id=None),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            dllm_config=None,
            tokenizer=None,
            disaggregation_mode=Mode.NULL,
            spec_algorithm=SimpleNamespace(
                is_dflash_family=lambda: False, is_uno=lambda: False
            ),
            max_req_input_len=100,
            tree_cache=SimpleNamespace(
                cache_controller=SimpleNamespace(pp_prefetch_command_group=None)
            ),
            grammar_manager=SimpleNamespace(process_req_with_grammar=lambda req: False),
            init_req_max_new_tokens=lambda req: None,
            _add_request_to_queue=self.queued.append,
            _maybe_namespace_elastic_radix_cache=lambda req: None,
            validate_dllm_request=lambda req: None,
            _get_multimodal_inputs=lambda mm: self.mm,
            _try_apply_padded_mm_input_ids=lambda *args: False,
            pad_input_ids_func=lambda ids, mm: array("q", [10, 9000001, 9000001, 20]),
            _maybe_compute_mrope_positions=lambda req: None,
        )
        module.SessionController = SimpleNamespace(adjust_mm_offsets=lambda *args: None)

    def request(self, **changes):
        args = dict(
            rid="mm-scheduler",
            input_text="prompt",
            input_ids=array("q", [10, 99, 20]),
            input_embeds=None,
            mm_inputs=self.mm,
            sampling_params=SimpleNamespace(),
            return_cache_input=True,
            return_logprob=False,
            return_sampling_mask=False,
            logprob_start_len=-1,
            return_routed_experts=False,
            routed_experts_start_len=0,
        )
        args.update(changes)
        return required_struct(IO.TokenizedGenerateReqInput, **args)

    def test_freeze_sees_expanded_then_truncated_ids_once(self):
        def truncate(req, max_len, allow):
            self.assertEqual(list(req.origin_input_ids), [10, 9000001, 9000001, 20])
            req.origin_input_ids = req.origin_input_ids[:3]
            return None

        self.validate.side_effect = truncate
        self.handle(self.scheduler, self.request())
        self.freeze.assert_called_once()
        self.assertEqual(list(self.freeze.call_args.args[0]), [10, 9000001, 9000001])
        self.assertIs(self.freeze.call_args.args[1], self.mm)
        self.assertEqual(
            self.queued[0].cache_input_metadata["input_ids"],
            array("q", [10, 9000001, 9000001]),
        )
        self.assertFalse(self.queued[0].cache_input_metadata_sent)

    def test_no_opt_in_does_not_freeze(self):
        self.handle(self.scheduler, self.request(return_cache_input=False))
        self.freeze.assert_not_called()
        self.assertIsNone(self.queued[0].cache_input_metadata)

    def test_invalid_length_does_not_freeze(self):
        self.validate.return_value = "too long"
        self.handle(self.scheduler, self.request())
        self.freeze.assert_not_called()
        self.assertEqual(self.queued[0].abort_message, "too long")

    def test_safe_metadata_error_is_kept_for_manager_fallback(self):
        self.freeze.side_effect = None
        self.freeze.return_value = {"error": "invalid_multimodal_metadata"}
        self.handle(self.scheduler, self.request())
        self.assertEqual(self.queued[0].cache_input_metadata, self.freeze.return_value)


if __name__ == "__main__":
    unittest.main()
