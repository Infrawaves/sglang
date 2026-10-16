"""Dependency-light tests for TokenizerManager's cache output boundary.

Extract the actual helper methods from the manager so these tests exercise its
request lifecycle without importing CUDA, torch, tokenizer, or model packages.
The page plan and usage types are loaded directly from their source module.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import sys
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

# CI registration is parsed statically; avoid importing heavyweight sglang.
if "sglang.test.ci.ci_register" in sys.modules:
    from sglang.test.ci.ci_register import register_cpu_ci
else:

    def register_cpu_ci(**kwargs):
        """Runtime no-op when testing without the serving package installed."""


register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[3]
_SRT = _ROOT / "python" / "sglang" / "srt"


def _load_page_cache():
    name = "_cache_tokenizer_test_page_cache"
    spec = importlib.util.spec_from_file_location(
        name, _SRT / "cache_accounting" / "page_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_CACHE = _load_page_cache()


def _load_disaggregation_mode():
    path = _SRT / "disaggregation/utils.py"
    tree = ast.parse(path.read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DisaggregationMode"
    )
    namespace = {"Enum": Enum}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["DisaggregationMode"]


_DISAGG_MODE = _load_disaggregation_mode()


def _load_manager_helpers(*, serving=None, schedule=None):
    path = _SRT / "managers" / "tokenizer_manager.py"
    source = ast.parse(path.read_text())
    manager = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "TokenizerManager"
    )
    names = {
        "_start_cache_accounting",
        "_attach_cache_usage",
        "_prepare_cache_plan",
        "_validate_cache_request",
    }
    methods = [node for node in manager.body if getattr(node, "name", None) in names]
    found = {node.name for node in methods}
    if found != names:
        raise AssertionError(f"Missing cache manager helpers: {names - found}")
    extracted = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            ast.ClassDef(
                name="ManagerHelpers",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    namespace = {
        "asyncio": asyncio,
        "json": json,
        "get_serving": lambda: serving,
        "get_schedule": lambda: schedule,
        "build_page_cache_plan": _CACHE.build_page_cache_plan,
        "CacheAccountingError": _CACHE.CacheAccountingError,
        "CacheUsage": _CACHE.CacheUsage,
        "DisaggregationMode": _DISAGG_MODE,
    }
    exec(compile(ast.fix_missing_locations(extracted), str(path), "exec"), namespace)
    return namespace["ManagerHelpers"]


class TestTokenizerCachePreparation(unittest.TestCase):
    def setUp(self):
        self.serving = SimpleNamespace(
            enable_cache_accounting=True,
            cache_namespace="deployment-v1",
            tokenizer_path="tokenizer-v1",
            chat_template=None,
            hf_chat_template_name=None,
        )
        self.schedule = SimpleNamespace(page_size=4)
        self.manager = _load_manager_helpers(
            serving=self.serving, schedule=self.schedule
        )()
        self.manager.server_args = SimpleNamespace(page_size=1)
        self.manager.model_config = SimpleNamespace(is_multimodal=False)
        self.manager.model_path = "model-v1"
        self.manager.is_generation = True
        self.manager.disaggregation_mode = _DISAGG_MODE.NULL
        self.manager.tokenizer = SimpleNamespace(chat_template="template-v1")
        self.weights = "weights-v1"
        self.manager.config_value = lambda name: self.weights
        self.state = SimpleNamespace(cache_plan=None)
        self.manager.rid_to_state = {"r1": self.state}
        self.obj = SimpleNamespace(
            rid="r1",
            cache_ttl_seconds=300,
            api_key="verified-api-key-hash",
            cache_required=True,
            sampling_params={"max_new_tokens": 4},
            text="text deliberately differs from final token IDs",
            input_ids=None,
            input_embeds=None,
            session_params=None,
            positional_embed_overrides=None,
            lora_path=None,
            lora_id=None,
            contains_mm_input=lambda: False,
        )

    def prepare(self, input_ids=None):
        if input_ids is None:
            input_ids = list(range(10))
        self.manager._prepare_cache_plan(self.obj, input_ids)
        return self.state.cache_plan

    def test_resolved_runtime_page_size_and_final_ids_determine_boundaries(self):
        plan = self.prepare()
        self.assertEqual(plan.page_size, 4)
        self.assertEqual(plan.prompt_tokens, 10)
        self.assertEqual(len(plan.page_keys), 2)
        self.assertEqual(plan.prompt_tokens % plan.page_size, 2)
        # Raw construction-time page_size=1 must not affect the result.
        self.assertEqual(self.manager.server_args.page_size, 1)

    def test_prefill_ignores_cache_options_before_configuration_and_identity(self):
        self.manager.disaggregation_mode = _DISAGG_MODE.PREFILL
        self.serving.enable_cache_accounting = False
        self.obj.api_key = None
        self.obj.cache_ttl_seconds = 3600
        self.assertIsNone(self.prepare())
        self.assertIsNone(self.obj.cache_ttl_seconds)
        self.assertIsNone(self.obj.api_key)
        self.assertIsNone(self.state.cache_plan)

    def test_decode_uses_selected_ttl_and_runtime_page_size(self):
        self.manager.disaggregation_mode = _DISAGG_MODE.DECODE
        for ttl in (300, 3600):
            with self.subTest(ttl=ttl):
                self.obj.cache_ttl_seconds = ttl
                plan = self.prepare()
                self.assertEqual(plan.ttl_seconds, ttl)
                self.assertEqual(plan.page_size, 4)
                self.assertEqual(len(plan.page_keys), 2)

    def test_engine_page_size_one_accepts_more_than_4096_complete_pages(self):
        self.schedule.page_size = 1
        plan = self.prepare(list(range(4097)))
        self.assertEqual(plan.page_size, 1)
        self.assertEqual(plan.prompt_tokens, 4097)
        self.assertEqual(len(plan.page_keys), 4097)

    def test_changing_resolved_engine_page_size_changes_page_boundaries(self):
        self.schedule.page_size = 8
        plan = self.prepare(list(range(19)))
        self.assertEqual(plan.page_size, 8)
        self.assertEqual(len(plan.page_keys), 2)
        self.assertEqual(plan.prompt_tokens % plan.page_size, 3)

    def test_content_page_keys_are_stable_across_requests(self):
        first = self.prepare()
        self.obj.rid = "r2"
        second_state = SimpleNamespace(cache_plan=None)
        self.manager.rid_to_state["r2"] = second_state
        self.manager._prepare_cache_plan(self.obj, list(range(10)))
        second = second_state.cache_plan
        self.assertEqual(first.page_keys, second.page_keys)

    def test_namespace_and_input_identity_are_frozen_before_output(self):
        ids = list(range(10))
        first = self.prepare(ids)
        original_keys = first.page_keys
        self.weights = "weights-v2"
        self.serving.cache_namespace = "deployment-v2"
        self.manager.tokenizer.chat_template = "template-v2"
        ids[0] = 99
        self.assertEqual(first.page_keys, original_keys)
        second = self.prepare(list(range(10)))
        self.assertNotEqual(first.page_keys, second.page_keys)

    def test_subpage_input_does_not_get_a_partial_page_key(self):
        plan = self.prepare([1, 2, 3])
        self.assertEqual(plan.page_keys, ())
        self.assertEqual(plan.prompt_tokens, 3)

    def test_ttl_selection_enters_cache_identity(self):
        short = self.prepare()
        self.obj.cache_ttl_seconds = 3600
        long = self.prepare()
        self.assertEqual((short.ttl_seconds, long.ttl_seconds), (300, 3600))
        self.assertNotEqual(short.page_keys, long.page_keys)

    def test_missing_input_ids_cannot_produce_a_plan(self):
        with self.assertRaisesRegex(ValueError, "input token IDs"):
            self.manager._prepare_cache_plan(self.obj, None)
        self.assertIsNone(self.state.cache_plan)

    def test_internal_adapter_id_is_rejected_as_well_as_adapter_path(self):
        self.obj.lora_id = "already-loaded-adapter"
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertIsNone(self.state.cache_plan)

    def test_implicit_unsupported_request_skips_accounting(self):
        self.obj.cache_required = False
        for name, value in (
            ("lora_id", "adapter"),
            ("session_params", {"id": "s"}),
            ("sampling_params", {"max_new_tokens": 4, "n": 2}),
            ("contains_mm_input", lambda: True),
        ):
            with self.subTest(name=name):
                original = getattr(self.obj, name)
                setattr(self.obj, name, value)
                self.obj.cache_ttl_seconds = 300
                self.obj.api_key = "verified-api-key-hash"
                self.assertIsNone(self.prepare())
                self.assertIsNone(self.obj.cache_ttl_seconds)
                self.assertIsNone(self.obj.api_key)
                setattr(self.obj, name, original)

    def test_multimodal_model_accepts_text_only_request(self):
        self.manager.model_config.is_multimodal = True
        self.assertIsNotNone(self.prepare())


class TestTokenizerCache(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = _load_manager_helpers()()
        self.plan = _CACHE.build_page_cache_plan(
            input_ids=list(range(10)),
            api_key="verified-api-key-hash",
            namespace="weights-v1/tokenizer-v1",
            page_size=4,
            ttl_seconds=300,
        )
        self.usage = _CACHE.CacheUsage(
            read_tokens=4, creation_5m_tokens=4, normal_input_tokens=2
        )
        self.account = AsyncMock(return_value=self.usage)
        self.manager._cache_store = SimpleNamespace(account=self.account)
        self.state = SimpleNamespace(
            cache_plan=self.plan,
            cache_task=None,
            cache_usage=None,
        )

    def start(self, *, prompt_tokens=10, completion_tokens=1, finish_reason=None):
        self.manager._start_cache_accounting(
            self.state, prompt_tokens, completion_tokens, finish_reason
        )

    async def attach(self, *, text="", cached_tokens=7):
        out = {
            "text": text,
            "meta_info": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "cached_tokens": cached_tokens,
            },
        }
        await self.manager._attach_cache_usage(self.state, out)
        return out

    async def test_success_uses_one_task_and_one_decision_across_chunks(self):
        self.start()
        task = self.state.cache_task
        self.assertIsInstance(task, asyncio.Task)
        first = await self.attach(text="a")
        self.start(completion_tokens=2)
        second = await self.attach(text="b")
        self.assertIs(self.state.cache_task, task)
        self.account.assert_awaited_once_with(self.plan)
        self.assertEqual(first["meta_info"]["cache_usage"], self.usage.to_dict())
        self.assertEqual(
            second["meta_info"]["cache_usage"],
            first["meta_info"]["cache_usage"],
        )

    async def test_eos_with_no_visible_text_still_accounts_generated_token(self):
        self.start(finish_reason={"type": "stop", "matched": 2})
        out = await self.attach(text="")
        self.account.assert_awaited_once_with(self.plan)
        self.assertEqual(out["meta_info"]["cache_usage"], self.usage.to_dict())

    async def test_no_generated_token_does_not_touch_store(self):
        self.start(completion_tokens=0, finish_reason={"type": "length"})
        await asyncio.sleep(0)
        self.assertIsNone(self.state.cache_task)
        self.account.assert_not_awaited()

    async def test_abort_or_error_does_not_create_cache(self):
        for reason in ("abort", "error"):
            with self.subTest(reason=reason):
                self.start(finish_reason={"type": reason, "message": "failed"})
                await asyncio.sleep(0)
                self.assertIsNone(self.state.cache_task)
                self.account.assert_not_awaited()

    async def test_disabled_request_never_calls_store_or_adds_usage(self):
        self.state.cache_plan = None
        self.start()
        out = await self.attach()
        self.account.assert_not_awaited()
        self.assertIsNone(self.state.cache_task)
        self.assertNotIn("cache_usage", out["meta_info"])

    async def test_redis_failure_remains_failed_and_does_not_retry_on_next_chunk(self):
        self.account.side_effect = _CACHE.CacheAccountingError(
            "unconfirmed Redis result"
        )
        self.start()
        task = self.state.cache_task
        for _ in range(2):
            with self.assertRaisesRegex(_CACHE.CacheAccountingError, "unconfirmed"):
                await self.attach()
            self.start(completion_tokens=2)
        self.assertIs(self.state.cache_task, task)
        self.account.assert_awaited_once_with(self.plan)

    async def test_prompt_count_mismatch_is_strict_and_never_writes(self):
        self.start(prompt_tokens=9)
        with self.assertRaises(_CACHE.CacheAccountingError):
            await self.attach()
        self.account.assert_not_awaited()

    async def test_tail_partition_and_physical_cache_metric_are_preserved(self):
        self.start()
        out = await self.attach(cached_tokens=7)
        cache = out["meta_info"]["cache_usage"]
        self.assertEqual(sum(cache.values()), self.plan.prompt_tokens)
        self.assertEqual(cache["normal_input_tokens"], 2)
        self.assertEqual(out["meta_info"]["cached_tokens"], 7)

    async def test_other_request_keeps_its_own_accounting_task(self):
        other_plan = _CACHE.build_page_cache_plan(
            input_ids=list(range(10)),
            api_key="verified-api-key-hash",
            namespace="weights-v1/tokenizer-v1",
            page_size=4,
            ttl_seconds=300,
        )
        other_state = SimpleNamespace(
            cache_plan=other_plan,
            cache_task=None,
            cache_usage=None,
        )
        self.start()
        self.manager._start_cache_accounting(other_state, 10, 1, None)
        self.assertIsNot(self.state.cache_task, other_state.cache_task)
        await asyncio.gather(self.state.cache_task, other_state.cache_task)
        self.assertEqual(self.account.await_count, 2)
        self.account.assert_any_await(self.plan)
        self.account.assert_any_await(other_plan)

    async def test_cancelled_client_wait_does_not_cancel_started_accounting(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def delayed_account(plan):
            started.set()
            await release.wait()
            return self.usage

        self.account.side_effect = delayed_account
        self.start()
        await started.wait()
        response_wait = asyncio.create_task(self.attach())
        await asyncio.sleep(0)
        response_wait.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await response_wait
        self.assertFalse(self.state.cache_task.cancelled())
        release.set()
        self.assertEqual(await self.state.cache_task, self.usage)
        self.account.assert_awaited_once_with(self.plan)


if __name__ == "__main__":
    unittest.main()
