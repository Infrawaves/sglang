"""Dependency-light tests for TokenizerManager's cache output boundary.

Extract the actual helper methods from the manager so these tests exercise its
request lifecycle without importing CUDA, torch, tokenizer, or model packages.
The page plan and usage types are loaded directly from their source module.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import importlib.util
import json
import sys
import threading
import time
import unittest
import uuid
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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


def _log_message(call):
    template, *values = call.args
    return template % tuple(values)


class _GenerateReqInput(SimpleNamespace):
    def regenerate_rid(self):
        self.rid = uuid.uuid4().hex
        return self.rid


def _load_manager_helpers(
    *,
    serving=None,
    schedule=None,
    disagg=None,
    cache_store_class=None,
    cache_metrics_class=None,
    cache_logger=None,
    prepare_media=None,
):
    path = _SRT / "managers" / "tokenizer_manager.py"
    source = ast.parse(path.read_text())
    manager = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "TokenizerManager"
    )
    names = {
        "_init_cache_accounting",
        "_start_cache_accounting",
        "_attach_cache_usage",
        "_prepare_cache_plan",
        "_validate_cache_request",
        "_resolve_cache_identity",
        "_init_req_state",
        "_handle_batch_request",
        "_collect_batch_responses",
        "_prepare_cache_media_digests",
        "_start_cache_renewal",
        "_finish_cache_accounting",
        "close_cache_accounting",
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
        "copy": copy,
        "json": json,
        "time": time,
        "os": SimpleNamespace(environ={}),
        "logger": cache_logger if cache_logger is not None else Mock(),
        "get_serving": lambda: serving,
        "get_schedule": lambda: schedule,
        "get_disagg": lambda: disagg,
        "RedisPageCacheStore": (
            cache_store_class if cache_store_class is not None else Mock()
        ),
        "CacheAccountingMetrics": (
            cache_metrics_class if cache_metrics_class is not None else Mock()
        ),
        "build_page_cache_plan": _CACHE.build_page_cache_plan,
        "CACHE_BLOCK_SIZE": _CACHE.CACHE_BLOCK_SIZE,
        "CacheAccountingError": _CACHE.CacheAccountingError,
        "CacheUsage": _CACHE.CacheUsage,
        "CacheMediaSpan": _CACHE.CacheMediaSpan,
        "prepare_media_digests": prepare_media if prepare_media is not None else Mock(),
        "DisaggregationMode": _DISAGG_MODE,
        "GenerateReqInput": _GenerateReqInput,
        "APIServerReqTimeStats": lambda **kwargs: SimpleNamespace(
            set_created_time=Mock(), set_finished_time=Mock()
        ),
        "ReqState": lambda out_list, finished, event, obj, time_stats: SimpleNamespace(
            out_list=out_list,
            finished=finished,
            event=event,
            obj=obj,
            time_stats=time_stats,
            cache_plan=None,
            cache_plan_context=None,
            cache_physical_ttl_seconds=None,
            cache_metrics_parent=None,
            cache_task=None,
            cache_usage=None,
            cache_finished_event=None,
            cache_renewal_task=None,
        ),
    }
    exec(compile(ast.fix_missing_locations(extracted), str(path), "exec"), namespace)
    return namespace["ManagerHelpers"]


class TestTokenizerCacheInitialization(unittest.TestCase):
    def test_enabled_startup_logs_responsibility_without_redis_details(self):
        for role, logged_role, action in (
            ("null", "standalone", "account"),
            ("decode", "decode", "account"),
            ("prefill", "prefill", "skip_prefill"),
        ):
            with self.subTest(role=role):
                logger = Mock()
                store_class = Mock()
                cfg = SimpleNamespace(
                    enable_cache_accounting=True,
                    cache_redis_url="redis://private-redis-host:6379/0",
                    cache_timeout=0.2,
                )
                manager = _load_manager_helpers(
                    serving=cfg,
                    disagg=SimpleNamespace(disaggregation_mode=role),
                    cache_store_class=store_class,
                    cache_logger=logger,
                )()
                manager.enable_metrics = False
                manager._init_cache_accounting()
                self.assertEqual(manager._cache_tasks, set())
                if role == "prefill":
                    store_class.assert_not_called()
                    self.assertIsNone(manager._cache_store)
                else:
                    store_class.assert_called_once_with(
                        cfg.cache_redis_url,
                        timeout_seconds=0.2,
                        username=None,
                        password=None,
                        metrics=None,
                    )
                    self.assertIs(manager._cache_store, store_class.return_value)
                logger.info.assert_called_once()
                self.assertEqual(
                    _log_message(logger.info.call_args),
                    "Cache accounting startup enabled=True "
                    f"role={logged_role} action={action} block_size=256 "
                    "metrics_enabled=False",
                )
                self.assertNotIn(
                    cfg.cache_redis_url, _log_message(logger.info.call_args)
                )
                logger.warning.assert_not_called()

    def test_disabled_startup_does_not_initialize_store_or_log(self):
        logger = Mock()
        store_class = Mock()
        manager = _load_manager_helpers(
            serving=SimpleNamespace(enable_cache_accounting=False),
            disagg=SimpleNamespace(disaggregation_mode="decode"),
            cache_store_class=store_class,
            cache_logger=logger,
        )()
        manager.enable_metrics = True
        manager._init_cache_accounting()
        self.assertIsNone(manager._cache_store)
        self.assertEqual(manager._cache_tasks, set())
        store_class.assert_not_called()
        logger.info.assert_not_called()
        logger.warning.assert_not_called()

    def test_metrics_follow_feature_role_and_enable_metrics_switch(self):
        for role, enabled in (
            ("decode", True),
            ("null", True),
            ("prefill", True),
            ("decode", False),
        ):
            with self.subTest(role=role, enabled=enabled):
                metrics_class = Mock()
                store_class = Mock()
                manager = _load_manager_helpers(
                    serving=SimpleNamespace(
                        enable_cache_accounting=True,
                        cache_redis_url="redis://unused",
                        cache_timeout=0.2,
                    ),
                    disagg=SimpleNamespace(disaggregation_mode=role),
                    cache_store_class=store_class,
                    cache_metrics_class=metrics_class,
                )()
                manager.enable_metrics = enabled
                manager._init_cache_accounting()
                if enabled and role != "prefill":
                    metrics_class.assert_called_once_with(
                        role="standalone" if role == "null" else role
                    )
                    self.assertIs(manager._cache_metrics, metrics_class.return_value)
                    self.assertIs(
                        store_class.call_args.kwargs["metrics"], manager._cache_metrics
                    )
                else:
                    metrics_class.assert_not_called()
                    self.assertIsNone(manager._cache_metrics)


class TestTokenizerCacheIdentity(unittest.TestCase):
    def setUp(self):
        self.manager = _load_manager_helpers()()
        self.manager.disaggregation_mode = _DISAGG_MODE.DECODE
        self.obj = _GenerateReqInput(
            cache_ttl_seconds=300, cache_user_id="body-supplied-value"
        )

    def test_adapter_identity_takes_precedence_over_shared_bearer(self):
        request = SimpleNamespace(
            state=SimpleNamespace(
                cache_identity="verified-user-digest", api_key="shared-bearer-digest"
            )
        )
        self.manager._resolve_cache_identity(self.obj, request)
        self.assertEqual(self.obj.cache_user_id, "verified-user-digest")

    def test_adapter_identity_does_not_require_a_bearer_identity(self):
        request = SimpleNamespace(
            state=SimpleNamespace(cache_identity="verified-user-digest")
        )
        self.manager._resolve_cache_identity(self.obj, request)
        self.assertEqual(self.obj.cache_user_id, "verified-user-digest")

    def test_native_generate_body_cannot_supply_trusted_http_identity(self):
        self.manager._resolve_cache_identity(
            self.obj, SimpleNamespace(state=SimpleNamespace())
        )
        self.assertIsNone(self.obj.cache_user_id)
        self.assertEqual(self.obj.cache_ttl_seconds, 300)

    def test_native_generate_ignores_bearer_identity_and_body_value(self):
        request = SimpleNamespace(
            state=SimpleNamespace(api_key="verified-bearer-digest")
        )
        self.manager._resolve_cache_identity(self.obj, request)
        self.assertIsNone(self.obj.cache_user_id)
        self.assertEqual(self.obj.cache_ttl_seconds, 300)

    def test_in_process_request_preserves_existing_digest(self):
        self.manager._resolve_cache_identity(self.obj, None)
        self.assertEqual(self.obj.cache_user_id, "body-supplied-value")

    def test_prefill_clears_cache_options_and_identity(self):
        self.manager.disaggregation_mode = _DISAGG_MODE.PREFILL
        self.manager._resolve_cache_identity(
            self.obj,
            SimpleNamespace(
                state=SimpleNamespace(cache_identity="verified-user-digest")
            ),
        )
        self.assertIsNone(self.obj.cache_user_id)
        self.assertIsNone(self.obj.cache_ttl_seconds)

    def test_batch_subobjects_and_parallel_clones_keep_physical_mode(self):
        class Batch(_GenerateReqInput):
            def __getitem__(self, index):
                return self.children[index]

        self.manager.enable_trace = False
        for ttl in (300, 3600):
            with self.subTest(ttl=ttl):
                self.manager.rid_to_state = {}
                child = _GenerateReqInput(
                    rid="base", received_time=1, return_prompt_token_ids=False
                )
                batch = Batch(
                    rid=["base"],
                    children=[child],
                    is_single=False,
                    bootstrap_room=None,
                    received_time=1,
                    cache_user_id=None,
                    cache_ttl_seconds=ttl,
                    cache_required=True,
                )
                self.manager._init_req_state(batch)
                self.assertIsNone(child.cache_user_id)
                self.assertEqual(child.cache_ttl_seconds, ttl)
                self.assertTrue(child.cache_required)
                self.assertEqual(
                    self.manager.rid_to_state["base"].cache_physical_ttl_seconds, ttl
                )
                # _handle_batch_request makes parallel samples by shallow copy
                # and initializes a new state without tokenizing each again.
                cloned = copy.copy(child)
                cloned.rid = "choice-1"
                self.manager._init_req_state(cloned)
                state = self.manager.rid_to_state[cloned.rid]
                self.assertEqual(state.cache_physical_ttl_seconds, ttl)
                self.assertIsNone(state.cache_plan)


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
            cache_user_id="verified-api-key-hash",
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
            input_ids = list(range(514))
        self.manager._prepare_cache_plan(self.obj, input_ids)
        return self.state.cache_plan

    def test_fixed_256_token_blocks_use_final_input_ids(self):
        plan = self.prepare()
        self.assertEqual(plan.page_size, 256)
        self.assertEqual(plan.prompt_tokens, 514)
        self.assertEqual(len(plan.page_keys), 2)
        self.assertEqual(plan.prompt_tokens % plan.page_size, 2)
        # Neither resolved nor construction-time KV page size affects accounting.
        self.assertEqual(self.schedule.page_size, 4)
        self.assertEqual(self.manager.server_args.page_size, 1)

    def test_prefill_ignores_cache_options_before_configuration_and_identity(self):
        self.manager.disaggregation_mode = _DISAGG_MODE.PREFILL
        self.serving.enable_cache_accounting = False
        self.obj.cache_user_id = None
        self.obj.cache_ttl_seconds = 3600
        self.assertIsNone(self.prepare())
        self.assertIsNone(self.obj.cache_ttl_seconds)
        self.assertIsNone(self.obj.cache_user_id)
        self.assertIsNone(self.state.cache_plan)

    def test_decode_uses_selected_ttl_and_256_token_blocks(self):
        self.manager.disaggregation_mode = _DISAGG_MODE.DECODE
        for ttl in (300, 3600):
            with self.subTest(ttl=ttl):
                self.obj.cache_ttl_seconds = ttl
                plan = self.prepare()
                self.assertEqual(plan.ttl_seconds, ttl)
                self.assertEqual(plan.page_size, 256)
                self.assertEqual(len(plan.page_keys), 2)

    def test_engine_page_size_one_does_not_change_accounting_blocks(self):
        self.schedule.page_size = 1
        plan = self.prepare(list(range(4097)))
        self.assertEqual(plan.page_size, 256)
        self.assertEqual(plan.prompt_tokens, 4097)
        self.assertEqual(len(plan.page_keys), 16)
        self.assertEqual(plan.prompt_tokens % plan.page_size, 1)

    def test_changing_engine_page_size_preserves_accounting_identity(self):
        first = self.prepare()
        self.schedule.page_size = 8
        plan = self.prepare()
        self.assertEqual(plan.page_size, 256)
        self.assertEqual(len(plan.page_keys), 2)
        self.assertEqual(plan.page_keys, first.page_keys)

    def test_content_page_keys_are_stable_across_requests(self):
        first = self.prepare()
        self.obj.rid = "r2"
        second_state = SimpleNamespace(cache_plan=None)
        self.manager.rid_to_state["r2"] = second_state
        self.manager._prepare_cache_plan(self.obj, list(range(514)))
        second = second_state.cache_plan
        self.assertEqual(first.page_keys, second.page_keys)

    def test_namespace_and_input_identity_are_frozen_before_output(self):
        ids = list(range(514))
        first = self.prepare(ids)
        original_keys = first.page_keys
        self.weights = "weights-v2"
        self.serving.cache_namespace = "deployment-v2"
        self.manager.tokenizer.chat_template = "template-v2"
        ids[0] = 99
        self.assertEqual(first.page_keys, original_keys)
        second = self.prepare(list(range(514)))
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

    def test_implicit_unsupported_request_with_identity_skips_accounting(self):
        self.obj.cache_required = False
        for name, value in (
            ("lora_id", "adapter"),
            ("session_params", {"id": "s"}),
            ("sampling_params", {"max_new_tokens": 4, "n": 2}),
        ):
            with self.subTest(name=name):
                original = getattr(self.obj, name)
                setattr(self.obj, name, value)
                self.obj.cache_ttl_seconds = 300
                self.obj.cache_user_id = "verified-api-key-hash"
                self.assertIsNone(self.prepare())
                self.assertIsNone(self.obj.cache_ttl_seconds)
                self.assertIsNone(self.obj.cache_user_id)
                setattr(self.obj, name, original)

    def test_unsupported_request_without_identity_uses_physical_mode(self):
        for name, value in (
            ("lora_id", "adapter"),
            ("session_params", {"id": "s"}),
            ("sampling_params", {"max_new_tokens": 4, "n": 2}),
            ("contains_mm_input", lambda: True),
        ):
            original = getattr(self.obj, name)
            for required in (False, True):
                for identity in (None, "", " "):
                    with self.subTest(name=name, identity=identity, required=required):
                        setattr(self.obj, name, value)
                        self.obj.cache_ttl_seconds = 300
                        self.obj.cache_required = required
                        self.obj.cache_user_id = identity
                        self.assertIsNone(self.prepare())
                        self.assertEqual(self.obj.cache_ttl_seconds, 300)
                        self.assertIsNone(self.obj.cache_user_id)
                        self.assertEqual(self.state.cache_physical_ttl_seconds, 300)
                        self.assertIsNone(self.state.cache_plan)
            setattr(self.obj, name, original)

    def test_missing_identity_never_builds_a_hash_or_requires_input_ids(self):
        builder = Mock(side_effect=AssertionError("must not hash anonymous input"))
        for ttl in (300, 3600):
            with (
                self.subTest(ttl=ttl),
                patch.dict(
                    self.manager._prepare_cache_plan.__globals__,
                    {"build_page_cache_plan": builder},
                ),
            ):
                self.obj.cache_user_id = None
                self.obj.cache_ttl_seconds = ttl
                self.manager._prepare_cache_plan(self.obj, None)
                self.assertEqual(self.state.cache_physical_ttl_seconds, ttl)
                self.assertIsNone(self.state.cache_plan)
        builder.assert_not_called()

    def test_multimodal_model_accepts_text_only_request(self):
        self.manager.model_config.is_multimodal = True
        self.assertIsNotNone(self.prepare())

    def test_multimodal_default_and_explicit_ttl_defer_until_final_input(self):
        self.obj.contains_mm_input = lambda: True
        for role in (_DISAGG_MODE.NULL, _DISAGG_MODE.DECODE):
            for ttl in (300, 3600):
                contexts = []
                for required in (False, True):
                    with self.subTest(role=role, ttl=ttl, required=required):
                        self.manager.disaggregation_mode = role
                        self.obj.cache_ttl_seconds = ttl
                        self.obj.cache_required = required
                        self.state.cache_plan = None
                        self.state.cache_plan_context = None
                        self.manager._prepare_cache_plan(self.obj, [1] * 20)
                        self.assertIsNone(self.state.cache_plan)
                        self.assertEqual(self.obj.cache_ttl_seconds, ttl)
                        self.assertEqual(
                            self.state.cache_plan_context[0], self.obj.cache_user_id
                        )
                        self.assertEqual(self.state.cache_plan_context[2], ttl)
                        contexts.append(self.state.cache_plan_context)
                self.assertEqual(contexts[0], contexts[1])

    def test_multimodal_waits_for_scheduler_when_processor_has_no_ids_yet(self):
        self.obj.contains_mm_input = lambda: True
        self.manager._prepare_cache_plan(self.obj, None)
        self.assertIsNone(self.state.cache_plan)
        self.assertIsNotNone(self.state.cache_plan_context)


class TestTokenizerCache(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.logger = Mock()
        self.manager = _load_manager_helpers(cache_logger=self.logger)()
        self.manager.disaggregation_mode = _DISAGG_MODE.DECODE
        self.metrics = Mock()
        self.manager._cache_metrics = self.metrics
        self.manager._cache_tasks = set()
        self.manager._cache_renewal_tasks = {}
        self.manager._cache_accounting_closing = False
        self.release_events = []
        self.plan = _CACHE.build_page_cache_plan(
            input_ids=list(range(10)),
            cache_user_id="verified-api-key-hash",
            namespace="weights-v1/tokenizer-v1",
            page_size=4,
            ttl_seconds=300,
        )
        self.usage = _CACHE.CacheUsage(ttl_seconds=300, read_tokens=4, write_tokens=4)
        self.account = AsyncMock(return_value=self.usage)
        self.renew = AsyncMock()
        self.close_store = AsyncMock()
        self.manager._cache_store = SimpleNamespace(
            account=self.account, renew=self.renew, aclose=self.close_store
        )
        self.state = SimpleNamespace(
            obj=SimpleNamespace(rid="cache-request-1"),
            cache_plan=self.plan,
            cache_task=None,
            cache_usage=None,
        )

    async def asyncTearDown(self):
        for event in self.release_events:
            event.set()
        await asyncio.wait_for(self.manager.close_cache_accounting(), timeout=1)

    async def test_multimodal_feature_hash_runs_off_the_event_loop(self):
        threads = []
        prepared = []

        def prepare(mm_inputs):
            threads.append(threading.get_ident())
            prepared.append(mm_inputs)

        manager = _load_manager_helpers(prepare_media=prepare)()
        obj = _GenerateReqInput(rid="mm", cache_user_id="user", cache_ttl_seconds=300)
        mm_inputs = SimpleNamespace(mm_items=[object()])
        await manager._prepare_cache_media_digests(obj, mm_inputs)
        self.assertEqual(prepared, [mm_inputs])
        self.assertNotEqual(threads, [threading.get_ident()])

    async def test_missing_identity_disabled_or_absent_media_never_hashes_features(
        self,
    ):
        prepare = Mock(side_effect=AssertionError("Must not hash these features"))
        manager = _load_manager_helpers(prepare_media=prepare)()
        for identity, ttl, mm_inputs in (
            (None, 300, object()),
            ("user", None, object()),
            ("user", 300, None),
        ):
            obj = _GenerateReqInput(
                rid="mm", cache_user_id=identity, cache_ttl_seconds=ttl
            )
            await manager._prepare_cache_media_digests(obj, mm_inputs)
        prepare.assert_not_called()

    async def test_multimodal_identity_error_keeps_inference_and_redacts_payload(self):
        prepare = Mock(side_effect=ValueError("sensitive media content"))
        logger = Mock()
        manager = _load_manager_helpers(prepare_media=prepare, cache_logger=logger)()
        obj = _GenerateReqInput(rid="mm", cache_user_id="user", cache_ttl_seconds=300)
        await manager._prepare_cache_media_digests(obj, object())
        self.assertEqual(obj.cache_ttl_seconds, 300)
        logger.warning.assert_called_once()
        message = _log_message(logger.warning.call_args)
        self.assertIn("identity unavailable", message)
        self.assertIn("ValueError", message)
        self.assertNotIn("sensitive", message)

    def start(
        self,
        *,
        prompt_tokens=10,
        completion_tokens=1,
        finish_reason=None,
        cached_tokens=0,
        cache_input_metadata=None,
    ):
        self.manager._start_cache_accounting(
            self.state,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            cached_tokens=cached_tokens,
            cache_input_metadata=cache_input_metadata,
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
        self.account.assert_awaited_once_with(self.plan, rid="cache-request-1")
        self.assertEqual(first["meta_info"]["cache_usage"], self.usage.to_dict())
        self.assertEqual(
            second["meta_info"]["cache_usage"],
            first["meta_info"]["cache_usage"],
        )
        self.logger.info.assert_called_once()
        self.logger.warning.assert_not_called()
        self.metrics.observe_usage.assert_called_once_with(300, self.usage, "redis")
        message = _log_message(self.logger.info.call_args)
        self.assertRegex(
            message,
            r"^Cache accounting complete rid=cache-request-1 role=decode ttl=300 "
            r"block_size=4 prompt=10 read=4 write=4 "
            r"source=redis elapsed_ms=\d+\.\d{3}$",
        )

    async def test_eos_with_no_visible_text_still_accounts_generated_token(self):
        self.start(finish_reason={"type": "stop", "matched": 2})
        out = await self.attach(text="")
        self.account.assert_awaited_once_with(self.plan, rid="cache-request-1")
        self.assertEqual(out["meta_info"]["cache_usage"], self.usage.to_dict())

    async def test_multimodal_uses_final_expansion_and_reuses_one_snapshot(self):
        self.state.cache_plan = None
        self.state.cache_plan_context = ("user", "model-v1", 300)
        metadata = {
            "input_ids": [7] * 256 + [1 << 40] * 512 + [8, 9],
            "media_spans": [(256, 768, "a" * 64)],
        }
        usage = _CACHE.CacheUsage(ttl_seconds=300, read_tokens=256, write_tokens=512)
        self.account.return_value = usage
        self.start(prompt_tokens=770, cache_input_metadata=metadata)
        task = self.state.cache_task
        first = await self.attach()
        plan = self.account.await_args.args[0]
        self.assertEqual(plan.prompt_tokens, 770)
        self.assertEqual(plan.page_size, 256)
        self.assertEqual(len(plan.page_keys), 3)
        self.assertIs(self.state.cache_plan, plan)
        self.assertIsNone(self.state.cache_plan_context)
        self.start(prompt_tokens=770, completion_tokens=2)
        second = await self.attach()
        self.assertIs(self.state.cache_task, task)
        self.account.assert_awaited_once()
        self.assertEqual(first["meta_info"]["cache_usage"], usage.to_dict())
        self.assertEqual(second["meta_info"]["cache_usage"], usage.to_dict())

    async def test_multimodal_identity_ttl_and_prefix_have_real_page_semantics(self):
        known_pages = set()

        async def account(plan, *, rid):
            read = 0
            for key in plan.page_keys:
                if key not in known_pages:
                    break
                read += plan.page_size
            known_pages.update(plan.page_keys)
            return _CACHE.CacheUsage(
                ttl_seconds=plan.ttl_seconds,
                read_tokens=read,
                write_tokens=len(plan.page_keys) * plan.page_size - read,
            )

        self.account.side_effect = account

        async def request(digest, *, ttl=300, user="user", pad=99, extra=0):
            self.state = SimpleNamespace(
                obj=SimpleNamespace(rid="mm-request"),
                cache_plan=None,
                cache_plan_context=(user, "model-v1", ttl),
                cache_task=None,
                cache_usage=None,
            )
            ids = [7] * 256 + [pad] * 512 + [8] * (2 + extra)
            self.start(
                prompt_tokens=len(ids),
                cache_input_metadata={
                    "input_ids": ids,
                    "media_spans": [(256, 768, digest)],
                },
            )
            return (await self.attach())["meta_info"]["cache_usage"]

        self.assertEqual(
            await request("a" * 64),
            {"ttl_seconds": 300, "read_tokens": 0, "write_tokens": 768},
        )
        self.assertEqual(
            await request("a" * 64, pad=1 << 40),
            {"ttl_seconds": 300, "read_tokens": 768, "write_tokens": 0},
        )
        self.assertEqual(
            await request("b" * 64),
            {"ttl_seconds": 300, "read_tokens": 256, "write_tokens": 512},
        )
        self.assertEqual(
            await request("a" * 64, extra=256),
            {"ttl_seconds": 300, "read_tokens": 768, "write_tokens": 256},
        )
        self.assertEqual((await request("a" * 64, ttl=3600))["write_tokens"], 768)
        self.assertEqual((await request("a" * 64, user="other"))["write_tokens"], 768)

    async def test_invalid_multimodal_metadata_falls_back_without_redis(self):
        for metadata in (
            None,
            {"error": "invalid_multimodal_metadata"},
            {"input_ids": [7] * 770, "media_spans": []},
            {"input_ids": [7] * 20, "media_spans": [(0, 20, "a" * 64)]},
            {"input_ids": [7] * 770, "media_spans": [(0, 800, "a" * 64)]},
            {"input_ids": [7] * 770, "media_spans": [(0, 512, "invalid")]},
        ):
            with self.subTest(metadata=metadata):
                self.metrics.reset_mock()
                self.state = SimpleNamespace(
                    obj=SimpleNamespace(rid="mm-invalid"),
                    cache_plan=None,
                    cache_plan_context=("user", "model-v1", 300),
                    cache_task=None,
                    cache_usage=None,
                )
                self.start(
                    prompt_tokens=770,
                    cached_tokens=333,
                    cache_input_metadata=metadata,
                )
                out = await self.attach()
                usage = _CACHE.CacheUsage(ttl_seconds=300, read_tokens=333)
                self.assertEqual(out["meta_info"]["cache_usage"], usage.to_dict())
                self.account.assert_not_awaited()
                self.assertIsNone(self.state.cache_plan_context)
                self.metrics.observe_usage.assert_called_once_with(
                    300, usage, "multimodal_unavailable"
                )
        self.assertEqual(self.logger.warning.call_count, 6)

    async def test_multimodal_no_successful_output_does_not_hash_or_write(self):
        self.state.cache_plan = None
        self.state.cache_plan_context = ("user", "model-v1", 300)
        for completion, reason in (
            (0, None),
            (0, {"type": "length"}),
            (1, {"type": "abort"}),
        ):
            self.start(completion_tokens=completion, finish_reason=reason)
            await asyncio.sleep(0)
            self.assertIsNone(self.state.cache_task)
        self.account.assert_not_awaited()

    async def test_multimodal_redis_failure_preserves_actual_physical_read(self):
        self.state.cache_plan = None
        self.state.cache_plan_context = ("user", "model-v1", 3600)
        self.account.side_effect = _CACHE.CacheAccountingError("unconfirmed")
        self.start(
            prompt_tokens=770,
            cached_tokens=333,
            cache_input_metadata={
                "input_ids": [7] * 770,
                "media_spans": [(256, 768, "a" * 64)],
            },
        )
        out = await self.attach()
        usage = _CACHE.CacheUsage(ttl_seconds=3600, read_tokens=333)
        self.assertEqual(out["meta_info"]["cache_usage"], usage.to_dict())
        self.account.assert_awaited_once()
        self.metrics.observe_usage.assert_called_once_with(3600, usage, "physical")

    async def test_no_generated_token_does_not_touch_store(self):
        self.start(completion_tokens=0, finish_reason={"type": "length"})
        await asyncio.sleep(0)
        self.assertIsNone(self.state.cache_task)
        self.account.assert_not_awaited()
        self.logger.info.assert_not_called()
        self.logger.warning.assert_not_called()
        self.metrics.observe_usage.assert_not_called()

    async def test_missing_identity_freezes_physical_read_without_store(self):
        for ttl in (300, 3600):
            with self.subTest(ttl=ttl):
                self.logger.reset_mock()
                self.metrics.reset_mock()
                self.state = SimpleNamespace(
                    obj=SimpleNamespace(rid="anonymous-request"),
                    cache_plan=None,
                    cache_physical_ttl_seconds=ttl,
                    cache_task=None,
                    cache_usage=None,
                )
                self.start(cached_tokens=7)
                task = self.state.cache_task
                first = await self.attach(cached_tokens=7)
                self.state.cache_physical_ttl_seconds = 3600 if ttl == 300 else 300
                self.start(completion_tokens=2, cached_tokens=9)
                second = await self.attach(cached_tokens=9)
                expected = _CACHE.CacheUsage(ttl_seconds=ttl, read_tokens=7)
                self.assertEqual(first["meta_info"]["cache_usage"], expected.to_dict())
                self.assertEqual(second["meta_info"]["cache_usage"], expected.to_dict())
                self.assertIs(self.state.cache_task, task)
                self.account.assert_not_called()
                self.logger.warning.assert_not_called()
                self.logger.info.assert_called_once()
                self.assertIn(
                    "source=missing_identity", _log_message(self.logger.info.call_args)
                )
                self.metrics.observe_usage.assert_called_once_with(
                    ttl, expected, "missing_identity"
                )

    async def test_missing_identity_still_waits_for_first_successful_output(self):
        self.state.cache_plan = None
        self.state.cache_physical_ttl_seconds = 300
        for completion, reason in (
            (0, None),
            (0, {}),
            (-1, {"type": "length"}),
            (0, {"type": "abort"}),
            (0, {"type": "error"}),
            (1, {"type": "abort"}),
            (1, {"type": "error"}),
        ):
            with self.subTest(completion=completion, reason=reason):
                self.start(completion_tokens=completion, finish_reason=reason)
                await asyncio.sleep(0)
                self.assertIsNone(self.state.cache_task)
        self.account.assert_not_called()
        self.metrics.observe_usage.assert_not_called()

    async def test_missing_identity_successful_finish_with_no_output_has_usage(self):
        for ttl in (300, 3600):
            for finish_type in ("stop", "length"):
                with self.subTest(ttl=ttl, finish_type=finish_type):
                    self.metrics.reset_mock()
                    self.state = SimpleNamespace(
                        obj=SimpleNamespace(rid="zero-output-request"),
                        cache_plan=None,
                        cache_physical_ttl_seconds=ttl,
                        cache_task=None,
                        cache_usage=None,
                    )
                    self.start(
                        completion_tokens=0,
                        finish_reason={"type": finish_type},
                        cached_tokens=7,
                    )
                    out = await self.attach()
                    usage = _CACHE.CacheUsage(ttl_seconds=ttl, read_tokens=7)
                    self.assertEqual(out["meta_info"]["cache_usage"], usage.to_dict())
                    self.metrics.observe_usage.assert_called_once_with(
                        ttl, usage, "missing_identity"
                    )
        self.account.assert_not_called()

    async def test_parallel_internal_warmup_does_not_account_or_change_sample_ttl(self):
        class Batch(_GenerateReqInput):
            def __getitem__(self, index):
                return self.children[index]

        self.manager.enable_trace = False
        for ttl in (300, 3600):
            with self.subTest(ttl=ttl):
                self.metrics.reset_mock()
                self.manager.rid_to_state = {}
                child = _GenerateReqInput(
                    rid="base", received_time=1, return_prompt_token_ids=False
                )
                batch = Batch(
                    rid=["base"],
                    children=[child],
                    is_single=False,
                    bootstrap_room=None,
                    received_time=1,
                    cache_user_id=None,
                    cache_ttl_seconds=ttl,
                    cache_required=True,
                    batch_size=1,
                    parallel_sample_num=2,
                    stream=False,
                )
                self.manager._init_req_state(batch)
                tokenized = SimpleNamespace(
                    rid="base",
                    sampling_params=SimpleNamespace(max_new_tokens=1),
                    stream=False,
                    mm_inputs=None,
                )
                self.manager._tokenize_one_request = AsyncMock(return_value=tokenized)
                self.manager._send_one_request = AsyncMock()
                warmups = []

                async def response(obj, request=None):
                    state = self.manager.rid_to_state[obj.rid]
                    if obj.cache_ttl_seconds is None:
                        warmups.append(obj)
                        self.assertIsNone(obj.cache_user_id)
                        self.assertFalse(obj.cache_required)
                        self.manager._start_cache_accounting(
                            state, 10, 0, {"type": "length"}, cached_tokens=7
                        )
                        self.assertIsNone(state.cache_task)
                        out = {"meta_info": {}}
                    else:
                        self.assertEqual(obj.cache_ttl_seconds, ttl)
                        self.manager._start_cache_accounting(
                            state, 10, 1, {"type": "stop"}, cached_tokens=7
                        )
                        out = {"meta_info": {}}
                        await self.manager._attach_cache_usage(state, out)
                    del self.manager.rid_to_state[obj.rid]
                    yield out

                self.manager._wait_one_response = response
                batches = [
                    out async for out in self.manager._handle_batch_request(batch)
                ]
                self.assertEqual(len(warmups), 1)
                self.assertEqual(child.cache_ttl_seconds, ttl)
                self.assertTrue(child.cache_required)
                self.assertEqual(len(batches), 1)
                self.assertEqual(len(batches[0]), 2)
                usage = _CACHE.CacheUsage(ttl_seconds=ttl, read_tokens=7)
                for choice in batches[0]:
                    self.assertEqual(
                        choice["meta_info"]["cache_usage"], usage.to_dict()
                    )
                self.metrics.observe_usage.assert_called_once_with(
                    ttl, usage, "missing_identity"
                )
        self.account.assert_not_called()

    async def test_missing_identity_keeps_physical_read_within_prompt(self):
        for physical_read, expected_read in ((-1, 0), (0, 0), (7, 7), (11, 10)):
            with self.subTest(physical_read=physical_read):
                self.state = SimpleNamespace(
                    obj=SimpleNamespace(rid="anonymous-request"),
                    cache_plan=None,
                    cache_physical_ttl_seconds=300,
                    cache_task=None,
                    cache_usage=None,
                )
                self.start(cached_tokens=physical_read)
                out = await self.attach()
                self.assertEqual(
                    out["meta_info"]["cache_usage"],
                    _CACHE.CacheUsage(
                        ttl_seconds=300, read_tokens=expected_read
                    ).to_dict(),
                )
        self.account.assert_not_called()

    async def test_missing_identity_parallel_samples_observe_one_input(self):
        parent = SimpleNamespace(cache_usage=None)
        samples = []
        for index, physical_read in enumerate((7, 9)):
            state = SimpleNamespace(
                obj=SimpleNamespace(rid=f"choice-{index}"),
                cache_plan=None,
                cache_physical_ttl_seconds=300,
                cache_metrics_parent=parent,
                cache_task=None,
                cache_usage=None,
            )
            self.manager._start_cache_accounting(
                state, 10, 1, None, cached_tokens=physical_read
            )
            samples.append(state)
        outputs = [{"meta_info": {}} for _ in samples]
        await asyncio.gather(
            *(
                self.manager._attach_cache_usage(state, out)
                for state, out in zip(samples, outputs)
            )
        )
        for out, physical_read in zip(outputs, (7, 9)):
            self.assertEqual(
                out["meta_info"]["cache_usage"],
                _CACHE.CacheUsage(ttl_seconds=300, read_tokens=physical_read).to_dict(),
            )
        self.assertEqual(
            parent.cache_usage, _CACHE.CacheUsage(ttl_seconds=300, read_tokens=7)
        )
        self.metrics.observe_usage.assert_called_once_with(
            300, parent.cache_usage, "missing_identity"
        )
        self.account.assert_not_called()

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

    async def test_redis_failure_freezes_physical_read_and_does_not_retry(self):
        self.account.side_effect = _CACHE.CacheAccountingError(
            "unconfirmed Redis result"
        )
        self.start(cached_tokens=7)
        task = self.state.cache_task
        first = await self.attach(cached_tokens=7)
        self.start(completion_tokens=2, cached_tokens=9)
        second = await self.attach(cached_tokens=9)
        expected = _CACHE.CacheUsage(ttl_seconds=300, read_tokens=7).to_dict()
        self.assertEqual(first["meta_info"]["cache_usage"], expected)
        self.assertEqual(second["meta_info"]["cache_usage"], expected)
        self.assertIs(self.state.cache_task, task)
        self.account.assert_awaited_once_with(self.plan, rid="cache-request-1")

    async def test_redis_fallback_logs_once_without_exception_contents_or_secrets(self):
        secrets = (
            "redis://private-redis-host:6379/0",
            "secret-api-key",
            "secret-input-content",
            "verified-api-key-hash",
            self.plan.page_keys[0],
        )
        error = _CACHE.CacheAccountingError(" ".join(secrets))
        error.__cause__ = TimeoutError(" ".join(secrets))
        self.account.side_effect = error
        self.start(cached_tokens=7)
        await self.attach()
        self.start(completion_tokens=2, cached_tokens=9)
        await self.attach()
        self.logger.warning.assert_called_once()
        self.logger.info.assert_called_once()
        warning = _log_message(self.logger.warning.call_args)
        complete = _log_message(self.logger.info.call_args)
        self.assertEqual(
            warning,
            "Cache accounting Redis fallback rid=cache-request-1 role=decode "
            "ttl=300 error_type=TimeoutError",
        )
        self.assertIn("read=7 write=0 source=physical", complete)
        for secret in secrets:
            self.assertNotIn(secret, warning + complete)
        self.assertEqual(self.logger.warning.call_args.kwargs, {})
        self.assertEqual(self.logger.info.call_args.kwargs, {})
        self.metrics.observe_usage.assert_called_once_with(
            300, _CACHE.CacheUsage(ttl_seconds=300, read_tokens=7), "physical"
        )

    async def test_no_complete_block_logs_without_claiming_redis_was_used(self):
        self.state.cache_plan = _CACHE.build_page_cache_plan(
            input_ids=[1, 2, 3],
            cache_user_id="verified-api-key-hash",
            namespace="weights-v1",
            page_size=_CACHE.CACHE_BLOCK_SIZE,
            ttl_seconds=3600,
        )
        self.account.return_value = _CACHE.CacheUsage(ttl_seconds=3600)
        self.manager.disaggregation_mode = _DISAGG_MODE.NULL
        self.start(prompt_tokens=3)
        await self.manager._attach_cache_usage(self.state, {"meta_info": {}})
        self.logger.info.assert_called_once()
        message = _log_message(self.logger.info.call_args)
        self.assertIn("role=standalone ttl=3600 block_size=256 prompt=3", message)
        self.assertIn("read=0 write=0 source=no_complete_block", message)
        self.logger.warning.assert_not_called()
        self.metrics.observe_usage.assert_called_once_with(
            3600, _CACHE.CacheUsage(ttl_seconds=3600), "no_complete_block"
        )

    async def test_redis_failure_keeps_physical_read_within_prompt(self):
        self.account.side_effect = _CACHE.CacheAccountingError("Redis unavailable")
        for physical_read, expected_read in ((-1, 0), (0, 0), (7, 7), (11, 10)):
            with self.subTest(physical_read=physical_read):
                self.state = SimpleNamespace(
                    obj=SimpleNamespace(rid="cache-request-1"),
                    cache_plan=self.plan,
                    cache_task=None,
                    cache_usage=None,
                )
                self.start(cached_tokens=physical_read)
                out = await self.attach(cached_tokens=physical_read)
                cache = out["meta_info"]["cache_usage"]
                self.assertEqual(cache["read_tokens"], expected_read)
                self.assertEqual(cache["write_tokens"], 0)
                self.assertEqual(cache["ttl_seconds"], self.plan.ttl_seconds)
                self.assertLessEqual(cache["read_tokens"], self.plan.prompt_tokens)

    async def test_prompt_count_mismatch_is_strict_and_never_writes(self):
        self.start(prompt_tokens=9)
        with self.assertRaises(_CACHE.CacheAccountingError):
            await self.attach()
        self.account.assert_not_awaited()

    async def test_local_accounting_error_does_not_fall_back_to_physical_read(self):
        self.account.side_effect = ValueError("Invalid cache plan")
        self.start(cached_tokens=7)
        with self.assertRaisesRegex(ValueError, "Invalid cache plan"):
            await self.attach()
        self.assertIsNone(self.state.cache_usage)
        self.account.assert_awaited_once_with(self.plan, rid="cache-request-1")
        self.logger.info.assert_not_called()
        self.logger.warning.assert_not_called()
        self.metrics.observe_usage.assert_not_called()

    async def test_complete_block_counts_and_physical_cache_metric_are_preserved(self):
        self.start()
        out = await self.attach(cached_tokens=7)
        cache = out["meta_info"]["cache_usage"]
        self.assertEqual(cache["read_tokens"] + cache["write_tokens"], 8)
        self.assertEqual(cache["ttl_seconds"], self.plan.ttl_seconds)
        self.assertEqual(out["meta_info"]["cached_tokens"], 7)

    async def test_other_request_keeps_its_own_accounting_task(self):
        other_plan = _CACHE.build_page_cache_plan(
            input_ids=list(range(10)),
            cache_user_id="verified-api-key-hash",
            namespace="weights-v1/tokenizer-v1",
            page_size=4,
            ttl_seconds=300,
        )
        other_state = SimpleNamespace(
            obj=SimpleNamespace(rid="cache-request-2"),
            cache_plan=other_plan,
            cache_task=None,
            cache_usage=None,
        )
        self.start()
        self.manager._start_cache_accounting(other_state, 10, 1, None)
        self.assertIsNot(self.state.cache_task, other_state.cache_task)
        await asyncio.gather(self.state.cache_task, other_state.cache_task)
        self.assertEqual(self.account.await_count, 2)
        self.account.assert_any_await(self.plan, rid="cache-request-1")
        self.account.assert_any_await(other_plan, rid="cache-request-2")

    async def test_cancelled_client_wait_does_not_cancel_started_accounting(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def delayed_account(plan, *, rid=None):
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
        self.account.assert_awaited_once_with(self.plan, rid="cache-request-1")

    async def test_renewal_runs_once_only_after_owner_finish(self):
        self.start()
        initial = await self.attach()
        renewal = self.state.cache_renewal_task
        for _ in range(3):
            await asyncio.sleep(0)
        self.assertFalse(renewal.done())
        self.renew.assert_not_awaited()
        self.manager._finish_cache_accounting(self.state)
        final = await self.attach()
        self.manager._finish_cache_accounting(self.state)
        await self.attach()
        self.assertTrue(renewal.done())
        self.renew.assert_awaited_once_with(self.plan, rid="cache-request-1")
        self.assertEqual(initial["meta_info"]["cache_usage"], self.usage.to_dict())
        self.assertEqual(final["meta_info"]["cache_usage"], self.usage.to_dict())
        self.account.assert_awaited_once()
        self.metrics.observe_usage.assert_called_once_with(300, self.usage, "redis")

    async def test_finish_before_account_uses_initial_eval_without_extra_renewal(self):
        self.start()
        self.manager._finish_cache_accounting(self.state)
        out = await self.attach()
        self.assertTrue(self.state.cache_finished_event.is_set())
        self.assertIsNone(getattr(self.state, "cache_renewal_task", None))
        self.renew.assert_not_awaited()
        self.account.assert_awaited_once_with(self.plan, rid="cache-request-1")
        self.assertEqual(out["meta_info"]["cache_usage"], self.usage.to_dict())

    async def test_finish_during_account_waits_for_final_renewal_before_response(self):
        entered, account_release = asyncio.Event(), asyncio.Event()
        renew_entered, renew_release = asyncio.Event(), asyncio.Event()
        self.release_events.extend((account_release, renew_release))

        async def account(plan, *, rid):
            entered.set()
            await account_release.wait()
            return self.usage

        async def renew(plan, *, rid):
            renew_entered.set()
            await renew_release.wait()

        self.account.side_effect = account
        self.renew.side_effect = renew
        self.start()
        await asyncio.wait_for(entered.wait(), 1)
        self.manager._finish_cache_accounting(self.state)
        response = asyncio.create_task(self.attach())
        account_release.set()
        await asyncio.wait_for(renew_entered.wait(), 1)
        self.assertFalse(response.done())
        renew_release.set()
        out = await asyncio.wait_for(response, 1)
        self.renew.assert_awaited_once_with(self.plan, rid="cache-request-1")
        self.assertEqual(out["meta_info"]["cache_usage"], self.usage.to_dict())

    async def test_cancelled_consumer_still_renews_when_owner_finishes(self):
        entered, release = asyncio.Event(), asyncio.Event()
        self.release_events.append(release)

        async def account(plan, *, rid):
            entered.set()
            await release.wait()
            return self.usage

        self.account.side_effect = account
        self.start()
        await asyncio.wait_for(entered.wait(), 1)
        response = asyncio.create_task(self.attach())
        await asyncio.sleep(0)
        response.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await response
        self.assertFalse(self.state.cache_task.cancelled())
        self.manager._finish_cache_accounting(self.state)
        release.set()
        await asyncio.wait_for(self.state.cache_task, 1)
        await asyncio.wait_for(self.state.cache_renewal_task, 1)
        self.renew.assert_awaited_once_with(self.plan, rid="cache-request-1")
        out = await self.attach()
        self.assertEqual(out["meta_info"]["cache_usage"], self.usage.to_dict())

    async def test_fallback_missing_identity_and_short_input_never_renew(self):
        for source in ("physical", "missing_identity", "no_complete_block", "media"):
            with self.subTest(source=source):
                state = SimpleNamespace(
                    obj=SimpleNamespace(rid=source),
                    cache_plan=self.plan,
                    cache_task=None,
                    cache_usage=None,
                )
                self.account.reset_mock(side_effect=True)
                self.account.return_value = self.usage
                if source == "physical":
                    self.account.side_effect = _CACHE.CacheAccountingError("offline")
                elif source == "missing_identity":
                    state.cache_plan = None
                    state.cache_physical_ttl_seconds = 300
                elif source == "media":
                    state.cache_plan = None
                    state.cache_plan_context = ("user", "model", 300)
                else:
                    state.cache_plan = _CACHE.build_page_cache_plan(
                        [1, 2], "user", "model", 256, 300
                    )
                    self.account.return_value = _CACHE.CacheUsage(ttl_seconds=300)
                self.manager._start_cache_accounting(
                    state,
                    2 if source == "no_complete_block" else 10,
                    1,
                    None,
                    cached_tokens=1,
                )
                await self.manager._attach_cache_usage(state, {"meta_info": {}})
                self.manager._finish_cache_accounting(state)
                self.assertIsNone(getattr(state, "cache_renewal_task", None))
                self.renew.assert_not_awaited()

    async def test_failed_final_renewal_keeps_usage_and_redacts_error(self):
        self.renew.side_effect = _CACHE.CacheAccountingError("private credential")
        self.start()
        initial = await self.attach()
        self.manager._finish_cache_accounting(self.state)
        final = await self.attach()
        self.assertEqual(final["meta_info"], initial["meta_info"])
        self.metrics.observe_usage.assert_called_once()
        warning = _log_message(self.logger.warning.call_args)
        self.assertIn("phase=finish", warning)
        self.assertIn("error_type=CacheAccountingError", warning)
        self.assertNotIn("private credential", warning)

    async def test_close_waits_for_final_renewal_without_cancelling_response(self):
        entered, release = asyncio.Event(), asyncio.Event()
        self.release_events.append(release)
        order = []

        async def renew(plan, *, rid):
            order.append("renew-start")
            entered.set()
            await release.wait()
            order.append("renew-end")

        async def close():
            order.append("close")

        self.renew.side_effect = renew
        self.close_store.side_effect = close
        self.start()
        await self.attach()
        self.manager._finish_cache_accounting(self.state)
        response = asyncio.create_task(self.attach())
        await asyncio.wait_for(entered.wait(), 1)
        closing = asyncio.create_task(self.manager.close_cache_accounting())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        self.assertFalse(self.state.cache_renewal_task.cancelled())
        self.close_store.assert_not_awaited()
        release.set()
        out, _ = await asyncio.wait_for(asyncio.gather(response, closing), 1)
        self.assertEqual(order, ["renew-start", "renew-end", "close"])
        self.assertEqual(out["meta_info"]["cache_usage"], self.usage.to_dict())
        self.assertEqual(self.manager._cache_tasks, set())
        self.assertEqual(self.manager._cache_renewal_tasks, {})

    async def test_close_finishes_concurrent_renewals_and_blocks_new_accounting(self):
        states = []
        for index in range(6):
            state = SimpleNamespace(
                obj=SimpleNamespace(rid=f"shutdown-{index}"),
                cache_plan=self.plan,
                cache_task=None,
                cache_usage=None,
            )
            self.manager._start_cache_accounting(state, 10, 1, None)
            states.append(state)
        await asyncio.gather(*(state.cache_task for state in states))
        self.renew.assert_not_awaited()
        await asyncio.wait_for(self.manager.close_cache_accounting(), 1)
        self.assertEqual(self.renew.await_count, len(states))
        self.assertTrue(all(state.cache_finished_event.is_set() for state in states))
        self.assertTrue(all(state.cache_renewal_task.done() for state in states))
        self.assertEqual(self.manager._cache_tasks, set())
        self.assertEqual(self.manager._cache_renewal_tasks, {})
        self.start()
        self.assertIsNone(self.state.cache_task)
        self.assertEqual(self.account.await_count, len(states))

    def test_real_output_and_abort_owners_signal_finish(self):
        source = ast.parse((_SRT / "managers/tokenizer_manager.py").read_text())
        manager = next(
            node
            for node in source.body
            if isinstance(node, ast.ClassDef) and node.name == "TokenizerManager"
        )

        def signals(node):
            return any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_finish_cache_accounting"
                for call in ast.walk(node)
            )

        output = next(
            node
            for node in manager.body
            if getattr(node, "name", None) == "_handle_batch_output"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Attribute)
                and node.test.attr == "finished"
                and any(signals(statement) for statement in node.body)
                for node in ast.walk(output)
            )
        )
        abort = next(
            node
            for node in manager.body
            if getattr(node, "name", None) == "_handle_abort_req"
        )
        finished_index = next(
            index
            for index, node in enumerate(abort.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute) and target.attr == "finished"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
        )
        signal_index = next(
            index for index, node in enumerate(abort.body) if signals(node)
        )
        self.assertGreater(signal_index, finished_index)


if __name__ == "__main__":
    unittest.main()
