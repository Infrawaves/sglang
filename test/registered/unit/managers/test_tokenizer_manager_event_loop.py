"""Event-loop responsiveness tests for TokenizerManager tokenization."""

import asyncio
import threading
from types import SimpleNamespace
import unittest
from contextvars import ContextVar
from unittest import mock

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _BlockingFastTokenizer:
    is_fast = True

    def __init__(self, tokenization_started, release_tokenization):
        self.tokenization_started = tokenization_started
        self.release_tokenization = release_tokenization

    def __call__(self, texts, **kwargs):
        self.tokenization_started.set()
        if not self.release_tokenization.wait(timeout=5):
            raise TimeoutError("test did not release tokenization")
        return {"input_ids": [[1] for _ in texts]}


class TestTokenizerManagerEventLoop(CustomTestCase):
    def test_shutdown_waits_for_pull_work_before_local_intake_and_after_output(self):
        """SIGTERM must keep schedulers alive while accepted pull work has no
        tokenizer rid yet, and while its outcome is still being reported."""
        from sglang.srt.managers import tokenizer_manager as module
        from sglang.srt.managers.io_struct import ShutdownReq
        from sglang.srt.managers.prefill_pull_agent import (
            PrefillPullAgent,
            PrefillPullAgentConfig,
        )

        async def run_shutdown():
            allow_generate = asyncio.Event()
            local_started = asyncio.Event()
            allow_output = asyncio.Event()
            outcome_pending = asyncio.Event()
            allow_outcome = asyncio.Event()
            polls = asyncio.Queue()
            next_poll = asyncio.Queue()
            finish_polling = False
            real_sleep = asyncio.sleep
            shutdown_requests = []
            shutdown_events = []

            manager = TokenizerManager.__new__(TokenizerManager)
            manager.gracefully_exit = True
            manager.rid_to_state = {}
            manager.server_status = module.ServerStatus.Up
            manager._cache_store = None
            manager._subprocess_watchdog = None
            manager.asyncio_tasks = set()
            manager._dispatch_to_scheduler = shutdown_requests.append
            manager._server_stop_hook = lambda: shutdown_events.append("server stopped")
            agent = PrefillPullAgent(
                PrefillPullAgentConfig(
                    router_urls=("http://router",),
                    worker_url="http://worker",
                    local_url="http://worker",
                    max_inflight=1,
                ),
                budget=mock.AsyncMock(),
                abort=mock.Mock(),
                ready=lambda: True,
            )
            manager.prefill_pull_agent = agent

            async def accepted_work():
                await allow_generate.wait()
                manager.rid_to_state["pull-rid"] = object()
                local_started.set()
                await allow_output.wait()
                manager.rid_to_state.pop("pull-rid")
                outcome_pending.set()
                await allow_outcome.wait()
                shutdown_events.append("outcome reported")

            async def controlled_sleep(delay):
                self.assertEqual(delay, 5)
                polls.put_nowait(None)
                if not finish_polling:
                    await next_poll.get()
                await real_sleep(0)

            assignment = asyncio.create_task(accepted_work())
            agent._assignments["assignment"] = assignment
            assignment.add_done_callback(
                lambda task: agent._assignment_done("assignment", task)
            )
            with mock.patch.object(
                module.asyncio, "sleep", side_effect=controlled_sleep
            ), mock.patch.object(
                module, "get_bool_env_var", return_value=False
            ), mock.patch.object(
                module, "collect_scheduler_processes", return_value=[]
            ), mock.patch.object(
                module, "kill_process_tree"
            ):
                watchdog = asyncio.create_task(manager.sigterm_watchdog())
                try:
                    await real_sleep(0)
                    self.assertEqual(manager.rid_to_state, {})
                    self.assertEqual(shutdown_requests, [])
                    await asyncio.wait_for(polls.get(), timeout=1)

                    allow_generate.set()
                    await asyncio.wait_for(local_started.wait(), timeout=1)
                    next_poll.put_nowait(None)
                    await asyncio.wait_for(polls.get(), timeout=1)
                    self.assertEqual(shutdown_requests, [])

                    allow_output.set()
                    await asyncio.wait_for(outcome_pending.wait(), timeout=1)
                    next_poll.put_nowait(None)
                    await asyncio.wait_for(polls.get(), timeout=1)
                    self.assertEqual(manager.rid_to_state, {})
                    self.assertEqual(shutdown_requests, [])
                    self.assertFalse(assignment.cancelled())

                    allow_outcome.set()
                    finish_polling = True
                    next_poll.put_nowait(None)
                    await asyncio.wait_for(watchdog, timeout=1)
                    self.assertEqual(len(shutdown_requests), 1)
                    self.assertIsInstance(shutdown_requests[0], ShutdownReq)
                    self.assertEqual(
                        shutdown_events, ["outcome reported", "server stopped"]
                    )
                finally:
                    assignment.cancel()
                    watchdog.cancel()
                    await asyncio.gather(assignment, watchdog, return_exceptions=True)

        asyncio.run(run_shutdown())

    def test_regular_tokenizer_does_not_block_event_loop(self):
        tokenization_started = threading.Event()
        loop_progressed = threading.Event()
        release_tokenization = threading.Event()
        loop_was_starved = threading.Event()
        tokenization_did_not_start = threading.Event()

        manager = TokenizerManager.__new__(TokenizerManager)
        manager.tokenizer = _BlockingFastTokenizer(
            tokenization_started, release_tokenization
        )
        manager.async_dynamic_batch_tokenizer = None
        # The post-tokenization EOS fix-up reads this (upstream).
        manager.model_config = SimpleNamespace(is_embedding_gemma=False)
        if hasattr(manager, "init_request_preprocessor"):
            manager.init_request_preprocessor()

        def release_after_loop_progress():
            if not tokenization_started.wait(timeout=2):
                tokenization_did_not_start.set()
                release_tokenization.set()
                return
            if not loop_progressed.wait(timeout=1):
                loop_was_starved.set()
            release_tokenization.set()

        watchdog = threading.Thread(target=release_after_loop_progress)
        watchdog.start()

        async def tokenize_and_probe_loop():
            tokenize_task = asyncio.create_task(manager._tokenize_texts("hello"))
            await asyncio.sleep(0)
            loop_progressed.set()
            return await tokenize_task

        try:
            input_ids, token_type_ids = asyncio.run(tokenize_and_probe_loop())
        finally:
            release_tokenization.set()
            watchdog.join(timeout=2)
            executor = getattr(manager, "_request_preprocessor_executor", None)
            if executor is not None:
                executor.shutdown(wait=True)

        self.assertEqual(input_ids, [1])
        self.assertIsNone(token_type_ids)
        self.assertFalse(tokenization_did_not_start.is_set())
        self.assertFalse(watchdog.is_alive())
        self.assertFalse(
            loop_was_starved.is_set(),
            "synchronous tokenization blocked the API event loop",
        )

    @unittest.skipUnless(
        hasattr(TokenizerManager, "init_request_preprocessor"),
        "requires the request preprocessor executor",
    )
    def test_request_preprocessing_is_serialized(self):
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.init_request_preprocessor()

        def first_call():
            first_started.set()
            if not release_first.wait(timeout=5):
                raise TimeoutError("test did not release first preprocessing call")

        def second_call():
            second_started.set()

        async def run_concurrent_calls():
            first_task = asyncio.create_task(
                manager.run_in_request_preprocessor(first_call)
            )
            started = await asyncio.to_thread(first_started.wait, 1)
            self.assertTrue(started)

            second_task = asyncio.create_task(
                manager.run_in_request_preprocessor(second_call)
            )
            await asyncio.sleep(0.05)
            self.assertFalse(second_started.is_set())

            release_first.set()
            await asyncio.gather(first_task, second_task)

        try:
            asyncio.run(run_concurrent_calls())
        finally:
            release_first.set()
            manager._request_preprocessor_executor.shutdown(wait=True)

        self.assertTrue(second_started.is_set())

    @unittest.skipUnless(
        hasattr(TokenizerManager, "init_request_preprocessor"),
        "requires the request preprocessor executor",
    )
    def test_request_preprocessing_preserves_context(self):
        request_context = ContextVar("request_context")
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.init_request_preprocessor()

        async def read_context_from_preprocessor():
            request_context.set("request-value")
            return await manager.run_in_request_preprocessor(request_context.get)

        try:
            result = asyncio.run(read_context_from_preprocessor())
        finally:
            manager._request_preprocessor_executor.shutdown(wait=True)

        self.assertEqual(result, "request-value")


if __name__ == "__main__":
    unittest.main()
