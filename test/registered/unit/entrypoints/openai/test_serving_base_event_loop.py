"""Event-loop responsiveness tests for OpenAI request preprocessing."""

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _BlockingServing(OpenAIServingBase):
    def __init__(self, conversion_started, release_conversion):
        tokenizer_manager = TokenizerManager.__new__(TokenizerManager)
        if hasattr(tokenizer_manager, "init_request_preprocessor"):
            tokenizer_manager.init_request_preprocessor()
        tokenizer_manager.request_logger = SimpleNamespace(
            log_requests=False, log_requests_level=0
        )
        tokenizer_manager.server_args = SimpleNamespace(
            tokenizer_metrics_allowed_custom_labels=None,
        )
        super().__init__(tokenizer_manager)
        self.conversion_started = conversion_started
        self.release_conversion = release_conversion

    def _request_id_prefix(self):
        return "test-"

    def _convert_to_internal_request(self, request, raw_request=None):
        self.conversion_started.set()
        if not self.release_conversion.wait(timeout=5):
            raise TimeoutError("test did not release request conversion")
        return object(), request

    async def _handle_non_streaming_request(
        self, adapted_request, request, raw_request
    ):
        return "ok"


class TestServingBaseEventLoop(CustomTestCase):
    def test_request_conversion_does_not_block_event_loop(self):
        conversion_started = threading.Event()
        loop_progressed = threading.Event()
        release_conversion = threading.Event()
        loop_was_starved = threading.Event()
        conversion_did_not_start = threading.Event()
        serving = _BlockingServing(conversion_started, release_conversion)

        def release_after_loop_progress():
            if not conversion_started.wait(timeout=2):
                conversion_did_not_start.set()
                release_conversion.set()
                return
            if not loop_progressed.wait(timeout=1):
                loop_was_starved.set()
            release_conversion.set()

        watchdog = threading.Thread(target=release_after_loop_progress)
        watchdog.start()

        async def run_request_and_probe_loop():
            request = SimpleNamespace(stream=False)
            request_task = asyncio.create_task(
                serving.handle_request(request, object())
            )
            await asyncio.sleep(0)
            loop_progressed.set()
            return await request_task

        try:
            result = asyncio.run(run_request_and_probe_loop())
        finally:
            release_conversion.set()
            watchdog.join(timeout=2)
            executor = getattr(
                serving.tokenizer_manager, "_request_preprocessor_executor", None
            )
            if executor is not None:
                executor.shutdown(wait=True)

        self.assertEqual(result, "ok")
        self.assertFalse(conversion_did_not_start.is_set())
        self.assertFalse(watchdog.is_alive())
        self.assertFalse(
            loop_was_starved.is_set(),
            "synchronous request conversion blocked the API event loop",
        )


class _SlowStepServing(_BlockingServing):
    """Validation and conversion each take ``validate_s`` / ``convert_s``."""

    def __init__(self, validate_s=0.0, convert_s=0.0):
        release = threading.Event()
        release.set()
        super().__init__(threading.Event(), release)
        self.validate_s = validate_s
        self.convert_s = convert_s

    def _validate_request(self, request):
        time.sleep(self.validate_s)
        return None

    def _convert_to_internal_request(self, request, raw_request=None):
        time.sleep(self.convert_s)
        return object(), request


class TestSlowPreprocessingWarning(CustomTestCase):
    logger_name = "sglang.srt.entrypoints.openai.serving_base"

    def _run(self, serving):
        request = SimpleNamespace(stream=False, rid="rid-1", tools=[1, 2, 3])
        try:
            return asyncio.run(serving.handle_request(request, object()))
        finally:
            executor = getattr(
                serving.tokenizer_manager, "_request_preprocessor_executor", None
            )
            if executor is not None:
                executor.shutdown(wait=True)

    def test_slow_validation_is_reported(self):
        with envs.SGLANG_LOG_SLOW_PREPROCESSING_MS.override(10):
            with self.assertLogs(self.logger_name, level="WARNING") as logs:
                self.assertEqual(self._run(_SlowStepServing(validate_s=0.05)), "ok")
        self.assertEqual(len(logs.output), 1)
        self.assertIn("Slow request validation on the event loop", logs.output[0])
        self.assertIn("rid=rid-1, tools=3", logs.output[0])

    def test_slow_thread_conversion_is_reported(self):
        with envs.SGLANG_LOG_SLOW_PREPROCESSING_MS.override(10):
            with self.assertLogs(self.logger_name, level="WARNING") as logs:
                self.assertEqual(self._run(_SlowStepServing(convert_s=0.05)), "ok")
        self.assertEqual(len(logs.output), 1)
        self.assertIn("conversion on the in-process thread", logs.output[0])

    def test_fast_or_disabled_is_silent(self):
        for threshold, serving in (
            (1000, _SlowStepServing(validate_s=0.02, convert_s=0.02)),
            (0, _SlowStepServing(validate_s=0.02, convert_s=0.02)),
        ):
            with envs.SGLANG_LOG_SLOW_PREPROCESSING_MS.override(threshold):
                with self.assertNoLogs(self.logger_name, level="WARNING"):
                    self.assertEqual(self._run(serving), "ok")


if __name__ == "__main__":
    unittest.main()
