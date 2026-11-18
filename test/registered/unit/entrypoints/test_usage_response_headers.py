"""HTTP cache headers must not change the serving model or its JSON usage."""

import asyncio
import json
import unittest
from types import SimpleNamespace

from fastapi.responses import JSONResponse, StreamingResponse

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.openai.protocol import (
    CacheUsage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    PromptTokensDetails,
    ResponsesRequest,
    ResponsesResponse,
    UsageInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestUsageResponseHeaders(CustomTestCase):
    def _request(self, kind, result):
        class Serving:
            async def handle_request(self, request, raw_request):
                return result

            create_responses = handle_request

        raw_request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    openai_serving_chat=Serving(),
                    openai_serving_responses=Serving(),
                )
            )
        )
        if kind == "chat":
            return asyncio.run(
                http_server.openai_v1_chat_completions(
                    ChatCompletionRequest(model="x", messages=[]), raw_request
                )
            )
        return asyncio.run(
            http_server.v1_responses_request(
                ResponsesRequest(model="x", input="hi"), raw_request
            )
        )

    def test_nonstream_headers_preserve_body_and_accounting(self):
        """Moving serialization to HTTP must keep physical and ledger counts separate."""
        for kind in ("chat", "responses"):
            for ttl in (None, 300, 3600):
                with self.subTest(kind=kind, ttl=ttl):
                    usage = UsageInfo(
                        prompt_tokens=2048,
                        completion_tokens=1,
                        total_tokens=2049,
                        prompt_tokens_details=PromptTokensDetails(cached_tokens=256),
                        cache_usage=(
                            CacheUsage(
                                ttl_seconds=ttl, read_tokens=256, write_tokens=512
                            )
                            if ttl is not None
                            else None
                        ),
                    )
                    result = (
                        ChatCompletionResponse(
                            id="c", model="x", choices=[], usage=usage
                        )
                        if kind == "chat"
                        else ResponsesResponse(
                            model="x", status="completed", usage=usage
                        )
                    )
                    result._response_headers = {"X-Sgl-Cached-Tokens": "1536"}
                    expected = json.loads(result.model_dump_json(by_alias=True))

                    response = self._request(kind, result)

                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers["x-sgl-cached-tokens"], "1536")
                    self.assertEqual(json.loads(response.body), expected)
                    self.assertNotIn("_response_headers", expected)
                    self.assertNotIn("cache_usage", expected["usage"])
                    for tier, seconds in (("5m", 300), ("1h", 3600)):
                        name = f"msh-usage-cache-write-tokens-{tier}"
                        if ttl is None:
                            self.assertNotIn(name, response.headers)
                        else:
                            self.assertEqual(
                                response.headers[name], "512" if ttl == seconds else "0"
                            )

    def test_http_layer_preserves_stream_and_error_response(self):
        """Transport responses must retain their status, body and streaming lifecycle."""
        for kind in ("chat", "responses"):
            with self.subTest(kind=kind):
                error = JSONResponse({"error": "bad request"}, status_code=400)
                self.assertIs(self._request(kind, error), error)

                async def chunks():
                    yield "data: [DONE]\n\n"

                stream = StreamingResponse(
                    chunks(), headers={"X-Sgl-Cached-Tokens": "0"}
                )
                self.assertIs(self._request(kind, stream), stream)
                self.assertEqual(stream.headers["x-sgl-cached-tokens"], "0")

    def test_background_ack_does_not_invent_cache_headers(self):
        """Queued Responses have no engine snapshot to expose as a physical hit count."""
        result = ResponsesResponse(model="x", status="queued")
        response = self._request("responses", result)
        self.assertNotIn("x-sgl-cached-tokens", response.headers)
        self.assertNotIn("msh-usage-cache-write-tokens-5m", response.headers)
        self.assertEqual(json.loads(response.body)["status"], "queued")


if __name__ == "__main__":
    unittest.main()
