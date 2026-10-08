"""Request validation errors must not echo the request back.

FastAPI attaches the offending value to every validation error as ``input``;
for a chat request that is the whole message list, once per Union branch.
Stringifying it ran on the HTTP event loop and stalled token delivery for
every stream on that tokenizer worker. These tests pin that the 400 body
stays small and free of the request content while still naming the bad field.
"""

import asyncio
import json
import time
import unittest

from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.requests import Request

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# sgl-model-gateway truncates upstream error bodies beyond this many bytes.
GATEWAY_ERROR_BODY_CAP = 2048
MARKER = "QQQQQQQQQQQQQQQQ"


def _request(path):
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "root_path": "",
        "query_string": b"",
        "headers": [],
    }
    return Request(scope)


def _handle(path, errors):
    response = asyncio.run(
        http_server.validation_exception_handler(
            _request(path), RequestValidationError(errors)
        )
    )
    return response.status_code, response.body


class TestValidationErrorResponse(CustomTestCase):
    def test_chat_completions_body_does_not_echo_large_request(self):
        big_text = MARKER * (4_000_000 // len(MARKER))
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": big_text},
                        {"type": "image_url", "image_url": 123},
                    ],
                }
            ],
        }
        client = TestClient(http_server.app)
        start = time.perf_counter()
        response = client.post("/v1/chat/completions", json=body)
        elapsed = time.perf_counter() - start

        self.assertEqual(response.status_code, 400)
        self.assertLess(len(response.content), GATEWAY_ERROR_BODY_CAP)
        self.assertNotIn(MARKER, response.text)
        message = response.json()["message"]
        self.assertIn("validation error", message)
        self.assertIn("messages", message)
        # Parsing the 4 MB body dominates; the error path itself adds little.
        self.assertLess(elapsed, 5.0)

    def test_responses_endpoint_uses_bounded_message(self):
        errors = [
            {
                "type": "string_type",
                "loc": ("body", "input"),
                "msg": "Input should be a valid string",
                "input": [{"content": MARKER * 100_000}],
            }
        ]
        status, raw = _handle("/v1/responses", errors)
        self.assertEqual(status, 400)
        self.assertLess(len(raw), GATEWAY_ERROR_BODY_CAP)
        body = json.loads(raw)
        self.assertNotIn(MARKER, body["error"]["message"])
        self.assertIn("('body', 'input')", body["error"]["message"])
        self.assertIn("Input should be a valid string", body["error"]["message"])

    def test_many_errors_are_truncated_with_marker(self):
        errors = [
            {"type": "missing", "loc": ("body", f"field_{i}"), "msg": "Field required"}
            for i in range(500)
        ]
        status, raw = _handle("/v1/chat/completions", errors)
        self.assertEqual(status, 400)
        self.assertLess(len(raw), GATEWAY_ERROR_BODY_CAP)
        message = json.loads(raw)["message"]
        self.assertTrue(message.startswith("500 validation errors: "))
        self.assertTrue(message.endswith("... [truncated]"))

    def test_small_error_is_kept_intact(self):
        errors = [
            {
                "type": "missing",
                "loc": ("body", "messages"),
                "msg": "Field required",
                "input": {"model": "m"},
            }
        ]
        status, raw = _handle("/v1/chat/completions", errors)
        self.assertEqual(status, 400)
        message = json.loads(raw)["message"]
        self.assertEqual(
            message,
            "1 validation error: [{'type': 'missing', 'loc': ('body', 'messages'), "
            "'msg': 'Field required'}]",
        )


if __name__ == "__main__":
    unittest.main()
