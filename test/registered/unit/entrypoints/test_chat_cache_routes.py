"""Exercise the production chat routes without loading model-serving workers."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import unittest
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from uuid import UUID

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.testclient import TestClient

# CI registration is parsed statically; avoid importing heavyweight sglang.
if "sglang.test.ci.ci_register" in sys.modules:
    from sglang.test.ci.ci_register import register_cpu_ci
else:

    def register_cpu_ci(**kwargs):
        """Runtime no-op when testing without the serving package installed."""


register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4]
_SRT = _ROOT / "python/sglang/srt"


def _load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _production_protocol():
    # The protocol itself is lightweight; only sglang.utils normally loads the
    # package's GPU-serving imports. Schema conversion is unused by these tests.
    utils = ModuleType("sglang.utils")
    utils.convert_json_schema_to_str = json.dumps
    with patch.dict(sys.modules, {"sglang.utils": utils}):
        return _load_source(
            "_cache_route_protocol", _SRT / "entrypoints/openai/protocol.py"
        )


_PROTOCOL = _production_protocol()
_AUTH = _load_source("_cache_route_auth", _SRT / "utils/auth.py")


def _register_production_routes(app):
    source = _SRT / "entrypoints/http_server.py"
    tree = ast.parse(source.read_text())
    selected = {"validate_json_request", "openai_v1_chat_completions"}
    nodes = [node for node in tree.body if getattr(node, "name", None) in selected]
    assert {node.name for node in nodes} == selected
    namespace = {
        "__name__": __name__,
        "app": app,
        "Depends": Depends,
        "Request": Request,
        "RequestValidationError": RequestValidationError,
        "ChatCompletionRequest": _PROTOCOL.ChatCompletionRequest,
        "ChatCompletionResponse": _PROTOCOL.ChatCompletionResponse,
        "ORJSONResponse": ORJSONResponse,
        "CORSMiddleware": CORSMiddleware,
    }
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *nodes,
            *[
                node
                for node in tree.body
                if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "app"
                and node.value.func.attr == "add_middleware"
                and node.value.args
                and isinstance(node.value.args[0], ast.Name)
                and node.value.args[0].id == "CORSMiddleware"
            ],
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(source), "exec"), namespace)
    return namespace["openai_v1_chat_completions"]


class _RecordingHandler:
    def __init__(self):
        self.calls = []
        self.response = None

    async def handle_request(self, request, raw_request):
        self.calls.append((request, raw_request.state.api_key))
        return self.response if self.response is not None else {"accepted": True}


class TestChatCacheRoutes(unittest.TestCase):
    path = "/v1/chat/completions"

    def setUp(self):
        self.app = FastAPI()
        self.handler = _RecordingHandler()
        self.app.state.openai_serving_chat = self.handler
        self.endpoint = _register_production_routes(self.app)
        _AUTH.add_api_key_middleware(
            self.app, api_key="verified-key", admin_api_key=None
        )
        self.client = TestClient(self.app)
        self.headers = {"Authorization": "Bearer verified-key"}

    def body(self, ttl="5m", stream=False):
        options = {"mode": "implicit"}
        if ttl is not None:
            options["ttl"] = ttl
        return {
            "model": "model-v1",
            "messages": [{"role": "user", "content": "cache this prompt"}],
            "max_tokens": 2,
            "prompt_cache_options": options,
            "stream": stream,
            "stream_options": {"include_usage": True} if stream else None,
        }

    def test_standard_path_uses_the_production_handler(self):
        routes = [r for r in self.app.routes if getattr(r, "path", None) == self.path]
        self.assertEqual(len(routes), 1)
        self.assertIs(routes[0].endpoint, self.endpoint)
        self.assertIn(self.path, self.app.openapi()["paths"])

    def test_both_ttl_tiers_and_authenticated_identity_reach_handler(self):
        for ttl in ("5m", "1h"):
            with self.subTest(ttl=ttl):
                response = self.client.post(
                    self.path, json=self.body(ttl), headers=self.headers
                )
                self.assertEqual(response.status_code, 200)
                request, api_key = self.handler.calls[-1]
                self.assertEqual(request.prompt_cache_options.mode, "implicit")
                self.assertEqual(request.prompt_cache_options.ttl, ttl)
                self.assertEqual(
                    api_key, "api-key:" + sha256(b"verified-key").hexdigest()
                )

    def test_missing_ttl_defaults_to_five_minutes(self):
        response = self.client.post(
            self.path, json=self.body(ttl=None), headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        request, _ = self.handler.calls[-1]
        self.assertEqual(request.prompt_cache_options.mode, "implicit")
        self.assertEqual(request.prompt_cache_options.ttl, "5m")

    def test_stream_usage_options_reach_handler(self):
        response = self.client.post(
            self.path, json=self.body(stream=True), headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        request, _ = self.handler.calls[-1]
        self.assertTrue(request.stream)
        self.assertTrue(request.stream_options.include_usage)

    def test_authentication_and_json_media_type_are_enforced(self):
        response = self.client.post(self.path, json=self.body())
        self.assertEqual(response.status_code, 401)
        response = self.client.post(
            self.path,
            content=json.dumps(self.body()),
            headers={**self.headers, "Content-Type": "text/plain"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.handler.calls, [])

    def test_invalid_bearer_header_is_rejected(self):
        for value in ("Bearer wrong-key", "", "Basic verified-key", "verified-key"):
            with self.subTest(value=value):
                response = self.client.post(
                    self.path, json=self.body(), headers={"Authorization": value}
                )
                self.assertEqual(response.status_code, 401)
        self.assertEqual(self.handler.calls, [])

    def test_bearer_scheme_is_case_insensitive_and_preserves_digest_identity(self):
        for headers in (
            {"Authorization": "Bearer verified-key"},
            {"Authorization": "bearer verified-key"},
        ):
            with self.subTest(headers=headers):
                response = self.client.post(
                    self.path, json=self.body(), headers=headers
                )
                self.assertEqual(response.status_code, 200)
                _, api_key = self.handler.calls[-1]
                self.assertEqual(
                    api_key, "api-key:" + sha256(b"verified-key").hexdigest()
                )

    def test_missing_cache_options_are_preserved_for_server_default(self):
        body = self.body()
        del body["prompt_cache_options"]
        response = self.client.post(self.path, json=body, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        request, _ = self.handler.calls[-1]
        self.assertIsNone(request.prompt_cache_options)

    def test_body_api_key_cannot_override_authenticated_identity(self):
        body = self.body()
        body["api_key"] = "untrusted-body-key"
        body["user"] = "another-user"
        response = self.client.post(self.path, json=body, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        _, api_key = self.handler.calls[-1]
        self.assertEqual(api_key, "api-key:" + sha256(b"verified-key").hexdigest())

    def test_invalid_cache_options_are_rejected_before_handler(self):
        for options in (
            {"mode": "implicit", "ttl": "2h"},
            {"mode": "explicit", "ttl": "5m"},
        ):
            with self.subTest(options=options):
                body = self.body()
                body["prompt_cache_options"] = options
                response = self.client.post(self.path, json=body, headers=self.headers)
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.handler.calls, [])

    def test_json_cache_headers_and_usage_come_from_the_same_snapshot(self):
        for ttl, read, write in (("5m", 200, 700), ("1h", 200, 700), ("1h", 900, 0)):
            with self.subTest(ttl=ttl, write=write):
                snapshot = _PROTOCOL.CacheUsage(
                    read_tokens=read,
                    creation_5m_tokens=write if ttl == "5m" else 0,
                    creation_1h_tokens=write if ttl == "1h" else 0,
                    normal_input_tokens=100,
                )
                self.handler.response = _PROTOCOL.ChatCompletionResponse(
                    id="cache-json",
                    model="model-v1",
                    metadata={"trace_id": UUID("12345678-1234-5678-1234-567812345678")},
                    choices=[
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    usage=_PROTOCOL.UsageInfo(
                        prompt_tokens=1000,
                        completion_tokens=100,
                        total_tokens=1100,
                        prompt_tokens_details={
                            "cached_tokens": read,
                            "cache_write_tokens": write,
                        },
                        cache_usage=snapshot,
                    ),
                )
                response = self.client.post(
                    self.path,
                    json=self.body(ttl),
                    headers={**self.headers, "Origin": "https://client.example"},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.headers["Msh-Usage-Cache-Write-Tokens-5m"],
                    str(snapshot.creation_5m_tokens),
                )
                self.assertEqual(
                    response.headers["Msh-Usage-Cache-Write-Tokens-1h"],
                    str(snapshot.creation_1h_tokens),
                )
                usage = response.json()["usage"]
                self.assertEqual(
                    response.json()["metadata"],
                    {"trace_id": "12345678-1234-5678-1234-567812345678"},
                )
                self.assertNotIn("cache_usage", usage)
                self.assertEqual(usage["prompt_tokens"], 1000)
                self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], read)
                self.assertEqual(
                    usage["prompt_tokens_details"]["cache_write_tokens"],
                    sum(
                        int(response.headers[header])
                        for header in snapshot.response_headers()
                    ),
                )
                exposed = response.headers["access-control-expose-headers"].lower()
                for header in snapshot.response_headers():
                    self.assertIn(header.lower(), exposed)

    def test_legacy_json_and_errors_do_not_gain_cache_headers(self):
        for result in (
            {"accepted": True},
            ORJSONResponse({"error": "unavailable"}, status_code=503),
        ):
            with self.subTest(result=result):
                self.handler.response = result
                response = self.client.post(
                    self.path, json=self.body(), headers=self.headers
                )
                self.assertNotIn("Msh-Usage-Cache-Write-Tokens-5m", response.headers)
                self.assertNotIn("Msh-Usage-Cache-Write-Tokens-1h", response.headers)


if __name__ == "__main__":
    unittest.main()
