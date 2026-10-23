"""CPU-only authentication, configuration and native-entry boundary tests."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

# CI registration is parsed statically; avoid importing heavyweight sglang.
if "sglang.test.ci.ci_register" in sys.modules:
    from sglang.test.ci.ci_register import register_cpu_ci
else:

    def register_cpu_ci(**kwargs):
        """Runtime no-op when testing without the serving package installed."""


register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[3]
_SRT = _ROOT / "python/sglang/srt"


def extract(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    source = tree.body
    if class_name:
        source = next(n for n in source if getattr(n, "name", None) == class_name).body
    node = next(n for n in source if getattr(n, "name", None) == name)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def load_auth():
    spec = importlib.util.spec_from_file_location(
        "_cache_test_auth", _SRT / "utils/auth.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestCacheIdentity(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        self.app = app

        @app.post("/health")
        @app.post("/ready")
        @app.post("/metrics")
        @app.post("/identity")
        async def identity(request: Request):
            return {
                "account": getattr(request.state, "api_key", None),
                "cache_identity": getattr(request.state, "cache_identity", None),
            }

        load_auth().add_api_key_middleware(
            app, api_key="verified-secret", admin_api_key=None
        )
        self.client = TestClient(app)

    def test_verified_credential_authenticates_without_cache_identity(self):
        response = self.client.post(
            "/identity", headers={"Authorization": "Bearer verified-secret"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"account": None, "cache_identity": None})
        self.assertNotIn("verified-secret", response.text)

    def test_body_and_untrusted_identity_header_do_not_create_state_identity(self):
        response = self.client.post(
            "/identity",
            headers={
                "Authorization": "Bearer verified-secret",
                "X-Account-Id": "victim",
            },
            json={"api_key": "victim", "user": "victim"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"account": None, "cache_identity": None})

    def test_invalid_credential_does_not_reach_handler(self):
        for credential in ("Bearer unknown", "Basic verified-secret", None):
            with self.subTest(credential=credential):
                headers = {"Authorization": credential} if credential else {}
                response = self.client.post("/identity", headers=headers)
                self.assertEqual(response.status_code, 401)

    def test_unverified_credentials_on_exempt_routes_do_not_create_identity(self):
        for path in ("/health", "/ready", "/metrics"):
            with self.subTest(path=path):
                response = self.client.post(
                    path, headers={"Authorization": "Bearer unknown"}
                )
                self.assertEqual(response.status_code, 200)
                self.assertIsNone(response.json()["account"])

    def test_non_ascii_bearer_on_exempt_routes_keeps_auth_bypass(self):
        for path in ("/health", "/ready", "/metrics"):
            with self.subTest(path=path):
                response = self.client.post(
                    path, headers=[(b"authorization", b"Bearer \xe9")]
                )
                self.assertEqual(response.status_code, 200)
                self.assertIsNone(response.json()["account"])

    def test_open_server_does_not_trust_arbitrary_bearer_identity(self):
        app = FastAPI()

        @app.post("/identity")
        async def identity(request: Request):
            return {"account": getattr(request.state, "api_key", None)}

        load_auth().add_api_key_middleware(app, api_key=None, admin_api_key=None)
        with TestClient(app) as client:
            response = client.post(
                "/identity", headers={"Authorization": "Bearer unknown"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["account"])

    def test_admin_bearer_authenticates_without_cache_identity(self):
        app = FastAPI()
        auth = load_auth()

        @app.post("/identity")
        @auth.auth_level(auth.AuthLevel.ADMIN_FORCE)
        async def identity(request: Request):
            return {"account": getattr(request.state, "api_key", None)}

        auth.add_api_key_middleware(app, api_key=None, admin_api_key="admin-secret")
        with TestClient(app) as client:
            response = client.post(
                "/identity", headers={"Authorization": "Bearer admin-secret"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["account"])

    def test_trusted_upstream_state_identity_is_preserved(self):
        @self.app.middleware("http")
        async def trusted_upstream(request, call_next):
            request.state.cache_identity = "trusted-tenant"
            return await call_next(request)

        response = self.client.post(
            "/identity", headers={"Authorization": "Bearer verified-secret"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"account": None, "cache_identity": "trusted-tenant"}
        )


class TestCacheConfiguration(unittest.TestCase):
    def setUp(self):
        self.validate = extract(
            _SRT / "arg_groups/validation_hook.py",
            "validate_cache_accounting",
            {"math": math, "urlparse": urlparse},
        )
        self.config = SimpleNamespace(
            enable_cache_accounting=True,
            cache_redis_url="redis://127.0.0.1:6379/0",
            cache_namespace="cache-test-namespace",
            cache_timeout=0.2,
            tokenizer_worker_num=8,
            detokenizer_worker_num=4,
            disaggregation_mode="null",
            dcp_size=4,
            checkpoint_engine_wait_weights_before_ready=False,
        )

    def test_valid_configuration(self):
        for mode in ("null", "decode"):
            with self.subTest(mode=mode):
                self.config.disaggregation_mode = mode
                self.validate(self.config)

    def test_disabled_needs_no_redis_or_namespace(self):
        self.validate(SimpleNamespace(enable_cache_accounting=False))

    def test_prefill_ignores_cache_configuration(self):
        self.validate(
            SimpleNamespace(enable_cache_accounting=True, disaggregation_mode="prefill")
        )

    def test_checkpoint_engine_can_wait_for_initial_weights(self):
        self.config.checkpoint_engine_wait_weights_before_ready = True
        self.validate(self.config)

    def test_invalid_configuration_fails_before_serving(self):
        for field, value in [
            ("cache_redis_url", None),
            ("cache_redis_url", "redis://"),
            ("cache_redis_url", "https://example.com"),
            ("cache_redis_url", "redis://user:secret@localhost/0"),
            ("cache_redis_url", "redis://localhost/0?password=secret"),
            ("cache_namespace", " "),
            ("cache_timeout", 0),
            ("cache_timeout", float("nan")),
        ]:
            with self.subTest(field=field, value=value):
                cfg = SimpleNamespace(**vars(self.config))
                setattr(cfg, field, value)
                with self.assertRaises(ValueError):
                    self.validate(cfg)


class TestNativeAccountBoundary(unittest.TestCase):
    def setUp(self):
        class GenerateReqInput:
            pass

        self.input_class = GenerateReqInput
        self.resolve_identity = extract(
            _SRT / "managers/tokenizer_manager.py",
            "_resolve_cache_identity",
            {
                "GenerateReqInput": GenerateReqInput,
                "DisaggregationMode": SimpleNamespace(PREFILL="prefill"),
            },
            "TokenizerManager",
        )
        self.obj = GenerateReqInput()
        self.obj.cache_ttl_seconds = 300
        self.obj.cache_user_id = "spoofed-from-body"
        self.manager = SimpleNamespace(disaggregation_mode="decode")

    def test_api_adapter_identity_replaces_native_body_identity(self):
        request = SimpleNamespace(
            state=SimpleNamespace(
                cache_identity="trusted-body-user", api_key="retired-bearer-identity"
            )
        )
        self.resolve_identity(self.manager, self.obj, request)
        self.assertEqual(self.obj.cache_user_id, "trusted-body-user")

    def test_native_http_body_cannot_supply_cache_identity(self):
        request = SimpleNamespace(state=SimpleNamespace())
        self.resolve_identity(self.manager, self.obj, request)
        self.assertIsNone(self.obj.cache_user_id)

    def test_legacy_bearer_state_does_not_replace_missing_body_identity(self):
        for identity in (None, "", 123):
            with self.subTest(identity=identity):
                self.obj.cache_user_id = "spoofed-from-body"
                request = SimpleNamespace(
                    state=SimpleNamespace(
                        cache_identity=identity, api_key="retired-bearer-identity"
                    )
                )
                self.resolve_identity(self.manager, self.obj, request)
                self.assertIsNone(self.obj.cache_user_id)

    def test_trusted_in_process_identity_is_preserved(self):
        self.obj.cache_user_id = "trusted-in-process-user"
        self.resolve_identity(self.manager, self.obj, None)
        self.assertEqual(self.obj.cache_user_id, "trusted-in-process-user")

    def test_prefill_ignores_cache_identity(self):
        self.manager.disaggregation_mode = "prefill"
        request = SimpleNamespace(state=SimpleNamespace(cache_identity="body-user"))
        self.resolve_identity(self.manager, self.obj, request)
        self.assertIsNone(self.obj.cache_user_id)
        self.assertIsNone(self.obj.cache_ttl_seconds)


if __name__ == "__main__":
    unittest.main()
