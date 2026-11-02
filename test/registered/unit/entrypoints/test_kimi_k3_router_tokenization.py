"""Run the router's actual request-schema parity tests in the CPU suite."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_TEST_PATH = (
    Path(__file__).resolve().parents[4]
    / "experimental/sgl-router/tests/test_kimi_k3_tokenization.py"
)
_SPEC = importlib.util.spec_from_file_location("router_kimi_k3_tests", _TEST_PATH)
_TESTS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TESTS)


class TestKimiK3RouterRealSchema(_TESTS.RealSchemaImageTests):
    @classmethod
    def setUpClass(cls):
        # CPU CI already installs SGLang; missing schema dependencies must fail.
        with patch.dict(os.environ, {"SGLANG_REQUIRE_KIMI_K3_SCHEMA_TESTS": "1"}):
            super().setUpClass()


if __name__ == "__main__":
    unittest.main()
