"""CPU-only contract tests for ``--dcp-kv-layout`` resolution."""

import argparse
import unittest

from sglang.srt.arg_groups.pd_disaggregation_hook import validate_dcp_kv_layout
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _page_args(**overrides) -> ServerArgs:
    values = {
        "model_path": "dummy",
        "dcp_kv_layout": "page",
        "disaggregation_mode": "decode",
        "disaggregation_transfer_backend": "mooncake",
        "dcp_size": 2,
        "decode_attention_backend": "cutedsl_mla",
    }
    values.update(overrides)
    return ServerArgs(**values)


class TestDcpKvLayout(CustomTestCase):
    def test_defaults_to_token_and_is_exposed_by_cli(self):
        self.assertEqual(ServerArgs(model_path="dummy").dcp_kv_layout, "token")

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(["--model-path", "dummy", "--dcp-kv-layout", "page"])
        self.assertEqual(parsed.dcp_kv_layout, "page")

    def test_token_keeps_existing_config_combinations(self):
        validate_dcp_kv_layout(ServerArgs(model_path="dummy"))

    def test_token_publishes_through_parallel_context(self):
        reset_context()
        try:
            publish(ServerArgs(model_path="dummy"), role="test")
            self.assertEqual(get_parallel().dcp_kv_layout, "token")
        finally:
            reset_context()

    def test_page_accepts_final_supported_contract(self):
        validate_dcp_kv_layout(_page_args())

    def test_page_prefill_does_not_require_cutedsl(self):
        validate_dcp_kv_layout(
            _page_args(
                disaggregation_mode="prefill",
                dcp_size=1,
                decode_attention_backend="trtllm_mla",
            )
        )

    def test_page_reads_the_resolved_decode_backend(self):
        server_args = _page_args(decode_attention_backend="aiter")
        server_args._resolved_overrides = [
            ("test", {"decode_attention_backend": "cutedsl_mla"})
        ]

        validate_dcp_kv_layout(server_args)

    def test_page_rejects_unsupported_static_combinations(self):
        cases = (
            ({"disaggregation_transfer_backend": "nixl"}, "backend mooncake"),
            ({"speculative_algorithm": "EAGLE"}, "speculative decoding"),
            ({"decode_attention_backend": "aiter"}, "'cutedsl_mla'"),
        )
        for overrides, message in cases:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                validate_dcp_kv_layout(_page_args(**overrides))


if __name__ == "__main__":
    unittest.main()
