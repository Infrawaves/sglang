"""Unit tests for request-level recent-image sampling."""

import argparse
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.serving_hook import handle_multimodal
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
    _recent_image_keep_count,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRecentImageSampling(CustomTestCase):
    @staticmethod
    def _manager():
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.mm_processor = SimpleNamespace(
            mm_tokens=SimpleNamespace(image_token_id=99, image_token="<image>")
        )
        manager.image_token_id = 99
        manager.recent_image_max_count = 50
        manager.recent_image_keep_ratio = 0.8
        manager.enable_recent_image_sampling_log = False
        return manager

    def test_keep_count_uses_configured_tail_cap_and_floor(self):
        defaults = (50, 0.8)
        self.assertEqual(_recent_image_keep_count(49, *defaults), 49)
        self.assertEqual(_recent_image_keep_count(50, *defaults), 50)
        self.assertEqual(_recent_image_keep_count(51, *defaults), 40)
        self.assertEqual(_recent_image_keep_count(62, *defaults), 49)
        self.assertEqual(_recent_image_keep_count(63, *defaults), 50)
        self.assertEqual(_recent_image_keep_count(106, *defaults), 50)

        # A request-specific startup configuration must control both the cap
        # and the ratio; the default 50/0.8 values must not be baked in.
        self.assertEqual(_recent_image_keep_count(9, 6, 0.5), 4)
        self.assertEqual(_recent_image_keep_count(12, 6, 0.5), 6)
        self.assertEqual(_recent_image_keep_count(21, 10, 0.25), 5)
        self.assertEqual(_recent_image_keep_count(21, 6, 0.8), 6)

    def test_custom_sampling_configuration_is_used_when_trimming(self):
        image_data = [f"image-{index}" for index in range(12)]
        req = GenerateReqInput(
            rid="custom-config",
            input_ids=[7, *([99] * 12), 8],
            image_data=image_data,
        )
        req.normalize_batch_and_arguments()

        manager = self._manager()
        manager.recent_image_max_count = 10
        manager.recent_image_keep_ratio = 0.5
        manager._truncate_recent_images(req)

        # floor(12 * .5) is six, so the ratio is applied before the custom
        # cap; the default 50/.8 values would have retained all 12 images.
        self.assertEqual(req.image_data, image_data[-6:])
        self.assertEqual(req.input_ids, [7, *([99] * 6), 8])

    def test_sampling_log_is_emitted_only_when_enabled(self):
        image_data = [f"image-{index}" for index in range(51)]
        req = GenerateReqInput(
            rid="logging-config",
            input_ids=[7, *([99] * 51), 8],
            image_data=image_data,
        )
        req.normalize_batch_and_arguments()

        manager = self._manager()
        with patch(
            "sglang.srt.managers.tokenizer_manager.logger"
        ) as logger_mock:
            manager._truncate_recent_images(req)
            logger_mock.info.assert_not_called()

        manager.enable_recent_image_sampling_log = True
        req = GenerateReqInput(
            rid="logging-enabled",
            input_ids=[7, *([99] * 51), 8],
            image_data=image_data,
        )
        req.normalize_batch_and_arguments()

        with patch(
            "sglang.srt.managers.tokenizer_manager.logger"
        ) as logger_mock:
            manager._truncate_recent_images(req)
            self.assertEqual(logger_mock.info.call_count, 2)
            logger_mock.info.assert_any_call(
                "Recent-image sampling received request %s with %d image(s).",
                "logging-enabled",
                51,
            )
            logger_mock.info.assert_any_call(
                "Recent-image sampling truncated request %s from %d to %d "
                "image(s); processing the latest images only.",
                "logging-enabled",
                51,
                40,
            )

    def test_single_request_trims_images_hashes_and_input_ids(self):
        image_data = [f"image-{index}" for index in range(106)]
        mm_hashes = [f"hash-{index}" for index in range(106)]
        mm_content_hashes = [f"content-{index}" for index in range(106)]
        req = GenerateReqInput(
            rid="request-1",
            input_ids=[7, *([99] * 106), 8],
            image_data=image_data,
            mm_hashes=mm_hashes,
            mm_content_hashes=mm_content_hashes,
        )
        req.normalize_batch_and_arguments()

        self._manager()._truncate_recent_images(req)

        self.assertEqual(req.image_data, image_data[-50:])
        self.assertEqual(req.mm_hashes, mm_hashes[-50:])
        self.assertEqual(req.mm_content_hashes, mm_content_hashes[-50:])
        self.assertEqual(req.input_ids, [7, *([99] * 50), 8])

    def test_batch_request_trims_text_placeholders_per_request(self):
        first_images = [f"first-{index}" for index in range(51)]
        second_images = [f"second-{index}" for index in range(63)]
        req = GenerateReqInput(
            rid=["first", "second"],
            text=[
                "prefix " + "<image>" * len(first_images),
                "prefix " + "<image>" * len(second_images),
            ],
            image_data=[first_images, second_images],
            mm_hashes=[
                [f"first-hash-{index}" for index in range(51)],
                [f"second-hash-{index}" for index in range(63)],
            ],
        )
        req.normalize_batch_and_arguments()

        self._manager()._truncate_recent_images(req)

        self.assertEqual(req.image_data[0], first_images[-40:])
        self.assertEqual(req.image_data[1], second_images[-50:])
        self.assertEqual(len(req.mm_hashes[0]), 40)
        self.assertEqual(len(req.mm_hashes[1]), 50)
        self.assertEqual(req.text[0].count("<image>"), 40)
        self.assertEqual(req.text[1].count("<image>"), 50)

    def test_language_only_worker_resolves_k3_media_placeholder(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.mm_processor = None
        manager.image_token_id = None
        manager.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(media_placeholder_token_id=99)
        )
        manager.tokenizer = SimpleNamespace(
            convert_ids_to_tokens=lambda token_ids: ["<|media_pad|>"]
        )

        self.assertEqual(
            manager._image_placeholder_spec(), (99, "<|media_pad|>")
        )

    @staticmethod
    def _parse_server_args(argv):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(["--model", "dummy", *argv])
        return ServerArgs.from_cli_args(args)

    def test_sampling_startup_args_have_defaults_and_parse_custom_values(self):
        defaults = self._parse_server_args([])
        self.assertEqual(defaults.recent_image_max_count, 50)
        self.assertEqual(defaults.recent_image_keep_ratio, 0.8)
        self.assertFalse(defaults.enable_recent_image_sampling_log)

        custom = self._parse_server_args(
            [
                "--recent-image-max-count",
                "12",
                "--recent-image-keep-ratio",
                "0.5",
                "--enable-recent-image-sampling-log",
            ]
        )
        self.assertEqual(custom.recent_image_max_count, 12)
        self.assertEqual(custom.recent_image_keep_ratio, 0.5)
        self.assertTrue(custom.enable_recent_image_sampling_log)

    def test_sampling_startup_args_validate_bounds(self):
        with self.assertRaisesRegex(ValueError, "recent-image-max-count"):
            handle_multimodal(
                ServerArgs(model_path="dummy", recent_image_max_count=0)
            )
        with self.assertRaisesRegex(ValueError, "recent-image-keep-ratio"):
            handle_multimodal(
                ServerArgs(model_path="dummy", recent_image_keep_ratio=0.0)
            )
        with self.assertRaisesRegex(ValueError, "recent-image-keep-ratio"):
            handle_multimodal(
                ServerArgs(model_path="dummy", recent_image_keep_ratio=1.01)
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
