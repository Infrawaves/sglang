"""Unit tests for detokenizer decode-id prefix compaction."""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ByteTokenizer:
    """Decode token ids as UTF-8 bytes to exercise partial characters."""

    is_fast = True

    @staticmethod
    def decode(token_ids, **kwargs):
        return bytes(token_ids).decode("utf-8", errors="replace")

    def batch_decode(self, token_ids, **kwargs):
        return [self.decode(ids) for ids in token_ids]


def _manager(*, disable_batch_decode=False):
    manager = object.__new__(DetokenizerManager)
    manager.tokenizer = _ByteTokenizer()
    manager.vocab_size = 256
    manager.decode_status = {}
    manager.disable_tokenizer_batch_decode = disable_batch_decode
    manager.is_tool_call_parser_gpt_oss = False
    return manager


def _recv(
    decode_ids,
    *,
    finished_reason=None,
    read_offset=1,
    no_stop_trim=False,
):
    return SimpleNamespace(
        rids=["rid"],
        decoded_texts=[""],
        decode_ids=[decode_ids],
        read_offsets=[read_offset],
        finished_reasons=[finished_reason],
        no_stop_trim=[no_stop_trim],
        skip_special_tokens=[False],
        spaces_between_special_tokens=[True],
    )


class TestDetokenizerDecodeIdsCompaction(CustomTestCase):
    def test_streaming_keeps_only_surrogate_window(self):
        for disable_batch_decode in (False, True):
            with self.subTest(disable_batch_decode=disable_batch_decode):
                manager = _manager(disable_batch_decode=disable_batch_decode)

                output = manager._decode_batch_token_id_output(
                    _recv([ord("^"), ord("A")])
                )
                self.assertEqual(output, ["A"])
                state = manager.decode_status["rid"]
                self.assertEqual(state.decode_ids, [ord("A")])
                self.assertEqual((state.surr_offset, state.read_offset), (0, 1))

                output = manager._decode_batch_token_id_output(_recv([ord("B")]))
                self.assertEqual(output, ["B"])
                state = manager.decode_status["rid"]
                self.assertEqual(state.decode_ids, [ord("B")])
                self.assertEqual((state.surr_offset, state.read_offset), (0, 1))

    def test_incomplete_utf8_is_retained_until_character_completes(self):
        manager = _manager()

        self.assertEqual(
            manager._decode_batch_token_id_output(_recv([ord("^"), 0xE4])), [""]
        )
        state = manager.decode_status["rid"]
        self.assertEqual(state.decode_ids, [ord("^"), 0xE4])
        self.assertEqual((state.surr_offset, state.read_offset), (0, 1))

        self.assertEqual(manager._decode_batch_token_id_output(_recv([0xB8])), [""])
        self.assertEqual(state.decode_ids, [ord("^"), 0xE4, 0xB8])

        self.assertEqual(manager._decode_batch_token_id_output(_recv([0xAD])), ["中"])
        self.assertEqual(state.decode_ids, [0xE4, 0xB8, 0xAD])
        self.assertEqual((state.surr_offset, state.read_offset), (0, 3))

        self.assertEqual(
            manager._decode_batch_token_id_output(_recv([ord("Z")])), ["Z"]
        )
        self.assertEqual(state.decode_ids, [ord("Z")])
        self.assertEqual((state.surr_offset, state.read_offset), (0, 1))

    def test_non_streaming_forced_update_then_finish(self):
        manager = _manager()

        first = manager._decode_batch_token_id_output(_recv([ord("^"), ord("A")]))
        final = manager._decode_batch_token_id_output(
            _recv([ord("B")], finished_reason={"type": "length"})
        )

        self.assertEqual(first + final, ["A", "B"])
        self.assertNotIn("rid", manager.decode_status)

    def test_stop_string_trim_after_compaction(self):
        for no_stop_trim, expected_final in ((False, ""), (True, "STOP")):
            with self.subTest(no_stop_trim=no_stop_trim):
                manager = _manager()
                first = manager._decode_batch_token_id_output(
                    _recv([ord("^"), ord("A")])
                )
                final = manager._decode_batch_token_id_output(
                    _recv(
                        list(b"STOP-trailing"),
                        finished_reason={"type": "stop", "matched": "STOP"},
                        no_stop_trim=no_stop_trim,
                    )
                )

                self.assertEqual(first, ["A"])
                self.assertEqual(final, [expected_final])
                self.assertNotIn("rid", manager.decode_status)


if __name__ == "__main__":
    unittest.main()
