"""Unit tests for serving_chat._truncate_kimi_k3_videos."""

import copy
import unittest

from sglang.srt.entrypoints.openai.serving_chat import _truncate_kimi_k3_videos
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def vid(i):
    return {
        "type": "video_url",
        "video_url": {"url": f"https://media.invalid/v{i}.mp4"},
    }


def img(i):
    return {
        "type": "image_url",
        "image_url": {"url": f"https://media.invalid/i{i}.png"},
    }


def txt(s):
    return {"type": "text", "text": s}


class NotIterated(str):
    """A str content the count must not walk character by character."""

    def __iter__(self):
        raise AssertionError("str content was iterated")


class TestTruncateKimiK3Videos(CustomTestCase):
    def test_keeps_the_last_two_videos_across_messages(self):
        messages = [
            {"role": "system", "content": NotIterated("sys " * 1000)},
            {"role": "user", "content": [vid(0), txt("a"), vid(1), img(0)]},
            {"role": "assistant", "content": None},
            {"role": "tool", "content": NotIterated('{"type": "video_url"}')},
            {"role": "user", "content": [txt("b"), vid(2), "raw", vid(3)]},
            {"role": "user", "content": [vid(4)]},
        ]
        _truncate_kimi_k3_videos(messages)
        self.assertEqual(messages[1]["content"], [txt("a"), img(0)])
        self.assertEqual(messages[4]["content"], [txt("b"), "raw", vid(3)])
        self.assertEqual(messages[5]["content"], [vid(4)])
        self.assertIsNone(messages[2]["content"])
        self.assertEqual(messages[3]["content"], '{"type": "video_url"}')

    def test_two_or_fewer_videos_leave_messages_unchanged(self):
        for messages in (
            [{"role": "user", "content": NotIterated("x" * 500_000)}],
            [{"role": "user"}, {"role": "user", "content": None}],
            [{"role": "user", "content": []}],
            [
                {"role": "user", "content": [vid(0), img(0)]},
                {"role": "user", "content": NotIterated("between")},
                {"role": "user", "content": [txt("t"), vid(1)]},
            ],
        ):
            before = copy.deepcopy(messages)
            _truncate_kimi_k3_videos(messages)
            self.assertEqual(messages, before)
            for got, want in zip(messages, before):
                self.assertIs(type(got.get("content")), type(want.get("content")))

    def test_str_and_text_list_content_count_the_same(self):
        text = "video_url " * 100
        as_str = [{"role": "user", "content": NotIterated(text)}] + [
            {"role": "user", "content": [vid(i)]} for i in range(3)
        ]
        as_list = [{"role": "user", "content": [txt(text)]}] + [
            {"role": "user", "content": [vid(i)]} for i in range(3)
        ]
        _truncate_kimi_k3_videos(as_str)
        _truncate_kimi_k3_videos(as_list)
        self.assertEqual(
            [m["content"] for m in as_str[1:]], [m["content"] for m in as_list[1:]]
        )
        self.assertEqual(as_list[1]["content"], [])
        self.assertEqual(as_str[0]["content"], text)


if __name__ == "__main__":
    unittest.main()
