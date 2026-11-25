"""Tests for resuming /generate requests a retiring decode instance handed off."""

import asyncio

import orjson
from fastapi.responses import ORJSONResponse
from sglang_router import mini_lb
from sglang_router.migration_resume import (
    ResumeStitcher,
    SseEventSplitter,
    build_resume_request,
    resumable,
)


def _hand_off(text, origin, output, completion_tokens, rid="orig"):
    return {
        "text": text,
        "output_ids": output,
        "meta_info": {
            "id": rid,
            "prompt_tokens": len(origin),
            "completion_tokens": completion_tokens,
            "finish_reason": {
                "type": "migrated",
                "origin_input_ids": origin,
                "output_ids": output,
                "kv_flushed_tokens": 64,
            },
        },
    }


def test_resume_request_continues_with_the_remaining_budget():
    """The resubmitted prompt holds every token so far, so the budget must
    shrink by what was generated, while the storage key inputs stay."""
    original = {
        "text": "hi",
        "rid": "r1",
        "extra_key": "tenant",
        "bootstrap_room": 7,
        "sampling_params": {"max_new_tokens": 10, "min_new_tokens": 4},
    }
    finish = {"origin_input_ids": [1, 2], "output_ids": [3, 4, 5]}

    resumed = build_resume_request(original, finish, attempt=1)

    assert resumed["input_ids"] == [1, 2, 3, 4, 5]
    assert "text" not in resumed and "bootstrap_room" not in resumed
    assert resumed["sampling_params"] == {"max_new_tokens": 7, "min_new_tokens": 1}
    assert resumed["extra_key"] == "tenant"
    assert resumed["rid"] == "r1-migrated-1"
    # A decode refuses to hand off a request that does not opt in.
    assert resumed["allow_migration"] is True


def test_two_hand_offs_stitch_into_one_request():
    """Each resumed segment reports only its own text and tokens; the client
    must see the whole generation under the original id and prompt size."""
    first = _hand_off("Hello", origin=[1, 2], output=[10, 11], completion_tokens=2)
    stitcher = ResumeStitcher.for_first_hand_off(
        first, first["meta_info"]["finish_reason"]
    )
    stitcher.absorb(first, first["meta_info"]["finish_reason"])
    second = _hand_off(
        " wor", origin=[1, 2, 10, 11], output=[12], completion_tokens=1, rid="x"
    )
    stitcher.absorb(second, second["meta_info"]["finish_reason"])

    final = stitcher.rewrite(
        {
            "text": "ld",
            "output_ids": [13],
            "meta_info": {
                "id": "y",
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "finish_reason": {"type": "stop"},
            },
        }
    )

    assert final["text"] == "Hello world"
    assert final["output_ids"] == [10, 11, 12, 13]
    assert final["meta_info"]["id"] == "orig"
    assert final["meta_info"]["prompt_tokens"] == 2
    assert final["meta_info"]["completion_tokens"] == 4
    assert final["meta_info"]["finish_reason"] == {"type": "stop"}


def test_sse_splitter_rejoins_events_split_across_reads():
    """The hand-off event carries the whole history and spans many reads."""
    splitter = SseEventSplitter()
    events = []
    for chunk in (b'data: {"a"', b": 1}\n", b"\ndata: [DO", b"NE]\n\n"):
        events += splitter.feed(chunk)
    assert events == [b'data: {"a": 1}\n\n', b"data: [DONE]\n\n"]
    assert splitter.flush() == []


class _FakeContent:
    def __init__(self, data: bytes):
        self._data = data

    async def iter_chunked(self, size):
        for i in range(0, len(self._data), 7):
            yield self._data[i : i + 7]


class _FakeResponse:
    def __init__(self, events):
        self.content = _FakeContent(
            b"".join(b"data: " + orjson.dumps(e) + b"\n\n" for e in events)
            + b"data: [DONE]\n\n"
        )


class _FakeSession:
    def __init__(self, decode_events):
        self.posts = []
        self._decode_events = decode_events

    async def post(self, url, json):
        self.posts.append((url, json))
        return _FakeResponse(self._decode_events)


def _load_balancer():
    lb = mini_lb.MiniLoadBalancer.__new__(mini_lb.MiniLoadBalancer)
    lb.timeout = 1
    lb.prefill_urls = ["http://p:1"]
    lb.prefill_bootstrap_ports = [8998]
    lb.decode_urls = ["http://d:1"]
    lb.test_external_dp_routing = False
    return lb


def test_stream_resumes_without_ending_the_client_stream():
    """The hand-off must not end the client's stream: its finish and [DONE]
    are swallowed and the resumed request's chunks follow, stitched."""
    lb = _load_balancer()
    hand_off = _hand_off("Hel", origin=[1], output=[10], completion_tokens=1)
    source = _FakeResponse([hand_off])
    session = _FakeSession(
        [
            {
                "text": "lo",
                "output_ids": [11],
                "meta_info": {
                    "id": "n",
                    "completion_tokens": 1,
                    "finish_reason": {"type": "stop"},
                },
            }
        ]
    )

    async def collect():
        return [
            chunk
            async for chunk in lb._stream_with_resume(
                session, {"text": "x", "stream": True}, source, "generate"
            )
        ]

    chunks = asyncio.run(collect())

    payloads = [orjson.loads(c[len(b"data: ") :]) for c in chunks if b"[DONE]" not in c]
    assert [p["meta_info"]["finish_reason"] for p in payloads] == [
        None,
        {"type": "stop"},
    ]
    assert payloads[-1]["text"] == "Hello"
    assert payloads[-1]["meta_info"]["completion_tokens"] == 2
    assert sum(b"[DONE]" in c for c in chunks) == 1
    resumed = [body for url, body in session.posts if url.startswith("http://d:1")]
    assert resumed[0]["input_ids"] == [1, 10]


def test_non_stream_resumes_on_another_pair(monkeypatch):
    lb = _load_balancer()
    hand_off = _hand_off("Hel", origin=[1], output=[10], completion_tokens=1)
    done = {
        "text": "lo",
        "output_ids": [11],
        "meta_info": {
            "id": "n",
            "completion_tokens": 1,
            "finish_reason": {"type": "length"},
        },
    }
    sent = []

    async def generate_once(request, prefill, decode, endpoint):
        sent.append(request)
        return ORJSONResponse(content=done)

    monkeypatch.setattr(lb, "_generate_once", generate_once)
    response = asyncio.run(
        lb._resume_migrated(
            {"text": "x", "sampling_params": {"max_new_tokens": 5}},
            ORJSONResponse(content=hand_off),
            "generate",
        )
    )

    body = orjson.loads(response.body)
    assert body["text"] == "Hello"
    assert body["meta_info"]["finish_reason"] == {"type": "length"}
    assert sent[0]["sampling_params"]["max_new_tokens"] == 4
    assert sent[0]["bootstrap_host"] == "p"


class _RecordingLoadBalancer:
    def __init__(self):
        self.dispatched = []

    def select_pair(self):
        return "http://p:1", 8998, "http://d:1"

    async def generate(self, request, prefill, decode, endpoint):
        self.dispatched.append(request)
        return ORJSONResponse(content={})

    async def generate_stream(self, request, prefill, decode, endpoint):
        self.dispatched.append(request)
        return ORJSONResponse(content={})


def test_generate_opts_in_only_requests_it_can_resume(monkeypatch):
    """Without allow_migration a retiring decode ends the request with a 503;
    with it on a batch or parallel-sampled request, the client would see the
    hand-off instead of a finish, since only single requests are resumed."""
    recording = _RecordingLoadBalancer()
    monkeypatch.setattr(mini_lb, "lb", recording)
    for body, expected in (
        ({"text": "x"}, True),
        ({"text": "x", "stream": True}, True),
        ({"text": "x", "sampling_params": {"n": 2}}, None),
        ({"text": ["x", "y"]}, None),
    ):
        asyncio.run(mini_lb.handle_generate_request(dict(body)))
        assert recording.dispatched[-1].get("allow_migration") is expected, body
    assert not resumable({"sampling_params": {"n": 3}})
