from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints.openai.kimi_token_provenance import (
    MISSING_LOGPROB,
    KimiTokenProvenance,
    align_token_logprobs,
)
from sglang.srt.function_call.kimik3_format import RESPONSE_OPEN, THINK_CLOSE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def ledger(pieces):
    tokenizer = SimpleNamespace(
        model=SimpleNamespace(decode_single_token_bytes=pieces.__getitem__)
    )
    result = KimiTokenProvenance(tokenizer)
    result.append(range(len(pieces)))
    return result


def ids(value):
    return value[0]["token_ids"]


def test_one_batch_thinking_and_answer_have_distinct_tokens():
    source = ledger([b"same", THINK_CLOSE.encode(), RESPONSE_OPEN.encode(), b"same"])
    assert ids(source.take("same", text=True, reasoning=True)) == [0]
    assert ids(source.take("same", text=True)) == [3]


def test_utf8_split_and_repeated_token_positions():
    source = ledger([b"\xe4\xb8", b"\xad", b"!", b"!"])
    assert ids(source.take("中", text=True)) == [0, 1]
    assert ids(source.take("!!", text=True)) == [2, 3]


def test_a_token_can_contribute_to_two_increments():
    source = ledger([b"abcdef"])
    assert ids(source.take("abc", text=True)) == [0]
    assert ids(source.take("def", text=True)) == [0]


def test_markers_in_argument_value_are_not_stripped():
    source = ledger([RESPONSE_OPEN.encode()])
    assert ids(source.take(RESPONSE_OPEN)) == [0]


def test_channel_marker_words_do_not_steal_content_tokens():
    source = ledger([RESPONSE_OPEN.encode(), b"response"])
    assert ids(source.take("response", text=True)) == [1]


def test_replayed_reasoning_uses_its_original_tokens():
    source = ledger([b"first", b" second", b" tail"])
    assert ids(source.take("first", text=True, reasoning=True)) == [0]
    assert ids(source.take(" second", text=True, reasoning=True)) == [1]
    assert ids(source.take("first second tail", text=True, replay_chars=12)) == [
        0,
        1,
        2,
    ]


def test_text_spanning_removed_wrappers():
    source = ledger([b"a", RESPONSE_OPEN.encode(), b"b"])
    assert ids(source.take("ab", text=True)) == [0, 2]


def test_mismatch_is_not_silently_assigned_arbitrary_ids():
    source = ledger([b"actual"])
    with pytest.raises(ValueError, match="no matching"):
        source.take("invented", text=True)


def test_same_token_id_generated_twice_is_not_deduplicated():
    source = ledger([b"a"])
    source.append([0])
    assert ids(source.take("aa", text=True)) == [0, 0]


def test_each_token_keeps_its_own_logprob():
    tokenizer = SimpleNamespace(
        model=SimpleNamespace(
            decode_single_token_bytes=[b"\xe4\xb8", b"\xad", b"!"].__getitem__
        )
    )
    source = KimiTokenProvenance(tokenizer)
    source.append([0, 1, 2], [-0.5, -1.0, -2.0])
    assert source.take("中", text=True) == [
        {"token_ids": [0, 1], "logprobs": [-0.5, -1.0]}
    ]
    assert source.take("!", text=True) == [{"token_ids": [2], "logprobs": [-2.0]}]


def test_token_without_a_logprob_reads_the_placeholder_never_zero():
    source = ledger([b"a", b"b"])
    source.append([0], [-0.25])
    entry = source.take("aba", text=True)[0]
    assert entry["token_ids"] == [0, 1, 0]
    assert entry["logprobs"] == [MISSING_LOGPROB, MISSING_LOGPROB, -0.25]
    assert len(entry["logprobs"]) == len(entry["token_ids"])


def test_align_token_logprobs_pairs_from_the_end_and_checks_ids():
    entries = [(-0.1, 7, "x"), (-0.2, 8, None)]
    assert align_token_logprobs([7, 8], entries) == [-0.1, -0.2]
    # The incremental abort event re-sends the last id against the whole list.
    assert align_token_logprobs([8], entries) == [-0.2]
    # No entry, an entry sampled for another id, or an unusable value.
    assert align_token_logprobs([5, 7, 8], entries) == [MISSING_LOGPROB, -0.1, -0.2]
    assert align_token_logprobs([7, 9], entries) == [-0.1, MISSING_LOGPROB]
    for bad in (None, (None, 7, None), (float("nan"), 7, None), (float("-inf"), 7)):
        assert align_token_logprobs([7], [bad]) == [MISSING_LOGPROB]
    assert align_token_logprobs([], entries) == []
    assert MISSING_LOGPROB == -9999.0
