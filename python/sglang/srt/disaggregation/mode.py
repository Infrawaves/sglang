"""Disaggregation mode, kept free of heavy imports for request and IPC code."""

from enum import Enum

FAKE_BOOTSTRAP_HOST = "2.2.2.2"
# Width of the PD top-logprobs wire buffers; the tokenizer rejects larger requests.
MAX_DISAGGREGATION_TOP_LOGPROBS = 128


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

    @staticmethod
    def to_engine_type(mode: str) -> str:
        if mode == DisaggregationMode.PREFILL.value:
            return "prefill"
        elif mode == DisaggregationMode.DECODE.value:
            return "decode"
        return "unified"
