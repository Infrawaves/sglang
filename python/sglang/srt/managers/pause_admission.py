"""Admission of new requests while generation is paused.

A paused engine normally holds new requests at the tokenizer until
`continue_generation`. A `migrate` pause retires the instance instead: no
continue follows, so a new request is refused at once and its router retries
it on another instance.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Optional

# The 503 a draining instance answers with. sgl-model-gateway takes the
# instance out of its own routing pool on seeing this header and retries the
# request elsewhere; any other 503 still counts as a worker failure.
DRAINING_HEADERS = {"X-SGLang-Draining": "1", "Retry-After": "0"}

_DRAINING_PAUSE_MODE = "migrate"


class ServerDrainingError(Exception):
    """A new request reached an instance that is handing off its requests."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE

    def __init__(self) -> None:
        super().__init__(
            "This instance is retiring and accepts no new requests; "
            "retry on another instance."
        )


class PauseAdmission:
    """The mode of the current pause, and whether it admits new requests."""

    def __init__(self) -> None:
        self._mode: Optional[str] = None

    def on_pause(self, mode: Optional[str]) -> None:
        self._mode = mode

    def on_continue(self) -> None:
        self._mode = None

    @property
    def draining(self) -> bool:
        return self._mode == _DRAINING_PAUSE_MODE

    def check(self) -> None:
        """Raise ServerDrainingError if a new request must be refused."""
        if self.draining:
            raise ServerDrainingError()
