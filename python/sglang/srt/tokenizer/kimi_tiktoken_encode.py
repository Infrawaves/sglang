"""Faster encode for the Kimi checkpoint's remote-code ``TikTokenTokenizer``.

``TikTokenTokenizer._encode_text_piece`` cuts text into 400k-char chunks, then
``_split_whitespaces_or_nonwhitespaces`` cuts each chunk so no whitespace or
non-whitespace run (``str.isspace``) exceeds 25k chars, and every slice goes to
``tiktoken.Encoding.encode``. Two per-instance overrides keep those exact
``encode`` calls:

* ``fast_split`` finds the same cuts with NumPy instead of a per-character
  Python loop (SGLANG_KIMI_TOKENIZER_FAST_SCAN).
* ``_ParallelChatSegments`` replaces ``_encode_chat_segments`` and encodes the
  long slices of one prompt on a per-process thread pool; tiktoken releases
  the GIL while encoding (SGLANG_KIMI_TOKENIZER_ENCODE_THREADS > 1).

They are installed only when the three methods involved match the verified
checkpoint source; any other version keeps its own methods and logs why.
"""

import hashlib
import inspect
import logging
import os
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from typing import Iterator, List

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# The constants of TikTokenTokenizer._encode_text_piece (pinned by its hash).
_MAX_ENCODE_CHARS = 400_000
_MAX_RUN_CHARS = 25_000
# Only slices this long go to the pool, and only when a prompt has two.
_PARALLEL_MIN_PIECE_CHARS = 16_384

# sha256 of textwrap.dedent(inspect.getsource(method)) in the verified
# tokenization_kimi.py (sha256 204ee39f...d936f9c6, tiktoken 0.14.0).
_VERIFIED_SOURCE_SHA256 = {
    "_split_whitespaces_or_nonwhitespaces": "b8f5acb3521037a488896765515d2569bc658ae7129ba1f202d5104565a9ad08",
    "_encode_text_piece": "9e77ff746d2babda36f13643cb1956a1572da92795ca932895f66decf402828c",
    "_encode_chat_segments": "46ab075f05531cffc00555ac428179378f71a5c9de03aafc06f55e1cfaddb2b2",
}

# Instance attribute listing the attributes this module set on a tokenizer.
_INSTALLED_ATTR = "_sglang_kimi_tiktoken_encode"


# ---------------------------------------------------------------------------
# Run scan
# ---------------------------------------------------------------------------

_tables = None


def _np_tables():
    """(numpy, isspace lookup table, largest whitespace code point)."""
    global _tables
    if _tables is None:
        import numpy as np

        spaces = [c for c in range(0x110000) if chr(c).isspace()]
        top = max(spaces)
        # Index top + 1 stands for every larger code point (not whitespace).
        lut = np.zeros(top + 2, dtype=bool)
        lut[spaces] = True
        _tables = (np, lut, top)
    return _tables


def fast_split(s: str, max_consecutive_slice_len: int) -> Iterator[str]:
    """Same slices as ``TikTokenTokenizer._split_whitespaces_or_nonwhitespaces``.

    The original cuts a maximal same-class run of length L starting at r at
    r + k*m for every k >= 1 with k*m < L, and nowhere else.
    """
    m = max_consecutive_slice_len
    if m < 1:
        raise ValueError(f"max_consecutive_slice_len must be >= 1, got {m}")
    n = len(s)
    if n <= m:
        yield s
        return
    np, lut, top = _np_tables()
    if s.isascii():
        mask = lut[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
    else:
        # surrogatepass keeps lone surrogates as their own code points.
        cps = np.frombuffer(s.encode("utf-32-le", "surrogatepass"), dtype="<u4")
        mask = lut[np.minimum(cps, top + 1)]
    starts = np.concatenate(([0], np.flatnonzero(mask[1:] != mask[:-1]) + 1))
    lengths = np.diff(np.append(starts, n))
    start = 0
    for j in np.flatnonzero(lengths > m).tolist():
        run_start, run_len = int(starts[j]), int(lengths[j])
        for cut in range(run_start + m, run_start + run_len, m):
            yield s[start:cut]
            start = cut
    yield s[start:]


# ---------------------------------------------------------------------------
# Parallel encode
# ---------------------------------------------------------------------------

_pool_lock = threading.Lock()
_pools = {}


def _reset_pools_in_child():
    # A forked child has none of the parent's pool threads; start over.
    global _pool_lock, _pools
    _pool_lock = threading.Lock()
    _pools = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_pools_in_child)


def _encode_pool(threads: int) -> ThreadPoolExecutor:
    """The process's pool for ``threads`` workers, created on first use."""
    pool = _pools.get(threads)
    if pool is None:
        with _pool_lock:
            pool = _pools.get(threads)
            if pool is None:
                pool = ThreadPoolExecutor(
                    max_workers=threads, thread_name_prefix="kimi-tiktoken-encode"
                )
                _pools[threads] = pool
    return pool


def shutdown_encode_pools() -> None:
    """Stop this process's encode pools; the next long prompt starts a new one."""
    with _pool_lock:
        pools = list(_pools.values())
        _pools.clear()
    for pool in pools:
        pool.shutdown(wait=True)


def _encode_piece(model, text: str, allow_special: bool) -> List[int]:
    # The two encode modes of TikTokenTokenizer._encode_text_piece.
    if allow_special:
        return model.encode(text, allowed_special="all")
    return model.encode(text, disallowed_special=())


class _ParallelChatSegments:
    """``_encode_chat_segments`` that encodes long slices on the pool.

    Holds only the tokenizer and the thread count, so a tokenizer carrying it
    pickles and deep-copies; the pool lives in the process, not here.
    """

    def __init__(self, tokenizer, threads: int):
        self.tokenizer = tokenizer
        self.threads = threads

    def __call__(self, segments) -> List[int]:
        tokenizer = self.tokenizer
        if sum(len(s.text) for s in segments) < 2 * _PARALLEL_MIN_PIECE_CHARS:
            return type(tokenizer)._encode_chat_segments(tokenizer, segments)

        split = tokenizer._split_whitespaces_or_nonwhitespaces
        pieces = []
        for segment in segments:
            text, allow = segment.text, segment.allow_special
            for i in range(0, len(text), _MAX_ENCODE_CHARS):
                chunk = text[i : i + _MAX_ENCODE_CHARS]
                pieces.extend((p, allow) for p in split(chunk, _MAX_RUN_CHARS))

        model = tokenizer.model
        big = [
            i for i, (p, _) in enumerate(pieces) if len(p) >= _PARALLEL_MIN_PIECE_CHARS
        ]
        if len(big) < 2:
            return list(chain.from_iterable(_encode_piece(model, *p) for p in pieces))

        pool = _encode_pool(self.threads)
        # Longest first keeps the pool's makespan short.
        big.sort(key=lambda i: -len(pieces[i][0]))
        futures = {i: pool.submit(_encode_piece, model, *pieces[i]) for i in big}
        try:
            results = [
                None if i in futures else _encode_piece(model, *p)
                for i, p in enumerate(pieces)
            ]
            for i, future in futures.items():
                results[i] = future.result()
        except BaseException:
            for future in futures.values():
                future.cancel()
            raise
        return list(chain.from_iterable(results))


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def _source_sha256(fn) -> str:
    return hashlib.sha256(
        textwrap.dedent(inspect.getsource(fn)).encode("utf-8")
    ).hexdigest()


def _unsupported_reason(tokenizer):
    """None when the tokenizer's class is the verified one, else why not."""
    cls = type(tokenizer)
    overridden = [n for n in _VERIFIED_SOURCE_SHA256 if n in vars(tokenizer)]
    if overridden:
        return f"instance already overrides {overridden}"
    for name, want in _VERIFIED_SOURCE_SHA256.items():
        attr = inspect.getattr_static(cls, name, None)
        if name == "_split_whitespaces_or_nonwhitespaces":
            if not isinstance(attr, staticmethod):
                return f"{name} is not a staticmethod"
            attr = attr.__func__
        if not inspect.isfunction(attr):
            return f"{name} is not a plain function"
        try:
            got = _source_sha256(attr)
        except (OSError, TypeError) as e:
            return f"source of {name} unavailable: {e}"
        if got != want:
            return f"{name} source sha256 {got[:12]} differs from the verified one"
    try:
        import tiktoken
    except ImportError:
        return "tiktoken is not importable"
    if not isinstance(getattr(tokenizer, "model", None), tiktoken.Encoding):
        return "tokenizer.model is not a tiktoken.Encoding"
    return None


def install_kimi_tiktoken_encode(tokenizer):
    """Install the enabled overrides on a verified Kimi TikTokenTokenizer.

    Other tokenizers, and repeated calls on the same one, are left unchanged.
    """
    cls = type(tokenizer)
    if cls.__name__ != "TikTokenTokenizer" or "tokenization_kimi" not in (
        cls.__module__ or ""
    ):
        return tokenizer
    if _INSTALLED_ATTR in vars(tokenizer):
        return tokenizer
    fast_scan = envs.SGLANG_KIMI_TOKENIZER_FAST_SCAN.get()
    threads = envs.SGLANG_KIMI_TOKENIZER_ENCODE_THREADS.get()
    if threads < 1:
        logger.warning(
            "SGLANG_KIMI_TOKENIZER_ENCODE_THREADS=%d is not >= 1; using 1", threads
        )
        threads = 1
    if not fast_scan and threads == 1:
        return tokenizer

    reason = _unsupported_reason(tokenizer)
    if reason is not None:
        logger.warning(
            "Kimi tiktoken encode overrides not installed on %s.%s (%s); "
            "using its own encode.",
            cls.__module__,
            cls.__qualname__,
            reason,
        )
        return tokenizer

    installed = []
    if fast_scan:
        _np_tables()  # build the lookup table now, not on a request
        tokenizer._split_whitespaces_or_nonwhitespaces = fast_split
        installed.append("_split_whitespaces_or_nonwhitespaces")
    if threads > 1:
        tokenizer._encode_chat_segments = _ParallelChatSegments(tokenizer, threads)
        installed.append("_encode_chat_segments")
    setattr(tokenizer, _INSTALLED_ATTR, tuple(installed))
    logger.info(
        "Kimi tiktoken encode: fast_scan=%s, encode_threads=%d", fast_scan, threads
    )
    return tokenizer


def uninstall_kimi_tiktoken_encode(tokenizer):
    """Remove what ``install_kimi_tiktoken_encode`` set on this tokenizer."""
    installed = vars(tokenizer).pop(_INSTALLED_ATTR, None)
    for name in installed or ():
        vars(tokenizer).pop(name, None)
    return tokenizer
