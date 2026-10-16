# SPDX-License-Identifier: Apache-2.0
"""Prompt-cache accounting in 256-token blocks, isolated by API key in Redis.

The caller supplies a model/tokenizer/template/adapter namespace
and calls ``account`` once, at the first successful output event. Redis operations
are not retried automatically; errors leave accounting unconfirmed so the caller
can fall back to the engine's actual cache usage.
"""

import asyncio
import hashlib
import math
import struct
from dataclasses import asdict, dataclass
from typing import Any, Sequence

CACHE_BLOCK_SIZE = 256

_KEY_PREFIX = "sglang:cache:v1"
_DOMAIN = b"sglang:cache:v1\x00"
_MAX_SAFE_INTEGER = (1 << 53) - 1
_TTLS = (300, 3600)


class CacheAccountingError(RuntimeError):
    """Accounting did not return a confirmed result; do not invent usage."""


@dataclass(frozen=True)
class CacheUsage:
    read_tokens: int = 0
    creation_5m_tokens: int = 0
    creation_1h_tokens: int = 0
    normal_input_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class PageCachePlan:
    page_keys: tuple[str, ...]
    prompt_tokens: int
    page_size: int
    ttl_seconds: int


def _positive_int(name: str, value: int, maximum: int = _MAX_SAFE_INTEGER) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")


def _text(name: str, value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    encoded = value.encode("utf-8")
    return struct.pack(">Q", len(encoded)) + encoded


def build_page_cache_plan(
    input_ids: Sequence[int],
    api_key: str,
    namespace: str,
    page_size: int,
    ttl_seconds: int,
) -> PageCachePlan:
    """Hash complete input pages with a cumulative, fixed-uint32 token encoding.

    Production callers pass ``CACHE_BLOCK_SIZE`` as ``page_size``; the low-level
    parameter remains available for testing. Partial final blocks remain normal
    input, independently of the engine's physical cache page size. ``namespace``
    isolates deployments and input interpretation. TTL is part of identity: 5m
    and 1h requests have independent pages and expiration timelines. Accounting
    adds no page-count limit. Each first-page prefix defines a Redis Cluster hash
    slot instead of pinning all prefixes for one API key.
    """
    _positive_int("page_size", page_size)
    if type(ttl_seconds) is not int or ttl_seconds not in _TTLS:
        raise ValueError("ttl_seconds must be 300 (5m) or 3600 (1h)")
    identity = _text("api_key", api_key) + _text("namespace", namespace)
    prompt_tokens = len(input_ids)
    if prompt_tokens > _MAX_SAFE_INTEGER:
        raise ValueError("prompt_tokens exceeds Redis Lua integer precision")
    page_count = prompt_tokens // page_size

    previous = hashlib.sha256(
        _DOMAIN + identity + struct.pack(">QQ", page_size, ttl_seconds)
    ).digest()
    digests = []
    # Validate tail tokens too: a plan must describe a valid token sequence.
    page_hash = None
    for index, token in enumerate(input_ids):
        if type(token) is not int or not 0 <= token <= 0xFFFFFFFF:
            raise ValueError("input_ids must contain unsigned 32-bit integer tokens")
        if index >= page_count * page_size:
            continue
        if index % page_size == 0:
            page_hash = hashlib.sha256(_DOMAIN + b"page\x00" + previous)
        page_hash.update(struct.pack(">I", token))
        if (index + 1) % page_size == 0:
            previous = page_hash.digest()
            digests.append(previous.hex())

    if digests:
        tag = digests[0]
        page_keys = tuple(f"{_KEY_PREFIX}:{{{tag}}}:page:{d}" for d in digests)
    else:
        page_keys = ()
    return PageCachePlan(
        page_keys=page_keys,
        prompt_tokens=prompt_tokens,
        page_size=page_size,
        ttl_seconds=ttl_seconds,
    )


# All keys are declared in KEYS, and share the first page's hash tag. Validate
# every argument and key type BEFORE writing: Redis scripts are atomic but do
# not roll back preceding writes when a later command fails.
_ACCOUNT_LUA = r"""
local ttl = tonumber(ARGV[1])
local page_size = tonumber(ARGV[2])
local prompt_tokens = tonumber(ARGV[3])
local pages = #KEYS
local function integer(n, minimum)
    return n and n >= minimum and n <= 9007199254740991 and n == math.floor(n)
end
if #ARGV ~= 3 or (ttl ~= 300 and ttl ~= 3600)
    or not integer(page_size, 1) or not integer(prompt_tokens, 0)
    or not integer(pages, 1)
    or math.floor(prompt_tokens / page_size) ~= pages then
    return redis.error_reply('invalid cache accounting arguments')
end
for i = 1, #KEYS do
    local kind = redis.call('TYPE', KEYS[i]).ok
    if kind ~= 'none' and kind ~= 'string' then
        return redis.error_reply('invalid cache key type')
    end
end
local hits = 0
for i = 1, pages do
    if redis.call('EXISTS', KEYS[i]) == 0 then
        break
    end
    hits = hits + 1
end
local read = hits * page_size
local written = (pages - hits) * page_size
local write_5m = 0
local write_1h = 0
if ttl == 300 then write_5m = written else write_1h = written end
local normal = prompt_tokens - pages * page_size
for i = 1, pages do
    if i <= hits then
        redis.call('EXPIRE', KEYS[i], ttl)
    else
        -- Existing pages after a hole cannot be used as a continuous prefix.
        redis.call('SET', KEYS[i], '1', 'EX', ttl)
    end
end
return {read, write_5m, write_1h, normal}
"""


def _validate_plan(plan: PageCachePlan) -> None:
    """Check caller-constructed plans before accessing Redis."""
    _positive_int("page_size", plan.page_size)
    _positive_int("prompt_tokens + 1", plan.prompt_tokens + 1)
    if type(plan.prompt_tokens) is not int or plan.prompt_tokens < 0:
        raise ValueError("prompt_tokens must be a nonnegative integer")
    if type(plan.ttl_seconds) is not int or plan.ttl_seconds not in _TTLS:
        raise ValueError("ttl_seconds must be 300 or 3600")
    if not isinstance(plan.page_keys, tuple):
        raise ValueError("page_keys must be a tuple")
    pages = len(plan.page_keys)
    if pages != plan.prompt_tokens // plan.page_size:
        raise ValueError("page_keys must cover exactly the complete prompt pages")
    if not pages:
        return
    first = plan.page_keys[0]
    prefix = f"{_KEY_PREFIX}:{{"
    if not isinstance(first, str) or not first.startswith(prefix):
        raise ValueError("invalid page key")
    tag, separator, _ = first[len(prefix) :].partition("}:page:")
    if not separator or len(tag) != 64 or any(c not in "0123456789abcdef" for c in tag):
        raise ValueError("invalid page hash tag")
    base = f"{_KEY_PREFIX}:{{{tag}}}:page:"
    if len(set(plan.page_keys)) != pages:
        raise ValueError("page keys must be unique")
    for key in plan.page_keys:
        if not isinstance(key, str) or not key.startswith(base):
            raise ValueError("all page keys must share one Redis Cluster hash tag")
        digest = key[len(base) :]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid page key digest")


class RedisPageCacheStore:
    """Lazy async Redis boundary; an error never becomes read/write success.

    ``account`` and its Redis client do not retry operations automatically.
    Cache keys require a dedicated Redis eviction/retention policy;
    eviction is equivalent to expiration for accounting. Memory exhaustion can
    still cause Redis runtime errors, so Redis failure remains explicitly
    unconfirmed rather than being billed as a cache miss.

    The key layout is Redis Cluster slot compatible. The default client targets
    a standalone Redis endpoint; Cluster discovery/MOVED routing is a separate
    deployment integration and is not provided by this framework.
    """

    def __init__(
        self,
        redis_url: str,
        timeout_seconds: float = 1.0,
        username: str | None = None,
        password: str | None = None,
    ):
        if not isinstance(redis_url, str) or not redis_url:
            raise ValueError("redis_url must be nonempty")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        self._redis_url = redis_url
        self._timeout_seconds = timeout_seconds
        self._username = username
        self._password = password
        self._redis: Any = None

    def _client(self):
        if self._redis is None:
            from redis import asyncio as redis_async
            from redis.asyncio.retry import Retry
            from redis.backoff import NoBackoff

            self._redis = redis_async.from_url(
                self._redis_url,
                socket_timeout=self._timeout_seconds,
                socket_connect_timeout=self._timeout_seconds,
                decode_responses=False,
                retry_on_timeout=False,
                retry=Retry(NoBackoff(), 0),
                username=self._username,
                password=self._password,
            )
        return self._redis

    async def account(self, plan: PageCachePlan) -> CacheUsage:
        _validate_plan(plan)
        if not plan.page_keys:
            return CacheUsage(normal_input_tokens=plan.prompt_tokens)
        try:
            result = await asyncio.wait_for(
                self._client().eval(
                    _ACCOUNT_LUA,
                    len(plan.page_keys),
                    *plan.page_keys,
                    plan.ttl_seconds,
                    plan.page_size,
                    plan.prompt_tokens,
                ),
                timeout=self._timeout_seconds,
            )
            if (
                not isinstance(result, (list, tuple))
                or len(result) != 4
                or any(type(value) is not int or value < 0 for value in result)
                or sum(result) != plan.prompt_tokens
                or result[3] != plan.prompt_tokens % plan.page_size
                or any(value % plan.page_size for value in result[:3])
                or (plan.ttl_seconds == 300 and result[2])
                or (plan.ttl_seconds == 3600 and result[1])
            ):
                raise ValueError("Invalid cache Redis result")
            return CacheUsage(*result)
        except Exception as exc:
            raise CacheAccountingError("Redis cache accounting is unconfirmed") from exc

    async def aclose(self) -> None:
        client, self._redis = self._redis, None
        if client is not None:
            await client.aclose()
