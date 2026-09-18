"""Transport-level retry wrapper (B-07).

`ai/sources.py`'s `fetch_wikipedia` catches any exception from the
per-title summary request and silently `continue`s to the next title:

    try:
        summ = await client.get(...)
        summ.raise_for_status()
    except Exception:
        continue  # one bad title shouldn't kill the whole fetch

That is reasonable behavior for *that* function's job (don't let one bad
title kill the whole batch), but it means a transient 500/429 on a single
summary request never reaches our outer retry policy at all: from
`AIFetchService`'s point of view, `fetch_wikipedia` simply returned
successfully with fewer (possibly zero) results. Retrying the whole
`fetch()` call blindly on an empty result is explicitly not the right fix
either -- a legitimately empty search is possible, and retrying would not
distinguish the two.

Since `ai/sources.py` is supplied/shared code this module does not modify,
the fix has to live one layer down: wrap the `httpx.AsyncClient` passed
*into* `fetch_wikipedia` with a transport that retries a transient HTTP
response (429/5xx) or transport-level exception *before* control returns
to `fetch_wikipedia`, so what it sees is either an eventual success or a
genuinely exhausted, non-transient failure -- never a silently-swallowed
transient one.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from researcher.services.retry import _is_transient_status

log = logging.getLogger(__name__)


class _RetryingTransport(httpx.AsyncBaseTransport):
    """Delegates to an existing `httpx.AsyncClient`'s `.send()`, retrying a
    transient HTTP status (429/5xx, via `retry.py`'s `_is_transient_status`)
    or a transport-level exception, with exponential backoff, up to
    `max_attempts` times. A permanent status (401, 404, ...) or a non-network
    exception is returned/raised on the first attempt, unretried.

    Does not own or close `inner_client` -- this wrapper's own `aclose()` is
    a no-op, since the client it wraps belongs to and outlives the caller
    (C passes it into `FetchService.fetch`; B never constructs or owns it).
    """

    def __init__(
        self,
        inner_client: httpx.AsyncClient,
        *,
        max_attempts: int = 3,
        initial_wait_seconds: float = 0.5,
        max_wait_seconds: float = 4.0,
    ) -> None:
        self._inner_client = inner_client
        self._max_attempts = max_attempts
        self._initial_wait_seconds = initial_wait_seconds
        self._max_wait_seconds = max_wait_seconds

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        wait = self._initial_wait_seconds
        last_exc: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._inner_client.send(request)
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt >= self._max_attempts:
                    raise
            else:
                if response.status_code < 400 or not _is_transient_status(
                    response.status_code
                ):
                    return response
                await response.aclose()
                if attempt >= self._max_attempts:
                    return response  # exhausted; caller sees the final status as-is
                last_exc = None

            log.warning(
                "wikipedia_transport_retry",
                extra={"attempt": attempt, "wait_seconds": round(wait, 3)},
            )
            await asyncio.sleep(min(wait, self._max_wait_seconds))
            wait *= 2

        if last_exc is not None:  # pragma: no cover - defensive, loop always returns/raises
            raise last_exc
        raise AssertionError("unreachable")  # pragma: no cover

    async def aclose(self) -> None:
        pass  # never close a client this wrapper does not own


def wikipedia_retrying_client(
    client: httpx.AsyncClient,
    *,
    max_attempts: int = 3,
    initial_wait_seconds: float = 0.5,
    max_wait_seconds: float = 4.0,
) -> httpx.AsyncClient:
    """Return a new `httpx.AsyncClient` that retries transient responses
    from `client`, for passing into `ai.sources.fetch_wikipedia`.

    The returned client must be `aclose()`d by the caller (its own
    transport's `aclose()` is a no-op, so this is cheap and never closes
    the wrapped `client`).
    """
    return httpx.AsyncClient(
        transport=_RetryingTransport(
            client,
            max_attempts=max_attempts,
            initial_wait_seconds=initial_wait_seconds,
            max_wait_seconds=max_wait_seconds,
        )
    )