"""Shared retry/backoff policy for AI provider calls (fetch + synthesis).

Owned by Part B (AI Service and Business Logic). This module has exactly one
job: decide, in one place, what counts as a retryable failure and how the
exponential backoff is computed, so `ai_service.AIFetchService` and
`ai_service.AISynthesisService` behave identically instead of each rolling
their own retry loop.

Security note
--------------
Provider SDK exceptions can embed sensitive material (API keys in headers,
full request URLs with query params, etc. -- see `ai/providers/*.py`, which
wraps every SDK exception as ``ProviderError(f"... : {e}")``). Nothing in
this module ever logs ``str(exc)`` or interpolates an exception into a log
message. Only the exception's *type* is logged, via a short, closed
vocabulary of error codes (`error_code_for`). Callers in `ai_service.py`
follow the same discipline.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ai.providers.base import ProviderError

log = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions considered transient and worth retrying. Anything else (bad
# input, malformed provider payloads mapped to UpstreamDataError, programming
# errors) propagates on the first attempt -- retrying those would only burn
# through the caller's timeout budget for no benefit.
#
# httpx.TransportError covers connection-level failures (ConnectError,
# ReadTimeout, WriteTimeout, PoolTimeout, NetworkError, ...) -- genuinely
# transient network problems, as distinct from httpx.HTTPStatusError (a
# real HTTP response with a 4xx/5xx status), which is not retried here as a
# blanket type: see `_should_retry` below for the HTTP-status classifier
# that distinguishes a permanent 401/404 from a transient 429/5xx within
# ProviderError specifically.
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ProviderError,
    asyncio.TimeoutError,
    ConnectionError,
    httpx.TransportError,
)


def _http_status_from(exc: BaseException, *, max_depth: int = 5) -> int | None:
    """Walk `exc`'s `__cause__` chain (bounded depth, cycle-guarded) looking
    for an HTTP status code, without ever reading or logging exception text.

    `ai/providers/*.py` and `ai/sources.py` wrap the original SDK/httpx
    exception as ``ProviderError(f"...: {e}") from e``, so the status code
    (if any) lives one or more hops down `__cause__`, not on `ProviderError`
    itself. Recognizes `httpx.HTTPStatusError`'s `.response.status_code` and
    SDK exceptions that expose `.status_code` directly (e.g. OpenAI's and
    Anthropic's `APIStatusError`). Returns `None` if no status code is
    discoverable, e.g. a plain connection failure or an SDK shape this
    wasn't written against.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    depth = 0
    while current is not None and depth < max_depth:
        if id(current) in seen:
            break  # defensive cycle guard; __cause__ chains shouldn't cycle
        seen.add(id(current))

        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status

        current = current.__cause__
        depth += 1
    return None


def _is_transient_status(status: int) -> bool:
    """429 (rate limited) and 5xx (server-side) are worth retrying; other
    4xx (401 unauthorized, 404 not found, 400 bad request, ...) are not --
    retrying a bad API key or a malformed request only wastes the caller's
    timeout budget on an outcome that cannot change.
    """
    return status == 429 or 500 <= status < 600


def _build_should_retry(
    retry_exceptions: tuple[type[Exception], ...],
) -> Callable[[BaseException], bool]:
    def _should_retry(exc: BaseException) -> bool:
        if not isinstance(exc, retry_exceptions):
            return False
        if isinstance(exc, ProviderError):
            status = _http_status_from(exc)
            if status is not None:
                return _is_transient_status(status)
            # No discoverable status (e.g. a plain connection failure, or an
            # SDK error shape not covered by `_http_status_from`): fall back
            # to the previous uniform-retry behavior rather than guessing.
            return True
        return True

    return _should_retry


def error_code_for(exc: BaseException) -> str:
    """Map an exception to a short, secret-free code for logs and warnings.

    Never include ``str(exc)`` in a log line or a user-facing message --
    that is the whole point of this function existing. An HTTP status code
    is not sensitive and is included when discoverable, since it is the
    single most useful piece of information for diagnosing a retry decision.
    """
    if isinstance(exc, asyncio.TimeoutError):
        return "provider_timeout"
    if isinstance(exc, ProviderError):
        status = _http_status_from(exc)
        if status is not None:
            return f"provider_error_status_{status}"
        return "provider_error"
    if isinstance(exc, httpx.TransportError):
        return "transport_error"
    if isinstance(exc, ConnectionError):
        return "provider_connection_error"
    return f"provider_unexpected_{type(exc).__name__.lower()}"


def _log_retry(retry_state: RetryCallState) -> None:
    """Tenacity ``before_sleep`` hook: log an upcoming retry, no secrets."""
    outcome = retry_state.outcome
    exc = outcome.exception() if outcome is not None else None
    wait_seconds = (
        round(retry_state.next_action.sleep, 3) if retry_state.next_action else None
    )
    log.warning(
        "provider_call_retry",
        extra={
            "attempt": retry_state.attempt_number,
            "wait_seconds": wait_seconds,
            "error_code": error_code_for(exc) if exc is not None else None,
        },
    )


def build_async_retrying(
    *,
    max_attempts: int = 3,
    initial_wait_seconds: float = 0.5,
    max_wait_seconds: float = 8.0,
    retry_exceptions: tuple[type[Exception], ...] = RETRYABLE_EXCEPTIONS,
) -> AsyncRetrying:
    """Return a configured ``AsyncRetrying`` controller.

    ``reraise=True`` means that once attempts are exhausted, the *original*
    exception propagates (never Tenacity's own ``RetryError`` wrapper), so
    callers upstream (C's orchestrator/research workflow) keep seeing the
    exception types they already know how to handle
    (``ProviderError`` / ``httpx.HTTPError`` / ``TimeoutError``).

    The retry predicate is `_should_retry` (see `_build_should_retry`), not
    a plain type check: a `ProviderError` wrapping a permanent HTTP status
    (401, 404, ...) is not retried even though `ProviderError` itself is in
    `retry_exceptions`, while one wrapping a transient status (429, 5xx) is.
    """
    return AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=initial_wait_seconds, max=max_wait_seconds),
        retry=retry_if_exception(_build_should_retry(retry_exceptions)),
        before_sleep=_log_retry,
    )


async def call_with_retry(
    func: Callable[[], Awaitable[T]],
    *,
    operation: str,
    timeout_seconds: float,
    max_attempts: int = 3,
    initial_wait_seconds: float = 0.5,
    max_wait_seconds: float = 8.0,
    retry_exceptions: tuple[type[Exception], ...] = RETRYABLE_EXCEPTIONS,
    request_id: str | None = None,
) -> T:
    """Run ``func()`` with a per-attempt timeout and exponential backoff.

    Parameters
    ----------
    func:
        A zero-argument async callable. Called fresh on every attempt --
        never pass an already-created coroutine object, since those cannot
        be re-awaited after a failed attempt.
    timeout_seconds:
        Wall-clock budget for a *single* attempt. This is the "timeout for
        every AI call" requirement: even if the caller's own outer deadline
        (C's ``asyncio.timeout``) is longer, no single attempt is allowed to
        run unbounded.
    operation, request_id:
        Logged as plain identifiers only (never exception text), so
        operators can correlate retries with a request without any risk of
        secret exposure.

    Raises
    ------
    The final attempt's exception, unmodified (see ``reraise=True`` in
    `build_async_retrying`). Non-retryable exceptions (e.g.
    ``UpstreamDataError``) propagate immediately on the first attempt.
    """
    retrying = build_async_retrying(
        max_attempts=max_attempts,
        initial_wait_seconds=initial_wait_seconds,
        max_wait_seconds=max_wait_seconds,
        retry_exceptions=retry_exceptions,
    )
    attempt_number = 0
    async for attempt in retrying:
        attempt_number += 1
        with attempt:
            log.info(
                "provider_call_attempt",
                extra={
                    "operation": operation,
                    "attempt": attempt_number,
                    "request_id": request_id,
                },
            )
            return await asyncio.wait_for(func(), timeout=timeout_seconds)
    # AsyncRetrying either returns from inside the loop or raises (reraise=True);
    # this line exists only to satisfy static type checkers.
    raise AssertionError("unreachable")  # pragma: no cover