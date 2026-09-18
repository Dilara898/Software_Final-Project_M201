"""Part B: AI Service and Business Logic.

Adapts the supplied, retry-free `ai/` package to the async ports Part C's
concurrency layer expects (`researcher.concurrency.contracts.FetchService`
and `.SynthesisService`), adding everything `ai/providers/base.py` explicitly
says is *not* its job: Tenacity retry with exponential backoff, a hard
timeout on every AI call, structured logging that never leaks secrets, and
(optionally) multi-provider failover for synthesis.

Ownership boundaries
---------------------
This module never edits `ai/*`, `researcher/concurrency/*`, or
`researcher/config.py`. It only *calls* them. `AIFetchService` and
`AISynthesisService` are the concrete objects A/D compose
`SourceOrchestrator` / `ResearchOrchestrator` with in production; in tests,
fakes are used instead (see `tests/test_c_research.py` for the pattern this
follows).

Two zero-argument factories are exported for `scripts/benchmark.py
--live --service-factory researcher.services.ai_service:make_fetch_service`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

import httpx

from ai.providers.base import LLMProvider, ProviderError
from ai.providers.factory import get_llm
from ai.schemas import AnswerWithCitations, Source
from ai.sources import fetch_arxiv, fetch_web, fetch_wikipedia
from ai.synthesizer import synthesize
from researcher.concurrency.contracts import UpstreamDataError
from researcher.exceptions import NoSourcesError
from researcher.services.retry import RateLimiter, call_with_retry, error_code_for
from researcher.services.transport import wikipedia_retrying_client

log = logging.getLogger(__name__)

T = TypeVar("T")

_Fetcher = Callable[..., Awaitable[list[Source]]]

_FETCHERS: dict[str, _Fetcher] = {
    "wikipedia": fetch_wikipedia,
    "arxiv": fetch_arxiv,
    "web": fetch_web,
}


class AIFetchService:
    """Adapts `ai.sources.fetch_*` coroutines to the `FetchService` port.

    `ai.sources` is already async/httpx-native, so no thread offloading is
    needed here (unlike synthesis). Each call gets its own retry-with-backoff
    cycle and a hard per-attempt timeout; C's own `timeout_seconds` /
    `max_concurrency` remain the *outer* budget this must fit inside.

    Pacing (B-05): `min_interval_seconds` (default 0, disabled) applies a
    shared `RateLimiter` per source name, so repeated calls -- including
    retries -- to the same provider/host don't exceed the configured rate.

    Wikipedia transport retry (B-07): when `source == "wikipedia"` and
    `wikipedia_transport_retry` is enabled (default), the `client` passed
    into `ai.sources.fetch_wikipedia` is wrapped so a transient HTTP status
    on the internal per-title summary request is retried before
    `fetch_wikipedia`'s own `except Exception: continue` ever sees it. See
    `researcher.services.transport` for why this can't be fixed by editing
    `ai/sources.py` itself.
    """

    def __init__(
        self,
        *,
        max_results: int = 3,
        timeout_seconds: float = 8.0,
        max_attempts: int = 3,
        initial_wait_seconds: float = 0.5,
        max_wait_seconds: float = 4.0,
        min_interval_seconds: float = 0.0,
        wikipedia_transport_retry: bool = True,
        wikipedia_transport_max_attempts: int = 3,
        wikipedia_transport_initial_wait_seconds: float = 0.5,
        wikipedia_transport_max_wait_seconds: float = 4.0,
    ) -> None:
        self._max_results = max_results
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._initial_wait_seconds = initial_wait_seconds
        self._max_wait_seconds = max_wait_seconds
        self._min_interval_seconds = min_interval_seconds
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._wikipedia_transport_retry = wikipedia_transport_retry
        self._wikipedia_transport_max_attempts = wikipedia_transport_max_attempts
        self._wikipedia_transport_initial_wait_seconds = (
            wikipedia_transport_initial_wait_seconds
        )
        self._wikipedia_transport_max_wait_seconds = wikipedia_transport_max_wait_seconds

    def _limiter_for(self, source: str) -> RateLimiter:
        limiter = self._rate_limiters.get(source)
        if limiter is None:
            limiter = RateLimiter(self._min_interval_seconds)
            self._rate_limiters[source] = limiter
        return limiter

    async def fetch(
        self,
        source: str,
        query: str,
        client: httpx.AsyncClient,
    ) -> list[Source]:
        fetcher = _FETCHERS.get(source)
        if fetcher is None:
            # Unknown source name is a caller/programming error, not a
            # transient provider failure -- never retried.
            raise UpstreamDataError(f"no fetcher registered for source={source!r}")

        limiter = self._limiter_for(source)

        async def _call() -> list[Source]:
            # B-05: pacing applies to every attempt, including retries.
            await limiter.acquire()

            # B-07: only wikipedia needs the transport-retry wrapper (its
            # internal summary-fetch swallows transient errors); other
            # sources already surface HTTP errors as ProviderError.
            effective_client = client
            wrapped_client: httpx.AsyncClient | None = None
            if source == "wikipedia" and self._wikipedia_transport_retry:
                wrapped_client = wikipedia_retrying_client(
                    client,
                    max_attempts=self._wikipedia_transport_max_attempts,
                    initial_wait_seconds=self._wikipedia_transport_initial_wait_seconds,
                    max_wait_seconds=self._wikipedia_transport_max_wait_seconds,
                )
                effective_client = wrapped_client

            try:
                # No broad `except Exception` here (B-04): the fetcher's own
                # exceptions -- ProviderError, transient network errors, or a
                # genuine programmer bug -- propagate unmodified.
                # `call_with_retry` already knows what is retryable
                # (retry.py's RETRYABLE_EXCEPTIONS); this function's only
                # responsibility is validating the *shape* of a successful
                # return, which is the one thing only this boundary can check.
                result = await fetcher(
                    query, max_results=self._max_results, client=effective_client
                )
            finally:
                if wrapped_client is not None:
                    await wrapped_client.aclose()

            if not isinstance(result, list) or not all(
                isinstance(item, Source) for item in result
            ):
                raise UpstreamDataError(
                    f"{source} fetcher returned {type(result).__name__}, "
                    "expected list[Source]"
                )
            return result

        start = time.monotonic()
        try:
            result = await call_with_retry(
                _call,
                operation=f"fetch:{source}",
                timeout_seconds=self._timeout_seconds,
                max_attempts=self._max_attempts,
                initial_wait_seconds=self._initial_wait_seconds,
                max_wait_seconds=self._max_wait_seconds,
            )
        except Exception as exc:
            log.warning(
                "fetch_exhausted",
                extra={"source": source, "error_code": error_code_for(exc)},
            )
            raise
        else:
            log.info(
                "fetch_succeeded",
                extra={
                    "source": source,
                    "result_count": len(result),
                    "duration_ms": round((time.monotonic() - start) * 1000, 1),
                },
            )
            return result


# ---------------------------------------------------------------------------
# LLM provider selection for synthesis, including the optional failover chain
# ---------------------------------------------------------------------------

def _llm_factory_for(name: str) -> Callable[[], LLMProvider]:
    """Build a zero-argument LLM factory for one provider name.

    Mirrors `ai.providers.factory.get_llm`'s branching, but as a reusable,
    named factory so several of these can be composed into a failover chain.
    Imports stay lazy so selecting a provider never requires every SDK to be
    installed.
    """
    normalized = name.lower().strip()
    if normalized == "anthropic":
        def _factory() -> LLMProvider:
            from ai.providers.anthropic import AnthropicLLM
            return AnthropicLLM()
        return _factory
    if normalized == "openai":
        def _factory() -> LLMProvider:
            from ai.providers.openai import OpenAILLM
            return OpenAILLM()
        return _factory
    if normalized in ("google", "gemini"):
        def _factory() -> LLMProvider:
            from ai.providers.google import GeminiLLM
            return GeminiLLM()
        return _factory
    raise ProviderError(
        f"Unknown provider name={name!r}. Expected anthropic | openai | gemini."
    )


def default_llm_chain(
    fallback_names: Sequence[str] = (),
) -> list[Callable[[], LLMProvider]]:
    """Primary provider (via `get_llm`, i.e. `LLM_PROVIDER`) plus fallbacks.

    `fallback_names` should list provider names in the order they should be
    tried after the primary fails. Duplicate/primary names are skipped so a
    misconfigured chain can't retry the same failing provider twice under a
    different name.
    """
    import os

    primary_name = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()
    chain: list[Callable[[], LLMProvider]] = [get_llm]
    seen = {primary_name if primary_name != "gemini" else "google"}
    for raw_name in fallback_names:
        normalized = raw_name.lower().strip()
        canonical = "google" if normalized == "gemini" else normalized
        if not normalized or canonical in seen:
            continue
        seen.add(canonical)
        chain.append(_llm_factory_for(normalized))
    return chain


class _BoundedThreadRunner:
    """Runs blocking calls through a small, dedicated thread pool instead of
    `asyncio.to_thread`'s shared, loop-wide default executor (capacity
    ``min(32, cpu_count + 4)`` by default).

    Why this exists (see docs/ROLE_B.md, B-02): a synchronous SDK call
    cannot be forcibly killed once it is running -- neither asyncio
    cancellation nor a caller `asyncio.wait_for` timeout stops the
    underlying OS thread. With `asyncio.to_thread`, a retry after a timeout
    therefore starts a brand-new thread *on top of* the one still running
    the previous attempt: three retries against a real LLM API means three
    real, concurrent, billed API calls for one logical request.

    Routing every attempt from one caller through a pool bounded to
    `max_workers` (default 1) fixes the *concurrency*, not the
    un-killability: a new attempt cannot begin executing until a worker
    slot is free, i.e. until the previous attempt's thread actually
    finishes. If a queued attempt's own timeout fires before it was ever
    given a worker, cancelling its (not-yet-started) `Future` succeeds and
    removes it from the queue without ever running -- so a chain of
    timeouts against a truly stuck call costs at most one real running
    thread, not one per retry.
    """

    def __init__(self, *, max_workers: int = 1) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ai-synthesis"
        )

    async def run(self, fn: Callable[..., T], *args: object, **kwargs: object) -> T:
        loop = asyncio.get_running_loop()
        future: concurrent.futures.Future = self._executor.submit(fn, *args, **kwargs)
        try:
            return await asyncio.wrap_future(future, loop=loop)
        except asyncio.CancelledError:
            # No-op if the work item already started running (it cannot be
            # stopped); removes it from the pool's queue if it had not.
            future.cancel()
            raise

    def shutdown(self) -> None:
        """Stop accepting new work; does not wait for a running thread."""
        self._executor.shutdown(wait=False, cancel_futures=True)


class AISynthesisService:
    """Adapts `ai.synthesizer.synthesize` (a blocking LLM call) to the async
    `SynthesisService` port.

    The underlying provider SDKs are synchronous, so each attempt runs in a
    worker thread (see `_BoundedThreadRunner`). Note the same caveat C's
    docs call out: `asyncio.wait_for` bounds how long *this caller* waits,
    but it cannot forcibly kill a blocking SDK thread -- what
    `_BoundedThreadRunner` guarantees is that at most `max_concurrent_calls`
    such threads are ever running at once, not that a timed-out one stops.
    Keep `timeout_seconds` here at or below the SDK client's own configured
    timeout so a stuck call eventually finishes on its own.

    Bonus: multi-provider failover. If `llm_factories` has more than one
    entry, each is tried in order; a provider only moves to the next after
    exhausting its own retry/backoff cycle. All providers share the same
    bounded thread pool, so a stuck primary-provider call also delays a
    fallback provider's attempts from actually starting (though each still
    observes its own `timeout_seconds` while queued).
    """

    def __init__(
        self,
        *,
        llm_factories: Sequence[Callable[[], LLMProvider]] | None = None,
        fallback_provider_names: Sequence[str] = (),
        timeout_seconds: float = 15.0,
        max_attempts: int = 3,
        initial_wait_seconds: float = 0.5,
        max_wait_seconds: float = 6.0,
        max_concurrent_calls: int = 1,
    ) -> None:
        self._llm_factories = (
            list(llm_factories)
            if llm_factories is not None
            else default_llm_chain(fallback_provider_names)
        )
        if not self._llm_factories:
            raise ValueError("at least one LLM factory is required")
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._initial_wait_seconds = initial_wait_seconds
        self._max_wait_seconds = max_wait_seconds
        self._thread_runner = _BoundedThreadRunner(max_workers=max_concurrent_calls)

    def shutdown(self) -> None:
        """Release this service's thread pool. Safe to call more than once;
        does not wait for a currently-running (un-killable) thread."""
        self._thread_runner.shutdown()

    async def synthesize(
        self,
        question: str,
        sources: list[Source],
    ) -> AnswerWithCitations:
        if not sources:
            # B-06: guard before touching any factory/SDK/retry machinery --
            # an empty source list can never produce a synthesizable answer,
            # so there is nothing a provider construction or retry could fix.
            raise NoSourcesError("No sources were provided to synthesize an answer from.")

        last_exc: Exception | None = None
        for provider_index, factory in enumerate(self._llm_factories):
            try:
                llm = factory()
            except ProviderError as exc:
                last_exc = exc
                log.warning(
                    "llm_provider_unavailable",
                    extra={
                        "provider_index": provider_index,
                        "error_code": "provider_construction_failed",
                    },
                )
                continue

            async def _call(llm: LLMProvider = llm) -> AnswerWithCitations:
                return await self._thread_runner.run(synthesize, question, sources, llm=llm)

            start = time.monotonic()
            try:
                result = await call_with_retry(
                    _call,
                    operation="synthesize",
                    timeout_seconds=self._timeout_seconds,
                    max_attempts=self._max_attempts,
                    initial_wait_seconds=self._initial_wait_seconds,
                    max_wait_seconds=self._max_wait_seconds,
                )
            except (ValueError, TypeError):
                # question/sources are invalid regardless of provider --
                # retrying with a different provider cannot fix this.
                raise
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "llm_provider_exhausted",
                    extra={
                        "provider_index": provider_index,
                        "error_code": error_code_for(exc),
                    },
                )
                continue
            else:
                log.info(
                    "synthesis_succeeded",
                    extra={
                        "provider_index": provider_index,
                        "citation_count": len(result.citations),
                        "duration_ms": round((time.monotonic() - start) * 1000, 1),
                    },
                )
                return result

        raise UpstreamDataError(
            "All configured LLM providers failed to synthesize an answer."
        ) from last_exc


# ---------------------------------------------------------------------------
# Zero-argument factories for wiring (D's CLI, C's live benchmark)
# ---------------------------------------------------------------------------

def make_fetch_service() -> AIFetchService:
    """Zero-argument factory: `module:callable` target for
    `scripts/benchmark.py --live --service-factory
    researcher.services.ai_service:make_fetch_service`.
    """
    return AIFetchService()


def make_synthesis_service() -> AISynthesisService:
    """Zero-argument factory mirroring `make_fetch_service`, for D's CLI
    wiring or any other place that composes `ResearchOrchestrator`.
    """
    return AISynthesisService()


__all__ = [
    "AIFetchService",
    "AISynthesisService",
    "default_llm_chain",
    "make_fetch_service",
    "make_synthesis_service",
]