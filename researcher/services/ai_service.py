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
import logging
from collections.abc import Awaitable, Callable, Sequence

import httpx

from ai.providers.base import LLMProvider, ProviderError
from ai.providers.factory import get_llm
from ai.schemas import AnswerWithCitations, Source
from ai.sources import fetch_arxiv, fetch_web, fetch_wikipedia
from ai.synthesizer import synthesize
from researcher.concurrency.contracts import UpstreamDataError
from researcher.services.retry import call_with_retry, error_code_for

log = logging.getLogger(__name__)

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
    """

    def __init__(
        self,
        *,
        max_results: int = 3,
        timeout_seconds: float = 8.0,
        max_attempts: int = 3,
        initial_wait_seconds: float = 0.5,
        max_wait_seconds: float = 4.0,
    ) -> None:
        self._max_results = max_results
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._initial_wait_seconds = initial_wait_seconds
        self._max_wait_seconds = max_wait_seconds

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

        async def _call() -> list[Source]:
            try:
                return await fetcher(query, max_results=self._max_results, client=client)
            except ProviderError:
                raise
            except UpstreamDataError:
                raise
            except Exception as exc:  # pragma: no cover - defensive boundary
                # A fetcher returning a shape we didn't expect is bad data,
                # not a transient failure; do not retry it.
                raise UpstreamDataError(
                    f"unexpected error from {source} fetcher"
                ) from exc

        try:
            return await call_with_retry(
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


class AISynthesisService:
    """Adapts `ai.synthesizer.synthesize` (a blocking LLM call) to the async
    `SynthesisService` port.

    The underlying provider SDKs are synchronous, so each attempt runs in a
    worker thread via `asyncio.to_thread`. Note the same caveat C's docs
    call out: `asyncio.wait_for` bounds how long *this caller* waits, but it
    cannot forcibly kill a blocking SDK thread. Keep `timeout_seconds` here
    at or below the SDK client's own configured timeout so a stuck call
    eventually finishes on its own rather than leaking a thread forever.

    Bonus: multi-provider failover. If `llm_factories` has more than one
    entry, each is tried in order; a provider only moves to the next after
    exhausting its own retry/backoff cycle.
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

    async def synthesize(
        self,
        question: str,
        sources: list[Source],
    ) -> AnswerWithCitations:
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

            async def _call() -> AnswerWithCitations:
                return await asyncio.to_thread(synthesize, question, sources, llm=llm)

            try:
                return await call_with_retry(
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