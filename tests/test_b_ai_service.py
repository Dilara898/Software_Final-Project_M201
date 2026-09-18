"""Part B: AI service layer tests -- retry, backoff, timeout, secret-safe
logging and multi-provider failover, all against fakes (no live API keys,
no network).
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from ai.providers.base import ProviderError
from ai.schemas import AnswerWithCitations, Citation, Source
from researcher.concurrency.contracts import UpstreamDataError
from researcher.services import ai_service
from researcher.services.ai_service import (
    AIFetchService,
    AISynthesisService,
    make_fetch_service,
    make_synthesis_service,
)


def sample_source(origin: str = "wikipedia") -> Source:
    return Source(
        title=f"{origin} title",
        url=f"https://example.invalid/{origin}",
        snippet="evidence",
        origin=origin,
    )


# ---------------------------------------------------------------------------
# AIFetchService
# ---------------------------------------------------------------------------


class TestAIFetchService:
    @pytest.mark.asyncio
    async def test_dispatches_to_correct_fetcher(self, monkeypatch):
        calls: list[str] = []

        async def fake_wikipedia(query, *, max_results, client):
            calls.append("wikipedia")
            return [sample_source("wikipedia")]

        monkeypatch.setitem(ai_service._FETCHERS, "wikipedia", fake_wikipedia)
        service = AIFetchService()
        result = await service.fetch("wikipedia", "q", client=object())
        assert calls == ["wikipedia"]
        assert result[0].origin == "wikipedia"

    @pytest.mark.asyncio
    async def test_unknown_source_raises_upstream_data_error_without_retry(self):
        service = AIFetchService(max_attempts=5)
        with pytest.raises(UpstreamDataError):
            await service.fetch("not-a-real-source", "q", client=object())

    @pytest.mark.asyncio
    async def test_retries_transient_failure_then_succeeds(self, monkeypatch):
        attempts = {"n": 0}

        async def flaky(query, *, max_results, client):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ProviderError("simulated transient outage: apikey=SECRET")
            return [sample_source("arxiv")]

        monkeypatch.setitem(ai_service._FETCHERS, "arxiv", flaky)
        service = AIFetchService(
            max_attempts=5, initial_wait_seconds=0.01, max_wait_seconds=0.02
        )
        result = await service.fetch("arxiv", "q", client=object())
        assert attempts["n"] == 3
        assert result[0].origin == "arxiv"

    @pytest.mark.asyncio
    async def test_exhausts_retries_and_reraises_original_type(self, monkeypatch):
        async def always_fails(query, *, max_results, client):
            raise ProviderError("always broken: apikey=SECRET")

        monkeypatch.setitem(ai_service._FETCHERS, "web", always_fails)
        service = AIFetchService(
            max_attempts=3, initial_wait_seconds=0.01, max_wait_seconds=0.02
        )
        with pytest.raises(ProviderError):
            await service.fetch("web", "q", client=object())

    @pytest.mark.asyncio
    async def test_per_call_timeout_is_enforced(self, monkeypatch):
        async def always_slow(query, *, max_results, client):
            await asyncio.sleep(30)

        monkeypatch.setitem(ai_service._FETCHERS, "wikipedia", always_slow)
        service = AIFetchService(
            timeout_seconds=0.05,
            max_attempts=2,
            initial_wait_seconds=0.01,
            max_wait_seconds=0.02,
        )
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                service.fetch("wikipedia", "q", client=object()), timeout=2
            )

    @pytest.mark.asyncio
    async def test_programmer_error_propagates_with_original_type(self, monkeypatch):
        """Regression test for B-04: a bug in the fetcher itself (not a
        provider/network failure) must surface as its own exception type,
        not get relabeled as UpstreamDataError -- that label is reserved for
        an actual malformed *return value*, checked separately below.
        """
        attempts = {"n": 0}

        async def broken(query, *, max_results, client):
            attempts["n"] += 1
            raise RuntimeError("programming error, not a provider outage")

        monkeypatch.setitem(ai_service._FETCHERS, "web", broken)
        service = AIFetchService(max_attempts=5)
        with pytest.raises(RuntimeError):
            await service.fetch("web", "q", client=object())
        assert attempts["n"] == 1

    @pytest.mark.asyncio
    async def test_malformed_return_shape_raises_upstream_data_error(self, monkeypatch):
        async def bad_shape(query, *, max_results, client):
            return "not a list of Source objects"

        monkeypatch.setitem(ai_service._FETCHERS, "web", bad_shape)
        service = AIFetchService(max_attempts=3, initial_wait_seconds=0.01, max_wait_seconds=0.01)
        with pytest.raises(UpstreamDataError):
            await service.fetch("web", "q", client=object())

    @pytest.mark.asyncio
    async def test_connection_error_is_retried_with_original_type(self, monkeypatch):
        """Regression test for B-04: a raw, retryable ConnectionError must
        reach the retry policy (and be retried) instead of being swallowed
        into UpstreamDataError by the fetch boundary before retry.py ever
        sees it.
        """
        attempts = {"n": 0}

        async def flaky(query, *, max_results, client):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ConnectionError("raw transient network failure")
            return [sample_source("web")]

        monkeypatch.setitem(ai_service._FETCHERS, "web", flaky)
        service = AIFetchService(
            max_attempts=5, initial_wait_seconds=0.01, max_wait_seconds=0.01
        )
        result = await service.fetch("web", "q", client=object())
        assert attempts["n"] == 3
        assert result[0].origin == "web"

    @pytest.mark.asyncio
    async def test_httpx_transport_error_is_retried_with_original_type(self, monkeypatch):
        import httpx

        attempts = {"n": 0}

        async def flaky(query, *, max_results, client):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise httpx.ConnectError("raw transport failure")
            return [sample_source("web")]

        monkeypatch.setitem(ai_service._FETCHERS, "web", flaky)
        service = AIFetchService(
            max_attempts=5, initial_wait_seconds=0.01, max_wait_seconds=0.01
        )
        result = await service.fetch("web", "q", client=object())
        assert attempts["n"] == 3
        assert result[0].origin == "web"

    @pytest.mark.asyncio
    async def test_no_secret_text_reaches_logs(self, monkeypatch, caplog):
        secret = "sk-live-SUPER-SECRET-VALUE"

        async def always_fails(query, *, max_results, client):
            raise ProviderError(f"upstream said: {secret}")

        monkeypatch.setitem(ai_service._FETCHERS, "web", always_fails)
        service = AIFetchService(
            max_attempts=2, initial_wait_seconds=0.01, max_wait_seconds=0.02
        )
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ProviderError):
                await service.fetch("web", "q", client=object())
        for record in caplog.records:
            assert secret not in record.getMessage()
            assert secret not in str(record.__dict__)


# ---------------------------------------------------------------------------
# AISynthesisService
# ---------------------------------------------------------------------------


def fake_answer(question: str, sources: list[Source]) -> AnswerWithCitations:
    return AnswerWithCitations(
        question=question,
        answer="Supported [1].",
        citations=[Citation(index=1, source=sources[0])],
    )


class TestAISynthesisService:
    @pytest.mark.asyncio
    async def test_happy_path_returns_answer(self, monkeypatch):
        monkeypatch.setattr(
            ai_service, "synthesize", lambda q, s, *, llm=None: fake_answer(q, s)
        )
        service = AISynthesisService(llm_factories=[lambda: object()])
        result = await service.synthesize("q", [sample_source("web")])
        assert result.answer == "Supported [1]."
        service.shutdown()

    @pytest.mark.asyncio
    async def test_retries_transient_failure_then_succeeds(self, monkeypatch):
        attempts = {"n": 0}

        def flaky(q, s, *, llm=None):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ProviderError("transient: apikey=SECRET")
            return fake_answer(q, s)

        monkeypatch.setattr(ai_service, "synthesize", flaky)
        service = AISynthesisService(
            llm_factories=[lambda: object()],
            max_attempts=5,
            initial_wait_seconds=0.01,
            max_wait_seconds=0.02,
        )
        result = await service.synthesize("q", [sample_source("web")])
        assert attempts["n"] == 3
        assert result.answer == "Supported [1]."
        service.shutdown()

    @pytest.mark.asyncio
    async def test_failover_to_second_provider_after_first_exhausts(self, monkeypatch):
        provider_used: list[int] = []

        def make_synth(fail_provider_index: int):
            def _synth(q, s, *, llm=None):
                idx = llm  # llm is the marker int for this fake test
                provider_used.append(idx)
                if idx == fail_provider_index:
                    raise ProviderError("primary provider down: apikey=SECRET")
                return fake_answer(q, s)

            return _synth

        monkeypatch.setattr(ai_service, "synthesize", make_synth(0))
        service = AISynthesisService(
            llm_factories=[lambda: 0, lambda: 1],
            max_attempts=2,
            initial_wait_seconds=0.01,
            max_wait_seconds=0.02,
        )
        result = await service.synthesize("q", [sample_source("web")])
        assert result.answer == "Supported [1]."
        # Provider 0 tried (and retried) before falling over to provider 1.
        assert provider_used.count(0) == 2
        assert provider_used.count(1) == 1
        service.shutdown()

    @pytest.mark.asyncio
    async def test_all_providers_exhausted_raises_upstream_data_error(self, monkeypatch):
        def always_fails(q, s, *, llm=None):
            raise ProviderError("down: apikey=SECRET")

        monkeypatch.setattr(ai_service, "synthesize", always_fails)
        service = AISynthesisService(
            llm_factories=[lambda: 0, lambda: 1],
            max_attempts=2,
            initial_wait_seconds=0.01,
            max_wait_seconds=0.02,
        )
        with pytest.raises(UpstreamDataError):
            await service.synthesize("q", [sample_source("web")])
        service.shutdown()

    @pytest.mark.asyncio
    async def test_value_error_is_not_retried_or_failed_over(self, monkeypatch):
        calls = {"n": 0}

        def bad_input(q, s, *, llm=None):
            calls["n"] += 1
            raise ValueError("question must be non-empty")

        monkeypatch.setattr(ai_service, "synthesize", bad_input)
        service = AISynthesisService(
            llm_factories=[lambda: 0, lambda: 1], max_attempts=5
        )
        with pytest.raises(ValueError):
            await service.synthesize("", [sample_source("web")])
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_per_call_timeout_is_enforced(self, monkeypatch):
        def always_slow(q, s, *, llm=None):
            import time

            # Short enough to keep the suite fast, long enough that the
            # 0.05s per-attempt timeout below reliably fires first.
            time.sleep(0.3)

        monkeypatch.setattr(ai_service, "synthesize", always_slow)
        service = AISynthesisService(
            llm_factories=[lambda: 0],
            timeout_seconds=0.05,
            max_attempts=1,
        )
        with pytest.raises(UpstreamDataError):
            await asyncio.wait_for(
                service.synthesize("q", [sample_source("web")]), timeout=2
            )
        service.shutdown()

    @pytest.mark.asyncio
    async def test_synthesis_timeout_retries_do_not_pile_up_threads(self, monkeypatch):
        """Regression test for B-02: a blocking synthesis call that keeps
        running past its own timeout must not accumulate one concurrently
        running thread per retry. `asyncio.to_thread`'s shared default
        executor made this possible (verified: 3 attempts -> 3 threads
        running at once); `AISynthesisService` must bound it instead.
        """
        import threading

        active = {"n": 0, "peak": 0}
        started = {"n": 0}
        lock = threading.Lock()

        def blocking_synth(q, s, *, llm=None):
            import time

            with lock:
                started["n"] += 1
                active["n"] += 1
                active["peak"] = max(active["peak"], active["n"])
            time.sleep(0.3)
            with lock:
                active["n"] -= 1
            return fake_answer(q, s)

        monkeypatch.setattr(ai_service, "synthesize", blocking_synth)
        service = AISynthesisService(
            llm_factories=[lambda: 0],
            timeout_seconds=0.03,
            max_attempts=3,
            initial_wait_seconds=0.01,
            max_wait_seconds=0.01,
        )
        start = asyncio.get_event_loop().time()
        with pytest.raises(UpstreamDataError):
            await service.synthesize("q", [sample_source("web")])
        elapsed = asyncio.get_event_loop().time() - start

        # The caller must not be blocked waiting for the stuck thread.
        assert elapsed < 0.3
        # At most one thread from this service is ever running at once.
        assert active["peak"] == 1
        # Only the first attempt ever actually started; the retries that
        # were still queued behind it got cancelled instead of piling up.
        assert started["n"] == 1

        service.shutdown()
        await asyncio.sleep(0.35)  # let the one real background thread finish

    def test_construction_requires_at_least_one_factory(self):
        with pytest.raises(ValueError):
            AISynthesisService(llm_factories=[])

    @pytest.mark.asyncio
    async def test_no_secret_text_reaches_logs(self, monkeypatch, caplog):
        secret = "sk-live-SUPER-SECRET-VALUE"

        def always_fails(q, s, *, llm=None):
            raise ProviderError(f"upstream said: {secret}")

        monkeypatch.setattr(ai_service, "synthesize", always_fails)
        service = AISynthesisService(
            llm_factories=[lambda: 0],
            max_attempts=2,
            initial_wait_seconds=0.01,
            max_wait_seconds=0.02,
        )
        with caplog.at_level(logging.WARNING):
            with pytest.raises(UpstreamDataError):
                await service.synthesize("q", [sample_source("web")])
        for record in caplog.records:
            assert secret not in record.getMessage()
            assert secret not in str(record.__dict__)
        service.shutdown()


# ---------------------------------------------------------------------------
# Factories (used by scripts/benchmark.py --live --service-factory ...)
# ---------------------------------------------------------------------------


def test_make_fetch_service_is_zero_argument_and_returns_fetch_service():
    service = make_fetch_service()
    assert isinstance(service, AIFetchService)


def test_make_synthesis_service_is_zero_argument_and_returns_synthesis_service():
    service = make_synthesis_service()
    assert isinstance(service, AISynthesisService)