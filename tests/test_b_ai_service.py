"""Part B: AI service layer tests -- retry, backoff, timeout, secret-safe
logging and multi-provider failover, all against fakes (no live API keys,
no network).
"""

from __future__ import annotations
from types import SimpleNamespace

import asyncio
import logging

import httpx
import pytest

from ai.providers.base import ProviderError
from ai.schemas import AnswerWithCitations, Citation, Source
from researcher.concurrency.contracts import UpstreamDataError
from researcher.exceptions import NoSourcesError
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
        result = await service.fetch("wikipedia", "q", client=SimpleNamespace(headers={}))
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
                service.fetch(
                    "wikipedia",
                    "q",
                    client=SimpleNamespace(headers={}),
                ),
                timeout=2,
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
        # Fake, secret-SHAPED fixture -- never a real credential. The
        # test asserts this exact text is redacted before it reaches a
        # log record, so it has to look like a key to be meaningful.
        secret = "sk-test-FAKE-CREDENTIAL-DO-NOT-USE"

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


class TestHttpStatusClassification:
    """Regression tests for B-03: a permanent HTTP status (401, 404, ...)
    wrapped as ProviderError must not be retried, while a transient one
    (429, 5xx) must be -- using the real `ai.sources.fetch_arxiv` against an
    `httpx.MockTransport`, exactly as the review reproduced the bug, rather
    than a fake fetcher, since the classifier reads the exception chain
    `fetch_arxiv` actually produces (ProviderError wrapping
    httpx.HTTPStatusError via `raise_for_status()`).
    """

    def _mock_client(self, status_sequence):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            status = status_sequence[min(calls["n"] - 1, len(status_sequence) - 1)]
            if status == 200:
                return httpx.Response(
                    200,
                    text='<?xml version="1.0"?>'
                    '<feed xmlns="http://www.w3.org/2005/Atom"></feed>',
                )
            return httpx.Response(status, text="error body")

        return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls

    @pytest.mark.asyncio
    async def test_401_is_not_retried(self, monkeypatch):
        monkeypatch.setitem(ai_service._FETCHERS, "arxiv", ai_service.fetch_arxiv)
        client, calls = self._mock_client([401])
        service = AIFetchService(max_attempts=3, initial_wait_seconds=0.01, max_wait_seconds=0.01)
        with pytest.raises(ProviderError):
            await service.fetch("arxiv", "q", client=client)
        await client.aclose()
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_404_is_not_retried(self, monkeypatch):
        monkeypatch.setitem(ai_service._FETCHERS, "arxiv", ai_service.fetch_arxiv)
        client, calls = self._mock_client([404])
        service = AIFetchService(max_attempts=3, initial_wait_seconds=0.01, max_wait_seconds=0.01)
        with pytest.raises(ProviderError):
            await service.fetch("arxiv", "q", client=client)
        await client.aclose()
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_429_is_retried_up_to_max_attempts(self, monkeypatch):
        monkeypatch.setitem(ai_service._FETCHERS, "arxiv", ai_service.fetch_arxiv)
        client, calls = self._mock_client([429, 429, 429])
        service = AIFetchService(max_attempts=3, initial_wait_seconds=0.01, max_wait_seconds=0.01)
        with pytest.raises(ProviderError):
            await service.fetch("arxiv", "q", client=client)
        await client.aclose()
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_500_is_retried_up_to_max_attempts(self, monkeypatch):
        monkeypatch.setitem(ai_service._FETCHERS, "arxiv", ai_service.fetch_arxiv)
        client, calls = self._mock_client([500, 500, 500])
        service = AIFetchService(max_attempts=3, initial_wait_seconds=0.01, max_wait_seconds=0.01)
        with pytest.raises(ProviderError):
            await service.fetch("arxiv", "q", client=client)
        await client.aclose()
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_429_then_200_recovers(self, monkeypatch):
        monkeypatch.setitem(ai_service._FETCHERS, "arxiv", ai_service.fetch_arxiv)
        client, calls = self._mock_client([429, 429, 200])
        service = AIFetchService(max_attempts=5, initial_wait_seconds=0.01, max_wait_seconds=0.01)
        result = await service.fetch("arxiv", "q", client=client)
        await client.aclose()
        assert calls["n"] == 3
        assert result == []  # empty <feed/> parses to no entries, not an error

    def test_error_code_for_includes_http_status(self):
        from researcher.services.retry import error_code_for

        exc = ProviderError("arXiv query failed: 401")
        exc.__cause__ = httpx.HTTPStatusError(
            "401", request=httpx.Request("GET", "https://x.invalid"),
            response=httpx.Response(401, request=httpx.Request("GET", "https://x.invalid")),
        )
        assert error_code_for(exc) == "provider_error_status_401"


class TestFetchServicePacing:
    """Integration-level regression test for B-05: pacing must apply to
    every attempt made through AIFetchService.fetch, including retries, not
    just the first call.
    """

    @pytest.mark.asyncio
    async def test_pacing_applies_across_retries(self, monkeypatch):
        import time

        call_times: list[float] = []

        async def flaky(query, *, max_results, client):
            call_times.append(time.monotonic())
            if len(call_times) < 3:
                raise ProviderError("transient")
            return []

        monkeypatch.setitem(ai_service._FETCHERS, "web", flaky)
        service = AIFetchService(
            max_attempts=5,
            min_interval_seconds=0.05,
            initial_wait_seconds=0.001,
            max_wait_seconds=0.001,
        )
        await service.fetch("web", "q", client=object())
        intervals = [call_times[i] - call_times[i - 1] for i in range(1, len(call_times))]
        assert all(interval >= 0.04 for interval in intervals)  # small tolerance

    @pytest.mark.asyncio
    async def test_pacing_disabled_by_default(self, monkeypatch):
        import time

        call_times: list[float] = []

        async def flaky(query, *, max_results, client):
            call_times.append(time.monotonic())
            if len(call_times) < 3:
                raise ProviderError("transient")
            return []

        monkeypatch.setitem(ai_service._FETCHERS, "web", flaky)
        service = AIFetchService(
            max_attempts=5, initial_wait_seconds=0.001, max_wait_seconds=0.001
        )
        start = time.monotonic()
        await service.fetch("web", "q", client=object())
        assert time.monotonic() - start < 0.1  # no artificial pacing delay


class TestWikipediaTransportRetryIntegration:
    """Integration-level regression test for B-07, through the full
    AIFetchService.fetch path (not just the transport wrapper in
    isolation -- see tests/test_b_transport.py for that).
    """

    @pytest.mark.asyncio
    async def test_wikipedia_summary_500_then_200_recovers_through_fetch_service(
        self, monkeypatch
    ):
        monkeypatch.setitem(ai_service._FETCHERS, "wikipedia", ai_service.fetch_wikipedia)
        calls = {"summary": 0}

        def handler(request):
            if "opensearch" in str(request.url):
                return httpx.Response(200, json=["q", ["Python"], [], []])
            calls["summary"] += 1
            if calls["summary"] < 2:
                return httpx.Response(500, text="server error")
            return httpx.Response(
                200,
                json={
                    "title": "Python",
                    "extract": "Python is a language.",
                    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Python"}},
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service = AIFetchService(max_attempts=1)  # outer retry off; transport retry only
        result = await service.fetch("wikipedia", "python", client=client)
        await client.aclose()

        assert len(result) == 1
        assert calls["summary"] == 2

    @pytest.mark.asyncio
    async def test_can_be_disabled(self, monkeypatch):
        monkeypatch.setitem(ai_service._FETCHERS, "wikipedia", ai_service.fetch_wikipedia)
        calls = {"summary": 0}

        def handler(request):
            if "opensearch" in str(request.url):
                return httpx.Response(200, json=["q", ["Python"], [], []])
            calls["summary"] += 1
            return httpx.Response(500, text="server error")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service = AIFetchService(max_attempts=1, wikipedia_transport_retry=False)
        result = await service.fetch("wikipedia", "python", client=client)
        await client.aclose()

        assert result == []
        assert calls["summary"] == 1  # not retried -- feature disabled


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
    async def test_failover_to_second_provider_after_first_exhausts(
        self, monkeypatch
    ):
        from ai.providers.base import LLMProvider

        provider_used: list[int] = []
        token_budgets: list[int] = []

        class FakeProvider(LLMProvider):
            def __init__(self, index: int):
                self.index = index

            def complete(
                self,
                prompt: str,
                *,
                json_schema: dict | None = None,
                max_tokens: int = 1024,
            ) -> str:
                provider_used.append(self.index)
                token_budgets.append(max_tokens)

                if self.index == 0:
                    raise ProviderError("primary provider down")

                return "Supported [1]."

        def fake_synthesize(q, s, *, llm=None):
            llm.complete(q)
            return fake_answer(q, s)

        monkeypatch.setattr(ai_service, "synthesize", fake_synthesize)

        service = AISynthesisService(
            llm_factories=[
                lambda: FakeProvider(0),
                lambda: FakeProvider(1),
            ],
            max_attempts=2,
            initial_wait_seconds=0.01,
            max_wait_seconds=0.02,
        )

        try:
            result = await service.synthesize(
                "q", [sample_source("web")]
            )
        finally:
            service.shutdown()

        assert result.answer == "Supported [1]."
        assert provider_used == [0, 0, 1]
        assert token_budgets == [4096, 4096, 4096]

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
    async def test_empty_sources_raises_no_sources_error_without_touching_factory(self):
        """Regression test for B-06: an empty source list must fail before
        any provider factory, SDK, or retry machinery is touched.
        """
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            return object()

        service = AISynthesisService(llm_factories=[factory])
        with pytest.raises(NoSourcesError):
            await service.synthesize("q", [])
        assert calls["n"] == 0
        service.shutdown()

    @pytest.mark.asyncio
    async def test_no_secret_text_reaches_logs(self, monkeypatch, caplog):
        # Fake, secret-SHAPED fixture -- never a real credential. The
        # test asserts this exact text is redacted before it reaches a
        # log record, so it has to look like a key to be meaningful.
        secret = "sk-test-FAKE-CREDENTIAL-DO-NOT-USE"

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