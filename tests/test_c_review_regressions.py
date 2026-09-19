"""Regression cases found by PR review; every test exercises a public boundary."""

import asyncio
import logging
from unittest.mock import AsyncMock

import httpx
import pytest

from ai.providers.base import ProviderError
from ai.schemas import AnswerWithCitations, Source
from researcher.concurrency.orchestrator import SourceOrchestrator, valid_source
from researcher.concurrency.research import InvalidAnswerError, validate_answer
from scripts import benchmark


def source(url="https://example.invalid/a"):
    return Source(title="Example", url=url, snippet="Evidence", origin="web")


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid:bad/a",
        "https://example.invalid:65536/a",
        "https://example.invalid:-1/a",
    ],
)
def test_invalid_url_ports_rejected(url):
    assert not valid_source(source(url), "web")


@pytest.mark.asyncio
async def test_cancellation_waits_for_asymmetric_async_cleanup():
    entered = asyncio.Event()
    cleanup_started = asyncio.Event()
    release = asyncio.Event()
    count = 0
    cleaned = []

    class Service:
        async def fetch(self, name, query, client):
            nonlocal count
            count += 1
            if count == 3:
                entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                if name != "wikipedia":
                    cleanup_started.set()
                    await release.wait()
                cleaned.append(name)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200))
    ) as client:
        worker = SourceOrchestrator(
            Service(), client=client, timeout_seconds=5, max_concurrency=3
        )
        task = asyncio.create_task(worker.collect("q"))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            await asyncio.wait_for(cleanup_started.wait(), 1)
            # Let gather propagate the fast child's cancellation before releasing cleanup.
            for _ in range(10):
                await asyncio.sleep(0)
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert set(cleaned) == {"wikipedia", "arxiv", "web"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [ProviderError("private detail"), TimeoutError("private detail")]
)
async def test_source_outage_logs_warning_without_exception_text(error, caplog):
    service = AsyncMock()
    service.fetch.side_effect = error
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200))
    ) as client:
        with caplog.at_level(logging.WARNING):
            result = await SourceOrchestrator(
                service, client=client, timeout_seconds=1, max_concurrency=3
            ).collect("q", ["web"])
    assert result.failed == ["web"]
    assert any(
        r.levelno == logging.WARNING and getattr(r, "source", None) == "web"
        for r in caplog.records
    )
    assert "private detail" not in caplog.text


@pytest.mark.asyncio
async def test_cache_truncation_is_visible():
    cache = AsyncMock()
    cache.get.return_value = [source(f"https://example.invalid/{i}") for i in range(4)]
    service = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200))
    ) as client:
        result = await SourceOrchestrator(
            service,
            client=client,
            cache=cache,
            cache_key=lambda n, q: n,
            timeout_seconds=1,
            max_concurrency=3,
        ).collect("q", ["web"])
    assert len(result.sources) == 3
    assert any(w.code == "source_limit_applied" for w in result.warnings)
    service.fetch.assert_not_called()


def test_huge_numeric_citation_uses_typed_error():
    answer = AnswerWithCitations(question="q", answer="[" + "9" * 5000 + "]")
    with pytest.raises(InvalidAnswerError):
        validate_answer(answer, "q", [source()])


def test_live_cli_passes_selected_timeout_to_http_client(monkeypatch, tmp_path, capsys):
    dataset = tmp_path / "questions.json"
    dataset.write_text('{"questions":[{"id":"q1","text":"q"}]}')
    service = AsyncMock()

    async def fetch(name, query, client):
        assert client.timeout.read == 9
        assert client.timeout.connect == 9
        return [
            Source(
                title=name,
                url=f"https://example.invalid/{name}",
                snippet="Evidence",
                origin=name,
            )
        ]

    service.fetch.side_effect = fetch
    monkeypatch.setattr(benchmark, "load_factory", lambda spec: lambda: service)
    original = benchmark.source_client

    def offline_transport_client(**kwargs):
        # Live benchmarking must reach arXiv, whose http:// URL 301s to https;
        # the shared factory is what guarantees that, so assert it here rather
        # than rebuilding a bare client the way this call site used to.
        client = original(
            **kwargs,
            transport=httpx.MockTransport(lambda r: httpx.Response(200)),
        )
        assert client.follow_redirects is True
        return client

    monkeypatch.setattr(benchmark, "source_client", offline_transport_client)
    assert (
        benchmark.main(
            [
                "--live",
                "--service-factory",
                "fake:create",
                "--timeout",
                "9",
                "--repeats",
                "1",
                "--questions",
                str(dataset),
            ]
        )
        == 0
    )
    assert "LIVE COLLECTION" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--out", "--csv"])
def test_benchmark_refuses_to_overwrite_input(flag, tmp_path, monkeypatch):
    dataset = tmp_path / "questions.json"
    original = '{"questions":[{"id":"q1","text":"q"}]}'
    dataset.write_text(original)
    run = AsyncMock()
    monkeypatch.setattr(benchmark, "run_benchmark", run)
    with pytest.raises(SystemExit) as caught:
        benchmark.main(["--offline", "--questions", str(dataset), flag, str(dataset)])
    assert caught.value.code == 2
    assert dataset.read_text() == original
    run.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid:443/a",
        "https://[::1]:8080/a",
        "https://example.invalid/a",
    ],
)
def test_valid_url_ports_still_accepted(url):
    assert valid_source(source(url), "web")


@pytest.mark.asyncio
async def test_sibling_error_does_not_interrupt_timeout_cleanup():
    cleanup_started = asyncio.Event()
    release = asyncio.Event()
    cleaned = []

    class Service:
        async def fetch(self, name, query, client):
            if name == "wikipedia":
                try:
                    # A service-level timeout has already cancelled this task.
                    async with asyncio.timeout(0.01):
                        try:
                            await asyncio.Event().wait()
                        finally:
                            cleanup_started.set()
                            await release.wait()
                            cleaned.append(name)
                except TimeoutError:
                    return []
            await cleanup_started.wait()
            raise TypeError("broken service contract")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200))
    ) as client:
        worker = SourceOrchestrator(
            Service(), client=client, timeout_seconds=2, max_concurrency=3
        )
        task = asyncio.create_task(worker.collect("q", ["wikipedia", "web"]))
        try:
            await asyncio.wait_for(cleanup_started.wait(), 1)
            for _ in range(10):
                await asyncio.sleep(0)
        finally:
            release.set()
            with pytest.raises(TypeError):
                await task
        assert cleaned == ["wikipedia"]
