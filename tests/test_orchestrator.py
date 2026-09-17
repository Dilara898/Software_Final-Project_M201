"""Network-free tests of C's collection behavior and resource cleanup."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from ai.providers.base import ProviderError
from ai.schemas import Source
from researcher.concurrency.orchestrator import SOURCE_NAMES, SourceOrchestrator


@pytest.fixture(autouse=True)
def no_http(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("tests must not send HTTP requests")

    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)


def sample(name):
    return Source(
        title=name,
        url=f"https://example.invalid/{name}",
        snippet="example",
        origin=name,
    )


def make(service, **kwargs):
    return SourceOrchestrator(
        service,
        timeout_seconds=kwargs.pop("timeout_seconds", 1),
        max_concurrency=kwargs.pop("max_concurrency", 3),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_tasks_overlap_share_client_and_keep_order():
    entered = set()
    clients = []
    all_entered = asyncio.Event()

    async def fetch(name, question, client):
        entered.add(name)
        clients.append(client)
        if len(entered) == 3:
            all_entered.set()
        await asyncio.wait_for(all_entered.wait(), 0.5)
        return [sample(name)]

    service = AsyncMock()
    service.fetch.side_effect = fetch
    result = await make(service).collect(
        "question", ["web", "arxiv", "wikipedia", "web"]
    )
    assert result.used == list(SOURCE_NAMES)
    assert [s.origin for s in result.sources] == list(SOURCE_NAMES)
    assert result.failed == []
    assert len({id(client) for client in clients}) == 1
    assert clients[0].is_closed


@pytest.mark.asyncio
async def test_source_timeout_preserves_fast_results_and_cancels_slow_task():
    cancelled = asyncio.Event()

    async def fetch(name, question, client):
        if name == "arxiv":
            try:
                await asyncio.sleep(30)
            finally:
                cancelled.set()
        return [sample(name)]

    service = AsyncMock()
    service.fetch.side_effect = fetch
    result = await asyncio.wait_for(make(service, timeout_seconds=0.03).collect("q"), 1)
    assert result.used == ["wikipedia", "web"]
    assert result.failed == ["arxiv"]
    assert result.outcomes[1].status == "timeout"
    assert cancelled.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2, 3])
async def test_failed_sources_are_reported_without_losing_successes(count):
    failed = SOURCE_NAMES[:count]

    async def fetch(name, question, client):
        if name in failed:
            raise ProviderError("simulated outage")
        return [sample(name)]

    service = AsyncMock()
    service.fetch.side_effect = fetch
    result = await make(service).collect("q")
    assert result.failed == list(failed)
    assert result.used == list(SOURCE_NAMES[count:])
    assert len(result.sources) == 3 - count


@pytest.mark.asyncio
async def test_http_failure_is_an_expected_source_failure():
    service = AsyncMock()
    service.fetch.side_effect = httpx.ConnectError("offline")
    result = await make(service).collect("q", ["web"])
    assert result.outcomes[0].status == "error"


@pytest.mark.asyncio
async def test_semaphore_is_shared_across_simultaneous_collections():
    active = peak = 0

    async def fetch(name, question, client):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return [sample(name)]

    service = AsyncMock()
    service.fetch.side_effect = fetch
    orchestrator = make(service, max_concurrency=2)
    first, second = await asyncio.gather(
        orchestrator.collect("one"), orchestrator.collect("two")
    )
    assert peak == 2
    assert first.failed == second.failed == []


@pytest.mark.asyncio
async def test_cache_hits_misses_and_bypass():
    service = AsyncMock()
    service.fetch.return_value = [sample("web")]
    cache = AsyncMock()
    cache.get.return_value = [sample("web")]
    orchestrator = make(service, cache=cache, cache_key=lambda name, q: f"{name}:{q}")
    result = await orchestrator.collect("q", ["web"])
    assert result.outcomes[0].cached
    service.fetch.assert_not_called()
    cache.get.assert_awaited_once_with("web:q")
    cache.get.return_value = None
    await orchestrator.collect("q", ["web"])
    cache.set.assert_awaited_once_with("web:q", [sample("web")])
    cache.reset_mock()
    await orchestrator.collect("q", ["web"], use_cache=False)
    cache.get.assert_not_called()
    cache.set.assert_not_called()
    assert service.fetch.await_count == 2


@pytest.mark.asyncio
async def test_empty_source_is_reported_and_not_cached():
    service = AsyncMock()
    service.fetch.return_value = []
    cache = AsyncMock()
    cache.get.return_value = None
    result = await make(service, cache=cache, cache_key=lambda n, q: n).collect(
        "q", ["web"]
    )
    assert result.outcomes[0].status == "empty"
    cache.set.assert_not_called()


@pytest.mark.asyncio
async def test_cache_write_timeout_preserves_successful_fetch():
    async def slow_write(key, sources):
        await asyncio.sleep(30)

    service = AsyncMock()
    service.fetch.return_value = [sample("web")]
    cache = AsyncMock()
    cache.get.return_value = None
    cache.set.side_effect = slow_write
    result = await asyncio.wait_for(
        make(
            service, timeout_seconds=0.03, cache=cache, cache_key=lambda name, q: name
        ).collect("q", ["web"]),
        2,
    )
    assert result.sources == [sample("web")]
    assert result.used == ["web"]
    assert result.failed == []
    assert result.outcomes[0].warnings[0].code == "cache_write_timeout"


@pytest.mark.asyncio
async def test_unexpected_error_cleans_up_siblings_before_closing_client():
    entered = asyncio.Event()
    stopped = asyncio.Event()
    clients = []

    async def fetch(name, question, client):
        clients.append(client)
        if name == "wikipedia":
            await entered.wait()
            raise TypeError("programming bug")
        entered.set()
        try:
            await asyncio.sleep(30)
        finally:
            assert not client.is_closed
            stopped.set()

    service = AsyncMock()
    service.fetch.side_effect = fetch
    with pytest.raises(TypeError, match="programming bug"):
        await asyncio.wait_for(make(service).collect("q"), 1)
    assert stopped.is_set()
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
async def test_caller_cancellation_is_propagated():
    entered = asyncio.Event()
    stopped = asyncio.Event()

    async def fetch(name, question, client):
        entered.set()
        try:
            await asyncio.sleep(30)
        finally:
            stopped.set()

    service = AsyncMock()
    service.fetch.side_effect = fetch
    task = asyncio.create_task(make(service).collect("q", ["web"]))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_sequential_mode_runs_one_at_a_time():
    active = peak = 0

    async def fetch(name, question, client):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return [sample(name)]

    service = AsyncMock()
    service.fetch.side_effect = fetch
    result = await make(service).collect("q", concurrent=False)
    assert peak == 1
    assert result.used == list(SOURCE_NAMES)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        make(AsyncMock(), timeout_seconds=timeout)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_rejects_invalid_concurrency(limit):
    with pytest.raises(ValueError):
        make(AsyncMock(), max_concurrency=limit)


def test_cache_requires_shared_key_function():
    with pytest.raises(ValueError, match="cache_key"):
        make(AsyncMock(), cache=AsyncMock())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question,wanted", [(" ", ["web"]), ("q", []), ("q", ["unknown"])]
)
async def test_invalid_collection_input(question, wanted):
    with pytest.raises(ValueError):
        await make(AsyncMock()).collect(question, wanted)
