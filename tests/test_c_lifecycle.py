"""Regression tests for cache isolation, input contracts and cancellation."""

import asyncio
import logging
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from ai.providers.base import ProviderError
from ai.schemas import Source
from researcher.concurrency.contracts import StorageUnavailableError, UpstreamDataError
from researcher.concurrency.models import SourceOutcome
from researcher.concurrency.orchestrator import SourceOrchestrator


def source(name="web", **updates):
    return Source(
        **dict(
            title=name,
            url=f"https://example.invalid/{name}",
            snippet="useful source",
            origin=name,
        )
        | updates
    )


@pytest_asyncio.fixture
async def client():
    async def forbidden(request):
        raise AssertionError("live HTTP is forbidden")

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as value:
        yield value


def collector(client, service=None, **kwargs):
    if service is None:
        service = AsyncMock()
        service.fetch.return_value = [source()]
    return SourceOrchestrator(
        service,
        client=client,
        timeout_seconds=kwargs.pop("timeout_seconds", 1),
        max_concurrency=kwargs.pop("max_concurrency", 3),
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get", "set"])
@pytest.mark.parametrize(
    "error",
    [
        StorageUnavailableError("secret-key"),
        ConnectionError("secret-key"),
        TimeoutError(),
    ],
)
async def test_expected_cache_errors_keep_live_result(client, operation, error, caplog):
    cache = AsyncMock()
    cache.get.return_value = None
    getattr(cache, operation).side_effect = error
    with caplog.at_level(logging.INFO):
        result = await collector(client, cache=cache, cache_key=lambda n, q: n).collect(
            "q", ["web"]
        )
    assert result.used == ["web"] and result.sources == [source()]
    assert result.warnings[0].stage == (
        "cache_read" if operation == "get" else "cache_write"
    )
    assert "secret-key" not in caplog.text
    assert caplog.records and caplog.records[-1].request_id == result.request_id


@pytest.mark.asyncio
async def test_slow_cache_read_has_own_budget_and_falls_back(client):
    cache = AsyncMock()

    async def slow(key):
        await asyncio.sleep(30)

    cache.get.side_effect = slow
    result = await asyncio.wait_for(
        collector(
            client, cache=cache, cache_key=lambda n, q: n, cache_timeout_seconds=0.01
        ).collect("q", ["web"]),
        1,
    )
    assert result.used == ["web"]
    assert result.cache_stats.read_errors == 1
    assert result.warnings[0].code == "cache_read_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload", [[], {"bad": "shape"}, [source("arxiv")], [source(snippet=" ")]]
)
async def test_invalid_cache_payload_is_refetched(client, payload):
    service = AsyncMock()
    service.fetch.return_value = [source()]
    cache = AsyncMock()
    cache.get.return_value = payload
    result = await collector(
        client, service, cache=cache, cache_key=lambda n, q: n
    ).collect("q", ["web"])
    service.fetch.assert_awaited_once()
    assert result.sources == [source()] and result.cache_stats.read_errors == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get", "set"])
async def test_programming_errors_in_cache_propagate(client, operation):
    cache = AsyncMock()
    cache.get.return_value = None
    getattr(cache, operation).side_effect = TypeError("bug")
    with pytest.raises(TypeError, match="bug"):
        await collector(client, cache=cache, cache_key=lambda n, q: n).collect(
            "q", ["web"]
        )


@pytest.mark.asyncio
async def test_cache_hit_skips_full_source_queue_and_stats_are_per_request(client):
    entered, release = asyncio.Event(), asyncio.Event()
    service = AsyncMock()

    async def fetch(name, question, client):
        entered.set()
        await release.wait()
        return [source()]

    service.fetch.side_effect = fetch
    cache = AsyncMock()
    cache.get.return_value = [source()]
    col = collector(
        client, service, max_concurrency=1, cache=cache, cache_key=lambda n, q: n
    )
    first = asyncio.create_task(col.collect("uncached", ["web"], use_cache=False))
    await asyncio.wait_for(entered.wait(), 1)
    try:
        hit = await asyncio.wait_for(col.collect("cached", ["web"]), 0.5)
        assert hit.cache_stats.hits == 1 and hit.cache_stats.bypasses == 0
        assert hit.outcomes[0].timings.queue_ms == 0
    finally:
        release.set()
        bypass = await first
    assert bypass.cache_stats.hits == 0 and bypass.cache_stats.bypasses == 1


@pytest.mark.asyncio
async def test_queue_timeout_does_not_invoke_service_or_leak_permit(client):
    entered, release = asyncio.Event(), asyncio.Event()
    service = AsyncMock()

    async def fetch(name, question, client):
        entered.set()
        await release.wait()
        return [source()]

    service.fetch.side_effect = fetch
    col = collector(client, service, max_concurrency=1, queue_timeout_seconds=0.01)
    first = asyncio.create_task(col.collect("first", ["web"]))
    await asyncio.wait_for(entered.wait(), 1)
    try:
        result = await col.collect("queued", ["web"])
        assert result.outcomes[0].status == "queue_timeout"
        assert service.fetch.await_count == 1
    finally:
        release.set()
        await first
    assert (await col.collect("next", ["web"])).used == ["web"]


@pytest.mark.asyncio
async def test_cancel_queued_caller_leaves_permits_usable(client):
    entered, release = asyncio.Event(), asyncio.Event()
    service = AsyncMock()

    async def fetch(name, question, client):
        entered.set()
        await release.wait()
        return [source()]

    service.fetch.side_effect = fetch
    col = collector(client, service, max_concurrency=1)
    first = asyncio.create_task(col.collect("one", ["web"]))
    await asyncio.wait_for(entered.wait(), 1)
    waiting = asyncio.create_task(col.collect("two", ["web"]))
    await asyncio.sleep(0)
    waiting.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await waiting
    finally:
        release.set()
        await first
    assert (await col.collect("three", ["web"])).used == ["web"]


@pytest.mark.asyncio
async def test_client_is_reused_and_only_owner_closes_it(client):
    service = AsyncMock()
    service.fetch.return_value = [source()]
    col = collector(client, service)
    await col.collect("one", ["web"])
    await col.collect("two", ["web"])
    assert not client.is_closed
    assert all(call.args[2] is client for call in service.fetch.await_args_list)
    await client.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await col.collect("three", ["web"])


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [None, {}, ["wrong type"]])
async def test_broken_service_contract_is_not_hidden(client, payload):
    service = AsyncMock()
    service.fetch.return_value = payload
    with pytest.raises(TypeError):
        await collector(client, service).collect("q", ["web"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        source("arxiv"),
        source(snippet=" "),
        source(url="file:///a"),
        source(url="https://["),
        source(url="https://bad host/a"),
        source(snippet="a" * 20_001),
    ],
)
async def test_invalid_source_data_is_explicit_and_good_subset_survives(client, bad):
    service = AsyncMock()
    service.fetch.return_value = [bad]
    col = collector(client, service)
    result = await col.collect("q", ["web"])
    assert result.outcomes[0].status == "invalid"
    service.fetch.return_value = [bad, source()]
    result = await col.collect("q", ["web"])
    assert result.sources == [source()]
    assert result.warnings[0].code == "source_invalid_items"


@pytest.mark.asyncio
async def test_empty_is_not_unavailable_and_result_limit_is_explicit(client):
    service = AsyncMock()
    service.fetch.return_value = []
    col = collector(client, service, max_results_per_source=1)
    empty = await col.collect("q", ["web"])
    assert empty.empty == ["web"] and empty.failed == []
    service.fetch.return_value = [source(), source(title="second")]
    result = await col.collect("q", ["web"])
    assert (
        len(result.sources) == 1 and result.warnings[0].code == "source_limit_applied"
    )


@pytest.mark.asyncio
async def test_explicit_invalid_upstream_and_provider_failure(client):
    service = AsyncMock()
    service.fetch.side_effect = UpstreamDataError()
    assert (await collector(client, service).collect("q", ["web"])).outcomes[
        0
    ].status == "invalid"
    service.fetch.side_effect = ProviderError("secret")
    assert (await collector(client, service).collect("q", ["web"])).failed == ["web"]


@pytest.mark.asyncio
@pytest.mark.parametrize("wanted", ["web", b"web", [None], None])
async def test_wrong_source_argument_type_is_rejected(client, wanted):
    with pytest.raises(ValueError):
        await collector(client).collect("q", wanted)


@pytest.mark.asyncio
async def test_cache_lists_are_not_mutated_by_caller(client):
    cached = [source()]
    cache = AsyncMock()
    cache.get.return_value = cached
    result = await collector(client, cache=cache, cache_key=lambda n, q: n).collect(
        "q", ["web"]
    )
    result.sources.clear()
    result.outcomes[0].sources.clear()
    assert cached == [source()]


def test_outcome_invariants():
    with pytest.raises(ValueError):
        SourceOutcome(name="web", status="ok", sources=[])
    with pytest.raises(ValueError):
        SourceOutcome(name="web", status="ok", sources=[source("arxiv")])


@pytest.mark.asyncio
async def test_network_guard_negative_control(client):
    with pytest.raises(AssertionError, match="forbidden"):
        await client.get("https://example.invalid")
