"""Offline compatibility tests against A's real storage implementations."""

import asyncio
from unittest.mock import AsyncMock

import asyncpg
import pytest

from researcher.concurrency.contracts import StorageUnavailableError
from researcher.concurrency.integration import (
    SessionHistoryAdapter,
    StorageCacheAdapter,
    sources_from_cli,
)
from researcher.concurrency.models import SessionRecord
from researcher.concurrency.orchestrator import SourceOrchestrator
from researcher.concurrency.research import ResearchOrchestrator
from researcher.exceptions import ValidationError
from researcher.storage.cache_store import (
    InMemoryCacheStore,
    PostgresCacheStore,
    cache_key,
)
from scripts.c_offline import OfflineService, offline_client


def record():
    return SessionRecord(
        request_id="request",
        question="q",
        answer="a",
        sources_used=["web"],
        sources_failed=[],
        duration_ms=12,
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, ["wikipedia", "arxiv", "web"]),
        ("wiki,web", ["wikipedia", "web"]),
        ("web, Wiki,web", ["wikipedia", "web"]),
        ("search,wikipedia,internet", ["wikipedia", "web"]),
    ],
)
def test_d_cli_names_map_to_c(value, expected):
    assert sources_from_cli(value) == expected


@pytest.mark.parametrize("value", ["", "wiki,", "unknown", "wiki,,web"])
def test_invalid_cli_names_rejected(value):
    with pytest.raises(ValidationError):
        sources_from_cli(value)


@pytest.mark.asyncio
async def test_real_a_memory_cache_and_history_round_trip():
    service = OfflineService()
    pool = AsyncMock()
    pool.fetchval.return_value = 7
    cache = StorageCacheAdapter(InMemoryCacheStore(ttl_seconds=60))
    async with offline_client() as client:
        collector = SourceOrchestrator(
            service,
            client=client,
            cache=cache,
            cache_key=cache_key,
            timeout_seconds=1,
            max_concurrency=3,
        )
        worker = ResearchOrchestrator(collector, service, SessionHistoryAdapter(pool))
        cold = await worker.research("What is X?", sources_from_cli("wiki,web"))
        warm = await worker.research(" what is x ", sources_from_cli("wiki,web"))
    assert cold.cache_stats.misses == 2 and warm.cache_stats.hits == 2
    assert warm.history_status == "saved" and warm.session_id == 7
    sql, *args = pool.fetchval.await_args.args
    assert "INSERT INTO research_sessions" in sql
    assert args == [
        " what is x ",
        warm.answer.answer,
        ["wikipedia", "web"],
        [],
        int(warm.timings.answer_ready_ms),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        asyncpg.ConnectionDoesNotExistError("private"),
        asyncpg.CannotConnectNowError("private"),
        asyncpg.TooManyConnectionsError("private"),
    ],
)
async def test_real_a_postgres_outages_preserve_answer(error):
    pool = AsyncMock()
    pool.fetchrow.side_effect = error
    pool.execute.side_effect = error
    pool.fetchval.side_effect = error
    cache = StorageCacheAdapter(PostgresCacheStore(pool, ttl_seconds=60))
    service = OfflineService()
    async with offline_client() as client:
        collector = SourceOrchestrator(
            service,
            client=client,
            cache=cache,
            cache_key=cache_key,
            timeout_seconds=1,
            max_concurrency=3,
        )
        result = await ResearchOrchestrator(
            collector, service, SessionHistoryAdapter(pool)
        ).research("q", ["web"])
    assert result.bundle.used == ["web"] and result.history_status == "failed"
    assert result.cache_stats.read_errors == 1 and result.cache_stats.write_errors == 1
    assert "private" not in str(result.warnings)
    pool.fetchval.assert_awaited_once()


@pytest.mark.asyncio
async def test_history_timeout_remains_timeout_and_is_not_retried():
    pool = AsyncMock()
    pool.fetchval.side_effect = TimeoutError()
    with pytest.raises(TimeoutError):
        await SessionHistoryAdapter(pool).save(record())
    pool.fetchval.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [asyncpg.UndefinedTableError("schema missing"), TypeError("bug")]
)
async def test_programming_and_schema_errors_propagate(error):
    pool = AsyncMock()
    pool.fetchrow.side_effect = error
    with pytest.raises(type(error)):
        await StorageCacheAdapter(PostgresCacheStore(pool, ttl_seconds=60)).get("key")
    pool.fetchval.side_effect = error
    with pytest.raises(type(error)):
        await SessionHistoryAdapter(pool).save(record())


@pytest.mark.asyncio
async def test_invalid_cached_source_becomes_cache_read_error():
    from datetime import datetime, timedelta, timezone

    pool = AsyncMock()
    pool.fetchrow.return_value = {
        "payload": '[{"title":"bad"}]',
        "expires_at": datetime.now(timezone.utc) + timedelta(seconds=60),
    }
    with pytest.raises(StorageUnavailableError):
        await StorageCacheAdapter(PostgresCacheStore(pool, ttl_seconds=60)).get("key")


@pytest.mark.asyncio
async def test_cancelled_history_propagates():
    pool = AsyncMock()
    pool.fetchval.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await SessionHistoryAdapter(pool).save(record())
