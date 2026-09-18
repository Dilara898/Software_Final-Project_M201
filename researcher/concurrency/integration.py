"""Adapters for A's storage and D's parsed sources; no settings or pool creation."""

from typing import Protocol

import asyncpg  # type: ignore[import-untyped]  # Driver ships no typing marker.
from pydantic import ValidationError

from ai.schemas import Source
from researcher.concurrency.contracts import Cache, StorageUnavailableError
from researcher.concurrency.models import SessionRecord, SourceName
from researcher.core.logic import select_sources
from researcher.models import ResearchSession
from researcher.storage.history import save_session


_CONNECTION_ERRORS = (
    asyncpg.PostgresConnectionError,
    asyncpg.CannotConnectNowError,
    asyncpg.TooManyConnectionsError,
    ConnectionError,
)


class HistoryPool(Protocol):
    async def fetchval(self, query: str, *args: object) -> int | None: ...


class StorageCacheAdapter:
    """Translate expected outages; preserve timeouts and programming errors."""

    def __init__(self, store: Cache) -> None:
        self._store = store

    async def get(self, key: str) -> list[Source] | None:
        try:
            return await self._store.get(key)
        except _CONNECTION_ERRORS as exc:
            raise StorageUnavailableError("Cache connection unavailable") from exc
        except ValidationError as exc:
            raise StorageUnavailableError("Cached source data is invalid") from exc

    async def set(self, key: str, sources: list[Source]) -> None:
        try:
            await self._store.set(key, sources)
        except _CONNECTION_ERRORS as exc:
            raise StorageUnavailableError("Cache connection unavailable") from exc


class SessionHistoryAdapter:
    """Map C's record to A's history function; the application owns the pool."""

    def __init__(self, pool: HistoryPool) -> None:
        self._pool = pool

    async def save(self, record: SessionRecord) -> int:
        session = ResearchSession(
            question=record.question,
            answer=record.answer,
            sources_used=list(record.sources_used),
            sources_failed=list(record.sources_failed),
            duration_ms=record.duration_ms,
        )
        try:
            session_id = await save_session(self._pool, session)
        except _CONNECTION_ERRORS as exc:
            raise StorageUnavailableError("History connection unavailable") from exc
        if type(session_id) is not int or session_id < 1:
            raise TypeError("History must return a positive integer session id")
        return session_id


def sources_from_cli(value: str | None) -> list[SourceName]:
    """Delegate CLI aliases/defaults and ValidationError to B's shared policy."""
    return select_sources(None if value is None else value.split(","))
