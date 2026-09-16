"""Explicit ports for A/B integration. No SQL, provider SDKs or env reads."""

from collections.abc import Awaitable, Callable
from typing import Protocol

import httpx

from ai.schemas import AnswerWithCitations, Source
from researcher.concurrency.models import SessionRecord


class FetchService(Protocol):
    async def fetch(
        self, source: str, query: str, client: httpx.AsyncClient
    ) -> list[Source]: ...


class SynthesisService(Protocol):
    async def synthesize(
        self, question: str, sources: list[Source]
    ) -> AnswerWithCitations: ...


class Cache(Protocol):
    async def get(self, key: str) -> list[Source] | None: ...

    async def set(self, key: str, sources: list[Source]) -> None: ...


class HistoryStore(Protocol):
    async def save(self, record: SessionRecord) -> int: ...


class StorageUnavailableError(Exception):
    """A's adapter maps expected storage connectivity failures to this port error."""


class UpstreamDataError(Exception):
    """B may raise this when a provider returned unusable data."""


class HistoryCallbackAdapter:
    """Connect A's save function using a typed callback, without importing A."""

    def __init__(self, save: Callable[[SessionRecord], Awaitable[int]]) -> None:
        self._save = save

    async def save(self, record: SessionRecord) -> int:
        return await self._save(record)
