import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

from ai.schemas import Source


class CacheStore(ABC):
    @abstractmethod
    async def get(self, key: str) -> list[Source] | None: ...

    @abstractmethod
    async def set(self, key: str, sources: list[Source]) -> None: ...


class InMemoryCacheStore(CacheStore):
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._data: dict[str, tuple[datetime, list[Source]]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key):
        async with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            expires_at, sources = hit
            if expires_at <= datetime.now(timezone.utc):
                self._data.pop(key, None)
                return None
            return sources

    async def set(self, key, sources):
        async with self._lock:
            expires = datetime.now(timezone.utc) + timedelta(seconds=self._ttl)
            self._data[key] = (expires, sources)


def cache_key(source: str, question: str) -> str:
    """(mənbə, normallaşdırılmış sual) → sabit açar."""
    norm = " ".join(question.lower().split()).rstrip("?.!")
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return f"{source}:{digest}"


class PostgresCacheStore(CacheStore):
    def __init__(self, pool, ttl_seconds: int) -> None:
        self._pool = pool
        self._ttl = ttl_seconds

    async def get(self, key):
        row = await self._pool.fetchrow(
            "SELECT payload, expires_at FROM cache_entries WHERE key = $1", key
        )
        if row is None:
            return None
        if row["expires_at"] <= datetime.now(timezone.utc):
            await self._pool.execute(
                "DELETE FROM cache_entries WHERE key = $1", key)
            return None
        return [Source(**item) for item in json.loads(row["payload"])]

    async def set(self, key, sources):
        payload = json.dumps([s.model_dump() for s in sources])
        expires = datetime.now(timezone.utc) + timedelta(seconds=self._ttl)
        await self._pool.execute(
            """INSERT INTO cache_entries (key, payload, expires_at)
               VALUES ($1, $2::jsonb, $3)
               ON CONFLICT (key) DO UPDATE
               SET payload = EXCLUDED.payload,
                   expires_at = EXCLUDED.expires_at""",
            key, payload, expires,
        )