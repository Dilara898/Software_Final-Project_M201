"""Covers researcher/storage/db.py's get_pool/close_pool directly. Existing
tests only ever monkeypatch these two functions wholesale (as a fetch/
history port dependency); nothing exercised the pool cache/lock logic
itself before this file.
"""
import pytest

import researcher.storage.db as db_module


class _FakePool:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_pool_state(monkeypatch):
    # db_module's `_pool` is process-global state; isolate each test from it.
    monkeypatch.setattr(db_module, "_pool", None)


@pytest.mark.asyncio
async def test_get_pool_creates_once_and_caches(monkeypatch):
    fake_pool = _FakePool()
    dsns_seen: list[str] = []

    async def fake_create_pool(dsn, **kwargs):
        dsns_seen.append(dsn)
        return fake_pool

    monkeypatch.setattr(db_module.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(
        db_module,
        "get_settings",
        lambda: type("S", (), {"database_url": "postgresql://fake/db"})(),
    )

    first = await db_module.get_pool()
    second = await db_module.get_pool()

    assert first is fake_pool
    assert second is fake_pool
    assert dsns_seen == ["postgresql://fake/db"]  # created exactly once


@pytest.mark.asyncio
async def test_close_pool_closes_and_clears_cache(monkeypatch):
    fake_pool = _FakePool()
    monkeypatch.setattr(db_module, "_pool", fake_pool)

    await db_module.close_pool()

    assert fake_pool.closed is True
    assert db_module._pool is None


@pytest.mark.asyncio
async def test_close_pool_is_a_noop_when_nothing_to_close():
    await db_module.close_pool()  # must not raise

    assert db_module._pool is None
