import pytest
from researcher.storage.cache_store import InMemoryCacheStore, cache_key


def test_key_is_normalized():
    assert cache_key("wikipedia", "What is X?") == \
        cache_key("wikipedia", " what is x ")


def test_key_differs_per_source():
    assert cache_key("wikipedia", "q") != cache_key("arxiv", "q")


@pytest.mark.asyncio
async def test_roundtrip(sample_source):
    store = InMemoryCacheStore(ttl_seconds=60)
    await store.set("k", [sample_source])
    got = await store.get("k")
    assert got[0].url == sample_source.url


@pytest.mark.asyncio
async def test_expired_returns_none(sample_source):
    store = InMemoryCacheStore(ttl_seconds=0)
    await store.set("k", [sample_source])
    assert await store.get("k") is None