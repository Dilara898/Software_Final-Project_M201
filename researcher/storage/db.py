import asyncio

import asyncpg

from researcher.config import get_settings

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:  # kilid daxilində ikinci yoxlama
                settings = get_settings()
                _pool = await asyncpg.create_pool(
                    settings.database_url, min_size=1, max_size=5
                )
    return _pool


async def close_pool() -> None:
    global _pool
    async with _pool_lock:
        if _pool is not None:
            await _pool.close()
            _pool = None