"""Tests for researcher/services/retry.py's RateLimiter (B-05)."""

from __future__ import annotations

import asyncio

import pytest

from researcher.services.retry import RateLimiter


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_disabled_by_default_no_wait(self):
        limiter = RateLimiter(0.0)
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        limiter._sleep = fake_sleep  # type: ignore[attr-defined]
        await limiter.acquire()
        await limiter.acquire()
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_paces_calls_with_fake_clock(self):
        clock = {"t": 0.0}
        sleeps: list[float] = []

        def fake_now() -> float:
            return clock["t"]

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock["t"] += seconds

        limiter = RateLimiter(1.0, _now=fake_now, _sleep=fake_sleep)
        await limiter.acquire()  # t=0, first call never waits
        await limiter.acquire()  # should wait ~1.0
        await limiter.acquire()  # should wait ~1.0
        assert sleeps == pytest.approx([1.0, 1.0])
        assert clock["t"] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_no_wait_if_enough_time_already_elapsed(self):
        clock = {"t": 0.0}
        sleeps: list[float] = []

        def fake_now() -> float:
            return clock["t"]

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock["t"] += seconds

        limiter = RateLimiter(1.0, _now=fake_now, _sleep=fake_sleep)
        await limiter.acquire()
        clock["t"] += 5.0  # plenty of time passes externally
        await limiter.acquire()
        assert sleeps == []  # no wait needed, interval already satisfied

    @pytest.mark.asyncio
    async def test_wait_responds_to_cancellation(self):
        limiter = RateLimiter(10.0)  # long interval
        await limiter.acquire()  # first call never waits

        task = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_shared_instance_serializes_concurrent_acquires(self):
        clock = {"t": 0.0}

        def fake_now() -> float:
            return clock["t"]

        async def fake_sleep(seconds: float) -> None:
            clock["t"] += seconds

        limiter = RateLimiter(1.0, _now=fake_now, _sleep=fake_sleep)
        await asyncio.gather(limiter.acquire(), limiter.acquire(), limiter.acquire())
        # Three acquires 1.0s apart each (the lock serializes them), so the
        # fake clock should have advanced by 2.0s total (first is free).
        assert clock["t"] == pytest.approx(2.0)