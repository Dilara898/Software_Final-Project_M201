"""Coverage for researcher.application.run_research (the UI's resource
lifecycle wrapper, added alongside researcher/ui.py) plus the previously
untested cleanup-failure branches of run_ask that share the same shape.
"""
from types import SimpleNamespace

import asyncpg
import httpx
import pytest
from unittest.mock import AsyncMock

from researcher.application import run_ask, run_research
from researcher.exceptions import NoSourcesError
from researcher.services.ai_service import AISynthesisService


def _fake_settings() -> SimpleNamespace:
    return SimpleNamespace(
        per_source_timeout_seconds=2,
        cache_ttl_seconds=60,
        max_sources_per_query=3,
    )


def _args() -> SimpleNamespace:
    return SimpleNamespace(question="Q?", sources=None, no_cache=False)


def _wire_happy_path(monkeypatch, *, build_result):
    monkeypatch.setattr("researcher.config.get_settings", _fake_settings)
    monkeypatch.setattr(
        "researcher.storage.db.get_pool", AsyncMock(return_value=object())
    )
    monkeypatch.setattr("researcher.storage.db.close_pool", AsyncMock())
    monkeypatch.setattr("researcher.application.build_result", build_result)


# ---- run_research: happy path and error propagation ----

@pytest.mark.asyncio
async def test_run_research_returns_build_result(monkeypatch):
    sentinel = object()

    async def fake_build_result(*a, **k):
        return sentinel

    _wire_happy_path(monkeypatch, build_result=fake_build_result)

    assert await run_research(_args()) is sentinel


@pytest.mark.asyncio
async def test_run_research_propagates_storage_failure(monkeypatch):
    monkeypatch.setattr("researcher.config.get_settings", _fake_settings)
    monkeypatch.setattr(
        "researcher.storage.db.get_pool",
        AsyncMock(side_effect=asyncpg.PostgresError("down")),
    )
    monkeypatch.setattr("researcher.storage.db.close_pool", AsyncMock())

    with pytest.raises(asyncpg.PostgresError):
        await run_research(_args())


@pytest.mark.asyncio
async def test_run_research_propagates_researcher_error(monkeypatch):
    async def fake_build_result(*a, **k):
        raise NoSourcesError("no usable sources")

    _wire_happy_path(monkeypatch, build_result=fake_build_result)

    with pytest.raises(NoSourcesError):
        await run_research(_args())


# ---- run_research: cleanup failures must not hide a good result ----

@pytest.mark.asyncio
async def test_run_research_db_cleanup_failure_does_not_replace_result(monkeypatch):
    sentinel = object()

    async def fake_build_result(*a, **k):
        return sentinel

    monkeypatch.setattr("researcher.config.get_settings", _fake_settings)
    monkeypatch.setattr(
        "researcher.storage.db.get_pool", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(
        "researcher.storage.db.close_pool",
        AsyncMock(side_effect=RuntimeError("cleanup boom")),
    )
    monkeypatch.setattr("researcher.application.build_result", fake_build_result)

    assert await run_research(_args()) is sentinel


@pytest.mark.asyncio
async def test_run_research_synthesis_shutdown_failure_does_not_replace_result(
    monkeypatch,
):
    sentinel = object()

    async def fake_build_result(*a, **k):
        return sentinel

    def boom_shutdown(self):
        raise RuntimeError("shutdown boom")

    _wire_happy_path(monkeypatch, build_result=fake_build_result)
    monkeypatch.setattr(AISynthesisService, "shutdown", boom_shutdown)

    assert await run_research(_args()) is sentinel


@pytest.mark.asyncio
async def test_run_research_http_cleanup_failure_does_not_replace_result(monkeypatch):
    sentinel = object()

    async def fake_build_result(*a, **k):
        return sentinel

    async def boom_aclose(self):
        raise RuntimeError("aclose boom")

    _wire_happy_path(monkeypatch, build_result=fake_build_result)
    monkeypatch.setattr(httpx.AsyncClient, "aclose", boom_aclose)

    assert await run_research(_args()) is sentinel


# ---- run_ask shares the exact same cleanup shape; these two branches
# (synthesis.shutdown() and client.aclose() raising) were untested before. ----

@pytest.mark.asyncio
async def test_run_ask_synthesis_shutdown_failure_does_not_replace_result(monkeypatch):
    def boom_shutdown(self):
        raise RuntimeError("shutdown boom")

    async def fake_execute_ask(*a, **k):
        return 0

    monkeypatch.setattr("researcher.config.get_settings", _fake_settings)
    monkeypatch.setattr(
        "researcher.storage.db.get_pool", AsyncMock(return_value=object())
    )
    monkeypatch.setattr("researcher.storage.db.close_pool", AsyncMock())
    monkeypatch.setattr("researcher.application.execute_ask", fake_execute_ask)
    monkeypatch.setattr(AISynthesisService, "shutdown", boom_shutdown)

    assert await run_ask(_args()) == 0


@pytest.mark.asyncio
async def test_run_ask_http_cleanup_failure_does_not_replace_result(monkeypatch):
    async def boom_aclose(self):
        raise RuntimeError("aclose boom")

    async def fake_execute_ask(*a, **k):
        return 0

    monkeypatch.setattr("researcher.config.get_settings", _fake_settings)
    monkeypatch.setattr(
        "researcher.storage.db.get_pool", AsyncMock(return_value=object())
    )
    monkeypatch.setattr("researcher.storage.db.close_pool", AsyncMock())
    monkeypatch.setattr("researcher.application.execute_ask", fake_execute_ask)
    monkeypatch.setattr(httpx.AsyncClient, "aclose", boom_aclose)

    assert await run_ask(_args()) == 0
