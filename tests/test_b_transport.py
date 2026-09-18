"""Tests for researcher/services/transport.py (B-07) -- transport-level
retry so ai.sources.fetch_wikipedia's internal per-title summary fetch
(which swallows exceptions with a bare `except Exception: continue`) still
gets the benefit of retrying a transient HTTP status, without modifying
ai/sources.py itself.
"""

from __future__ import annotations

import httpx
import pytest

from ai.sources import fetch_wikipedia
from researcher.services.transport import wikipedia_retrying_client


def _search_response(titles: list[str]) -> httpx.Response:
    return httpx.Response(200, json=["q", titles, [], []])


def _summary_response(title: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "title": title,
            "extract": f"{title} is a thing.",
            "content_urls": {"desktop": {"page": f"https://en.wikipedia.org/wiki/{title}"}},
        },
    )


class TestWikipediaRetryingClient:
    @pytest.mark.asyncio
    async def test_summary_500_then_200_recovers(self):
        calls = {"summary": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if "opensearch" in str(request.url):
                return _search_response(["Python"])
            calls["summary"] += 1
            if calls["summary"] < 2:
                return httpx.Response(500, text="server error")
            return _summary_response("Python")

        base_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        wrapped = wikipedia_retrying_client(
            base_client, max_attempts=3, initial_wait_seconds=0.01, max_wait_seconds=0.01
        )
        try:
            result = await fetch_wikipedia("python", max_results=1, client=wrapped)
        finally:
            await wrapped.aclose()
            await base_client.aclose()

        assert len(result) == 1
        assert calls["summary"] == 2

    @pytest.mark.asyncio
    async def test_summary_404_is_not_retried(self):
        calls = {"summary": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if "opensearch" in str(request.url):
                return _search_response(["NoSuchPage"])
            calls["summary"] += 1
            return httpx.Response(404, text="not found")

        base_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        wrapped = wikipedia_retrying_client(
            base_client, max_attempts=3, initial_wait_seconds=0.01, max_wait_seconds=0.01
        )
        try:
            result = await fetch_wikipedia("nosuchpage", max_results=1, client=wrapped)
        finally:
            await wrapped.aclose()
            await base_client.aclose()

        # ai.sources.fetch_wikipedia's own except/continue means a
        # permanently-failing title is simply omitted, not an exception.
        assert result == []
        assert calls["summary"] == 1

    @pytest.mark.asyncio
    async def test_legitimately_empty_search_is_not_retried_into_something(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _search_response([])  # no matching titles at all

        base_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        wrapped = wikipedia_retrying_client(base_client)
        try:
            result = await fetch_wikipedia("asdkfjhaslkdfjh", max_results=3, client=wrapped)
        finally:
            await wrapped.aclose()
            await base_client.aclose()

        assert result == []

    @pytest.mark.asyncio
    async def test_wrapper_never_closes_the_wrapped_client(self):
        base_client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: _search_response([]))
        )
        wrapped = wikipedia_retrying_client(base_client)
        await wrapped.aclose()
        # The original client must still be usable after the wrapper closes.
        response = await base_client.get("https://en.wikipedia.org/w/api.php")
        assert response.status_code == 200
        await base_client.aclose()

    @pytest.mark.asyncio
    async def test_exhausted_transient_status_returns_final_response_as_is(self):
        calls = {"summary": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if "opensearch" in str(request.url):
                return _search_response(["AlwaysDown"])
            calls["summary"] += 1
            return httpx.Response(500, text="still down")

        base_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        wrapped = wikipedia_retrying_client(
            base_client, max_attempts=3, initial_wait_seconds=0.01, max_wait_seconds=0.01
        )
        try:
            result = await fetch_wikipedia("alwaysdown", max_results=1, client=wrapped)
        finally:
            await wrapped.aclose()
            await base_client.aclose()

        assert result == []  # exhausted -> fetch_wikipedia's continue applies
        assert calls["summary"] == 3  # exactly max_attempts, not more