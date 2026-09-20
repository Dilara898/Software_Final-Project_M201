"""Unit tests for the Wikipedia full-text search fallback, in isolation.

See tests/test_b_ai_service.py::TestWikipediaFulltextFallbackIntegration
for the same fix exercised through AIFetchService.fetch.
"""

from __future__ import annotations

import httpx
import pytest

from researcher.services.wikipedia_fulltext_fallback import search_wikipedia_fulltext


def _search_response(pages: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"pages": pages})


def _summary_response(title: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "title": title,
            "extract": f"{title} is a thing.",
            "content_urls": {"desktop": {"page": f"https://en.wikipedia.org/wiki/{title}"}},
        },
    )


@pytest.mark.asyncio
async def test_empty_query_returns_empty_without_a_network_call():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"pages": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await search_wikipedia_fulltext("   ", max_results=3, client=client)

    assert result == []
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_finds_and_summarizes_matching_pages():
    def handler(request: httpx.Request) -> httpx.Response:
        if "search/page" in str(request.url):
            return _search_response([{"title": "Photosynthesis"}, {"title": "Cell (biology)"}])
        return _summary_response(request.url.path.rsplit("/", 1)[-1])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await search_wikipedia_fulltext(
            "What is photosynthesis?", max_results=3, client=client
        )

    assert [s.title for s in result] == ["Photosynthesis", "Cell_(biology)"]
    assert all(s.origin == "wikipedia" for s in result)


@pytest.mark.asyncio
async def test_malformed_page_entries_are_skipped_not_fatal():
    def handler(request: httpx.Request) -> httpx.Response:
        if "search/page" in str(request.url):
            return _search_response([{"no_title_field": True}, {"title": "Real Page"}])
        return _summary_response("Real_Page")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await search_wikipedia_fulltext("q", max_results=3, client=client)

    assert len(result) == 1
    assert result[0].title == "Real_Page"


@pytest.mark.asyncio
async def test_a_bad_summary_does_not_kill_the_whole_fallback():
    def handler(request: httpx.Request) -> httpx.Response:
        if "search/page" in str(request.url):
            return _search_response([{"title": "Broken"}, {"title": "Fine"}])
        if request.url.path.endswith("/Broken"):
            return httpx.Response(500, text="server error")
        return _summary_response("Fine")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await search_wikipedia_fulltext("q", max_results=3, client=client)

    assert len(result) == 1
    assert result[0].title == "Fine"


@pytest.mark.asyncio
async def test_no_extract_is_skipped():
    def handler(request: httpx.Request) -> httpx.Response:
        if "search/page" in str(request.url):
            return _search_response([{"title": "Empty"}])
        return httpx.Response(200, json={"title": "Empty", "extract": ""})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await search_wikipedia_fulltext("q", max_results=3, client=client)

    assert result == []


@pytest.mark.asyncio
async def test_search_http_error_returns_empty_not_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await search_wikipedia_fulltext("q", max_results=3, client=client)

    assert result == []


@pytest.mark.asyncio
async def test_malformed_search_body_returns_empty_not_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "a", "dict"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await search_wikipedia_fulltext("q", max_results=3, client=client)

    assert result == []
