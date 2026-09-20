"""Regression tests for the shared source HTTP client (arXiv 301/406 path).

`ai/sources.py` requests arXiv over ``http://export.arxiv.org/api/query``.
arXiv answers that with a 301 to the https URL, and `fetch_arxiv` calls
`raise_for_status()`, which in httpx raises on a redirect too. A client
built without ``follow_redirects=True`` therefore turns every arXiv call
into a ProviderError before a single byte of Atom is parsed.

`ai/` is supplied code we must not modify, so the guarantee has to be
enforced where we build the client instead.
"""

from __future__ import annotations

import httpx
import pytest

from ai.sources import fetch_arxiv
from researcher.concurrency.orchestrator import SourceOrchestrator
from researcher.services.ai_service import AIFetchService
from researcher.services.http_client import (
    SOURCE_USER_AGENT,
    accept_for_host,
    source_client,
)


@pytest.mark.asyncio
async def test_shared_client_follows_redirects() -> None:
    async with source_client(timeout_seconds=5.0) as client:
        assert client.follow_redirects is True


@pytest.mark.asyncio
async def test_shared_client_sends_descriptive_user_agent() -> None:
    async with source_client(timeout_seconds=5.0) as client:
        assert client.headers["User-Agent"] == SOURCE_USER_AGENT
        assert "http" in SOURCE_USER_AGENT


@pytest.mark.asyncio
async def test_arxiv_redirect_is_followed_end_to_end() -> None:
    """Even after HTTPS upgrade, a server redirect must reach the payload."""
    atom = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    )

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/api/query":
            return httpx.Response(301, headers={"Location": "/api/result"})
        return httpx.Response(200, text=atom)

    client = source_client(timeout_seconds=5.0, transport=httpx.MockTransport(handler))
    async with client:
        response = await client.get("http://export.arxiv.org/api/query")

    assert response.status_code == 200
    assert response.text == atom
    assert len(seen) == 2  # original request plus the followed redirect


@pytest.mark.asyncio
async def test_arxiv_request_declares_an_atom_accept_header() -> None:
    """406 Not Acceptable is a content-negotiation refusal; state what we take."""
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["accept"] = request.headers["Accept"]
        return httpx.Response(200, text="<feed/>")

    client = source_client(timeout_seconds=5.0, transport=httpx.MockTransport(handler))
    async with client:
        await client.get("https://export.arxiv.org/api/query")

    assert "application/atom+xml" in captured["accept"]
    assert "*/*" in captured["accept"]  # never a hard refusal of other types


@pytest.mark.asyncio
async def test_wikipedia_request_still_accepts_json() -> None:
    """The client is shared, so arXiv's Accept must not break Wikipedia's JSON."""
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["accept"] = request.headers["Accept"]
        return httpx.Response(200, json={})

    client = source_client(timeout_seconds=5.0, transport=httpx.MockTransport(handler))
    async with client:
        await client.get("https://en.wikipedia.org/api/rest_v1/page/summary/Foo")

    assert "application/json" in captured["accept"]


def test_unknown_host_keeps_the_httpx_default() -> None:
    assert accept_for_host("example.invalid") is None


@pytest.mark.asyncio
async def test_real_arxiv_fetcher_uses_https_and_parses_evidence() -> None:
    seen = []

    def handler(request):
        seen.append(request)
        # Simulate an intermediary rejecting the plaintext hop.
        if request.url.scheme == "http":
            return httpx.Response(406)
        assert request.url.params["search_query"] == "all:quantum computing"
        assert request.url.params["max_results"] == "2"
        assert "application/atom+xml" in request.headers["Accept"]
        assert request.headers["User-Agent"] == SOURCE_USER_AGENT
        return httpx.Response(200, text=(
            '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
            '<title>Quantum paper</title><id>https://arxiv.org/abs/1234.5678</id>'
            '<summary>Research evidence.</summary></entry></feed>'
        ))

    async with source_client(timeout_seconds=2, transport=httpx.MockTransport(handler)) as client:
        sources = await fetch_arxiv("quantum computing", max_results=2, client=client)
        assert not client.is_closed
    assert len(seen) == 1
    assert len(sources) == 1 and sources[0].origin == "arxiv"
    assert sources[0].title == "Quantum paper"


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "http://example.invalid/api/query?a=1",
    "http://export.arxiv.org/other?a=1",
    "http://export.arxiv.org:8080/api/query?a=1",
    "https://export.arxiv.org/api/query?a=1",
])
async def test_https_upgrade_is_scoped_to_the_public_arxiv_endpoint(url) -> None:
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200)

    async with source_client(timeout_seconds=2, transport=httpx.MockTransport(handler)) as client:
        await client.get(url)
    assert seen == [url]


@pytest.mark.asyncio
async def test_persistent_arxiv_406_is_not_retried_and_other_sources_survive(caplog) -> None:
    arxiv_calls = []

    def handler(request):
        if request.url.host == "export.arxiv.org":
            arxiv_calls.append(request)
            return httpx.Response(406, text="private diagnostic body")
        if request.url.path == "/w/api.php":
            return httpx.Response(200, json=["q", ["Example"], [], []])
        return httpx.Response(200, json={
            "title": "Example", "extract": "Valid evidence.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Example"}},
        })

    service = AIFetchService(max_attempts=3, initial_wait_seconds=0, max_wait_seconds=0)
    async with source_client(timeout_seconds=2, transport=httpx.MockTransport(handler)) as client:
        result = await SourceOrchestrator(
            service, client=client, timeout_seconds=2, max_concurrency=2,
        ).collect("q", ["wikipedia", "arxiv"], use_cache=False)
        assert not client.is_closed
    assert len(arxiv_calls) == 1
    assert result.used == ["wikipedia"] and result.failed == ["arxiv"]
    assert result.outcomes[1].status == "error"  # Never report this as empty success.
    assert "private diagnostic body" not in caplog.text
