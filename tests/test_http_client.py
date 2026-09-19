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

from researcher.services.http_client import (
    SOURCE_USER_AGENT,
    accept_for_host,
    source_client,
)


def test_shared_client_follows_redirects() -> None:
    client = source_client(timeout_seconds=5.0)
    try:
        assert client.follow_redirects is True
    finally:
        pass


def test_shared_client_sends_descriptive_user_agent() -> None:
    client = source_client(timeout_seconds=5.0)
    assert client.headers["User-Agent"] == SOURCE_USER_AGENT
    assert "http" in SOURCE_USER_AGENT  # contact URL, per Wikipedia UA policy


@pytest.mark.asyncio
async def test_arxiv_redirect_is_followed_end_to_end() -> None:
    """A 301 from the http:// arXiv URL must resolve to the https payload."""
    atom = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    )

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.scheme == "http":
            https_url = str(request.url).replace("http://", "https://", 1)
            return httpx.Response(301, headers={"Location": https_url})
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
