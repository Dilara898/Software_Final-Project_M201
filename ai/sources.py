"""Async source fetchers for the research pipeline.

Three coroutines, all returning `list[Source]`:

  fetch_wikipedia(query)  — Wikipedia REST API (no key)
  fetch_arxiv(query)      — arXiv public Atom API (no key)
  fetch_web(query)        — pluggable web search (Tavily / Serper / DuckDuckGo)

Design notes
------------
- We use `httpx.AsyncClient` because it gives us a single HTTP library that
  works async out of the box. The import is lazy so `import ai` works even
  if httpx is not installed (e.g. running smoke tests offline).

- All three coroutines accept an optional `client` parameter. In production,
  the SE layer should pass a single shared `httpx.AsyncClient` to amortize
  connection setup. In tests, students can pass a fake.

- The web search provider abstraction follows the same pattern as the LLM/VLM
  providers in `ai.providers`. Set `WEB_SEARCH_PROVIDER` env var to pick.
"""

from __future__ import annotations

import abc
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

from ai.providers.base import ProviderError
from ai.schemas import Source


# ---------------------------------------------------------------------------
# httpx is imported lazily so the package stays importable without it.
# ---------------------------------------------------------------------------

def _require_httpx():
    try:
        import httpx  # type: ignore
    except ImportError as e:
        raise ProviderError(
            "The `httpx` package is required for live source fetching. "
            "Install with `pip install httpx`."
        ) from e
    return httpx


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

_WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


async def fetch_wikipedia(
    query: str,
    *,
    max_results: int = 3,
    client: Any = None,
    timeout: float = 10.0,
) -> list[Source]:
    """Search Wikipedia and return the top-N article summaries.

    No API key required.
    """
    if not query.strip():
        return []
    httpx = _require_httpx()

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout)

    try:
        # Step 1: search for matching titles.
        try:
            r = await client.get(
                _WIKI_SEARCH_URL,
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": max_results,
                    "namespace": 0,
                    "format": "json",
                },
            )
            r.raise_for_status()
        except Exception as e:  # pragma: no cover - network path
            raise ProviderError(f"Wikipedia search failed: {e}") from e

        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            return []
        titles = data[1]

        # Step 2: pull summary for each title.
        sources: list[Source] = []
        for title in titles:
            try:
                summ = await client.get(
                    _WIKI_SUMMARY_URL.format(title=title.replace(" ", "_"))
                )
                summ.raise_for_status()
            except Exception:
                continue  # one bad title shouldn't kill the whole fetch
            body = summ.json()
            extract = (body.get("extract") or "").strip()
            if not extract:
                continue
            url = (
                body.get("content_urls", {}).get("desktop", {}).get("page")
                or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            )
            sources.append(Source(
                title=body.get("title", title),
                url=url,
                snippet=extract,
                origin="wikipedia",
            ))
        return sources
    finally:
        if own_client:
            await client.aclose()


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

_ARXIV_URL = "http://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


async def fetch_arxiv(
    query: str,
    *,
    max_results: int = 3,
    client: Any = None,
    timeout: float = 10.0,
) -> list[Source]:
    """Search arXiv and return the top-N matching paper abstracts.

    No API key required. Returns an empty list rather than raising on a
    well-formed but empty response.
    """
    if not query.strip():
        return []
    httpx = _require_httpx()

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        try:
            r = await client.get(
                _ARXIV_URL,
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
            )
            r.raise_for_status()
        except Exception as e:  # pragma: no cover - network path
            raise ProviderError(f"arXiv query failed: {e}") from e

        return _parse_arxiv_atom(r.text)
    finally:
        if own_client:
            await client.aclose()


def _parse_arxiv_atom(xml_text: str) -> list[Source]:
    """Parse arXiv's Atom-format response into `Source` objects."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ProviderError(f"arXiv: malformed XML response: {e}")
    out: list[Source] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        title_el = entry.find(f"{_ATOM_NS}title")
        summary_el = entry.find(f"{_ATOM_NS}summary")
        id_el = entry.find(f"{_ATOM_NS}id")
        if title_el is None or summary_el is None or id_el is None:
            continue
        title = re.sub(r"\s+", " ", (title_el.text or "").strip())
        snippet = re.sub(r"\s+", " ", (summary_el.text or "").strip())
        url = (id_el.text or "").strip()
        if not (title and snippet and url):
            continue
        out.append(Source(title=title, url=url, snippet=snippet, origin="arxiv"))
    return out


# ---------------------------------------------------------------------------
# Web search — pluggable provider
# ---------------------------------------------------------------------------

class WebSearchProvider(abc.ABC):
    """Contract for web search providers used by `fetch_web`."""

    @abc.abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: int = 3,
        client: Any = None,
    ) -> list[Source]:
        raise NotImplementedError


class TavilyProvider(WebSearchProvider):
    """Tavily search API. Free tier: 1000 req/month. Sign up: tavily.com"""

    URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None, *, timeout: float = 10.0) -> None:
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self._api_key:
            raise ProviderError("TAVILY_API_KEY is not set.")
        self._timeout = timeout

    async def search(
        self,
        query: str,
        *,
        max_results: int = 3,
        client: Any = None,
    ) -> list[Source]:
        httpx = _require_httpx()
        own = client is None
        if own:
            client = httpx.AsyncClient(timeout=self._timeout)
        try:
            try:
                r = await client.post(
                    self.URL,
                    json={
                        "api_key": self._api_key,
                        "query": query,
                        "max_results": max_results,
                    },
                )
                r.raise_for_status()
            except Exception as e:  # pragma: no cover - network path
                raise ProviderError(f"Tavily search failed: {e}") from e
            body = r.json()
            return [
                Source(
                    title=item.get("title", "(untitled)"),
                    url=item.get("url", ""),
                    snippet=item.get("content", "") or item.get("snippet", ""),
                    origin="web",
                )
                for item in (body.get("results") or [])
                if item.get("url")
            ]
        finally:
            if own:
                await client.aclose()


class SerperProvider(WebSearchProvider):
    """Serper.dev (Google search proxy). Free tier: 2500 queries. Sign up: serper.dev"""

    URL = "https://google.serper.dev/search"

    def __init__(self, api_key: str | None = None, *, timeout: float = 10.0) -> None:
        self._api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self._api_key:
            raise ProviderError("SERPER_API_KEY is not set.")
        self._timeout = timeout

    async def search(
        self,
        query: str,
        *,
        max_results: int = 3,
        client: Any = None,
    ) -> list[Source]:
        httpx = _require_httpx()
        own = client is None
        if own:
            client = httpx.AsyncClient(timeout=self._timeout)
        try:
            try:
                r = await client.post(
                    self.URL,
                    headers={"X-API-KEY": self._api_key},
                    json={"q": query, "num": max_results},
                )
                r.raise_for_status()
            except Exception as e:  # pragma: no cover - network path
                raise ProviderError(f"Serper search failed: {e}") from e
            body = r.json()
            return [
                Source(
                    title=item.get("title", "(untitled)"),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    origin="web",
                )
                for item in (body.get("organic") or [])[:max_results]
                if item.get("link")
            ]
        finally:
            if own:
                await client.aclose()


class DuckDuckGoProvider(WebSearchProvider):
    """DuckDuckGo search via the `duckduckgo-search` package. No API key.

    This provider runs the (sync) `duckduckgo-search` library inside a thread
    so it presents the same async interface as the others.
    """

    def __init__(self) -> None:
        try:
            import duckduckgo_search  # type: ignore  # noqa: F401
        except ImportError as e:
            raise ProviderError(
                "The `duckduckgo-search` package is required for DuckDuckGoProvider. "
                "Install with `pip install duckduckgo-search`."
            ) from e

    async def search(
        self,
        query: str,
        *,
        max_results: int = 3,
        client: Any = None,
    ) -> list[Source]:
        # `client` is unused — this provider doesn't speak HTTP directly.
        import asyncio
        from duckduckgo_search import DDGS  # type: ignore

        def _run() -> list[Source]:
            results: list[Source] = []
            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=max_results):
                    if not item.get("href"):
                        continue
                    results.append(Source(
                        title=item.get("title", "(untitled)"),
                        url=item["href"],
                        snippet=item.get("body", ""),
                        origin="web",
                    ))
            return results

        return await asyncio.to_thread(_run)


def get_web_search_provider() -> WebSearchProvider:
    """Factory: select the configured web-search provider.

    Reads `WEB_SEARCH_PROVIDER` env var. Default: tavily.
    """
    name = os.getenv("WEB_SEARCH_PROVIDER", "tavily").lower().strip()
    if name == "tavily":
        return TavilyProvider()
    if name == "serper":
        return SerperProvider()
    if name in ("duckduckgo", "ddg"):
        return DuckDuckGoProvider()
    raise ProviderError(
        f"Unknown WEB_SEARCH_PROVIDER={name!r}. "
        "Expected tavily | serper | duckduckgo."
    )


async def fetch_web(
    query: str,
    *,
    max_results: int = 3,
    provider: WebSearchProvider | None = None,
    client: Any = None,
) -> list[Source]:
    """Fetch web search results via the configured provider."""
    if not query.strip():
        return []
    provider = provider or get_web_search_provider()
    return await provider.search(query, max_results=max_results, client=client)
