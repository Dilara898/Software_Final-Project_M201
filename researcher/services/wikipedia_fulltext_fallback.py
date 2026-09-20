"""Full-text Wikipedia search fallback for natural-language questions.

`ai/sources.py`'s `fetch_wikipedia` searches via the legacy MediaWiki
``action=opensearch`` endpoint, which is a *title-prefix* matcher, not a
full-text search. Verified directly, reproducibly, from multiple networks:

    "photosynthesis"                          -> ["Photosynthesis", ...]
    "What is photosynthesis?"                 -> []
    "What is photosynthesis and what are its main stages?" -> []

Both are genuine 200 OK responses; the endpoint simply cannot match a
sentence-shaped query against a page title. This is not a rate limit, a
User-Agent problem, or an environment-specific block -- earlier hypotheses
along those lines (recorded in this project's own incident notes) were
ruled out by direct reproduction before this fix was written. Since most
real research questions are phrased exactly like the failing examples
above, this affects the common case, not an edge case.

`ai/sources.py` cannot be modified (project constraint), so this fallback
lives entirely in the SE layer: when `AIFetchService` sees an empty,
non-error Wikipedia result, it retries the search via Wikipedia's modern
REST search endpoint (a genuine full-text search -- verified to correctly
match every example question in this project's own dataset), then reuses
the exact same summary endpoint and `Source` shape `ai.sources.fetch_wikipedia`
itself uses, so callers cannot tell the two search strategies apart.
"""

from __future__ import annotations

import httpx

from ai.schemas import Source

_REST_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


async def search_wikipedia_fulltext(
    query: str,
    *,
    max_results: int,
    client: httpx.AsyncClient,
) -> list[Source]:
    """Full-text Wikipedia search. Returns `[]` on any failure -- this is a
    fallback path; it must never raise past the caller, and a fallback that
    also finds nothing is simply a fallback that found nothing, not an error.
    """
    if not query.strip():
        return []

    try:
        r = await client.get(_REST_SEARCH_URL, params={"q": query, "limit": max_results})
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    pages = data.get("pages") if isinstance(data, dict) else None
    if not isinstance(pages, list):
        return []

    sources: list[Source] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        title = page.get("title")
        if not isinstance(title, str) or not title:
            continue

        try:
            summ = await client.get(_SUMMARY_URL.format(title=title.replace(" ", "_")))
            summ.raise_for_status()
            body = summ.json()
        except Exception:
            continue  # one bad title shouldn't kill the whole fallback

        extract = (body.get("extract") or "").strip()
        if not extract:
            continue
        url = (
            body.get("content_urls", {}).get("desktop", {}).get("page")
            or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        )
        sources.append(
            Source(title=body.get("title", title), url=url, snippet=extract, origin="wikipedia")
        )
    return sources
