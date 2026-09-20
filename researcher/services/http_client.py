"""The one place a live source `httpx.AsyncClient` is constructed.

`ai/sources.py` is supplied code we do not modify, and it makes two
assumptions the SE layer has to satisfy from the outside:

1. It requests arXiv over ``http://export.arxiv.org/api/query``. arXiv
   may answer with a redirect to HTTPS, and `fetch_arxiv` then calls
   `raise_for_status()` -- which in httpx raises on a redirect as well as
   on 4xx/5xx. A client built without ``follow_redirects=True`` turns
   a redirected arXiv call into ``ProviderError: arXiv query failed`` before any
   Atom is parsed. Three separate call sites used to build their own
   client and two of them omitted the flag, so the live benchmark and the
   orchestrator's own fallback client could never reach arXiv at all.

   The request hook now upgrades this exact public API endpoint before
   sending it, avoiding an unnecessary HTTP hop. Other redirects remain
   supported; no supplied ai/ code is modified.

2. It issues `client.get(...)` itself, so one way to negotiate
   content per source is to reach the request after it is built. httpx
   otherwise sends ``Accept: */*``. Declaring Atom for arXiv and JSON for
   Wikipedia makes the expected representation explicit. This does NOT
   establish the cause of a live 406 or guarantee its resolution: proxies
   and upstream policy can also reject requests. A persistent 406 must
   remain visible as a source failure, not be converted into empty success.

Because one client is shared across all three fetchers, Accept cannot be a
client-level default: arXiv's Atom would then be sent to Wikipedia's JSON
endpoint too. It is applied per request, keyed on host, via a request
event hook. A hook rather than a custom transport is deliberate -- it
leaves the `transport=` slot free for the tests that inject
`httpx.MockTransport`, and it still runs when `transport.py`'s Wikipedia
retry wrapper re-sends a request through this client.
"""

from __future__ import annotations

import httpx

#: Wikipedia's API etiquette asks for a descriptive agent with a contact URL;
#: arXiv asks the same. Sent on every source request.
SOURCE_USER_AGENT = (
    "AsyncResearchAssistant/1.0 "
    "(AI-ENG-110 student project; contact: "
    "https://github.com/Dilara898/Software_Final-Project_M201)"
)

#: Host -> Accept. Every value ends in ``*/*`` so a server that prefers a
#: different representation may still serve one.
_ACCEPT_BY_HOST = {
    "export.arxiv.org": (
        "application/atom+xml, application/xml;q=0.9, text/xml;q=0.9, */*;q=0.8"
    ),
    "en.wikipedia.org": "application/json, */*;q=0.8",
}


def accept_for_host(host: str) -> str | None:
    """Accept header for `host`, or None to leave httpx's default alone."""
    return _ACCEPT_BY_HOST.get(host.lower())


async def set_source_accept_header(request: httpx.Request) -> None:
    """Use HTTPS for the supplied arXiv API URL and set host-specific Accept."""
    if (
        request.url.scheme == "http"
        and request.url.host == "export.arxiv.org"
        and request.url.path == "/api/query"
        and request.url.port in (None, 80)
    ):
        request.url = request.url.copy_with(scheme="https", port=None)
    accept = accept_for_host(request.url.host)
    if accept is not None:
        request.headers["Accept"] = accept


def source_client(
    *,
    timeout_seconds: float,
    **client_kwargs: object,
) -> httpx.AsyncClient:
    """Build the shared client handed to `ai.fetch_*`.

    Extra keyword arguments go straight to `httpx.AsyncClient`, which is
    how tests inject an `httpx.MockTransport`. The caller owns the
    returned client and must close it.
    """
    return httpx.AsyncClient(
        timeout=timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": SOURCE_USER_AGENT},
        event_hooks={"request": [set_source_accept_header]},
        **client_kwargs,  # type: ignore[arg-type]
    )
