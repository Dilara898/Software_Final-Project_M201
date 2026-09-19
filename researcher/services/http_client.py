"""The one place a live source `httpx.AsyncClient` is constructed.

`ai/sources.py` is supplied code we do not modify, and it makes two
assumptions the SE layer has to satisfy from the outside:

1. It requests arXiv over ``http://export.arxiv.org/api/query``. arXiv
   answers with a 301 to the https URL, and `fetch_arxiv` then calls
   `raise_for_status()` -- which in httpx raises on a redirect as well as
   on 4xx/5xx. A client built without ``follow_redirects=True`` turns
   every arXiv call into ``ProviderError: arXiv query failed`` before any
   Atom is parsed. Three separate call sites used to build their own
   client and two of them omitted the flag, so the live benchmark and the
   orchestrator's own fallback client could never reach arXiv at all.

2. It issues `client.get(...)` itself, so the only way to negotiate
   content per source is to reach the request after it is built. httpx
   otherwise sends ``Accept: */*``, which some intermediaries answer with
   406 Not Acceptable. Declaring what each host actually serves -- Atom
   for arXiv, JSON for Wikipedia -- removes that failure mode; every value
   keeps a ``*/*`` fallback so we never turn a servable response into a
   refusal ourselves.

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
    """Request event hook: declare what this host is known to serve."""
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
