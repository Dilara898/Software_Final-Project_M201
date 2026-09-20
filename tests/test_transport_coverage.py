"""Covers researcher/services/transport.py's exhausted-transport-exception
branch (a raised httpx.TransportError, as opposed to a transient HTTP
status), which test_b_transport.py's fetch_wikipedia-level tests never
observe directly since fetch_wikipedia swallows whatever finally escapes.
"""
import logging

import httpx
import pytest

from researcher.services.transport import _RetryingTransport, wikipedia_retrying_client


class _AlwaysFailsClient:
    def __init__(self) -> None:
        self.calls = 0

    async def send(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        raise httpx.ConnectError("boom", request=request)


@pytest.mark.asyncio
async def test_transport_error_exhausts_retries_then_raises():
    inner = _AlwaysFailsClient()
    transport = _RetryingTransport(
        inner, max_attempts=3, initial_wait_seconds=0.01, max_wait_seconds=0.01
    )
    request = httpx.Request("GET", "https://en.wikipedia.org/api/rest_v1/page/summary/X")

    with pytest.raises(httpx.ConnectError):
        await transport.handle_async_request(request)

    assert inner.calls == 3


@pytest.mark.asyncio
async def test_permanent_rejection_is_logged_not_swallowed_silently(caplog):
    """A 403 on a per-title summary must leave a trace.

    `ai.fetch_wikipedia` catches a failed summary request with a bare
    `except Exception: continue`, so a rejection never reaches the caller --
    the source simply returns fewer, or zero, results and is indistinguishable
    from a genuinely empty search. Wikimedia returns 403 to requests it
    considers policy-violating, which is exactly how a deployment can report
    "fetch succeeded, 0 results" forever. The transport is the last layer that
    still sees the status code.
    """
    inner = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(403, text="nope"))
    )
    wrapper = wikipedia_retrying_client(inner, max_attempts=2)
    with caplog.at_level(logging.WARNING, logger="researcher.services.transport"):
        async with wrapper:
            response = await wrapper.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/Gravity"
            )
    await inner.aclose()

    assert response.status_code == 403  # passed through unchanged
    records = [r for r in caplog.records if r.message == "wikipedia_request_rejected"]
    assert len(records) == 1
    assert records[0].status_code == 403
    assert records[0].path.endswith("/Gravity")
