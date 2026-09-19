"""Covers researcher/services/transport.py's exhausted-transport-exception
branch (a raised httpx.TransportError, as opposed to a transient HTTP
status), which test_b_transport.py's fetch_wikipedia-level tests never
observe directly since fetch_wikipedia swallows whatever finally escapes.
"""
import httpx
import pytest

from researcher.services.transport import _RetryingTransport


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
