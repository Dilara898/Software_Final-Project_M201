"""Bounded collection with independent cache, queue and source deadlines."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable, Sequence
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from ai.providers.base import ProviderError
from ai.schemas import Source
from researcher.concurrency.contracts import (
    Cache,
    FetchService,
    StorageUnavailableError,
    UpstreamDataError,
)
from researcher.concurrency.models import (
    SOURCE_NAMES,
    CacheStatus,
    CollectionResult,
    OperationWarning,
    SourceName,
    SourceOutcome,
    SourceStatus,
    SourceTimings,
)

log = logging.getLogger(__name__)


def positive_seconds(value: float, name: str) -> float:
    """Validate constructor values even when used without A's Settings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def selected_sources(wanted: Sequence[str]) -> list[SourceName]:
    if isinstance(wanted, (str, bytes)) or not isinstance(wanted, Sequence):
        raise ValueError("sources must be a sequence of canonical names")
    if any(not isinstance(name, str) or name not in SOURCE_NAMES for name in wanted):
        raise ValueError("sources must be wikipedia, arxiv or web")
    names = [name for name in SOURCE_NAMES if name in wanted]
    if not names:
        raise ValueError("at least one source must be selected")
    return names


def valid_source(source: Source, name: str) -> bool:
    try:
        url = urlsplit(source.url)
        # urlsplit defers malformed/out-of-range port validation until access.
        _ = url.port
        return (
            source.origin == name
            and bool(source.title.strip())
            and bool(source.snippet.strip())
            and len(source.snippet) <= 20_000
            and url.scheme in ("http", "https")
            and bool(url.hostname)
            and not any(c.isspace() for c in source.url)
        )
    except ValueError:
        return False


class SourceOrchestrator:
    """Reuse within one event loop; callers own an injected HTTP client.

    If no client is supplied, collect owns a short-lived client for convenience.
    Retry and provider rate limits belong to the injected service, not this class.
    Expected storage outages use StorageUnavailableError; programming errors
    propagate. Cache failures never erase a successful fetch.
    """

    def __init__(
        self,
        service: FetchService,
        *,
        timeout_seconds: float,
        max_concurrency: int,
        client: httpx.AsyncClient | None = None,
        cache: Cache | None = None,
        cache_key: Callable[[str, str], str] | None = None,
        queue_timeout_seconds: float = 10,
        cache_timeout_seconds: float = 0.5,
        max_results_per_source: int = 3,
    ) -> None:
        self._timeout = positive_seconds(timeout_seconds, "timeout_seconds")
        self._queue_timeout = positive_seconds(
            queue_timeout_seconds, "queue_timeout_seconds"
        )
        self._cache_timeout = positive_seconds(
            cache_timeout_seconds, "cache_timeout_seconds"
        )
        self._semaphore = asyncio.Semaphore(
            positive_integer(max_concurrency, "max_concurrency")
        )
        self._max_results = positive_integer(
            max_results_per_source, "max_results_per_source"
        )
        if cache is not None and cache_key is None:
            raise ValueError("cache_key is required when a cache is supplied")
        self._service, self._client = service, client
        self._cache, self._cache_key = cache, cache_key

    @staticmethod
    def _warning(code: str, stage: str, name: SourceName) -> OperationWarning:
        # Never include exception strings: SDK errors can contain keys or URLs.
        return OperationWarning(
            code=code, stage=stage, source=name, message=code.replace("_", " ")
        )

    async def _one(
        self,
        name: SourceName,
        question: str,
        client: httpx.AsyncClient,
        use_cache: bool,
        request_id: str,
    ) -> SourceOutcome:
        started = time.perf_counter()
        measurements: dict[str, float] = {}
        warnings: list[OperationWarning] = []
        cache_status: CacheStatus = "bypass"
        sources: list[Source] = []
        status: SourceStatus = "empty"
        error_code: str | None = None

        def finish() -> SourceOutcome:
            timings = SourceTimings(
                **measurements, total_ms=(time.perf_counter() - started) * 1000
            )
            for warning in warnings:
                log.warning(
                    warning.code,
                    extra={
                        "request_id": request_id,
                        "source": name,
                        "stage": warning.stage,
                        "error_code": warning.code,
                    },
                )
            if error_code is not None:
                log.warning(
                    error_code,
                    extra={
                        "request_id": request_id,
                        "source": name,
                        "stage": "collection",
                        "status": status,
                        "error_code": error_code,
                    },
                )
            log.info(
                "source_completed",
                extra={
                    "request_id": request_id,
                    "source": name,
                    "status": status,
                    "cache_status": cache_status,
                    "duration_ms": timings.total_ms,
                    "result_count": len(sources),
                    "error_code": error_code,
                },
            )
            return SourceOutcome(
                name=name,
                sources=list(sources),
                status=status,
                cache_status=cache_status,
                warnings=warnings,
                timings=timings,
                error_code=error_code,
            )

        key = None
        if use_cache and self._cache is not None and self._cache_key is not None:
            key = self._cache_key(name, question)
            cache_status = "miss"
            stage_started = time.perf_counter()
            try:
                async with asyncio.timeout(self._cache_timeout):
                    hit = await self._cache.get(key)
                if hit is not None:
                    if (
                        isinstance(hit, list)
                        and hit
                        and all(
                            isinstance(s, Source) and valid_source(s, name) for s in hit
                        )
                    ):
                        if len(hit) > self._max_results:
                            warnings.append(
                                self._warning(
                                    "source_limit_applied", "cache_read", name
                                )
                            )
                        sources, status, cache_status = (
                            list(hit[: self._max_results]),
                            "ok",
                            "hit",
                        )
                    else:
                        cache_status = "read_error"
                        warnings.append(
                            self._warning("cache_entry_invalid", "cache_read", name)
                        )
            except (TimeoutError, StorageUnavailableError, ConnectionError) as exc:
                cache_status = "read_error"
                code = (
                    "cache_read_timeout"
                    if isinstance(exc, TimeoutError)
                    else "cache_read_failed"
                )
                warnings.append(self._warning(code, "cache_read", name))
            finally:
                measurements["cache_read_ms"] = (
                    time.perf_counter() - stage_started
                ) * 1000
            if cache_status == "hit":
                return finish()

        queue_started = time.perf_counter()
        try:
            async with asyncio.timeout(self._queue_timeout):
                await self._semaphore.acquire()
        except TimeoutError:
            measurements["queue_ms"] = (time.perf_counter() - queue_started) * 1000
            status, error_code = "queue_timeout", "source_queue_timeout"
            return finish()
        measurements["queue_ms"] = (time.perf_counter() - queue_started) * 1000
        fetch_started = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout):
                found = await self._service.fetch(name, question, client)
            if not isinstance(found, list) or any(
                not isinstance(s, Source) for s in found
            ):
                raise TypeError("FetchService.fetch must return list[Source]")
            accepted = [s for s in found if valid_source(s, name)]
            if len(accepted) != len(found):
                warnings.append(self._warning("source_invalid_items", "fetch", name))
            if len(accepted) > self._max_results:
                warnings.append(self._warning("source_limit_applied", "fetch", name))
            sources = accepted[: self._max_results]
            status = "ok" if sources else ("invalid" if found else "empty")
            if status == "invalid":
                error_code = "source_invalid_data"
        except TimeoutError:
            status, error_code = "timeout", "source_timeout"
        except UpstreamDataError:
            status, error_code = "invalid", "source_invalid_data"
        except (ProviderError, httpx.HTTPError):
            status, error_code = "error", "source_unavailable"
        finally:
            measurements["fetch_ms"] = (time.perf_counter() - fetch_started) * 1000
            self._semaphore.release()

        if sources and key is not None and self._cache is not None:
            stage_started = time.perf_counter()
            try:
                async with asyncio.timeout(self._cache_timeout):
                    await self._cache.set(key, list(sources))
            except (TimeoutError, StorageUnavailableError, ConnectionError) as exc:
                code = (
                    "cache_write_timeout"
                    if isinstance(exc, TimeoutError)
                    else "cache_write_failed"
                )
                warnings.append(self._warning(code, "cache_write", name))
            finally:
                measurements["cache_write_ms"] = (
                    time.perf_counter() - stage_started
                ) * 1000
        return finish()

    async def collect(
        self,
        question: str,
        wanted: Sequence[str] = SOURCE_NAMES,
        *,
        use_cache: bool = True,
        concurrent: bool = True,
        request_id: str | None = None,
    ) -> CollectionResult:
        """Return every selected outcome; empty results are not provider failures."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        names = selected_sources(wanted)
        request_id = request_id or uuid4().hex
        started = time.perf_counter()

        async def run(client: httpx.AsyncClient) -> CollectionResult:
            if client.is_closed:
                raise RuntimeError("the injected client is closed")
            if concurrent:
                tasks = [
                    asyncio.create_task(
                        self._one(name, question, client, use_cache, request_id),
                        name=f"source:{request_id}:{name}",
                    )
                    for name in names
                ]
                try:
                    outcomes = await asyncio.gather(*tasks)
                finally:
                    # Await cleanup on success, errors and caller cancellation.
                    for task in tasks:
                        # gather already forwards caller cancellation. Do not
                        # interrupt an existing asynchronous finally block.
                        if not task.done() and task.cancelling() == 0:
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
            else:
                outcomes = [
                    await self._one(name, question, client, use_cache, request_id)
                    for name in names
                ]
            return CollectionResult(
                request_id=request_id,
                outcomes=outcomes,
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        if self._client is not None:
            return await run(self._client)
        async with httpx.AsyncClient(timeout=self._timeout) as owned_client:
            return await run(owned_client)
