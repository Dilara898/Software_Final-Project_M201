"""Real B services, supplied AI functions and C workflow over offline HTTP."""

from collections import Counter
from unittest.mock import AsyncMock

import httpx
import pytest

from ai.providers.base import LLMProvider
from researcher.cli import build_parser
from researcher.concurrency.integration import (
    SessionHistoryAdapter,
    StorageCacheAdapter,
    sources_from_cli,
)
from researcher.concurrency.orchestrator import SourceOrchestrator
from researcher.concurrency.research import NoSourcesError, ResearchOrchestrator
from researcher.services.ai_service import AIFetchService, AISynthesisService
from researcher.storage.cache_store import InMemoryCacheStore, cache_key


class OfflineLLM(LLMProvider):
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, *, json_schema=None, max_tokens=1024):
        self.calls += 1
        return "Supported by the supplied evidence [1]."


def http_handler(counts, *, arxiv_status=200, wiki_status=200):
    def handle(request):
        path = request.url.path
        counts[path] += 1
        if path == "/w/api.php":
            return httpx.Response(wiki_status, json=["q", ["Example"], [], []])
        if path.startswith("/api/rest_v1/page/summary/"):
            # Exercise B's transport retry beneath the real Wikipedia fetcher.
            if counts[path] == 1:
                return httpx.Response(500)
            return httpx.Response(200, json={
                "title": "Example", "extract": "Evidence about the question.",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Example"}},
            })
        assert path == "/api/query", f"Unexpected offline HTTP request: {path}"
        return httpx.Response(arxiv_status, text=(
            '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
            '<title>Paper</title><id>https://arxiv.org/abs/1234.5678</id>'
            '<summary>Evidence from a paper.</summary></entry></feed>'
        ))
    return handle


def fetch_service():
    return AIFetchService(
        max_attempts=2, initial_wait_seconds=0, max_wait_seconds=0,
        wikipedia_transport_initial_wait_seconds=0,
        wikipedia_transport_max_wait_seconds=0,
    )


@pytest.mark.asyncio
async def test_real_abcd_contracts_cache_bypass_citations_and_history():
    counts = Counter()
    llm = OfflineLLM()
    synthesis = AISynthesisService(llm_factories=[lambda: llm], max_attempts=1)
    pool = AsyncMock()
    pool.fetchval.return_value = 17
    args = build_parser().parse_args([
        "ask", "What is evidence?", "--sources", "arxiv,wiki,wiki",
    ])
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http_handler(counts))) as client:
            collector = SourceOrchestrator(
                fetch_service(), client=client,
                cache=StorageCacheAdapter(InMemoryCacheStore(60)), cache_key=cache_key,
                timeout_seconds=2, max_concurrency=3,
            )
            runner = ResearchOrchestrator(collector, synthesis, SessionHistoryAdapter(pool))
            wanted = sources_from_cli(args.sources)
            cold = await runner.research(args.question, wanted, use_cache=not args.no_cache)
            before_warm = counts.copy()
            warm = await runner.research(args.question, wanted)
            assert counts == before_warm
            bypass = build_parser().parse_args([
                "ask", args.question, "--sources", args.sources, "--no-cache",
            ])
            fresh = await runner.research(
                bypass.question, sources_from_cli(bypass.sources), use_cache=not bypass.no_cache,
            )
            assert not client.is_closed  # B's Wikipedia wrapper never owns this client.
    finally:
        synthesis.shutdown()
    assert cold.bundle.used == ["wikipedia", "arxiv"]
    assert cold.cache_stats.misses == 2 and warm.cache_stats.hits == 2
    assert fresh.cache_stats.bypasses == 2
    assert counts["/api/query"] == 2
    assert counts["/api/rest_v1/page/summary/Example"] == 3
    assert llm.calls == pool.fetchval.await_count == 3
    assert warm.session_id == 17 and warm.history_status == "saved"
    assert warm.answer.citations[0].source == warm.bundle.sources[0]


@pytest.mark.asyncio
async def test_real_b_permanent_source_failure_degrades_in_c():
    counts = Counter()
    synthesis = AISynthesisService(llm_factories=[OfflineLLM], max_attempts=1)
    pool = AsyncMock()
    pool.fetchval.return_value = 1
    try:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(http_handler(counts, arxiv_status=404)),
        ) as client:
            collector = SourceOrchestrator(fetch_service(), client=client, timeout_seconds=2, max_concurrency=2)
            result = await ResearchOrchestrator(
                collector, synthesis, SessionHistoryAdapter(pool),
            ).research("What is evidence?", ["wikipedia", "arxiv"], use_cache=False)
    finally:
        synthesis.shutdown()
    assert counts["/api/query"] == 1
    assert result.bundle.failed == ["arxiv"]
    assert result.bundle.used == ["wikipedia"]
    assert result.answer.citations and any(w.source == "arxiv" for w in result.warnings)


@pytest.mark.asyncio
async def test_all_sources_failed_keeps_c_diagnostics_without_starting_b_llm():
    counts = Counter()
    factories = []

    def factory():
        factories.append(1)
        return OfflineLLM()

    synthesis = AISynthesisService(llm_factories=[factory], max_attempts=1)
    pool = AsyncMock()
    try:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(http_handler(counts, wiki_status=404)),
        ) as client:
            collector = SourceOrchestrator(fetch_service(), client=client, timeout_seconds=2, max_concurrency=1)
            with pytest.raises(NoSourcesError) as error:
                await ResearchOrchestrator(
                    collector, synthesis, SessionHistoryAdapter(pool),
                ).research("What is evidence?", ["wikipedia"], use_cache=False)
    finally:
        synthesis.shutdown()
    assert error.value.collection.failed == ["wikipedia"]
    assert not factories
    pool.fetchval.assert_not_awaited()
