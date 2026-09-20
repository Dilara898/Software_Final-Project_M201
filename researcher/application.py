"""D-owned composition of A storage, B services and C research workflow."""
import asyncio
import logging
import re
import sys

import asyncpg

from researcher.concurrency.integration import (
    SessionHistoryAdapter,
    StorageCacheAdapter,
    sources_from_cli,
)
from researcher.concurrency.orchestrator import SourceOrchestrator
from researcher.concurrency.research import ResearchOrchestrator
from researcher.services.ai_service import AIFetchService, AISynthesisService
from researcher.services.http_client import source_client
from researcher.storage.cache_store import PostgresCacheStore, cache_key

log = logging.getLogger(__name__)


def clean(value: object) -> str:
    """Remove terminal control sequences while preserving normal text."""
    text = re.sub(
        r"\x1b\][\s\S]*?(?:\x07|\x1b\\)",
        "",
        str(value),
    )
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)

    return "".join(
        character
        for character in text
        if character in "\n\t"
        or (
            ord(character) >= 32
            and not 127 <= ord(character) <= 159
        )
    )


def render_answer(result) -> str:
    """Render the result without changing citation numbers."""
    answer = result.answer
    bundle = result.bundle
    stats = result.cache_stats

    lines = [
        f"Question: {answer.question}",
        "",
        answer.answer,
        "",
        "References:",
    ]

    for citation in sorted(
        answer.citations,
        key=lambda item: item.index,
    ):
        source = citation.source
        lines.extend(
            [
                f"[{citation.index}] ({source.origin}) {source.title}",
                source.url,
            ]
        )

    if bundle.failed:
        lines.append(
            "Unavailable sources: " + ", ".join(bundle.failed)
        )

    if bundle.empty:
        lines.append(
            "Sources with no results: " + ", ".join(bundle.empty)
        )

    for warning in result.warnings:
        lines.append(
            f"Warning: {warning.code}: {warning.message}"
        )

    lines.append(f"History: {result.history_status}")

    if result.history_status == "unknown":
        lines.append(
            "History write is unconfirmed; it may have committed. "
            "Do not retry automatically."
        )

    if result.session_id is not None:
        lines.append(f"Session: {result.session_id}")

    lines.append(
        f"{len(bundle.sources)} documents; "
        f"{len(bundle.used)} providers; "
        f"total {result.timings.total_ms / 1000:.2f} seconds; "
        f"answer ready "
        f"{result.timings.answer_ready_ms / 1000:.2f} seconds; "
        f"cache: {stats.hits} hits, "
        f"{stats.misses} misses, "
        f"{stats.bypasses} bypasses; "
        f"{stats.read_errors} read errors, "
        f"{stats.write_errors} write errors"
    )

    return clean("\n".join(lines))


async def build_result(args, settings, pool, client, fetch, synthesis):
    """Run the research workflow and return the raw `ResearchResult`.

    Shared by `execute_ask` (CLI) and `run_research` (UI) so both present
    the same underlying pipeline instead of two parallel implementations.
    """
    wanted = sources_from_cli(args.sources)

    cache = StorageCacheAdapter(
        PostgresCacheStore(
            pool,
            settings.cache_ttl_seconds,
        )
    )
    history = SessionHistoryAdapter(pool)

    collector = SourceOrchestrator(
        fetch,
        timeout_seconds=settings.per_source_timeout_seconds,
        max_concurrency=3,
        client=client,
        cache=cache,
        cache_key=cache_key,
        max_results_per_source=settings.max_sources_per_query,
    )

    workflow = ResearchOrchestrator(
        collector,
        synthesis,
        history,
        synthesis_timeout_seconds=55,
        research_timeout_seconds=(
            settings.per_source_timeout_seconds + 65
        ),
    )

    return await workflow.research(
        args.question,
        wanted,
        use_cache=not args.no_cache,
    )


async def execute_ask(args, settings, pool, client, fetch, synthesis):
    """Connect injected services using C's existing adapters."""
    result = await build_result(args, settings, pool, client, fetch, synthesis)
    print(render_answer(result))
    return 0


async def run_ask(args) -> int:
    """Own resources and release them on success, error or cancellation."""
    from researcher.config import get_settings
    from researcher.storage.db import close_pool, get_pool

    synthesis = None
    client = None

    try:
        settings = get_settings()
        pool = await asyncio.wait_for(get_pool(), timeout=10)

        synthesis = AISynthesisService()
        budget = settings.per_source_timeout_seconds

        # The source deadline bounds attempts, pacing and retry waits.
        fetch = AIFetchService(
            max_results=settings.max_sources_per_query,
            timeout_seconds=budget / 4,
            initial_wait_seconds=budget / 40,
            max_wait_seconds=budget / 20,
            min_interval_seconds=1.0,
        )

        client = source_client(timeout_seconds=budget)

        return await execute_ask(
            args,
            settings,
            pool,
            client,
            fetch,
            synthesis,
        )

    except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
        log.error(
            "Research storage/network failure (%s)",
            type(exc).__name__,
        )
        print(
            "Research failed: check database connectivity, "
            "tables and network access.",
            file=sys.stderr,
        )
        return 1

    finally:
        # Cleanup errors must not replace the original result or error.
        if synthesis is not None:
            try:
                synthesis.shutdown()
            except Exception as exc:
                log.warning(
                    "Synthesis cleanup failed (%s)",
                    type(exc).__name__,
                )

        if client is not None:
            try:
                await asyncio.wait_for(
                    client.aclose(),
                    timeout=5,
                )
            except Exception as exc:
                log.warning(
                    "HTTP cleanup failed (%s)",
                    type(exc).__name__,
                )

        try:
            await asyncio.wait_for(close_pool(), timeout=5)
        except Exception as exc:
            log.warning(
                "Database cleanup failed (%s)",
                type(exc).__name__,
            )


async def run_research(args):
    """Own resources and release them, returning the raw `ResearchResult`.

    Same resource lifecycle as `run_ask`, but for callers that render their
    own presentation instead of printing to stdout/stderr and returning a
    process exit code -- namely the Streamlit UI in `researcher/ui.py`.
    Every exception propagates uncaught so each caller maps it to its own
    presentation, exactly like `researcher.cli.main()` does around `run_ask`.
    """
    from researcher.config import get_settings
    from researcher.storage.db import close_pool, get_pool

    synthesis = None
    client = None

    try:
        settings = get_settings()
        pool = await asyncio.wait_for(get_pool(), timeout=10)

        synthesis = AISynthesisService()
        budget = settings.per_source_timeout_seconds

        fetch = AIFetchService(
            max_results=settings.max_sources_per_query,
            timeout_seconds=budget / 4,
            initial_wait_seconds=budget / 40,
            max_wait_seconds=budget / 20,
            min_interval_seconds=1.0,
        )

        # Same factory the CLI path uses. Building a client here by hand is
        # what caused the arXiv outage in artefacts/incident-arxiv-redirect.txt:
        # two near-identical snippets drifted apart. One factory, one behaviour.
        client = source_client(timeout_seconds=budget)

        return await build_result(args, settings, pool, client, fetch, synthesis)

    finally:
        if synthesis is not None:
            try:
                synthesis.shutdown()
            except Exception as exc:
                log.warning("Synthesis cleanup failed (%s)", type(exc).__name__)

        if client is not None:
            try:
                await asyncio.wait_for(client.aclose(), timeout=5)
            except Exception as exc:
                log.warning("HTTP cleanup failed (%s)", type(exc).__name__)

        try:
            await asyncio.wait_for(close_pool(), timeout=5)
        except Exception as exc:
            log.warning("Database cleanup failed (%s)", type(exc).__name__)
