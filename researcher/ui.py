"""Streamlit UI for the Async Research Assistant.

Reuses the exact same business logic the CLI uses -- `researcher.application
.run_research` calls the same `build_result` pipeline as `researcher.cli`'s
`ask` command (A's storage adapters, B's retrying AI services, C's
concurrent orchestrators) -- so this is a second presentation of one
pipeline, not a parallel implementation.

Run with:
    streamlit run researcher/ui.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from types import SimpleNamespace

import asyncpg
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError as SettingsError

# One-time process bootstrap, mirroring researcher.cli.main()'s own setup:
# existing process environment values take priority over .env, and empty
# optional values must not override a provider's own default.
load_dotenv(".env", override=False)
for _name in ("LLM_MODEL", "LLM_PROVIDER", "WEB_SEARCH_PROVIDER"):
    if _name in os.environ and not os.environ[_name].strip():
        os.environ.pop(_name)
os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("WEB_SEARCH_PROVIDER", "tavily")

from ai.providers.base import ProviderError  # noqa: E402
from researcher.application import run_research  # noqa: E402
from researcher.concurrency.models import ResearchResult  # noqa: E402
from researcher.concurrency.research import OrchestrationError  # noqa: E402
from researcher.exceptions import ResearcherError, ValidationError  # noqa: E402
from researcher.services.logging_setup import configure_logging  # noqa: E402
from researcher.storage.db import close_pool, get_pool  # noqa: E402
from researcher.storage.history import list_sessions  # noqa: E402
from researcher.validation import validate_limit, validate_question  # noqa: E402

configure_logging()
log = logging.getLogger(__name__)

st.set_page_config(page_title="Async Research Assistant", page_icon="\U0001f50e")


def _ask(question: str, sources: list[str], no_cache: bool) -> ResearchResult:
    args = SimpleNamespace(
        question=question,
        sources=",".join(sources) if sources else None,
        no_cache=no_cache,
    )
    return asyncio.run(run_research(args))


async def _fetch_history(limit: int):
    pool = await get_pool()
    try:
        return await list_sessions(pool, limit=limit)
    finally:
        await close_pool()


def render_result(result: ResearchResult) -> None:
    st.subheader("Answer")
    st.write(result.answer.answer)

    if result.answer.citations:
        st.subheader("References")
        for citation in sorted(result.answer.citations, key=lambda c: c.index):
            source = citation.source
            st.markdown(
                f"**[{citation.index}]** ({source.origin}) "
                f"[{source.title}]({source.url})"
            )

    bundle = result.bundle
    if bundle.failed:
        st.warning("Unavailable sources: " + ", ".join(bundle.failed))
    if bundle.empty:
        st.info("Sources with no results: " + ", ".join(bundle.empty))
    for warning in result.warnings:
        st.warning(f"{warning.code}: {warning.message}")

    stats = result.cache_stats
    cols = st.columns(4)
    cols[0].metric("Documents", len(bundle.sources))
    cols[1].metric("Providers used", len(bundle.used))
    cols[2].metric("Total time (s)", f"{result.timings.total_ms / 1000:.2f}")
    cols[3].metric("Cache hits", stats.hits)

    caption = f"History: {result.history_status}"
    if result.session_id is not None:
        caption += f" (session #{result.session_id})"
    st.caption(caption)
    if result.history_status == "unknown":
        st.info(
            "History write is unconfirmed; it may have committed. "
            "Do not retry automatically."
        )


def render_error(exc: Exception) -> None:
    # Never surface str(exc) directly: provider/DB errors can embed secrets
    # (API keys, connection strings). Only the exception type is logged,
    # matching researcher.cli.main()'s own discipline.
    log.error("UI request failed (%s)", type(exc).__name__)

    if isinstance(exc, ValidationError):
        st.error(str(exc))
    elif isinstance(exc, (ResearcherError, OrchestrationError, ProviderError)):
        st.error(
            "Research failed. Check provider settings, available sources "
            "and connectivity."
        )
    elif isinstance(exc, SettingsError):
        st.error("Invalid configuration. Check DATABASE_URL and .env settings.")
    elif isinstance(exc, ModuleNotFoundError):
        st.error(f"Missing dependency: {exc.name}. Install the project requirements.")
    elif isinstance(exc, (asyncpg.PostgresError, OSError, TimeoutError)):
        st.error("Research failed: check database connectivity, tables and network access.")
    else:
        st.error("Unexpected error. Check the application logs.")


def ask_tab() -> None:
    st.subheader("Ask a research question")
    question = st.text_input("Question", placeholder="What is photosynthesis?")
    sources = st.multiselect(
        "Sources",
        options=["wiki", "arxiv", "web"],
        default=["wiki", "arxiv", "web"],
        help="Same aliases the CLI accepts: wiki, arxiv, web.",
    )
    no_cache = st.checkbox("Skip cache", value=False)

    if not st.button("Ask", type="primary", disabled=not question.strip()):
        return

    try:
        clean_question = validate_question(question)
    except ValidationError as exc:
        render_error(exc)
        return

    with st.spinner("Researching..."):
        try:
            result = _ask(clean_question, sources, no_cache)
        except Exception as exc:  # noqa: BLE001 - single, logged UI error boundary
            render_error(exc)
            return

    render_result(result)


def history_tab() -> None:
    st.subheader("Past queries")
    limit = st.number_input("Show last N", min_value=1, max_value=100, value=10, step=1)

    if not st.button("Refresh"):
        return

    try:
        clean_limit = validate_limit(int(limit))
        sessions = asyncio.run(_fetch_history(clean_limit))
    except Exception as exc:  # noqa: BLE001 - single, logged UI error boundary
        render_error(exc)
        return

    if not sessions:
        st.info("No previous queries found.")
        return

    for session in sessions:
        with st.expander(f"#{session.id} · {session.created_at} -- {session.question}"):
            st.write(session.answer)


def main() -> None:
    st.title("Async Research Assistant")
    st.caption("Wikipedia + arXiv + web search, synthesized into one cited answer.")

    tab_ask, tab_history = st.tabs(["Ask", "History"])
    with tab_ask:
        ask_tab()
    with tab_history:
        history_tab()


main()
